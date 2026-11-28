"""Strictly causal temporal graph feature engineering.

Every feature computed for a transaction at `tx.tx_timestamp` uses ONLY:
1. Graph edges with `edge.timestamp < tx.tx_timestamp` (no same-timestamp or future edges).
2. Fraud labels with `label.label_available_at <= tx.tx_timestamp` and `label.tx_timestamp < tx.tx_timestamp`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from graph_fraud_engine.config import FeatureConfig
from graph_fraud_engine.graph import TemporalEntityGraph
from graph_fraud_engine.schemas import TemporalLeakageError, TransactionRecord

HIGH_RISK_MCCS = {"gift_cards", "crypto_onramp", "digital_goods", "electronics"}

FEATURE_NAMES: List[str] = [
    "log_amount_usd",
    "is_high_risk_mcc",
    "is_emulator_or_kiosk",
    "account_prior_tx_24h",
    "account_prior_tx_7d",
    "account_prior_tx_30d",
    "account_amount_zscore",
    "device_prior_tx_24h",
    "device_prior_distinct_accounts_30d",
    "device_distinct_accounts_24h",
    "device_high_risk_mcc_ratio_30d",
    "device_matured_fraud_count",
    "device_matured_fraud_rate",
    "merchant_matured_fraud_rate",
    "two_hop_ring_account_count",
    "two_hop_ring_matured_fraud_count",
    "shared_merchant_with_co_accounts_30d",
    "device_tenure_days",
    "account_unseen_cold_start",
    "device_unseen_cold_start",
]

# Tabular-only subset (no multi-hop graph features) for clean baseline comparison
TABULAR_ONLY_FEATURE_NAMES: List[str] = [
    "log_amount_usd",
    "is_high_risk_mcc",
    "is_emulator_or_kiosk",
    "account_prior_tx_24h",
    "account_prior_tx_7d",
    "account_prior_tx_30d",
    "account_amount_zscore",
    "merchant_matured_fraud_rate",
    "account_unseen_cold_start",
    "device_unseen_cold_start",
]


@dataclass(frozen=True)
class FeatureExtractionResult:
    tx_id: str
    as_of: datetime
    features: Dict[str, float]
    max_edge_timestamp_used: Optional[datetime]
    max_label_available_at_used: Optional[datetime]
    cold_start_flags: Dict[str, bool]

    def to_vector(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        names = feature_names or FEATURE_NAMES
        return np.array([self.features[name] for name in names], dtype=np.float64)


def extract_transaction_features(
    graph: TemporalEntityGraph,
    tx: TransactionRecord,
    feature_config: Optional[FeatureConfig] = None,
) -> FeatureExtractionResult:
    """Compute strictly leak-free temporal & graph features for a single transaction."""
    cfg = feature_config or FeatureConfig()
    as_of = tx.tx_timestamp
    short_w = timedelta(hours=cfg.short_window_hours)
    med_w = timedelta(days=cfg.medium_window_days)
    long_w = timedelta(days=cfg.long_window_days)

    max_edge_ts: Optional[datetime] = None
    max_label_ts: Optional[datetime] = None

    def _track_edges(edges) -> None:
        nonlocal max_edge_ts
        for e in edges:
            if e.timestamp >= as_of:
                raise TemporalLeakageError(
                    f"Future edge {e.tx_id} ({e.timestamp}) leaked into {tx.tx_id} ({as_of})"
                )
            if max_edge_ts is None or e.timestamp > max_edge_ts:
                max_edge_ts = e.timestamp

    def _track_labels(labels) -> None:
        nonlocal max_label_ts
        for l in labels:
            if l.label_available_at > as_of or l.tx_timestamp >= as_of:
                raise TemporalLeakageError(
                    f"Immature/future label {l.tx_id} (avail={l.label_available_at}) leaked into {tx.tx_id} ({as_of})"
                )
            if max_label_ts is None or l.label_available_at > max_label_ts:
                max_label_ts = l.label_available_at

    # Entity metadata (static attributes known at entity onboarding, with cold-start defaults)
    merchant = graph.merchants.get(tx.merchant_id)
    device = graph.devices.get(tx.device_id)
    is_high_risk_mcc = 1.0 if (merchant is not None and merchant.category in HIGH_RISK_MCCS) else 0.0
    if device is None:
        is_emulator_or_kiosk = 0.0
    elif device.device_type == "emulator":
        is_emulator_or_kiosk = 1.0
    elif device.device_type == "kiosk":
        is_emulator_or_kiosk = 0.25
    else:
        is_emulator_or_kiosk = 0.0

    # Cold-start flags (unseen prior to as_of)
    acc_unseen = graph.is_unseen_prior_to(tx.account_id, as_of)
    dev_unseen = graph.is_unseen_prior_to(tx.device_id, as_of)
    mer_unseen = graph.is_unseen_prior_to(tx.merchant_id, as_of)

    # Account prior history
    acc_edges_30d = graph.prior_account_tx_edges(
        tx.account_id, as_of=as_of, window=long_w, exclude_tx_id=tx.tx_id
    )
    _track_edges(acc_edges_30d)
    acc_edges_7d = [e for e in acc_edges_30d if e.timestamp >= as_of - med_w]
    acc_edges_24h = [e for e in acc_edges_7d if e.timestamp >= as_of - short_w]

    if acc_edges_30d:
        amounts = np.array([e.amount_usd for e in acc_edges_30d], dtype=np.float64)
        mean_amt = float(np.mean(amounts))
        std_amt = float(np.std(amounts))
        account_amount_zscore = float(np.clip((tx.amount_usd - mean_amt) / (std_amt + 10.0), -5.0, 10.0))
    else:
        account_amount_zscore = 0.0

    # Device prior history
    dev_edges_all = graph.prior_edges(
        tx.device_id, as_of=as_of, relation="USES_DEVICE", exclude_tx_id=tx.tx_id
    )
    _track_edges(dev_edges_all)
    dev_edges_30d = [e for e in dev_edges_all if e.timestamp >= as_of - long_w]
    dev_edges_24h = [e for e in dev_edges_30d if e.timestamp >= as_of - short_w]

    distinct_other_accs_30d = {e.src for e in dev_edges_30d if e.src != tx.account_id}
    distinct_other_accs_24h = {e.src for e in dev_edges_24h if e.src != tx.account_id}

    # Device merchant mix in 30d (distinguishes legitimate household devices from fraud ring devices!)
    dev_mer_edges_30d = graph.prior_edges(
        tx.device_id, as_of=as_of, relation="OBSERVED_AT", window=long_w, exclude_tx_id=tx.tx_id
    )
    _track_edges(dev_mer_edges_30d)
    if dev_mer_edges_30d:
        hr_count = sum(
            1
            for e in dev_mer_edges_30d
            if (m := graph.merchants.get(e.dst)) is not None and m.category in HIGH_RISK_MCCS
        )
        device_hr_ratio = hr_count / float(len(dev_mer_edges_30d))
    else:
        device_hr_ratio = 0.0

    if dev_edges_all:
        earliest_dev_ts = min(e.timestamp for e in dev_edges_all)
        raw_tenure_days = max(0.0, (as_of - earliest_dev_ts).total_seconds() / 86400.0)
        # Cap tenure at long_window_days so the feature remains temporally stationary across splits.
        device_tenure_days = min(float(cfg.long_window_days), raw_tenure_days)
    else:
        device_tenure_days = 0.0

    # Matured fraud signals on device and merchant (strictly label_available_at <= as_of, 30d window)
    long_min_ts = as_of - long_w
    dev_matured_labels = [
        l
        for l in graph.matured_labels_for_entity(tx.device_id, as_of=as_of, exclude_tx_id=tx.tx_id)
        if l.tx_timestamp >= long_min_ts
    ]
    _track_labels(dev_matured_labels)
    dev_matured_fraud = sum(1 for l in dev_matured_labels if l.is_fraud == 1)
    dev_matured_rate = (dev_matured_fraud + 0.05) / (len(dev_matured_labels) + 5.0)

    mer_matured_labels = [
        l
        for l in graph.matured_labels_for_entity(tx.merchant_id, as_of=as_of, exclude_tx_id=tx.tx_id)
        if l.tx_timestamp >= long_min_ts
    ]
    _track_labels(mer_matured_labels)
    mer_matured_fraud = sum(1 for l in mer_matured_labels if l.is_fraud == 1)
    mer_matured_rate = (mer_matured_fraud + 0.1) / (len(mer_matured_labels) + 10.0)

    # 2-hop coordinated ring features (strictly < as_of)
    co_accounts, _ = graph.compute_two_hop_account_ring(
        account_id=tx.account_id,
        device_id=tx.device_id,
        as_of=as_of,
        window=long_w,
        exclude_tx_id=tx.tx_id,
    )
    ring_matured_fraud_txs = set()
    shared_merchant_co_accounts = 0
    for co_acc in co_accounts:
        co_labels = [
            l
            for l in graph.matured_labels_for_entity(co_acc, as_of=as_of, exclude_tx_id=tx.tx_id)
            if l.tx_timestamp >= long_min_ts
        ]
        _track_labels(co_labels)
        for l in co_labels:
            if l.is_fraud == 1:
                ring_matured_fraud_txs.add(l.tx_id)

        co_mer_neighbors = graph.prior_neighbors(
            co_acc, as_of=as_of, relation="TRANSACTS_AT", window=long_w, exclude_tx_id=tx.tx_id
        )
        if tx.merchant_id in co_mer_neighbors:
            shared_merchant_co_accounts += 1

    features: Dict[str, float] = {
        "log_amount_usd": math.log1p(tx.amount_usd),
        "is_high_risk_mcc": is_high_risk_mcc,
        "is_emulator_or_kiosk": is_emulator_or_kiosk,
        "account_prior_tx_24h": float(len(acc_edges_24h)),
        "account_prior_tx_7d": float(len(acc_edges_7d)),
        "account_prior_tx_30d": float(len(acc_edges_30d)),
        "account_amount_zscore": account_amount_zscore,
        "device_prior_tx_24h": float(len(dev_edges_24h)),
        "device_prior_distinct_accounts_30d": float(len(distinct_other_accs_30d)),
        "device_distinct_accounts_24h": float(len(distinct_other_accs_24h)),
        "device_high_risk_mcc_ratio_30d": float(device_hr_ratio),
        "device_matured_fraud_count": float(dev_matured_fraud),
        "device_matured_fraud_rate": float(dev_matured_rate),
        "merchant_matured_fraud_rate": float(mer_matured_rate),
        "two_hop_ring_account_count": float(len(co_accounts)),
        "two_hop_ring_matured_fraud_count": float(len(ring_matured_fraud_txs)),
        "shared_merchant_with_co_accounts_30d": float(shared_merchant_co_accounts),
        "device_tenure_days": float(device_tenure_days),
        "account_unseen_cold_start": 1.0 if acc_unseen else 0.0,
        "device_unseen_cold_start": 1.0 if dev_unseen else 0.0,
    }

    return FeatureExtractionResult(
        tx_id=tx.tx_id,
        as_of=as_of,
        features=features,
        max_edge_timestamp_used=max_edge_ts,
        max_label_available_at_used=max_label_ts,
        cold_start_flags={
            "account_unseen": acc_unseen,
            "device_unseen": dev_unseen,
            "merchant_unseen": mer_unseen,
        },
    )


def extract_feature_matrix(
    graph: TemporalEntityGraph,
    transactions: List[TransactionRecord],
    feature_config: Optional[FeatureConfig] = None,
) -> Tuple[np.ndarray, List[FeatureExtractionResult]]:
    """Extract feature matrix (N x F) and per-transaction audit records in chronological order."""
    results = [extract_transaction_features(graph, tx, feature_config) for tx in transactions]
    if not results:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64), results
    matrix = np.vstack([res.to_vector(FEATURE_NAMES) for res in results])
    return matrix, results
