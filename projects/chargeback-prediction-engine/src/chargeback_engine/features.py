"""Causal point-in-time merchant and transaction feature engineering using Polars."""

from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from chargeback_engine.config import FeatureConfig
from chargeback_engine.contracts import (
    SUPPORTED_MCC_GROUPS,
    PaymentChannel,
    ReasonCategory,
)

CHANNELS_ORDERED: Tuple[str, ...] = tuple(c.value for c in PaymentChannel)
MCC_GROUPS_ORDERED: Tuple[str, ...] = tuple(sorted(SUPPORTED_MCC_GROUPS))


@dataclass(frozen=True)
class FeatureSchemaMetadata:
    feature_set_version: str
    feature_columns: Tuple[str, ...]
    feature_schema_hash: str
    prior_dispute_rate: float


def get_feature_columns(windows_days: Sequence[int]) -> Tuple[str, ...]:
    """Return canonical ordered list of numeric model feature names."""
    cols: List[str] = [
        "amount_usd",
        "log_amount_usd",
        "is_cross_border_num",
        "has_3ds_num",
        "has_avs_match_num",
        "customer_tenure_days",
        "log_customer_tenure_days",
        "merchant_tenure_days",
        "avg_fulfillment_days",
    ]
    for w in sorted(windows_days):
        cols.extend(
            [
                f"m_tx_count_{w}d",
                f"m_tx_amount_sum_{w}d",
                f"m_tx_amount_mean_{w}d",
            ]
        )
    cols.extend(
        [
            "amount_to_m_mean_30d_ratio",
            "m_disputes_reported_30d",
            "m_disputes_reported_90d",
            "m_fraud_disputes_90d",
            "m_consumer_disputes_90d",
            "m_unknown_reason_disputes_90d",
            "m_reported_dispute_rate_90d",
            "m_prior_mature_tx_count",
            "m_prior_mature_dispute_count",
            "m_is_cold_start",
            "m_smoothed_dispute_rate",
        ]
    )
    for ch in CHANNELS_ORDERED:
        cols.append(f"ch_{ch}")
    for mcc in MCC_GROUPS_ORDERED:
        cols.append(f"mcc_{mcc}")
    return tuple(cols)


def compute_feature_schema_hash(
    feature_set_version: str, feature_columns: Sequence[str]
) -> str:
    payload = f"{feature_set_version}:" + ",".join(feature_columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _to_epoch_seconds(dt: datetime) -> float:
    return dt.timestamp()


def _count_and_sum_in_window(
    timestamps: List[float],
    prefix_sums: List[float],
    target_ts: float,
    window_seconds: float,
) -> Tuple[int, float]:
    """Count and sum values for events with (target_ts - window_seconds) <= ts < target_ts."""
    hi = bisect.bisect_left(timestamps, target_ts)
    lo = bisect.bisect_left(timestamps, target_ts - window_seconds, 0, hi)
    count = hi - lo
    total = prefix_sums[hi] - prefix_sums[lo]
    return count, total


def _count_in_window(
    timestamps: List[float], target_ts: float, window_seconds: float
) -> int:
    hi = bisect.bisect_left(timestamps, target_ts)
    lo = bisect.bisect_left(timestamps, target_ts - window_seconds, 0, hi)
    return hi - lo


def build_causal_features(
    labeled_df: pl.DataFrame,
    feature_cfg: FeatureConfig,
    *,
    feature_set_version: str = "1.0.0",
    prior_dispute_rate: Optional[float] = None,
) -> Tuple[pl.DataFrame, FeatureSchemaMetadata]:
    """Compute strictly causal point-in-time transaction and merchant features.

    Anti-leakage guarantees for every transaction row at time t = transaction_ts:
      - Prior merchant transaction counts/sums include only transactions with tx_ts < t.
      - Reported dispute counts in trailing 30d/90d include only disputes with dispute_ts < t.
      - Mature merchant dispute counts include only prior transactions with maturity_date <= t
        (and tx_ts < t) whose observed dispute (if any) occurred at dispute_ts <= t.
      - Empirical Bayes shrinkage blends mature merchant history with `prior_dispute_rate`
        using `feature_cfg.cold_start_prior_weight`.
    """
    windows = tuple(sorted(feature_cfg.merchant_windows_days))
    feature_cols = get_feature_columns(windows)
    effective_prior = (
        float(prior_dispute_rate)
        if prior_dispute_rate is not None
        else float(feature_cfg.fallback_prior_dispute_rate)
    )
    meta = FeatureSchemaMetadata(
        feature_set_version=feature_set_version,
        feature_columns=feature_cols,
        feature_schema_hash=compute_feature_schema_hash(
            feature_set_version, feature_cols
        ),
        prior_dispute_rate=effective_prior,
    )

    if labeled_df.is_empty():
        empty_cols = {c: pl.Series(c, [], dtype=pl.Float64) for c in feature_cols}
        out = labeled_df.with_columns(list(empty_cols.values()))
        return out, meta

    sorted_df = labeled_df.sort(["transaction_ts", "transaction_id"])
    rows = sorted_df.to_dicts()

    # Pre-index per-merchant event histories sorted by event time.
    # 1) Transaction events: (tx_ts_sec, amount_usd)
    # 2) Reported dispute events: (dispute_ts_sec, reason_category)
    # 3) Maturity events: (maturity_ts_sec, tx_ts_sec, dispute_ts_sec_or_inf)
    m_tx_ts: Dict[str, List[float]] = {}
    m_tx_prefix_amt: Dict[str, List[float]] = {}
    m_disp_ts: Dict[str, List[float]] = {}
    m_fraud_disp_ts: Dict[str, List[float]] = {}
    m_consumer_disp_ts: Dict[str, List[float]] = {}
    m_unknown_disp_ts: Dict[str, List[float]] = {}
    m_maturity_events: Dict[str, List[Tuple[float, float, float]]] = {}

    for r in rows:
        mid = r["merchant_id"]
        t_sec = _to_epoch_seconds(r["transaction_ts"])
        mat_sec = _to_epoch_seconds(r["maturity_date"])
        amt = float(r["amount_usd"])

        m_tx_ts.setdefault(mid, []).append(t_sec)
        prefix = m_tx_prefix_amt.setdefault(mid, [0.0])
        prefix.append(prefix[-1] + amt)

        d_ts = r["dispute_ts"]
        if d_ts is not None:
            d_sec = _to_epoch_seconds(d_ts)
            m_disp_ts.setdefault(mid, []).append(d_sec)
            cat = r["reason_category"]
            if cat == ReasonCategory.FRAUD.value:
                m_fraud_disp_ts.setdefault(mid, []).append(d_sec)
            elif cat == ReasonCategory.CONSUMER_DISPUTE.value:
                m_consumer_disp_ts.setdefault(mid, []).append(d_sec)
            elif cat == ReasonCategory.UNKNOWN.value:
                m_unknown_disp_ts.setdefault(mid, []).append(d_sec)
            m_maturity_events.setdefault(mid, []).append((mat_sec, t_sec, d_sec))
        else:
            m_maturity_events.setdefault(mid, []).append((mat_sec, t_sec, float("inf")))

    # Sort dispute arrays and maturity arrays by their respective event timestamps.
    for dmap in (
        m_disp_ts,
        m_fraud_disp_ts,
        m_consumer_disp_ts,
        m_unknown_disp_ts,
    ):
        for mid in dmap:
            dmap[mid].sort()

    m_mat_ts_sorted: Dict[str, List[float]] = {}
    m_mat_tx_ts_sorted: Dict[str, List[float]] = {}
    m_mat_disp_ts_sorted: Dict[str, List[float]] = {}
    for mid, evs in m_maturity_events.items():
        evs.sort(key=lambda x: (x[0], x[1]))
        m_mat_ts_sorted[mid] = [e[0] for e in evs]
        m_mat_tx_ts_sorted[mid] = [e[1] for e in evs]
        m_mat_disp_ts_sorted[mid] = [e[2] for e in evs]

    sec_per_day = 86400.0
    prior_w = float(feature_cfg.cold_start_prior_weight)
    min_mature_cold = int(feature_cfg.cold_start_min_mature_txns)

    computed: Dict[str, List[float]] = {c: [] for c in feature_cols}

    for r in rows:
        mid = r["merchant_id"]
        t_sec = _to_epoch_seconds(r["transaction_ts"])
        onboard_sec = _to_epoch_seconds(r["merchant_onboarding_ts"])
        amt = float(r["amount_usd"])
        cust_tenure = float(r["customer_tenure_days"])

        computed["amount_usd"].append(amt)
        computed["log_amount_usd"].append(math.log1p(amt))
        computed["is_cross_border_num"].append(1.0 if r["is_cross_border"] else 0.0)
        computed["has_3ds_num"].append(1.0 if r["has_3ds"] else 0.0)
        computed["has_avs_match_num"].append(1.0 if r["has_avs_match"] else 0.0)
        computed["customer_tenure_days"].append(cust_tenure)
        computed["log_customer_tenure_days"].append(math.log1p(cust_tenure))
        computed["merchant_tenure_days"].append(
            max(0.0, (t_sec - onboard_sec) / sec_per_day)
        )
        computed["avg_fulfillment_days"].append(float(r["avg_fulfillment_days"]))

        tx_times = m_tx_ts.get(mid, [])
        tx_prefix = m_tx_prefix_amt.get(mid, [0.0])

        mean_30d = 0.0
        count_90d = 0
        for w in windows:
            cnt, s_amt = _count_and_sum_in_window(
                tx_times, tx_prefix, t_sec, w * sec_per_day
            )
            mean_amt = (s_amt / cnt) if cnt > 0 else 0.0
            computed[f"m_tx_count_{w}d"].append(float(cnt))
            computed[f"m_tx_amount_sum_{w}d"].append(float(s_amt))
            computed[f"m_tx_amount_mean_{w}d"].append(float(mean_amt))
            if w == 30:
                mean_30d = mean_amt
            if w == 90:
                count_90d = cnt

        ratio_30d = (amt / mean_30d) if mean_30d > 0.0 else 1.0
        computed["amount_to_m_mean_30d_ratio"].append(float(ratio_30d))

        disp_times = m_disp_ts.get(mid, [])
        d_30 = _count_in_window(disp_times, t_sec, 30.0 * sec_per_day)
        d_90 = _count_in_window(disp_times, t_sec, 90.0 * sec_per_day)
        d_fraud_90 = _count_in_window(
            m_fraud_disp_ts.get(mid, []), t_sec, 90.0 * sec_per_day
        )
        d_cons_90 = _count_in_window(
            m_consumer_disp_ts.get(mid, []), t_sec, 90.0 * sec_per_day
        )
        d_unk_90 = _count_in_window(
            m_unknown_disp_ts.get(mid, []), t_sec, 90.0 * sec_per_day
        )

        computed["m_disputes_reported_30d"].append(float(d_30))
        computed["m_disputes_reported_90d"].append(float(d_90))
        computed["m_fraud_disputes_90d"].append(float(d_fraud_90))
        computed["m_consumer_disputes_90d"].append(float(d_cons_90))
        computed["m_unknown_reason_disputes_90d"].append(float(d_unk_90))
        computed["m_reported_dispute_rate_90d"].append(
            float(d_90 / count_90d) if count_90d > 0 else 0.0
        )

        # Prior mature transactions as of time t: maturity_date <= t AND tx_ts < t
        mat_times = m_mat_ts_sorted.get(mid, [])
        mat_hi = bisect.bisect_right(mat_times, t_sec)
        mat_tx_times = m_mat_tx_ts_sorted.get(mid, [])
        mat_disp_times = m_mat_disp_ts_sorted.get(mid, [])

        mature_tx_cnt = 0
        mature_disp_cnt = 0
        for idx in range(mat_hi):
            if mat_tx_times[idx] < t_sec:
                mature_tx_cnt += 1
                if mat_disp_times[idx] <= t_sec:
                    mature_disp_cnt += 1

        smoothed_rate = (mature_disp_cnt + prior_w * effective_prior) / (
            mature_tx_cnt + prior_w
        )
        computed["m_prior_mature_tx_count"].append(float(mature_tx_cnt))
        computed["m_prior_mature_dispute_count"].append(float(mature_disp_cnt))
        computed["m_is_cold_start"].append(
            1.0 if mature_tx_cnt < min_mature_cold else 0.0
        )
        computed["m_smoothed_dispute_rate"].append(float(smoothed_rate))

        ch_val = r["channel"]
        for ch in CHANNELS_ORDERED:
            computed[f"ch_{ch}"].append(1.0 if ch_val == ch else 0.0)

        mcc_val = r["mcc_group"]
        for mcc in MCC_GROUPS_ORDERED:
            computed[f"mcc_{mcc}"].append(1.0 if mcc_val == mcc else 0.0)

    # Avoid duplicating amount_usd column which already exists in labeled_df
    new_series = [
        pl.Series(name=col, values=vals, dtype=pl.Float64)
        for col, vals in computed.items()
        if col != "amount_usd"
    ]
    out_df = sorted_df.with_columns(new_series)
    return out_df, meta


def extract_feature_matrix(
    df: pl.DataFrame, feature_columns: Sequence[str]
) -> np.ndarray:
    """Extract float64 feature matrix matching the exact ordered feature schema."""
    return df.select(list(feature_columns)).to_numpy().astype(np.float64)
