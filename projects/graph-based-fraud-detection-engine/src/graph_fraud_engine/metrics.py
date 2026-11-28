"""Temporal splitting with label maturity enforcement and fraud evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from graph_fraud_engine.config import EngineConfig
from graph_fraud_engine.schemas import DatasetBundle, FraudLabelRecord, TransactionRecord


@dataclass(frozen=True)
class TemporalSplitData:
    train_txs: List[TransactionRecord]
    train_y: np.ndarray
    train_immature_excluded: int
    val_txs: List[TransactionRecord]
    val_y: np.ndarray
    val_immature_excluded: int
    test_txs: List[TransactionRecord]
    test_y: np.ndarray
    train_cutoff: datetime
    val_cutoff: datetime
    test_cutoff: datetime


def build_temporal_splits(bundle: DatasetBundle, config: EngineConfig) -> TemporalSplitData:
    """Partition transactions into strict chronological train/val/test splits.

    When `enforce_label_maturity` is True:
    - Training includes only transactions whose `tx_timestamp <= train_cutoff` AND
      `label_available_at <= train_cutoff` (delayed labels maturing after train_cutoff are excluded).
    - Validation includes only transactions with `train_cutoff < tx_timestamp <= val_cutoff` AND
      `label_available_at <= val_cutoff`.
    - Test includes transactions with `val_cutoff < tx_timestamp <= test_cutoff` evaluated against
      their eventual matured ground-truth labels (post-maturity evaluation).
    """
    start_dt = datetime.fromisoformat(
        config.synthetic_data.start_time_utc.replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    train_cutoff = start_dt + timedelta(days=config.temporal_split.train_end_day)
    val_cutoff = start_dt + timedelta(days=config.temporal_split.val_end_day)
    test_cutoff = start_dt + timedelta(days=config.temporal_split.test_end_day)

    labels_by_tx: Dict[str, FraudLabelRecord] = {l.tx_id: l for l in bundle.labels}
    enforce_maturity = config.temporal_split.enforce_label_maturity

    train_txs: List[TransactionRecord] = []
    train_y_list: List[int] = []
    train_immature_excluded = 0

    val_txs: List[TransactionRecord] = []
    val_y_list: List[int] = []
    val_immature_excluded = 0

    test_txs: List[TransactionRecord] = []
    test_y_list: List[int] = []

    for tx in sorted(bundle.transactions, key=lambda t: (t.tx_timestamp, t.tx_id)):
        lbl = labels_by_tx.get(tx.tx_id)
        if lbl is None:
            continue
        if tx.tx_timestamp <= train_cutoff:
            if enforce_maturity and lbl.label_available_at > train_cutoff:
                train_immature_excluded += 1
                continue
            train_txs.append(tx)
            train_y_list.append(lbl.is_fraud)
        elif tx.tx_timestamp <= val_cutoff:
            if enforce_maturity and lbl.label_available_at > val_cutoff:
                val_immature_excluded += 1
                continue
            val_txs.append(tx)
            val_y_list.append(lbl.is_fraud)
        elif tx.tx_timestamp <= test_cutoff:
            test_txs.append(tx)
            test_y_list.append(lbl.is_fraud)

    return TemporalSplitData(
        train_txs=train_txs,
        train_y=np.array(train_y_list, dtype=np.int64),
        train_immature_excluded=train_immature_excluded,
        val_txs=val_txs,
        val_y=np.array(val_y_list, dtype=np.int64),
        val_immature_excluded=val_immature_excluded,
        test_txs=test_txs,
        test_y=np.array(test_y_list, dtype=np.int64),
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
        test_cutoff=test_cutoff,
    )


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) over uniform probability bins."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    if len(y) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(y))
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        avg_conf = float(np.mean(p[mask]))
        avg_acc = float(np.mean(y[mask]))
        ece += (count / n) * abs(avg_acc - avg_conf)
    return float(ece)


def recall_at_fixed_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float,
) -> Dict[str, float]:
    """Compute the highest recall achieved with false-positive rate <= target_fpr."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return {"target_fpr": target_fpr, "achieved_fpr": 0.0, "recall": 0.0, "threshold": 1.0}

    fpr, tpr, thresholds = roc_curve(y, p)
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) == 0:
        return {"target_fpr": target_fpr, "achieved_fpr": 0.0, "recall": 0.0, "threshold": 1.0}

    best_idx = int(valid_indices[np.argmax(tpr[valid_indices])])
    thresh = float(thresholds[best_idx])
    if not np.isfinite(thresh):
        thresh = 1.0
    return {
        "target_fpr": float(target_fpr),
        "achieved_fpr": float(fpr[best_idx]),
        "recall": float(tpr[best_idx]),
        "threshold": thresh,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fixed_fpr_targets: Sequence[float] = (0.01, 0.05),
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute PR-AUC, ROC-AUC, Brier score, ECE, and recall at fixed FPR targets."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    has_both = len(np.unique(y)) >= 2

    pr_auc = float(average_precision_score(y, p)) if has_both else 0.0
    roc_auc = float(roc_auc_score(y, p)) if has_both else 0.0
    brier = float(brier_score_loss(y, p)) if len(y) > 0 else 0.0
    ece = compute_ece(y, p, n_bins=n_bins)

    fpr_results = {
        f"fpr_{int(round(target * 100)):02d}pct": recall_at_fixed_fpr(y, p, target)
        for target in fixed_fpr_targets
    }

    return {
        "n_samples": int(len(y)),
        "n_positives": int(np.sum(y)),
        "positive_rate": float(np.mean(y)) if len(y) > 0 else 0.0,
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc_auc, 6),
        "brier_score": round(brier, 6),
        "ece": round(ece, 6),
        "recall_at_fixed_fpr": fpr_results,
    }
