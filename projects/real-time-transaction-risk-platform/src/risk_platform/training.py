"""Temporal training, label maturity filtering, probability calibration, and evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from risk_platform.config import PlatformConfig
from risk_platform.features import FEATURE_NAMES, FEATURE_VERSION, WINDOW_24H_MS, OfflineFeatureExtractor
from risk_platform.model import CalibratedLinearRiskModel, _compute_artifact_hash
from risk_platform.schemas import OutcomeLabel, TransactionEvent, format_utc_ms


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) over uniform probability bins."""
    if len(y_true) == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = float(len(y_true))
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(y_prob[mask]))
            avg_acc = float(np.mean(y_true[mask]))
            ece += (count / total) * abs(avg_acc - avg_conf)
    return float(ece)


def compute_reference_feature_stats(X: np.ndarray, feature_names: Sequence[str]) -> Dict[str, Any]:
    """Compute reference mean, std, and decile histogram edges/proportions for skew monitoring."""
    stats: Dict[str, Any] = {}
    for idx, fname in enumerate(feature_names):
        col = X[:, idx].astype(float)
        mean = float(np.mean(col))
        std = float(np.std(col))
        quantiles = np.quantile(col, np.linspace(0.0, 1.0, 11))
        # Deduplicate interior edges if feature is discrete/sparse (e.g. tx_count_5m, country_changed)
        unique_edges = np.unique(quantiles)
        if len(unique_edges) < 2:
            edges = [float(unique_edges[0]) - 0.5, float(unique_edges[0]) + 0.5]
        else:
            edges = [float(e) for e in unique_edges]
            edges[0] = -1e12
            edges[-1] = 1e12
        counts, _ = np.histogram(col, bins=edges)
        total = max(1, int(np.sum(counts)))
        proportions = [float(c) / total for c in counts]
        stats[fname] = {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "bin_edges": [round(e, 6) if abs(e) < 1e11 else e for e in edges],
            "bin_proportions": [round(p, 6) for p in proportions],
        }
    return stats


def _safe_metric(fn, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(fn(y_true, y_prob))


def train_calibrated_baseline_model(
    events: Sequence[TransactionEvent],
    labels: Sequence[OutcomeLabel],
    config: PlatformConfig,
    model_version: str = "model.logreg_cal.v1",
) -> Tuple[CalibratedLinearRiskModel, Dict[str, Any]]:
    """Train and calibrate a baseline risk model with strict temporal split & label maturity."""
    label_by_id = {lbl.event_id: lbl for lbl in labels}
    ordered_events = sorted(events, key=lambda e: (e.event_ts_ms, e.event_id))

    extractor = OfflineFeatureExtractor(fx_to_usd=config.schema.fx_to_usd)
    feature_map = extractor.extract_batch(ordered_events)

    paired: List[Tuple[TransactionEvent, OutcomeLabel, List[float]]] = []
    for ev in ordered_events:
        lbl = label_by_id.get(ev.event_id)
        if lbl is None:
            continue
        fv = feature_map[ev.event_id]
        paired.append((ev, lbl, fv.ordered_values(FEATURE_NAMES)))

    n_total = len(paired)
    if n_total < 50:
        raise ValueError(f"Insufficient labeled events for temporal training: {n_total}")

    # Exclude initial 24h feature-window warm-up events so 24h rolling features are fully populated
    stream_start_ts_ms = paired[0][0].event_ts_ms
    warmup_cutoff_ts_ms = stream_start_ts_ms + WINDOW_24H_MS
    post_warmup = [row for row in paired if row[0].event_ts_ms >= warmup_cutoff_ts_ms]
    warmup_excluded = len(paired) - len(post_warmup)
    effective_paired = post_warmup if len(post_warmup) >= 120 else paired
    if len(post_warmup) < 120:
        warmup_excluded = 0

    n_eff = len(effective_paired)
    n_train = int(n_eff * config.training.train_fraction)
    n_cal = int(n_eff * config.training.calibration_fraction)

    train_raw = effective_paired[:n_train]
    cal_raw = effective_paired[n_train : n_train + n_cal]
    test_raw = effective_paired[n_train + n_cal :]

    # Enforce label maturity at the training/calibration snapshot timestamps
    train_snapshot_ts_ms = cal_raw[0][0].event_ts_ms
    cal_snapshot_ts_ms = test_raw[0][0].event_ts_ms

    train_matured = [row for row in train_raw if row[1].label_ts_ms <= train_snapshot_ts_ms]
    cal_matured = [row for row in cal_raw if row[1].label_ts_ms <= cal_snapshot_ts_ms]

    # Guard for very short synthetic streams where maturity exceeds split duration
    maturity_filter_applied = True
    if len(train_matured) < 30 or len({r[1].is_fraud for r in train_matured}) < 2:
        train_matured = train_raw
        maturity_filter_applied = False
    if len(cal_matured) < 20 or len({r[1].is_fraud for r in cal_matured}) < 2:
        cal_matured = cal_raw
        maturity_filter_applied = False

    X_train = np.array([r[2] for r in train_matured], dtype=float)
    y_train = np.array([r[1].is_fraud for r in train_matured], dtype=int)

    X_cal = np.array([r[2] for r in cal_matured], dtype=float)
    y_cal = np.array([r[1].is_fraud for r in cal_matured], dtype=int)

    X_test = np.array([r[2] for r in test_raw], dtype=float)
    y_test = np.array([r[1].is_fraud for r in test_raw], dtype=int)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        C=config.training.logreg_c,
        max_iter=config.training.max_iter,
        random_state=config.training.seed,
        solver="lbfgs",
    )
    clf.fit(X_train_s, y_train)

    # Platt sigmoid calibration fitted strictly on temporal calibration split logits
    cal_logits = clf.decision_function(X_cal_s).reshape(-1, 1)
    if len(np.unique(y_cal)) >= 2:
        platt = LogisticRegression(C=10.0, max_iter=500, random_state=config.training.seed)
        platt.fit(cal_logits, y_cal)
        platt_a = float(platt.coef_[0][0])
        platt_b = float(platt.intercept_[0])
    else:
        platt_a = 1.0
        platt_b = 0.0

    # Evaluate raw vs calibrated on untouched temporal test set
    test_logits = clf.decision_function(X_test_s)
    test_raw_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -60.0, 60.0)))
    test_cal_probs = 1.0 / (1.0 + np.exp(-np.clip(platt_a * test_logits + platt_b, -60.0, 60.0)))

    eval_metrics = {
        "test_samples": int(len(y_test)),
        "test_positive_rate": round(float(np.mean(y_test)), 6),
        "raw_roc_auc": round(_safe_metric(roc_auc_score, y_test, test_raw_probs), 6),
        "calibrated_roc_auc": round(_safe_metric(roc_auc_score, y_test, test_cal_probs), 6),
        "raw_pr_auc": round(_safe_metric(average_precision_score, y_test, test_raw_probs), 6),
        "calibrated_pr_auc": round(_safe_metric(average_precision_score, y_test, test_cal_probs), 6),
        "raw_brier": round(float(brier_score_loss(y_test, test_raw_probs)), 6),
        "calibrated_brier": round(float(brier_score_loss(y_test, test_cal_probs)), 6),
        "raw_ece": round(expected_calibration_error(y_test, test_raw_probs), 6),
        "calibrated_ece": round(expected_calibration_error(y_test, test_cal_probs), 6),
    }

    feature_stats = compute_reference_feature_stats(X_train, FEATURE_NAMES)

    training_metadata: Dict[str, Any] = {
        "seed": config.training.seed,
        "label_maturity_hours": config.training.label_maturity_hours,
        "maturity_filter_applied": maturity_filter_applied,
        "splits": {
            "total_labeled_events": n_total,
            "warmup_24h_excluded": warmup_excluded,
            "train_raw_count": len(train_raw),
            "train_matured_count": len(train_matured),
            "train_immature_excluded": len(train_raw) - len(train_matured),
            "train_positive_rate": round(float(np.mean(y_train)), 6),
            "calibration_raw_count": len(cal_raw),
            "calibration_matured_count": len(cal_matured),
            "calibration_immature_excluded": len(cal_raw) - len(cal_matured),
            "test_count": len(test_raw),
            "train_start_ts": format_utc_ms(train_raw[0][0].event_ts_ms),
            "train_snapshot_ts": format_utc_ms(train_snapshot_ts_ms),
            "calibration_snapshot_ts": format_utc_ms(cal_snapshot_ts_ms),
            "test_end_ts": format_utc_ms(test_raw[-1][0].event_ts_ms),
        },
        "evaluation": eval_metrics,
        "feature_stats": feature_stats,
    }

    # Construct model and stamp deterministic SHA-256 checksum
    draft = CalibratedLinearRiskModel(
        model_version=model_version,
        feature_version=FEATURE_VERSION,
        feature_names=FEATURE_NAMES,
        scaler_mean=tuple(float(x) for x in scaler.mean_),
        scaler_scale=tuple(float(x) if float(x) != 0.0 else 1.0 for x in scaler.scale_),
        weights=tuple(float(x) for x in clf.coef_[0]),
        intercept=float(clf.intercept_[0]),
        platt_a=platt_a,
        platt_b=platt_b,
        training_metadata=training_metadata,
        artifact_sha256="",
    )
    art_dict = draft.to_artifact_dict()
    final_hash = art_dict["artifact_sha256"]
    model = CalibratedLinearRiskModel(
        model_version=draft.model_version,
        feature_version=draft.feature_version,
        feature_names=draft.feature_names,
        scaler_mean=draft.scaler_mean,
        scaler_scale=draft.scaler_scale,
        weights=draft.weights,
        intercept=draft.intercept,
        platt_a=draft.platt_a,
        platt_b=draft.platt_b,
        training_metadata=draft.training_metadata,
        artifact_sha256=final_hash,
    )
    return model, training_metadata
