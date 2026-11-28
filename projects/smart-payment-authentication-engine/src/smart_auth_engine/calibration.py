"""Probability calibration and calibration quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


@dataclass
class CalibrationMetrics:
    brier_score: float
    log_loss: float
    ece: float
    roc_auc: float
    base_rate: float
    mean_predicted_prob: float
    reliability_bins: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brier_score": round(self.brier_score, 6),
            "log_loss": round(self.log_loss, 6),
            "ece": round(self.ece, 6),
            "roc_auc": round(self.roc_auc, 6),
            "base_rate": round(self.base_rate, 6),
            "mean_predicted_prob": round(self.mean_predicted_prob, 6),
            "reliability_bins": self.reliability_bins,
        }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute Brier score, log-loss, ECE, ROC-AUC, and reliability bin table."""
    y_t = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if len(y_t) == 0:
        raise ValueError("Cannot compute calibration metrics on empty arrays.")

    brier = float(brier_score_loss(y_t, p))
    ll = float(log_loss(y_t, p, labels=[0, 1]))
    roc = float(roc_auc_score(y_t, p)) if len(np.unique(y_t)) > 1 else 0.5

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: List[Dict[str, float]] = []
    n = float(len(y_t))

    for i in range(n_bins):
        lo = float(bin_edges[i])
        hi = float(bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(p[mask].mean())
        avg_acc = float(y_t[mask].mean())
        ece += (count / n) * abs(avg_acc - avg_conf)
        bins.append(
            {
                "bin_lower": round(lo, 4),
                "bin_upper": round(hi, 4),
                "count": float(count),
                "mean_predicted": round(avg_conf, 6),
                "empirical_fraud_rate": round(avg_acc, 6),
            }
        )

    return CalibrationMetrics(
        brier_score=brier,
        log_loss=ll,
        ece=float(ece),
        roc_auc=roc,
        base_rate=float(y_t.mean()),
        mean_predicted_prob=float(p.mean()),
        reliability_bins=bins,
    )


class PostHocProbabilityCalibrator:
    """Calibrator trained strictly on a chronological held-out calibration window."""

    def __init__(self, method: Literal["sigmoid", "isotonic", "none"] = "sigmoid") -> None:
        if method not in ("sigmoid", "isotonic", "none"):
            raise ValueError(f"Unsupported calibration method: {method!r}")
        self.method = method
        self._platt: Optional[LogisticRegression] = None
        self._iso: Optional[IsotonicRegression] = None
        self._is_fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "PostHocProbabilityCalibrator":
        p = np.clip(np.asarray(raw_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        y = np.asarray(y_true, dtype=np.int64)
        if len(y) == 0:
            raise ValueError("Cannot fit calibrator on an empty calibration split.")

        if self.method == "none" or len(np.unique(y)) < 2:
            self._is_fitted = True
            return self

        if self.method == "sigmoid":
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            lr = LogisticRegression(C=1e6, solver="lbfgs")
            lr.fit(logits, y)
            self._platt = lr
        elif self.method == "isotonic":
            iso = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
            iso.fit(p, y)
            self._iso = iso

        self._is_fitted = True
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before transform().")
        p = np.clip(np.asarray(raw_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        if self.method == "none":
            return p
        if self.method == "sigmoid" and self._platt is not None:
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            return np.clip(self._platt.predict_proba(logits)[:, 1], 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic" and self._iso is not None:
            return np.clip(self._iso.predict(p), 1e-6, 1.0 - 1e-6)
        return p
