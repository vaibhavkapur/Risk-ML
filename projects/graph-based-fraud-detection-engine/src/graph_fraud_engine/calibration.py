"""Probability calibration (Platt scaling, Isotonic regression, or identity)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _clip_probs(probs: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def _logit(probs: np.ndarray) -> np.ndarray:
    p = _clip_probs(probs)
    return np.log(p / (1.0 - p))


class ProbabilityCalibrator:
    """Post-hoc probability calibrator fitted strictly on a held-out temporal validation split."""

    def __init__(self, method: str = "platt") -> None:
        if method not in {"platt", "isotonic", "none"}:
            raise ValueError(f"Unsupported calibration method: {method}")
        self.method = method
        self._platt_model: Optional[LogisticRegression] = None
        self._iso_model: Optional[IsotonicRegression] = None
        self.is_fitted: bool = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        y = np.asarray(y_true, dtype=np.int64)
        p = _clip_probs(raw_probs)

        # If validation set has only one class or method is 'none', fall back gracefully to identity
        if self.method == "none" or len(np.unique(y)) < 2:
            self.is_fitted = True
            return self

        if self.method == "platt":
            logits = _logit(p).reshape(-1, 1)
            lr = LogisticRegression(C=10.0, solver="lbfgs", max_iter=500)
            lr.fit(logits, y)
            # Enforce monotonic non-decreasing calibration; never invert rank order
            if float(lr.coef_[0, 0]) > 0.0:
                self._platt_model = lr
            else:
                self._platt_model = None
        elif self.method == "isotonic":
            iso = IsotonicRegression(y_min=1e-4, y_max=1.0 - 1e-4, out_of_bounds="clip")
            iso.fit(p, y)
            self._iso_model = iso

        self.is_fitted = True
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        p = _clip_probs(raw_probs)
        if not self.is_fitted or self.method == "none":
            return p
        if self.method == "platt" and self._platt_model is not None:
            logits = _logit(p).reshape(-1, 1)
            return _clip_probs(self._platt_model.predict_proba(logits)[:, 1])
        if self.method == "isotonic" and self._iso_model is not None:
            return _clip_probs(self._iso_model.predict(p))
        return p
