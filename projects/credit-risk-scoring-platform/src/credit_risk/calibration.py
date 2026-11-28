"""Probability-of-default calibration, monotonicity verification, and calibration plots."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from credit_risk.errors import CalibrationMonotonicityError, UnsupportedInputError


def _safe_logit(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(probs, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class PDCalibrator:
    """Post-hoc probability calibrator supporting Platt (sigmoid) and Isotonic scaling."""

    def __init__(
        self,
        method: str = "isotonic",
        clip_epsilon: float = 1e-4,
        monotonicity_tolerance: float = 1e-6,
    ) -> None:
        if method not in ("sigmoid", "isotonic"):
            raise UnsupportedInputError(
                f"Unsupported calibration method '{method}'. Expected 'sigmoid' or 'isotonic'.",
                details={"method": method},
            )
        self.method = method
        self.clip_epsilon = float(clip_epsilon)
        self.monotonicity_tolerance = float(monotonicity_tolerance)
        self._platt: Optional[LogisticRegression] = None
        self._isotonic: Optional[IsotonicRegression] = None
        self.is_fitted_ = False

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "PDCalibrator":
        scores = np.asarray(raw_scores, dtype=np.float64)
        labels = np.asarray(y_true, dtype=np.int64)
        if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
            raise ValueError("raw_scores and y_true must be 1-D arrays of equal length.")
        if len(np.unique(labels)) < 2:
            raise ValueError("Calibration split must contain both default (1) and non-default (0) labels.")

        if self.method == "sigmoid":
            logits = _safe_logit(scores).reshape(-1, 1)
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
            lr.fit(logits, labels)
            # Enforce non-negative slope for monotonicity.
            if lr.coef_[0, 0] < 0.0:
                raise CalibrationMonotonicityError(
                    "Platt/sigmoid calibrator fitted a negative slope, violating monotonicity.",
                    details={"slope": float(lr.coef_[0, 0])},
                )
            self._platt = lr
        else:
            iso = IsotonicRegression(
                y_min=self.clip_epsilon,
                y_max=1.0 - self.clip_epsilon,
                increasing=True,
                out_of_bounds="clip",
            )
            iso.fit(scores, labels.astype(np.float64))
            self._isotonic = iso

        self.is_fitted_ = True
        self.verify_monotonicity()
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("PDCalibrator must be fitted before transform().")
        scores = np.asarray(raw_scores, dtype=np.float64)
        if self.method == "sigmoid":
            assert self._platt is not None
            logits = _safe_logit(scores).reshape(-1, 1)
            calibrated = self._platt.predict_proba(logits)[:, 1]
        else:
            assert self._isotonic is not None
            calibrated = self._isotonic.predict(scores)
        return np.clip(np.asarray(calibrated, dtype=np.float64), self.clip_epsilon, 1.0 - self.clip_epsilon)

    def verify_monotonicity(self, grid_points: int = 200) -> Dict[str, Any]:
        """Verify that calibrated PD is non-decreasing across [0.001, 0.999]."""
        grid = np.linspace(0.001, 0.999, grid_points)
        cal = self.transform(grid)
        diffs = np.diff(cal)
        min_diff = float(np.min(diffs))
        if min_diff < -self.monotonicity_tolerance:
            raise CalibrationMonotonicityError(
                f"Calibrator '{self.method}' violated monotonicity: min_diff={min_diff:.8f}",
                details={"method": self.method, "min_diff": min_diff, "tolerance": self.monotonicity_tolerance},
            )
        return {
            "method": self.method,
            "is_monotone": True,
            "grid_points": grid_points,
            "min_step_delta": round(min_diff, 10),
            "tolerance": self.monotonicity_tolerance,
        }


@dataclass(frozen=True)
class ReliabilityBin:
    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_predicted_pd: Optional[float]
    observed_default_rate: Optional[float]
    abs_calibration_error: Optional[float]


def compute_reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[List[Dict[str, Any]], float]:
    """Compute equal-width reliability diagram bins and Expected Calibration Error (ECE)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_prob, dtype=np.float64)
    n = len(y_t)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins_out: List[Dict[str, Any]] = []
    ece = 0.0
    for idx in range(n_bins):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx == n_bins - 1:
            mask = (y_p >= lo) & (y_p <= hi)
        else:
            mask = (y_p >= lo) & (y_p < hi)
        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(y_p[mask]))
            obs_rate = float(np.mean(y_t[mask]))
            abs_err = abs(mean_pred - obs_rate)
            ece += (count / max(n, 1)) * abs_err
            bins_out.append(
                {
                    "bin_index": idx,
                    "lower_bound": round(lo, 4),
                    "upper_bound": round(hi, 4),
                    "count": count,
                    "mean_predicted_pd": round(mean_pred, 6),
                    "observed_default_rate": round(obs_rate, 6),
                    "abs_calibration_error": round(abs_err, 6),
                }
            )
        else:
            bins_out.append(
                {
                    "bin_index": idx,
                    "lower_bound": round(lo, 4),
                    "upper_bound": round(hi, 4),
                    "count": 0,
                    "mean_predicted_pd": None,
                    "observed_default_rate": None,
                    "abs_calibration_error": None,
                }
            )
    return bins_out, float(ece)


def save_calibration_plots(
    y_true: np.ndarray,
    curves: Dict[str, np.ndarray],
    output_dir: Union[str, Path],
    n_bins: int = 10,
    title_prefix: str = "Credit Risk PD Calibration (Temporal Test Split)",
) -> Dict[str, str]:
    """Generate reliability diagram and score distribution plots in PNG and SVG."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Ensure matplotlib uses a persistent writable cache dir in sandboxed environments.
    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = Path(tempfile.gettempdir()) / "credit_risk_mpl_cache"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_rel, ax_hist) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=140)

    # Panel 1: Reliability curves
    ax_rel.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.2, label="Perfect calibration")
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    for idx, (label, probs) in enumerate(curves.items()):
        bins, ece = compute_reliability_bins(y_true, probs, n_bins=n_bins)
        xs = [b["mean_predicted_pd"] for b in bins if b["count"] > 0]
        ys = [b["observed_default_rate"] for b in bins if b["count"] > 0]
        color = palette[idx % len(palette)]
        ax_rel.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=color,
            label=f"{label} (ECE={ece:.4f})",
        )
        ax_hist.hist(
            probs,
            bins=20,
            range=(0.0, 1.0),
            alpha=0.35,
            histtype="stepfilled",
            edgecolor=color,
            color=color,
            label=label,
        )

    ax_rel.set_title(f"{title_prefix}\nReliability Diagram")
    ax_rel.set_xlabel("Mean Predicted Probability of Default (PD)")
    ax_rel.set_ylabel("Observed Default Rate")
    ax_rel.set_xlim(0.0, 1.0)
    ax_rel.set_ylim(0.0, 1.0)
    ax_rel.grid(True, linestyle=":", alpha=0.5)
    ax_rel.legend(loc="upper left", fontsize=8.5)

    ax_hist.set_title("Predicted PD Distribution")
    ax_hist.set_xlabel("Predicted Probability of Default (PD)")
    ax_hist.set_ylabel("Applicant Count")
    ax_hist.grid(True, linestyle=":", alpha=0.5)
    ax_hist.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()

    png_path = out_path / "calibration_plot.png"
    svg_path = out_path / "calibration_plot.svg"
    fig.savefig(png_path, format="png", bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    return {
        "png": str(png_path),
        "svg": str(svg_path),
    }
