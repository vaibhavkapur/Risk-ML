"""Training-serving skew detection using PSI and standardized mean shift."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence
import numpy as np

from risk_platform.features import FEATURE_NAMES
from risk_platform.model import CalibratedLinearRiskModel
from risk_platform.schemas import FeatureVector


def population_stability_index(
    expected_props: Sequence[float], actual_props: Sequence[float], eps: float = 1e-6
) -> float:
    """Compute Population Stability Index (PSI) between reference and serving histograms."""
    psi = 0.0
    for exp_p, act_p in zip(expected_props, actual_props):
        e = max(float(exp_p), eps)
        a = max(float(act_p), eps)
        psi += (a - e) * math.log(a / e)
    return float(psi)


def detect_training_serving_skew(
    model: CalibratedLinearRiskModel,
    serving_vectors: Sequence[FeatureVector],
    psi_warning: float = 0.10,
    psi_critical: float = 0.25,
    mean_shift_sigma_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Compare serving feature vectors against reference training distributions in the model artifact."""
    ref_stats = model.training_metadata.get("feature_stats", {})
    if not serving_vectors:
        return {
            "status": "NO_DATA",
            "samples": 0,
            "alert_triggered": False,
            "features": {},
        }

    per_feature: Dict[str, Any] = {}
    max_psi = 0.0
    max_mean_shift = 0.0
    warning_features = []
    critical_features = []

    for fname in FEATURE_NAMES:
        vals = np.array([float(fv.values[fname]) for fv in serving_vectors], dtype=float)
        ref = ref_stats[fname]
        edges = ref["bin_edges"]
        exp_props = ref["bin_proportions"]

        counts, _ = np.histogram(vals, bins=edges)
        total = max(1, int(np.sum(counts)))
        act_props = [float(c) / total for c in counts]
        psi = population_stability_index(exp_props, act_props)

        serve_mean = float(np.mean(vals))
        serve_std = float(np.std(vals))
        ref_mean = float(ref["mean"])
        ref_std = float(ref["std"])
        mean_shift_sigma = abs(serve_mean - ref_mean) / max(ref_std, 1e-6)

        if psi >= psi_critical or mean_shift_sigma >= mean_shift_sigma_threshold * 1.5:
            severity = "CRITICAL"
            critical_features.append(fname)
        elif psi >= psi_warning or mean_shift_sigma >= mean_shift_sigma_threshold:
            severity = "WARNING"
            warning_features.append(fname)
        else:
            severity = "OK"

        max_psi = max(max_psi, psi)
        max_mean_shift = max(max_mean_shift, mean_shift_sigma)
        per_feature[fname] = {
            "psi": round(psi, 6),
            "train_mean": round(ref_mean, 6),
            "serving_mean": round(serve_mean, 6),
            "train_std": round(ref_std, 6),
            "serving_std": round(serve_std, 6),
            "mean_shift_sigma": round(mean_shift_sigma, 6),
            "severity": severity,
        }

    if critical_features:
        overall_status = "CRITICAL_SKEW"
    elif warning_features:
        overall_status = "WARNING_SKEW"
    else:
        overall_status = "OK"

    return {
        "model_version": model.model_version,
        "feature_version": model.feature_version,
        "samples": len(serving_vectors),
        "status": overall_status,
        "alert_triggered": overall_status != "OK",
        "max_psi": round(max_psi, 6),
        "max_mean_shift_sigma": round(max_mean_shift, 6),
        "critical_features": critical_features,
        "warning_features": warning_features,
        "thresholds": {
            "psi_warning": psi_warning,
            "psi_critical": psi_critical,
            "mean_shift_sigma": mean_shift_sigma_threshold,
        },
        "features": per_feature,
    }
