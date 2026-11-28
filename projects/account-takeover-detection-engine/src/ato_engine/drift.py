"""Distribution drift detection and alerting (PSI, missingness, and feature expiry)."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from .config import DriftConfig
from .models import ATORiskModelBundle
from .schemas import DriftAlert, DriftReport, FeatureVector, PolicyDecision


def population_stability_index(
    expected: Sequence[float],
    actual: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index (PSI) between reference and observed distributions."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    exp_arr = np.asarray(expected, dtype=float)
    act_arr = np.asarray(actual, dtype=float)

    # Use quantile edges from expected distribution with clipped [0, 1] fallback for constant data
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(exp_arr, quantiles))
    if len(edges) < 3:
        lo = min(float(np.min(exp_arr)), float(np.min(act_arr)))
        hi = max(float(np.max(exp_arr)), float(np.max(act_arr)))
        if hi <= lo:
            return 0.0
        edges = np.linspace(lo, hi, n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts, _ = np.histogram(exp_arr, bins=edges)
    act_counts, _ = np.histogram(act_arr, bins=edges)

    eps = 1e-4
    exp_pct = np.maximum(exp_counts / max(len(exp_arr), 1), eps)
    act_pct = np.maximum(act_counts / max(len(act_arr), 1), eps)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(max(0.0, psi), 6)


def evaluate_drift(
    decisions: Sequence[PolicyDecision],
    feature_vectors: Sequence[FeatureVector],
    model_bundle: ATORiskModelBundle,
    config: DriftConfig,
) -> DriftReport:
    """Evaluate score PSI, key feature shift, missingness drift, and expiry rate."""
    n = len(decisions)
    if n < config.min_sample_size or len(feature_vectors) < config.min_sample_size:
        return DriftReport(
            sample_size=n,
            score_psi=0.0,
            feature_psi={},
            missingness_rate=0.0,
            baseline_missingness_rate=model_bundle.reference_missingness_rate,
            expiry_rate=0.0,
            alerts=[],
            critical_drift_detected=False,
        )

    observed_scores = [d.calibrated_risk_score for d in decisions]
    score_psi = population_stability_index(
        model_bundle.reference_score_distribution,
        observed_scores,
        n_bins=10,
    )

    # Evaluate standardized mean shift (converted to PSI-like magnitude) on key monitored features
    monitored_features = (
        "ip_reputation_risk",
        "device_trust_score",
        "is_new_device_for_user",
        "failed_attempts_1h",
        "sensitive_action_count",
    )
    feature_psi: Dict[str, float] = {}
    for fname in monitored_features:
        obs_vals = np.asarray([fv.values[fname] for fv in feature_vectors], dtype=float)
        ref_stats = model_bundle.reference_feature_stats.get(fname, {})
        ref_mean = float(ref_stats.get("cal_mean", ref_stats.get("mean", 0.0)))
        ref_std = max(float(ref_stats.get("std", 0.25)), 0.05)
        # Approximate Gaussian KL-symmetric shift (comparable scale to PSI)
        z_shift = abs(float(np.mean(obs_vals)) - ref_mean) / ref_std
        feature_psi[fname] = round(float(min(2.0, 0.25 * (z_shift ** 2))), 6)

    missingness_rate = round(
        float(np.mean([fv.values["missing_feature_ratio"] for fv in feature_vectors])),
        6,
    )
    baseline_miss = model_bundle.reference_missingness_rate
    missingness_delta = missingness_rate - baseline_miss

    expiry_rate = round(
        float(np.mean([1.0 if fv.profile_expired else 0.0 for fv in feature_vectors])),
        6,
    )

    alerts: List[DriftAlert] = []
    if score_psi >= config.psi_critical_threshold:
        alerts.append(
            DriftAlert(
                metric="score_psi",
                severity="critical",
                observed_value=score_psi,
                threshold=config.psi_critical_threshold,
                message=f"Calibrated risk score PSI ({score_psi:.3f}) exceeded critical threshold ({config.psi_critical_threshold:.3f}).",
            )
        )
    elif score_psi >= config.psi_warning_threshold:
        alerts.append(
            DriftAlert(
                metric="score_psi",
                severity="warning",
                observed_value=score_psi,
                threshold=config.psi_warning_threshold,
                message=f"Calibrated risk score PSI ({score_psi:.3f}) exceeded warning threshold ({config.psi_warning_threshold:.3f}).",
            )
        )

    for fname, shift_val in feature_psi.items():
        if shift_val >= config.psi_critical_threshold:
            alerts.append(
                DriftAlert(
                    metric=f"feature_shift:{fname}",
                    severity="critical",
                    observed_value=shift_val,
                    threshold=config.psi_critical_threshold,
                    message=f"Feature '{fname}' distribution shift ({shift_val:.3f}) exceeded critical threshold.",
                )
            )
        elif shift_val >= config.psi_warning_threshold:
            alerts.append(
                DriftAlert(
                    metric=f"feature_shift:{fname}",
                    severity="warning",
                    observed_value=shift_val,
                    threshold=config.psi_warning_threshold,
                    message=f"Feature '{fname}' distribution shift ({shift_val:.3f}) exceeded warning threshold.",
                )
            )

    if missingness_delta >= config.missingness_increase_critical:
        alerts.append(
            DriftAlert(
                metric="missingness_delta",
                severity="critical",
                observed_value=round(missingness_delta, 6),
                threshold=config.missingness_increase_critical,
                message=f"Telemetry missingness increased by {missingness_delta:.3f} (critical).",
            )
        )
    elif missingness_delta >= config.missingness_increase_warning:
        alerts.append(
            DriftAlert(
                metric="missingness_delta",
                severity="warning",
                observed_value=round(missingness_delta, 6),
                threshold=config.missingness_increase_warning,
                message=f"Telemetry missingness increased by {missingness_delta:.3f} (warning).",
            )
        )

    if expiry_rate >= config.expiry_rate_warning:
        alerts.append(
            DriftAlert(
                metric="feature_expiry_rate",
                severity="warning",
                observed_value=expiry_rate,
                threshold=config.expiry_rate_warning,
                message=f"Feature profile expiry rate ({expiry_rate:.3f}) exceeded warning threshold ({config.expiry_rate_warning:.3f}).",
            )
        )

    critical_detected = any(a.severity == "critical" for a in alerts)
    return DriftReport(
        sample_size=n,
        score_psi=score_psi,
        feature_psi=feature_psi,
        missingness_rate=missingness_rate,
        baseline_missingness_rate=baseline_miss,
        expiry_rate=expiry_rate,
        alerts=alerts,
        critical_drift_detected=critical_detected,
    )
