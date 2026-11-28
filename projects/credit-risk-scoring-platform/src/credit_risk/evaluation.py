"""Discrimination, calibration, stability (PSI/CSI), subgroup diagnostics, and baseline comparison."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from credit_risk.calibration import compute_reliability_bins
from credit_risk.config import BaselineComparisonConfig
from credit_risk.contracts import ApplicantRecord
from credit_risk.errors import InsufficientSubgroupSampleError


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov (KS) separation statistic between defaults and non-defaults."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))


def compute_discrimination_and_calibration_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    n_bins: int = 10,
    decision_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Compute AUROC, PR-AUC, KS, Brier score, log loss, ECE, and calibration-in-the-large."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)

    unique_classes = np.unique(y_t)
    has_both_classes = len(unique_classes) == 2

    auroc = float(roc_auc_score(y_t, y_p)) if has_both_classes else None
    pr_auc = float(average_precision_score(y_t, y_p)) if has_both_classes else None
    ks = compute_ks_statistic(y_t, y_p) if has_both_classes else None
    brier = float(brier_score_loss(y_t, y_p))
    ll = float(log_loss(y_t, y_p, labels=[0, 1]))
    bins, ece = compute_reliability_bins(y_t, y_p, n_bins=n_bins)

    observed_rate = float(np.mean(y_t))
    mean_pred = float(np.mean(y_p))
    cal_ratio = (mean_pred / observed_rate) if observed_rate > 0 else None

    flagged = y_p >= decision_threshold
    flagged_rate = float(np.mean(flagged))

    return {
        "sample_size": int(len(y_t)),
        "default_count": int(np.sum(y_t)),
        "observed_default_rate": round(observed_rate, 6),
        "mean_predicted_pd": round(mean_pred, 6),
        "calibration_in_the_large_gap": round(mean_pred - observed_rate, 6),
        "calibration_in_the_large_ratio": round(cal_ratio, 6) if cal_ratio is not None else None,
        "auroc": round(auroc, 6) if auroc is not None else None,
        "pr_auc": round(pr_auc, 6) if pr_auc is not None else None,
        "ks_statistic": round(ks, 6) if ks is not None else None,
        "brier_score": round(brier, 6),
        "log_loss": round(ll, 6),
        "ece": round(ece, 6),
        "decision_threshold": float(decision_threshold),
        "elevated_review_rate_at_threshold": round(flagged_rate, 6),
        "reliability_bins": bins,
    }


def compute_psi_from_proportions(
    expected_proportions: Sequence[float],
    actual_proportions: Sequence[float],
    eps: float = 1e-6,
) -> float:
    """Compute Population Stability Index (PSI) from pre-binned proportions."""
    exp_arr = np.asarray(expected_proportions, dtype=np.float64)
    act_arr = np.asarray(actual_proportions, dtype=np.float64)
    if exp_arr.shape != act_arr.shape:
        raise ValueError("Expected and actual proportion vectors must have identical shape.")
    exp_safe = np.clip(exp_arr, eps, None)
    act_safe = np.clip(act_arr, eps, None)
    exp_safe = exp_safe / np.sum(exp_safe)
    act_safe = act_safe / np.sum(act_safe)
    return float(np.sum((act_safe - exp_safe) * np.log(act_safe / exp_safe)))


def compute_psi(
    expected_values: Sequence[float] | np.ndarray,
    actual_values: Sequence[float] | np.ndarray,
    n_bins: int = 10,
    warning_threshold: float = 0.10,
    alert_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Compute quantile-binned PSI between baseline (expected) and target (actual) distributions."""
    exp = np.asarray(expected_values, dtype=np.float64)
    act = np.asarray(actual_values, dtype=np.float64)
    exp = exp[np.isfinite(exp)]
    act = act[np.isfinite(act)]
    if len(exp) == 0 or len(act) == 0:
        return {"psi": None, "status": "INSUFFICIENT_DATA", "bin_edges": []}

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(exp, quantiles))
    if len(edges) < 2:
        return {"psi": 0.0, "status": "STABLE", "bin_edges": edges.tolist()}

    edges[0] = -np.inf
    edges[-1] = np.inf
    exp_counts, _ = np.histogram(exp, bins=edges)
    act_counts, _ = np.histogram(act, bins=edges)

    exp_props = exp_counts / max(len(exp), 1)
    act_props = act_counts / max(len(act), 1)
    psi_val = compute_psi_from_proportions(exp_props, act_props)

    if psi_val >= alert_threshold:
        status = "ALERT"
    elif psi_val >= warning_threshold:
        status = "WARNING"
    else:
        status = "STABLE"

    return {
        "psi": round(psi_val, 6),
        "status": status,
        "n_bins_effective": int(len(edges) - 1),
        "expected_proportions": [round(float(p), 6) for p in exp_props],
        "actual_proportions": [round(float(p), 6) for p in act_props],
    }


def compute_feature_stability_csi(
    X_expected: np.ndarray,
    X_actual: np.ndarray,
    feature_names: Sequence[str],
    n_bins: int = 10,
    warning_threshold: float = 0.10,
    alert_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Compute Characteristic Stability Index (CSI) across numeric features."""
    per_feature: Dict[str, Dict[str, Any]] = {}
    alerts: List[str] = []
    warnings: List[str] = []

    for idx, name in enumerate(feature_names):
        res = compute_psi(
            X_expected[:, idx],
            X_actual[:, idx],
            n_bins=n_bins,
            warning_threshold=warning_threshold,
            alert_threshold=alert_threshold,
        )
        per_feature[name] = {
            "csi": res["psi"],
            "status": res["status"],
        }
        if res["status"] == "ALERT":
            alerts.append(name)
        elif res["status"] == "WARNING":
            warnings.append(name)

    return {
        "per_feature": per_feature,
        "warning_features": warnings,
        "alert_features": alerts,
        "max_csi": max(
            (v["csi"] for v in per_feature.values() if v["csi"] is not None),
            default=0.0,
        ),
    }


def evaluate_subgroups(
    applicants: Sequence[ApplicantRecord],
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    subgroup_columns: Sequence[str],
    min_sample_size: int = 30,
    decision_threshold: float = 0.15,
    strict: bool = False,
) -> Dict[str, Any]:
    """Evaluate calibration, discrimination, and review-routing rates per subgroup.

    Explicitly enforces `min_sample_size`:
    - If `strict=True`, raises `InsufficientSubgroupSampleError` when any subgroup < min_sample_size.
    - If `strict=False`, marks the subgroup as `INSUFFICIENT_SAMPLE`, omits unreliable metrics,
      and records an explicit diagnostic warning (never silent success).
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_prob, dtype=np.float64)
    if len(applicants) != len(y_t) or len(y_t) != len(y_p):
        raise ValueError("applicants, y_true, and y_prob must have the same length.")

    report: Dict[str, Any] = {
        "min_sample_size": int(min_sample_size),
        "decision_threshold": float(decision_threshold),
        "subgroups": {},
        "insufficient_sample_warnings": [],
        "disclaimer": (
            "Educational subgroup diagnostics only; NOT proof of legal fairness, "
            "disparate-impact compliance, or fair-lending approval."
        ),
    }

    for col in subgroup_columns:
        groups: Dict[str, List[int]] = {}
        for idx, rec in enumerate(applicants):
            val = str(getattr(rec, col))
            groups.setdefault(val, []).append(idx)

        col_results: Dict[str, Any] = {}
        valid_selection_rates: List[float] = []
        valid_tprs: List[float] = []
        valid_cal_gaps: List[float] = []

        for group_val in sorted(groups.keys()):
            indices = groups[group_val]
            n_g = len(indices)
            if n_g < min_sample_size:
                msg = (
                    f"Subgroup {col}='{group_val}' has n={n_g} < min_sample_size={min_sample_size}; "
                    "metrics withheld to avoid unstable estimates."
                )
                if strict:
                    raise InsufficientSubgroupSampleError(
                        msg,
                        details={
                            "column": col,
                            "group": group_val,
                            "sample_size": n_g,
                            "min_sample_size": min_sample_size,
                        },
                    )
                report["insufficient_sample_warnings"].append(msg)
                col_results[group_val] = {
                    "status": "INSUFFICIENT_SAMPLE",
                    "sample_size": n_g,
                    "min_sample_size": min_sample_size,
                    "metrics": None,
                    "warning": msg,
                }
                continue

            yt_g = y_t[indices]
            yp_g = y_p[indices]
            preds_g = (yp_g >= decision_threshold).astype(np.int64)

            pos_mask = yt_g == 1
            neg_mask = yt_g == 0
            n_pos = int(np.sum(pos_mask))
            n_neg = int(np.sum(neg_mask))

            tpr = float(np.mean(preds_g[pos_mask])) if n_pos > 0 else None
            fpr = float(np.mean(preds_g[neg_mask])) if n_neg > 0 else None
            sel_rate = float(np.mean(preds_g))
            obs_rate = float(np.mean(yt_g))
            mean_pd = float(np.mean(yp_g))
            cal_gap = mean_pd - obs_rate
            auroc = float(roc_auc_score(yt_g, yp_g)) if (n_pos > 0 and n_neg > 0) else None
            brier = float(brier_score_loss(yt_g, np.clip(yp_g, 1e-7, 1.0 - 1e-7)))

            valid_selection_rates.append(sel_rate)
            if tpr is not None:
                valid_tprs.append(tpr)
            valid_cal_gaps.append(cal_gap)

            col_results[group_val] = {
                "status": "EVALUATED",
                "sample_size": n_g,
                "default_count": n_pos,
                "metrics": {
                    "observed_default_rate": round(obs_rate, 6),
                    "mean_predicted_pd": round(mean_pd, 6),
                    "calibration_gap": round(cal_gap, 6),
                    "elevated_review_rate": round(sel_rate, 6),
                    "tpr_at_threshold": round(tpr, 6) if tpr is not None else None,
                    "fpr_at_threshold": round(fpr, 6) if fpr is not None else None,
                    "auroc": round(auroc, 6) if auroc is not None else None,
                    "brier_score": round(brier, 6),
                },
            }

        disparity = {
            "max_elevated_review_rate_gap": (
                round(max(valid_selection_rates) - min(valid_selection_rates), 6)
                if len(valid_selection_rates) >= 2
                else None
            ),
            "max_tpr_gap": (
                round(max(valid_tprs) - min(valid_tprs), 6)
                if len(valid_tprs) >= 2
                else None
            ),
            "max_abs_calibration_gap": (
                round(max(abs(g) for g in valid_cal_gaps), 6)
                if valid_cal_gaps
                else None
            ),
        }

        report["subgroups"][col] = {
            "groups": col_results,
            "disparity_summary": disparity,
        }

    return report


def compare_against_baseline(
    logistic_calibrated: Dict[str, Any],
    boosted_calibrated: Dict[str, Any],
    tolerances: BaselineComparisonConfig,
) -> Dict[str, Any]:
    """Compare boosted calibrated model against logistic reference baseline with explicit tolerances."""
    lr_auroc = logistic_calibrated.get("auroc")
    gb_auroc = boosted_calibrated.get("auroc")
    lr_brier = logistic_calibrated["brier_score"]
    gb_brier = boosted_calibrated["brier_score"]
    gb_ece = boosted_calibrated["ece"]

    auroc_delta = (
        round(gb_auroc - lr_auroc, 6)
        if (lr_auroc is not None and gb_auroc is not None)
        else None
    )
    brier_delta = round(gb_brier - lr_brier, 6)

    checks = {
        "auroc_within_tolerance": (
            auroc_delta >= tolerances.min_auroc_improvement_or_parity
            if auroc_delta is not None
            else False
        ),
        "brier_within_tolerance": brier_delta <= tolerances.max_brier_degradation,
        "calibrated_ece_within_tolerance": gb_ece <= tolerances.max_ece_calibrated,
    }

    return {
        "reference_model": "logistic_regression_calibrated",
        "candidate_model": "boosted_trees_calibrated",
        "auroc_delta": auroc_delta,
        "brier_delta": brier_delta,
        "candidate_ece": gb_ece,
        "tolerances": {
            "min_auroc_improvement_or_parity": tolerances.min_auroc_improvement_or_parity,
            "max_brier_degradation": tolerances.max_brier_degradation,
            "max_ece_calibrated": tolerances.max_ece_calibrated,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
