"""Runtime metadata capture and model card report generation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Union

import joblib
import matplotlib
import numpy as np
import scipy
import sklearn
import yaml

from credit_risk import __version__
from credit_risk.config import PlatformConfig


def collect_runtime_metadata(cfg: PlatformConfig) -> Dict[str, Any]:
    """Capture pinned environment, versions, seed, and config digest for reproducibility."""
    cfg_dict = asdict(cfg)
    cfg_digest = hashlib.sha256(
        json.dumps(cfg_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    optional_packages: Dict[str, Any] = {}
    for pkg in ("xgboost", "shap"):
        try:
            mod = __import__(pkg)
            optional_packages[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            optional_packages[pkg] = "not_installed"

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "project_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": cfg.project.seed,
        "config_sha256": cfg_digest,
        "installed_libraries": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
            "pyyaml": yaml.__version__,
        },
        "optional_libraries": optional_packages,
    }


def render_model_card_markdown(
    cfg: PlatformConfig,
    evaluation_report: Dict[str, Any],
    runtime_metadata: Dict[str, Any],
) -> str:
    """Render a transparent, evidence-linked Model Card in Markdown."""
    splits = evaluation_report["splits"]
    lr_raw = evaluation_report["models"]["logistic_regression"]["uncalibrated_test"]
    lr_cal = evaluation_report["models"]["logistic_regression"]["calibrated_test"]
    gb_raw = evaluation_report["models"]["boosted_trees"]["uncalibrated_test"]
    gb_cal = evaluation_report["models"]["boosted_trees"]["calibrated_test"]
    primary_name = evaluation_report["primary_model"]
    cal_method = evaluation_report["selected_calibration_method"]
    psi_info = evaluation_report["stability"]["score_psi_train_vs_test"]
    csi_info = evaluation_report["stability"]["feature_csi_train_vs_test"]
    subgroup = evaluation_report["subgroup_diagnostics"]
    baseline_cmp = evaluation_report["baseline_comparison"]

    lines = [
        "# Model Card — Credit Risk Scoring Platform (Educational Prototype)",
        "",
        "> **Important Notice:** This model card documents an educational/research prototype trained exclusively on deterministic synthetic loan data. **Automated lending decisions are strictly disabled (`automated_decision_allowed = false`).** Outputs are decision-support estimates for manual review prioritization only and are NOT proof of legal fairness, regulatory compliance, or creditworthiness in production.",
        "",
        "## 1. Model Details",
        "",
        f"- **Project / Version:** `{cfg.project.name}` v`{cfg.project.version}`",
        f"- **Schema / Feature / Policy Versions:** `schema={cfg.project.schema_version}`, `features={cfg.project.feature_version}`, `policy={cfg.project.policy_version}`",
        f"- **Primary Model:** `{primary_name}` (backend: `{cfg.models.boosted_trees.backend}`)",
        f"- **Reference Baseline:** `logistic_regression` (`sklearn.linear_model.LogisticRegression`)",
        f"- **Selected Calibration Method:** `{cal_method}` (compared against `{', '.join(cfg.calibration.comparison_methods)}`)",
        f"- **Random Seed / Config SHA-256:** `{cfg.project.seed}` / `{runtime_metadata['config_sha256'][:16]}...`",
        f"- **Python / scikit-learn:** `{runtime_metadata['python_version']}` / `{runtime_metadata['installed_libraries']['scikit_learn']}`",
        f"- **Optional Backends:** `xgboost={runtime_metadata['optional_libraries']['xgboost']}`, `shap={runtime_metadata['optional_libraries']['shap']}`",
        "",
        "## 2. Intended Use and Out-of-Scope Uses",
        "",
        "- **Intended Use:** Educational demonstration of point-in-time feature contracts, temporal validation with label maturity windows, post-hoc PD calibration, stability monitoring (PSI/CSI), subgroup diagnostics with minimum-sample guards, local explanations, and human manual-review triage.",
        "- **Out-of-Scope Uses:** Automated credit approval/denial, adverse-action notice generation without legal/compliance review, pricing/interest-rate setting, or scoring real customer records.",
        "",
        "## 3. Data Provenance and Temporal Splits",
        "",
        "- **Data Source:** Deterministic synthetic generator (`synthetic_generator_v1`) with controlled missingness and post-`2024-01-01` temporal distribution shift.",
        "- **Real Customer Records:** None (`contains_real_customer_data = false`).",
        f"- **Observation Window:** `{cfg.data.observation_window_days}` days; label cutoff `{cfg.data.as_of_cutoff_date}`.",
        f"- **Train Split (`<= {cfg.temporal_split.train_end_date}`):** `n={splits['train']['count']}`, default rate=`{splits['train']['default_rate']:.4f}`",
        f"- **Calibration Split (`<= {cfg.temporal_split.calibration_end_date}`):** `n={splits['calibration']['count']}`, default rate=`{splits['calibration']['default_rate']:.4f}`",
        f"- **Test Split (`<= {cfg.temporal_split.test_end_date}`):** `n={splits['test']['count']}`, default rate=`{splits['test']['default_rate']:.4f}`",
        f"- **Right-Censored Immature Excluded:** `n={splits['censored_excluded']['count']}`",
        "- **Protected Subgroup Attributes:** `age_band` and `region_tier` are strictly excluded from model feature matrices and retained solely for post-hoc subgroup diagnostics.",
        "",
        "## 4. Measured Performance (Held-Out Temporal Test Split)",
        "",
        "| Model Variant | AUROC | PR-AUC | KS | Brier Score | ECE | Mean Pred PD | Observed Default Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Logistic Regression (Uncalibrated) | {lr_raw['auroc']:.4f} | {lr_raw['pr_auc']:.4f} | {lr_raw['ks_statistic']:.4f} | {lr_raw['brier_score']:.4f} | {lr_raw['ece']:.4f} | {lr_raw['mean_predicted_pd']:.4f} | {lr_raw['observed_default_rate']:.4f} |",
        f"| Logistic Regression (Calibrated: {cal_method}) | {lr_cal['auroc']:.4f} | {lr_cal['pr_auc']:.4f} | {lr_cal['ks_statistic']:.4f} | {lr_cal['brier_score']:.4f} | {lr_cal['ece']:.4f} | {lr_cal['mean_predicted_pd']:.4f} | {lr_cal['observed_default_rate']:.4f} |",
        f"| Boosted Trees (Uncalibrated) | {gb_raw['auroc']:.4f} | {gb_raw['pr_auc']:.4f} | {gb_raw['ks_statistic']:.4f} | {gb_raw['brier_score']:.4f} | {gb_raw['ece']:.4f} | {gb_raw['mean_predicted_pd']:.4f} | {gb_raw['observed_default_rate']:.4f} |",
        f"| Boosted Trees (Calibrated: {cal_method}) | {gb_cal['auroc']:.4f} | {gb_cal['pr_auc']:.4f} | {gb_cal['ks_statistic']:.4f} | {gb_cal['brier_score']:.4f} | {gb_cal['ece']:.4f} | {gb_cal['mean_predicted_pd']:.4f} | {gb_cal['observed_default_rate']:.4f} |",
        "",
        f"- **Baseline Comparison (`boosted_trees` vs `logistic_regression`):** AUROC delta=`{baseline_cmp['auroc_delta']:+.4f}`, Brier delta=`{baseline_cmp['brier_delta']:+.4f}`, Candidate ECE=`{baseline_cmp['candidate_ece']:.4f}`, All tolerance checks passed=`{baseline_cmp['all_checks_passed']}`.",
        "- **Calibration Plots:** `reports/calibration_plot.png` and `reports/calibration_plot.svg`.",
        "",
        "## 5. Stability and Distribution Shift Diagnostics (Train vs Temporal Test)",
        "",
        f"- **Calibrated Score PSI:** `{psi_info['psi']}` (status: `{psi_info['status']}`; warning >= `{cfg.evaluation.psi_warning_threshold}`, alert >= `{cfg.evaluation.psi_alert_threshold}`)",
        f"- **Max Feature CSI:** `{csi_info['max_csi']}`",
        f"- **Features with Warning CSI:** `{csi_info['warning_features'] or 'None'}`",
        f"- **Features with Alert CSI:** `{csi_info['alert_features'] or 'None'}`",
        "",
        "## 6. Subgroup Diagnostics (Test Split, Monitoring Only)",
        "",
        f"Minimum subgroup sample threshold: `n >= {subgroup['min_sample_size']}` (groups below this threshold have metrics withheld and raise explicit warnings/errors).",
        "",
    ]

    for col, col_data in subgroup["subgroups"].items():
        lines.append(f"### Subgroup Dimension: `{col}`")
        lines.append("")
        lines.append("| Group | Status | N | Observed DR | Mean Pred PD | Cal Gap | Elevated Review Rate | TPR | FPR | AUROC |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for gname, ginfo in col_data["groups"].items():
            if ginfo["status"] != "EVALUATED":
                lines.append(f"| `{gname}` | `{ginfo['status']}` | {ginfo['sample_size']} | — | — | — | — | — | — | — |")
            else:
                m = ginfo["metrics"]
                tpr_s = f"{m['tpr_at_threshold']:.4f}" if m["tpr_at_threshold"] is not None else "N/A"
                fpr_s = f"{m['fpr_at_threshold']:.4f}" if m["fpr_at_threshold"] is not None else "N/A"
                auc_s = f"{m['auroc']:.4f}" if m["auroc"] is not None else "N/A"
                lines.append(
                    f"| `{gname}` | `EVALUATED` | {ginfo['sample_size']} | {m['observed_default_rate']:.4f} | "
                    f"{m['mean_predicted_pd']:.4f} | {m['calibration_gap']:+.4f} | {m['elevated_review_rate']:.4f} | "
                    f"{tpr_s} | {fpr_s} | {auc_s} |"
                )
        disp = col_data["disparity_summary"]
        lines.append("")
        lines.append(
            f"- Disparity summary: max elevated-review-rate gap=`{disp['max_elevated_review_rate_gap']}`, "
            f"max TPR gap=`{disp['max_tpr_gap']}`, max |calibration gap|=`{disp['max_abs_calibration_gap']}`."
        )
        lines.append("")

    lines.extend(
        [
            "## 7. Limitations, Uncertainty, and Governance Notes",
            "",
            "- **Synthetic Data Only:** Performance numbers reflect synthetic feature-to-default relationships and do not generalize to real borrowers or macroeconomic regimes.",
            "- **Calibration Under Shift:** Post-hoc calibration is fitted on the pre-shift calibration window; under severe macroeconomic shift (`PSI >= 0.25`), recalibration and underwriting review are required.",
            "- **Missingness Uncertainty:** Median imputation with missingness indicators stabilizes inference, but applicants with multiple missing bureau fields carry wider epistemic uncertainty and trigger mandatory manual verification items.",
            "- **Subgroup Uncertainty:** Small subgroups (`n < min_subgroup_sample_size`) produce wide confidence intervals; metrics are deliberately withheld rather than silently reported.",
            "- **No Automated Lending:** Risk bands only govern human manual-review queue priority and SLA.",
            "",
        ]
    )

    return "\n".join(lines)


def write_model_card(
    cfg: PlatformConfig,
    evaluation_report: Dict[str, Any],
    runtime_metadata: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = render_model_card_markdown(cfg, evaluation_report, runtime_metadata)
    target.write_text(content, encoding="utf-8")
    return target
