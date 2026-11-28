"""Domain correctness tests: reference oracle, missing values, and distribution shift."""

import math
from pathlib import Path
import numpy as np

from credit_risk.config import DataConfig, load_config
from credit_risk.contracts import validate_applicant_record
from credit_risk.data import generate_synthetic_dataset, load_json_fixture, temporal_split_dataset
from credit_risk.evaluation import (
    compute_discrimination_and_calibration_metrics,
    compute_feature_stability_csi,
    compute_psi,
    compute_psi_from_proportions,
)
from credit_risk.features import FeaturePipeline
from credit_risk.policy import ManualReviewPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"


def test_hand_calculated_reference_oracle():
    oracle = load_json_fixture(FIXTURES_DIR / "reference_oracle.json")
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    policy = ManualReviewPolicy(cfg.policy)

    # 1. Logistic PD + Risk Band Routing Oracle
    lr_oracle = oracle["logistic_oracle"]
    intercept = lr_oracle["intercept"]
    weights = lr_oracle["weights"]
    for case in lr_oracle["cases"]:
        z = intercept + sum(weights[k] * case["features"][k] for k in weights)
        pd_val = 1.0 / (1.0 + math.exp(-z))
        assert abs(z - case["expected_logit"]) < 1e-12
        assert abs(pd_val - case["expected_pd"]) < 1e-12
        routed = policy.route(pd_val)
        assert routed.risk_band == case["expected_risk_band"]
        assert routed.automated_decision_allowed is False
        assert routed.manual_review_required is True

    # 2. Discrimination & Brier Oracle
    m_oracle = oracle["metrics_oracle"]
    tol = m_oracle["tolerance"]
    metrics = compute_discrimination_and_calibration_metrics(
        y_true=m_oracle["y_true"],
        y_prob=m_oracle["y_prob"],
        n_bins=4,
    )
    assert abs(metrics["brier_score"] - m_oracle["expected_brier_score"]) < tol
    assert abs(metrics["auroc"] - m_oracle["expected_auroc"]) < tol
    assert abs(metrics["pr_auc"] - m_oracle["expected_pr_auc"]) < tol

    # 3. PSI Oracle
    psi_oracle = oracle["psi_oracle"]
    psi_id = compute_psi_from_proportions(
        psi_oracle["expected_proportions"], psi_oracle["actual_proportions_identical"]
    )
    psi_sh = compute_psi_from_proportions(
        psi_oracle["expected_proportions"], psi_oracle["actual_proportions_shifted"]
    )
    assert abs(psi_id - psi_oracle["expected_psi_identical"]) < psi_oracle["tolerance"]
    assert abs(psi_sh - psi_oracle["expected_psi_shifted"]) < psi_oracle["tolerance"]


def test_missing_value_imputation_and_indicators():
    valid_apps = [validate_applicant_record(r) for r in load_json_fixture(FIXTURES_DIR / "valid_applicants.json")]
    pipe = FeaturePipeline()
    X_raw = pipe.fit_transform(valid_apps, standardized=False)
    X_std = pipe.transform(valid_apps, standardized=True)

    assert np.all(np.isfinite(X_raw))
    assert np.all(np.isfinite(X_std))

    # Record index 1 (FIX-MISSING-002) has null credit_score and annual_income_usd.
    idx_missing_cs = pipe.feature_names_.index("missing_credit_score")
    idx_missing_inc = pipe.feature_names_.index("missing_annual_income_usd")
    idx_cs = pipe.feature_names_.index("credit_score")

    assert X_raw[0, idx_missing_cs] == 0.0
    assert X_raw[1, idx_missing_cs] == 1.0
    assert X_raw[1, idx_missing_inc] == 1.0
    # Imputed credit_score should equal median of non-null scores [760, 300, 850] -> 760.0
    assert X_raw[1, idx_cs] == 760.0


def test_distribution_shift_triggers_psi_and_csi_diagnostics():
    base_data_cfg = DataConfig(
        n_applicants=1600,
        start_date="2022-01-01",
        end_date="2024-06-30",
        observation_window_days=365,
        as_of_cutoff_date="2025-07-01",
        missing_rate=0.04,
        shift_start_date="2024-01-01",
        shift_magnitude=0.0,
    )
    shifted_data_cfg = DataConfig(
        n_applicants=1600,
        start_date="2022-01-01",
        end_date="2024-06-30",
        observation_window_days=365,
        as_of_cutoff_date="2025-07-01",
        missing_rate=0.04,
        shift_start_date="2024-01-01",
        shift_magnitude=1.1,
    )

    ds_stable = generate_synthetic_dataset(base_data_cfg, seed=11)
    ds_shifted = generate_synthetic_dataset(shifted_data_cfg, seed=11)

    splits_stable = temporal_split_dataset(
        ds_stable.applicants, ds_stable.labels, "2023-06-30", "2023-12-31", "2024-06-30"
    )
    splits_shifted = temporal_split_dataset(
        ds_shifted.applicants, ds_shifted.labels, "2023-06-30", "2023-12-31", "2024-06-30"
    )

    pipe_stable = FeaturePipeline().fit(splits_stable["train"][0])
    csi_stable = compute_feature_stability_csi(
        pipe_stable.transform(splits_stable["train"][0], standardized=False)[:, :10],
        pipe_stable.transform(splits_stable["test"][0], standardized=False)[:, :10],
        pipe_stable.feature_names_[:10],
        n_bins=8,
    )

    pipe_shifted = FeaturePipeline().fit(splits_shifted["train"][0])
    csi_shifted = compute_feature_stability_csi(
        pipe_shifted.transform(splits_shifted["train"][0], standardized=False)[:, :10],
        pipe_shifted.transform(splits_shifted["test"][0], standardized=False)[:, :10],
        pipe_shifted.feature_names_[:10],
        n_bins=8,
    )

    assert csi_shifted["max_csi"] > csi_stable["max_csi"]
    assert csi_shifted["max_csi"] >= 0.10
    assert len(csi_shifted["warning_features"]) + len(csi_shifted["alert_features"]) >= 1

    # Also verify direct PSI on shifted DTI values flags WARNING or ALERT.
    train_dti = [r.debt_to_income_ratio for r in splits_shifted["train"][0] if r.debt_to_income_ratio is not None]
    test_dti = [r.debt_to_income_ratio for r in splits_shifted["test"][0] if r.debt_to_income_ratio is not None]
    dti_psi = compute_psi(train_dti, test_dti, n_bins=8)
    assert dti_psi["status"] in ("WARNING", "ALERT")
