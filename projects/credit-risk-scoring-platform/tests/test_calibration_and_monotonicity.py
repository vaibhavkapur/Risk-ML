"""Integration tests for PD calibration and policy threshold monotonicity."""

from pathlib import Path
import numpy as np
import pytest

from credit_risk.calibration import PDCalibrator
from credit_risk.config import PolicyConfig, load_config
from credit_risk.data import generate_synthetic_dataset, temporal_split_dataset
from credit_risk.errors import CalibrationMonotonicityError, UnsupportedInputError
from credit_risk.evaluation import compute_discrimination_and_calibration_metrics
from credit_risk.features import FeaturePipeline
from credit_risk.models import train_baseline_models
from credit_risk.policy import ManualReviewPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_sigmoid_and_isotonic_calibration_monotonicity_integration():
    cfg = load_config(PROJECT_ROOT / "config" / "reference.yaml")
    dataset = generate_synthetic_dataset(cfg.data, seed=cfg.project.seed)
    splits = temporal_split_dataset(
        dataset.applicants,
        dataset.labels,
        cfg.temporal_split.train_end_date,
        cfg.temporal_split.calibration_end_date,
        cfg.temporal_split.test_end_date,
    )

    train_apps, train_lbls = splits["train"]
    cal_apps, cal_lbls = splits["calibration"]
    test_apps, test_lbls = splits["test"]

    y_train = np.array([int(l.default_flag) for l in train_lbls])
    y_cal = np.array([int(l.default_flag) for l in cal_lbls])
    y_test = np.array([int(l.default_flag) for l in test_lbls])

    pipe = FeaturePipeline().fit(train_apps)
    models = train_baseline_models(
        X_train_std=pipe.transform(train_apps, standardized=True),
        X_train_raw=pipe.transform(train_apps, standardized=False),
        y_train=y_train,
        models_cfg=cfg.models,
        seed=cfg.project.seed,
    )

    gb_cal_raw = models["boosted_trees"].predict_raw_proba(pipe.transform(cal_apps, standardized=False))
    gb_test_raw = models["boosted_trees"].predict_raw_proba(pipe.transform(test_apps, standardized=False))

    grid = np.linspace(0.001, 0.999, 250)
    for method in ("sigmoid", "isotonic"):
        cal = PDCalibrator(method=method).fit(gb_cal_raw, y_cal)
        check = cal.verify_monotonicity()
        assert check["is_monotone"] is True

        cal_grid = cal.transform(grid)
        assert np.all(np.diff(cal_grid) >= -1e-6)
        assert np.all((cal_grid > 0.0) & (cal_grid < 1.0))

        cal_test = cal.transform(gb_test_raw)
        metrics = compute_discrimination_and_calibration_metrics(y_test, cal_test, n_bins=5)
        assert metrics["auroc"] > 0.65
        assert metrics["brier_score"] < 0.25


def test_inverted_scores_raise_calibration_monotonicity_error():
    # Deliberately anti-monotone synthetic scores should fail Platt monotonicity guard.
    scores = np.linspace(0.05, 0.95, 100)
    labels = np.array([1 if s < 0.5 else 0 for s in scores])
    with pytest.raises(CalibrationMonotonicityError):
        PDCalibrator(method="sigmoid").fit(scores, labels)


def test_policy_threshold_monotonicity_and_no_automated_lending():
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    policy = ManualReviewPolicy(cfg.policy)
    res = policy.verify_threshold_monotonicity(grid_points=500)
    assert res["band_severity_monotone"] is True
    assert res["sla_urgency_monotone"] is True
    assert res["automated_decision_allowed"] is False

    # Attempting to enable automated lending must be explicitly rejected.
    bad_policy_cfg = PolicyConfig(automated_decision_allowed=True, risk_bands=cfg.policy.risk_bands)
    with pytest.raises(UnsupportedInputError):
        ManualReviewPolicy(bad_policy_cfg)
