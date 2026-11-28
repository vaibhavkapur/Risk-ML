"""Boundary and failure behavior tests for insufficient subgroup sample sizes."""

from pathlib import Path
import numpy as np
import pytest

from credit_risk.config import DataConfig
from credit_risk.data import generate_synthetic_dataset
from credit_risk.errors import InsufficientSubgroupSampleError
from credit_risk.evaluation import evaluate_subgroups


def test_insufficient_subgroup_sample_strict_raises_explicit_error():
    data_cfg = DataConfig(
        n_applicants=40,
        start_date="2023-01-01",
        end_date="2023-06-30",
        observation_window_days=365,
        as_of_cutoff_date="2025-07-01",
        missing_rate=0.05,
        shift_start_date="2024-01-01",
        shift_magnitude=0.0,
    )
    ds = generate_synthetic_dataset(data_cfg, seed=19)
    y_true = np.array([int(l.default_flag) for l in ds.labels])
    y_prob = np.array(ds.true_pd)

    # With only 40 applicants across 4 age_bands, at least one subgroup will have n < 30.
    with pytest.raises(InsufficientSubgroupSampleError) as exc_info:
        evaluate_subgroups(
            applicants=ds.applicants,
            y_true=y_true,
            y_prob=y_prob,
            subgroup_columns=["age_band"],
            min_sample_size=30,
            strict=True,
        )
    assert exc_info.value.error_code == "INSUFFICIENT_SUBGROUP_SAMPLE_ERROR"
    assert exc_info.value.details["min_sample_size"] == 30
    assert exc_info.value.details["sample_size"] < 30


def test_insufficient_subgroup_sample_non_strict_withholds_metrics_and_warns():
    data_cfg = DataConfig(
        n_applicants=50,
        start_date="2023-01-01",
        end_date="2023-06-30",
        observation_window_days=365,
        as_of_cutoff_date="2025-07-01",
        missing_rate=0.05,
        shift_start_date="2024-01-01",
        shift_magnitude=0.0,
    )
    ds = generate_synthetic_dataset(data_cfg, seed=23)
    y_true = np.array([int(l.default_flag) for l in ds.labels])
    y_prob = np.array(ds.true_pd)

    report = evaluate_subgroups(
        applicants=ds.applicants,
        y_true=y_true,
        y_prob=y_prob,
        subgroup_columns=["age_band", "region_tier"],
        min_sample_size=30,
        strict=False,
    )

    assert len(report["insufficient_sample_warnings"]) >= 1
    age_groups = report["subgroups"]["age_band"]["groups"]
    insufficient = [g for g in age_groups.values() if g["status"] == "INSUFFICIENT_SAMPLE"]
    assert len(insufficient) >= 1
    for g in insufficient:
        assert g["metrics"] is None
        assert "withheld" in g["warning"]
