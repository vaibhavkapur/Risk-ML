"""Integration tests: calibrated models, expected loss, class imbalance, cold start, reproducibility."""

from __future__ import annotations

import pytest

from chargeback_engine.pipeline import run_pipeline
from chargeback_engine.synthetic import generate_synthetic_dataset


def test_default_pipeline_calibration_and_metrics(default_config):
    merchants, txs, disputes = generate_synthetic_dataset(
        default_config.synthetic, seed=default_config.seed
    )
    result = run_pipeline(default_config, merchants, txs, disputes)

    cal = result.metrics_primary_calibrated
    uncal = result.metrics_primary_uncalibrated
    base_rate = cal["observed_dispute_rate"]

    # Model meaningfully outperforms random baseline on PR-AUC and precision@budget
    assert cal["pr_auc"] > base_rate * 2.0
    assert cal["precision_at_review_budget_by_value"] > base_rate * 2.0
    # Calibration improves Brier score and ECE over class-weighted uncalibrated raw probabilities
    assert cal["brier_score"] < uncal["brier_score"]
    assert cal["ece"] < uncal["ece"]
    # Calibrated expected loss is strictly positive and reasonably aligned with realized loss
    assert cal["total_expected_loss_usd"] > 0.0
    assert 0.35 <= cal["expected_to_realized_loss_ratio"] <= 2.2
    # Immature negatives were strictly excluded
    assert result.maturity_diagnostics.immature_negative_count > 0
    assert len(result.sample_explanations) == 5


def test_class_imbalance_and_merchant_cold_start_integration(cold_start_config):
    merchants, txs, disputes = generate_synthetic_dataset(
        cold_start_config.synthetic, seed=cold_start_config.seed
    )
    result = run_pipeline(cold_start_config, merchants, txs, disputes)

    cal = result.metrics_primary_calibrated
    assert cal["observed_dispute_rate"] < 0.05  # imbalanced scenario
    assert cal["pr_auc"] > cal["observed_dispute_rate"] * 1.5
    assert cal["ece"] < 0.05

    # Cold-start slice must exist and produce valid non-NaN calibrated metrics
    assert "cold_start_merchants" in result.cold_start_slice_metrics
    assert "established_merchants" in result.cold_start_slice_metrics
    cold_slice = result.cold_start_slice_metrics["cold_start_merchants"]
    assert cold_slice["n_samples"] > 0
    assert 0.0 < cold_slice["mean_predicted_prob"] < 0.5


def test_reproducibility_bit_for_bit_identical(default_config):
    m1, t1, d1 = generate_synthetic_dataset(
        default_config.synthetic, seed=default_config.seed
    )
    r1 = run_pipeline(default_config, m1, t1, d1)

    m2, t2, d2 = generate_synthetic_dataset(
        default_config.synthetic, seed=default_config.seed
    )
    r2 = run_pipeline(default_config, m2, t2, d2)

    assert r1.run_id == r2.run_id
    assert r1.to_dict() == r2.to_dict()
