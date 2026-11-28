"""Tests for temporal splits, label maturity, calibration, and model predictions."""

from __future__ import annotations

import json
from pathlib import Path

from payment_auth_engine.config import load_config
from payment_auth_engine.features import filter_matured_label_indices
from payment_auth_engine.models import AuthorizationModelBundle
from payment_auth_engine.schemas import PolicyAction
from payment_auth_engine.synthetic import generate_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_temporal_split_and_label_maturity_prevent_leakage() -> None:
    config = load_config(ROOT / "config" / "default.yaml")
    dataset = generate_synthetic_dataset(config)
    bundle, train_idx, cal_idx, test_idx = AuthorizationModelBundle.train(
        dataset.attempts, dataset.outcomes, config, dataset.as_of_utc
    )

    assert len(train_idx) + len(cal_idx) + len(test_idx) == len(dataset.attempts)
    assert dataset.attempts[int(train_idx[-1])].timestamp_utc <= dataset.attempts[int(cal_idx[0])].timestamp_utc
    assert dataset.attempts[int(cal_idx[-1])].timestamp_utc <= dataset.attempts[int(test_idx[0])].timestamp_utc

    matured_train = filter_matured_label_indices(
        dataset.outcomes, train_idx, dataset.as_of_utc, config.data.min_label_maturity_days
    )
    assert len(matured_train) > 0
    for i in matured_train:
        out = dataset.outcomes[int(i)]
        assert out.authorized is True
        assert out.fraud_confirmed is not None
        assert out.label_maturity_days >= config.data.min_label_maturity_days

    # Calibration quality checks on held-out calibration fold
    acc_cal = bundle.summary.acceptance_calibration
    loss_cal = bundle.summary.loss_calibration
    assert 0.0 <= acc_cal["ece_after"] < 0.08
    assert acc_cal["roc_auc"] > 0.65
    assert 0.0 <= loss_cal["ece_after"] < 0.05
    assert loss_cal["roc_auc"] > 0.70


def test_economics_oracle_matches_exact_formula() -> None:
    oracle = json.loads((ROOT / "data" / "fixtures" / "reference_oracle.json").read_text(encoding="utf-8"))[
        "economics_oracle"
    ]
    amt = oracle["amount_cents"]
    margin_rate = oracle["merchant_margin_bps"] / 10_000.0
    lgd = amt * oracle["loss_given_fraud_multiplier"] + oracle["chargeback_fee_cents"]
    exp_loss = oracle["p_acc"] * oracle["p_fraud"] * lgd
    exp_rev = oracle["p_acc"] * (amt * margin_rate)
    exp_net = exp_rev - exp_loss - oracle["action_cost_cents"]
    loss_bps = 10_000.0 * (exp_loss / (oracle["p_acc"] * amt))

    tol = oracle["tolerance"]
    assert abs(lgd - oracle["expected"]["lgd_cents"]) <= tol
    assert abs(exp_loss - oracle["expected"]["expected_loss_cents"]) <= tol
    assert abs(exp_rev - oracle["expected"]["expected_revenue_cents"]) <= tol
    assert abs(exp_net - oracle["expected"]["expected_net_value_cents"]) <= tol
    assert abs(loss_bps - oracle["expected"]["expected_loss_bps_on_approved"]) <= tol


def test_model_predictions_valid_ranges_and_decline_zeroes() -> None:
    config = load_config(ROOT / "config" / "default.yaml")
    dataset = generate_synthetic_dataset(config)
    bundle, _, _, test_idx = AuthorizationModelBundle.train(
        dataset.attempts, dataset.outcomes, config, dataset.as_of_utc
    )
    sample = dataset.attempts[int(test_idx[0])]
    preds = bundle.predict_all_actions(sample)

    for action, pred in preds.items():
        assert 0.0 <= pred.acceptance_prob <= 1.0
        assert 0.0 <= pred.fraud_prob_given_approved <= 1.0
        assert pred.expected_loss_cents >= 0.0
        assert pred.estimated_latency_ms >= 0.0

    dec = preds[PolicyAction.DECLINE_POLICY]
    assert dec.acceptance_prob == 0.0
    assert dec.expected_loss_cents == 0.0
    assert dec.expected_net_value_cents == 0.0
