"""Tests for counterfactual estimators, assumption violations, and constrained policies."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from payment_auth_engine.config import load_config
from payment_auth_engine.counterfactual import (
    CounterfactualAssumptionError,
    compute_exact_estimators,
)
from payment_auth_engine.engine import run_full_experiment
from payment_auth_engine.schemas import PolicyAction

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "data" / "fixtures"


def test_counterfactual_estimators_match_hand_calculated_oracle() -> None:
    oracle = json.loads((FIXTURES_DIR / "reference_oracle.json").read_text(encoding="utf-8"))[
        "counterfactual_oracle"
    ]
    res = compute_exact_estimators(
        logged_actions=[PolicyAction(a) for a in oracle["logged_actions"]],
        candidate_actions=[PolicyAction(a) for a in oracle["candidate_actions"]],
        propensities=oracle["propensities"],
        observed_rewards=oracle["observed_rewards"],
        dm_candidate_rewards=oracle["dm_candidate_rewards"],
        dm_logged_rewards=oracle["dm_logged_rewards"],
        ips_clip_min=oracle["ips_clip_min"],
        min_propensity_threshold=oracle["min_propensity_threshold"],
        min_ess_ratio=oracle["min_ess_ratio"],
    )
    tol = oracle["tolerance"]
    exp = oracle["expected"]
    assert abs(res["dm"] - exp["dm"]) <= tol
    assert abs(res["ips"] - exp["ips"]) <= tol
    assert abs(res["snips"] - exp["snips"]) <= tol
    assert abs(res["dr"] - exp["dr"]) <= tol
    assert abs(res["ess"] - exp["ess"]) <= tol
    assert abs(res["ess_ratio"] - exp["ess_ratio"]) <= tol
    assert res["matched_samples"] == exp["matched_samples"]


def test_counterfactual_positivity_violation_raises_explicit_error() -> None:
    with pytest.raises(CounterfactualAssumptionError) as exc_info:
        compute_exact_estimators(
            logged_actions=[PolicyAction.APPROVE_PRIMARY, PolicyAction.APPROVE_PRIMARY],
            candidate_actions=[PolicyAction.APPROVE_PRIMARY, PolicyAction.APPROVE_PRIMARY],
            propensities=[0.5, 0.0],
            observed_rewards=[1.0, 1.0],
            dm_candidate_rewards=[0.8, 0.8],
            dm_logged_rewards=[0.8, 0.8],
        )
    assert exc_info.value.code == "POSITIVITY_VIOLATION"

    with pytest.raises(CounterfactualAssumptionError) as exc_info_low:
        compute_exact_estimators(
            logged_actions=[PolicyAction.APPROVE_PRIMARY, PolicyAction.APPROVE_PRIMARY],
            candidate_actions=[PolicyAction.APPROVE_PRIMARY, PolicyAction.APPROVE_PRIMARY],
            propensities=[0.5, 0.005],
            observed_rewards=[1.0, 1.0],
            dm_candidate_rewards=[0.8, 0.8],
            dm_logged_rewards=[0.8, 0.8],
            min_propensity_threshold=0.02,
        )
    assert exc_info_low.value.code == "POSITIVITY_VIOLATION"


def test_counterfactual_nan_propensity_and_unsupported_action_fail_explicitly() -> None:
    with pytest.raises(CounterfactualAssumptionError) as exc_nan:
        compute_exact_estimators(
            logged_actions=[PolicyAction.APPROVE_PRIMARY],
            candidate_actions=[PolicyAction.APPROVE_PRIMARY],
            propensities=[math.nan],
            observed_rewards=[1.0],
            dm_candidate_rewards=[0.9],
            dm_logged_rewards=[0.9],
        )
    assert exc_nan.value.code == "INVALID_PROPENSITY"

    with pytest.raises(CounterfactualAssumptionError) as exc_sup:
        compute_exact_estimators(
            logged_actions=[PolicyAction.APPROVE_PRIMARY, PolicyAction.APPROVE_PRIMARY],
            candidate_actions=[PolicyAction.STEP_UP_3DS, PolicyAction.APPROVE_PRIMARY],
            propensities=[0.5, 0.5],
            observed_rewards=[1.0, 1.0],
            dm_candidate_rewards=[0.8, 0.8],
            dm_logged_rewards=[0.8, 0.8],
        )
    assert exc_sup.value.code == "INSUFFICIENT_SUPPORT"


def test_counterfactual_ess_collapse_raises_explicit_error() -> None:
    # 10 samples, only 1 match -> ESS = 1.0 -> ess_ratio = 0.10 < 0.25
    logged = [PolicyAction.APPROVE_PRIMARY] + [PolicyAction.DECLINE_POLICY] * 9
    cand = [PolicyAction.APPROVE_PRIMARY] * 10
    with pytest.raises(CounterfactualAssumptionError) as exc_ess:
        compute_exact_estimators(
            logged_actions=logged,
            candidate_actions=cand,
            propensities=[0.5] * 10,
            observed_rewards=[1.0] + [0.0] * 9,
            dm_candidate_rewards=[0.8] * 10,
            dm_logged_rewards=[0.8] + [0.0] * 9,
            min_ess_ratio=0.25,
            strict_ess=True,
        )
    assert exc_ess.value.code == "ESS_COLLAPSE"


def test_end_to_end_constrained_policy_meets_constraints_and_reports_bounds() -> None:
    config = load_config(ROOT / "config" / "default.yaml")
    result = run_full_experiment(config)

    constrained_audit = result["policy_audit"]["constrained_risk_aware"]
    assert constrained_audit["passed"] is True, f"Hidden/unexpected breach: {constrained_audit}"
    assert constrained_audit["decision_level_breaches"] == 0
    assert constrained_audit["portfolio_breaches"] == []
    assert constrained_audit["expected_loss_bps"] <= config.policy.max_expected_loss_bps
    assert constrained_audit["p95_latency_ms"] <= config.policy.max_p95_latency_ms

    for policy_name in ("baseline_threshold", "unconstrained_ev", "constrained_risk_aware"):
        rep = result["policy_comparison"][policy_name]
        acc_b = rep["net_acceptance_bounds"]
        loss_b = rep["expected_loss_bps_bounds"]
        assert 0.0 <= acc_b["lower"] <= acc_b["estimate"] <= acc_b["upper"] <= 1.0
        assert 0.0 <= loss_b["lower"] <= loss_b["estimate"] <= loss_b["upper"]
        # DR estimator should closely track synthetic ground-truth oracle within documented tolerance
        assert rep["dr_vs_oracle_acceptance_abs_err"] < 0.06
