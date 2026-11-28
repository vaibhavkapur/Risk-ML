"""Integration and resilience tests: processor outage, missing features, and safety invariants."""

from __future__ import annotations

import json
from pathlib import Path

from payment_auth_engine.config import load_config
from payment_auth_engine.engine import AuthorizationEngine
from payment_auth_engine.models import AuthorizationModelBundle
from payment_auth_engine.policy import ConstrainedRiskAwarePolicy
from payment_auth_engine.schemas import PolicyAction, ProcessorHealthStatus, validate_attempt
from payment_auth_engine.synthetic import generate_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "data" / "fixtures"


def _build_engine() -> AuthorizationEngine:
    config = load_config(ROOT / "config" / "default.yaml")
    dataset = generate_synthetic_dataset(config)
    bundle, _, cal_idx, _ = AuthorizationModelBundle.train(
        dataset.attempts, dataset.outcomes, config, dataset.as_of_utc
    )
    cal_attempts = [dataset.attempts[int(i)] for i in cal_idx]
    cal_preds = bundle.predict_batch_all_actions(cal_attempts)
    policy = ConstrainedRiskAwarePolicy(config)
    policy.fit_dual_multipliers(cal_attempts, cal_preds)
    return AuthorizationEngine(config=config, model_bundle=bundle, policy=policy)


def test_processor_outage_triggers_secondary_failover_and_total_outage_fails_closed() -> None:
    engine = _build_engine()
    valid_raw = json.loads((FIXTURES_DIR / "valid_attempts.json").read_text(encoding="utf-8"))
    low_risk = validate_attempt(valid_raw[0])  # primary is proc_a

    # Simulate primary processor outage
    engine.health_registry.set_health("proc_a", ProcessorHealthStatus.OUTAGE, reason="simulated_outage")
    decision_failover = engine.decide(low_risk, record_in_ledger=False)
    assert decision_failover.action == PolicyAction.ROUTE_SECONDARY
    assert decision_failover.processor_selected in ("proc_b", "proc_c")
    assert decision_failover.processor_selected != "proc_a"
    assert "PROCESSOR_OUTAGE_FAILOVER" in decision_failover.explanation.reason_codes

    # Simulate total outage across ALL configured processors -> must fail closed!
    for pid in list(engine.config.processors):
        engine.health_registry.set_health(pid, ProcessorHealthStatus.OUTAGE, reason="total_outage")
    decision_closed = engine.decide(low_risk, record_in_ledger=False)
    assert decision_closed.action == PolicyAction.BLOCK_PROCESSOR_OUTAGE
    assert decision_closed.processor_selected is None
    assert "ALL_PROCESSORS_OUTAGE_FAIL_CLOSED" in decision_closed.explanation.reason_codes

    snap = engine.diagnostics.snapshot()
    assert snap["counters"]["outage_failovers"] >= 1
    assert snap["counters"]["outage_fail_closed_blocks"] >= 1


def test_missing_features_handled_conservatively_with_explicit_diagnostics() -> None:
    engine = _build_engine()
    valid_raw = json.loads((FIXTURES_DIR / "valid_attempts.json").read_text(encoding="utf-8"))
    missing_attempt = validate_attempt(valid_raw[4])  # all 5 optional features null

    decision = engine.decide(missing_attempt)
    assert len(decision.explanation.missing_features) == 5
    assert "MISSING_FEATURES_CONSERVATIVE_PENALTY" in decision.explanation.reason_codes
    assert decision.any_constraint_breach is False

    # Compare against an identical attempt with clean low-risk observed features
    complete_attempt = missing_attempt.model_copy(
        update={
            "attempt_id": "att_complete_cmp",
            "idempotency_key": "idem_complete_cmp",
            "risk_score": 0.05,
            "velocity_1h": 0,
            "velocity_24h": 1,
            "account_age_days": 500,
            "bin_approval_rate_30d": 0.96,
        }
    )
    complete_dec = engine.decide(complete_attempt)
    assert complete_dec.explanation.missing_features == []
    # Conservative handling means missing features cannot have higher expected net value than clean low-risk features
    assert decision.expected_net_value_cents <= complete_dec.expected_net_value_cents

    snap = engine.diagnostics.snapshot()
    assert snap["counters"]["missing_feature_requests"] >= 1


def test_engine_enforces_hard_decline_duplicate_and_uncertain_safety_end_to_end() -> None:
    engine = _build_engine()
    valid_raw = json.loads((FIXTURES_DIR / "valid_attempts.json").read_text(encoding="utf-8"))
    boundary_raw = json.loads((FIXTURES_DIR / "boundary_attempts.json").read_text(encoding="utf-8"))

    first = validate_attempt(valid_raw[0])
    d1 = engine.decide(first)
    # Re-sending identical idempotency key must deduplicate without double-submitting
    d2 = engine.decide(first)
    assert d2.action == PolicyAction.DEDUPLICATE_RETURN_PRIOR
    assert d2.processor_selected is None

    # Hard decline retry (code 54)
    hard_decline = validate_attempt(boundary_raw[2]["payload"])
    d_hard = engine.decide(hard_decline)
    assert d_hard.action == PolicyAction.BLOCK_HARD_DECLINE
    assert d_hard.processor_selected is None

    # Uncertain timeout retry
    uncertain = validate_attempt(boundary_raw[3]["payload"])
    d_unc = engine.decide(uncertain)
    assert d_unc.action == PolicyAction.HOLD_RECONCILE
    assert d_unc.processor_selected is None

    snap = engine.diagnostics.snapshot()
    assert snap["counters"]["duplicate_attempt_blocks"] == 1
    assert snap["counters"]["hard_decline_blocks"] == 1
    assert snap["counters"]["uncertain_outcome_holds"] == 1
    assert d1.decision_id.startswith("dec_")
