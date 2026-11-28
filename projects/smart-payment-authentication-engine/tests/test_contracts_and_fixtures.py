"""M1 tests: Canonical contracts, valid/invalid/boundary fixtures, and hand-calculated reference oracle."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from smart_auth_engine.config import load_config
from smart_auth_engine.contracts import (
    AuthAction,
    PaymentEvent,
    RiskLabel,
    build_stable_decision_id,
)
from smart_auth_engine.features import extract_features
from smart_auth_engine.policy import PolicyThresholds, decide_authentication_action
from smart_auth_engine.training import ReferenceOracleScorer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def test_valid_payment_fixtures_parse_cleanly() -> None:
    raw = json.loads((FIXTURES_DIR / "valid_payments.json").read_text(encoding="utf-8"))
    events = [PaymentEvent.model_validate(item) for item in raw]
    assert len(events) == 3
    assert events[0].amount_major == pytest.approx(45.0)
    assert events[0].is_cross_border is False
    # pay_valid_002 has DE billing and FR IP -> derived cross-border True
    assert events[1].is_cross_border is True
    # pay_valid_003_jpy is zero-decimal currency
    assert events[2].currency == "JPY"
    assert events[2].amount_major == pytest.approx(8500.0)


def test_invalid_payment_fixtures_fail_with_expected_errors() -> None:
    cases = json.loads((FIXTURES_DIR / "invalid_payments.json").read_text(encoding="utf-8"))
    assert len(cases) >= 7
    for case in cases:
        with pytest.raises(ValidationError) as exc_info:
            PaymentEvent.model_validate(case["payload"])
        assert case["expected_error_substring"] in str(exc_info.value), (
            f"Case {case['case_name']} error did not contain {case['expected_error_substring']!r}"
        )


def test_boundary_payment_fixtures_behavior() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    oracle = ReferenceOracleScorer()
    thresholds = PolicyThresholds(challenge_threshold=0.12, decline_threshold=0.55)
    cases = json.loads((FIXTURES_DIR / "boundary_payments.json").read_text(encoding="utf-8"))
    by_name = {c["case_name"]: PaymentEvent.model_validate(c["payload"]) for c in cases}

    # Exact minimum amount (1 minor unit) -> approve
    ev_min = by_name["exact_min_amount_1_minor"]
    ext_min = extract_features(ev_min)
    p_min, _, _ = oracle.predict_proba_event(ev_min)
    dec_min = decide_authentication_action(
        ev_min, ext_min, thresholds, cfg, oracle.model_version, p_min, p_min
    )
    assert dec_min.action == AuthAction.APPROVE
    assert dec_min.mandatory_rule_triggered is False

    # Exact mandatory high amount threshold (50,000 minor) -> mandatory challenge
    ev_eq = by_name["exact_mandatory_high_amount_threshold"]
    ext_eq = extract_features(ev_eq)
    p_eq, _, _ = oracle.predict_proba_event(ev_eq)
    dec_eq = decide_authentication_action(
        ev_eq, ext_eq, thresholds, cfg, oracle.model_version, p_eq, p_eq
    )
    assert dec_eq.action == AuthAction.CHALLENGE
    assert dec_eq.mandatory_rule_triggered is True
    assert "MANDATORY_HIGH_AMOUNT" in dec_eq.reason_codes

    # 49,999 minor (1 minor below mandatory threshold) -> no mandatory rule
    ev_below = by_name["just_below_mandatory_high_amount_threshold"]
    ext_below = extract_features(ev_below)
    p_below, _, _ = oracle.predict_proba_event(ev_below)
    dec_below = decide_authentication_action(
        ev_below, ext_below, thresholds, cfg, oracle.model_version, p_below, p_below
    )
    assert dec_below.mandatory_rule_triggered is False
    assert dec_below.action == AuthAction.APPROVE

    # All optional telemetry missing -> abstain via fallback
    ev_missing = by_name["all_optional_telemetry_missing_abstain"]
    ext_missing = extract_features(ev_missing)
    dec_missing = decide_authentication_action(
        ev_missing, ext_missing, thresholds, cfg, oracle.model_version, 0.05, 0.05
    )
    assert dec_missing.action == AuthAction.ABSTAIN
    assert dec_missing.fallback_used is True


def test_hand_calculated_reference_oracle_matches_tolerance() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    oracle = ReferenceOracleScorer()
    fixture = json.loads((FIXTURES_DIR / "reference_oracle.json").read_text(encoding="utf-8"))
    thresholds = PolicyThresholds(
        challenge_threshold=fixture["thresholds"]["challenge_threshold"],
        decline_threshold=fixture["thresholds"]["decline_threshold"],
    )

    for case in fixture["cases"]:
        ev = PaymentEvent.model_validate(case["payment"])
        ext = extract_features(ev)
        p_cal, p_raw, explanations = oracle.predict_proba_event(ev)

        if case["expected_oracle_prob"] is not None:
            assert p_cal == pytest.approx(
                case["expected_oracle_prob"], abs=cfg.evaluation.oracle_probability_tolerance
            ), f"Oracle probability mismatch for {case['name']}"

        dec = decide_authentication_action(
            event=ev,
            extracted=ext,
            thresholds=thresholds,
            cfg=cfg,
            model_version=oracle.model_version,
            calibrated_prob=p_cal,
            raw_prob=p_raw,
            explanations=explanations,
        )
        assert dec.action.value == case["expected_action"], f"Action mismatch for {case['name']}"
        assert dec.mandatory_rule_triggered is case["expected_mandatory_rule_triggered"]
        assert dec.override_triggered is case["expected_override_triggered"]
        assert dec.fallback_used is case["expected_fallback_used"]


def test_risk_label_causality_and_stable_decision_id() -> None:
    ts = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    lbl = RiskLabel(
        payment_id="pay_lbl_1",
        event_timestamp=ts,
        is_fraud=True,
        label_available_at=ts + timedelta(days=7),
        label_source="chargeback",
        fraud_loss_minor=12000,
    )
    assert lbl.is_mature_at(ts + timedelta(days=6)) is False
    assert lbl.is_mature_at(ts + timedelta(days=7)) is True

    with pytest.raises(ValidationError):
        RiskLabel(
            payment_id="pay_lbl_bad",
            event_timestamp=ts,
            is_fraud=True,
            label_available_at=ts - timedelta(seconds=1),
            label_source="chargeback",
            fraud_loss_minor=12000,
        )

    id1 = build_stable_decision_id("pay_1", ts, "m1", "1.0.0", "1.0.0")
    id2 = build_stable_decision_id("pay_1", ts, "m1", "1.0.0", "1.0.0")
    id3 = build_stable_decision_id("pay_2", ts, "m1", "1.0.0", "1.0.0")
    assert id1 == id2
    assert id1 != id3
