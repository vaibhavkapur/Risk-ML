"""Domain correctness tests: hard declines, duplicate attempts, and uncertain outcomes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from payment_auth_engine.schemas import AuthorizationAttempt, PolicyAction, validate_attempt
from payment_auth_engine.taxonomy import (
    ResponseCategory,
    classify_response_code,
    evaluate_retry_eligibility,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def _make_attempt(**overrides: object) -> AuthorizationAttempt:
    base = {
        "attempt_id": "att_test_1",
        "transaction_id": "txn_test_1",
        "idempotency_key": "idem_test_1",
        "timestamp_utc": datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc),
        "amount_cents": 5000,
        "currency": "USD",
        "merchant_id": "merch_001",
        "merchant_category_code": "5411",
        "card_network": "visa",
        "issuer_country": "US",
        "merchant_country": "US",
        "entry_mode": "ecommerce",
        "attempt_number": 2,
        "prior_response_code": "05",
        "prior_outcome_known": True,
        "risk_score": 0.10,
        "processor_id": "proc_a",
    }
    base.update(overrides)
    return AuthorizationAttempt.model_validate(base)


def test_reference_taxonomy_oracle_matches() -> None:
    oracle = json.loads((FIXTURES_DIR / "reference_oracle.json").read_text(encoding="utf-8"))["retry_taxonomy_oracle"]
    for code in oracle["hard_decline_codes"]:
        spec = classify_response_code(code)
        assert spec.category == ResponseCategory.HARD_DECLINE
        assert spec.retry_eligible is False

    for code in oracle["uncertain_codes"]:
        spec = classify_response_code(code)
        assert spec.category == ResponseCategory.UNCERTAIN
        assert spec.retry_eligible is False
        assert spec.requires_reconciliation is True

    for code in oracle["technical_retriable_codes"]:
        spec = classify_response_code(code)
        assert spec.category == ResponseCategory.TECHNICAL_RETRIABLE
        assert spec.retry_eligible is True


@pytest.mark.parametrize("code", ["14", "54", "57", "59", "63"])
def test_never_retry_hard_decline(code: str) -> None:
    attempt = _make_attempt(prior_response_code=code, attempt_number=2, risk_score=0.01)
    result = evaluate_retry_eligibility(attempt)
    assert result.eligible is False
    assert result.recommended_action == PolicyAction.BLOCK_HARD_DECLINE
    assert result.is_hard_stop is True
    assert result.reason_code == "HARD_DECLINE_NEVER_RETRY"


@pytest.mark.parametrize(
    "code,known",
    [
        ("TIMEOUT", False),
        ("TIMEOUT", True),
        ("NETWORK_ERROR", False),
        ("UNKNOWN", True),
        ("UNMAPPED_CODE", True),
        (None, False),
    ],
)
def test_never_retry_uncertain_authorization_outcome_blindly(code: object, known: bool) -> None:
    attempt = _make_attempt(prior_response_code=code, prior_outcome_known=known, attempt_number=2)
    result = evaluate_retry_eligibility(attempt)
    assert result.eligible is False
    assert result.recommended_action == PolicyAction.HOLD_RECONCILE
    assert result.requires_reconciliation is True
    assert result.reason_code == "UNCERTAIN_OUTCOME_NEVER_BLIND_RETRY"


def test_duplicate_idempotency_key_and_prior_approved_blocked() -> None:
    attempt = _make_attempt(attempt_number=1, prior_response_code=None)
    dup_idem = evaluate_retry_eligibility(attempt, idempotency_seen=True)
    assert dup_idem.eligible is False
    assert dup_idem.recommended_action == PolicyAction.DEDUPLICATE_RETURN_PRIOR
    assert dup_idem.reason_code == "DUPLICATE_IDEMPOTENCY_KEY"

    dup_txn = evaluate_retry_eligibility(attempt, prior_approved=True)
    assert dup_txn.eligible is False
    assert dup_txn.recommended_action == PolicyAction.DEDUPLICATE_RETURN_PRIOR
    assert dup_txn.reason_code == "ALREADY_APPROVED_TRANSACTION"

    retry_after_00 = _make_attempt(attempt_number=2, prior_response_code="00")
    res_00 = evaluate_retry_eligibility(retry_after_00)
    assert res_00.eligible is False
    assert res_00.recommended_action == PolicyAction.DEDUPLICATE_RETURN_PRIOR


def test_retriable_soft_and_technical_declines() -> None:
    tech = _make_attempt(attempt_number=2, prior_response_code="91")
    res_tech = evaluate_retry_eligibility(tech)
    assert res_tech.eligible is True
    assert res_tech.recommended_action == PolicyAction.ROUTE_SECONDARY

    sca = _make_attempt(attempt_number=2, prior_response_code="65", has_3ds=False)
    res_sca = evaluate_retry_eligibility(sca)
    assert res_sca.eligible is True
    assert res_sca.recommended_action == PolicyAction.STEP_UP_3DS

    nsf_non_recurring = _make_attempt(attempt_number=2, prior_response_code="51", is_recurring=False)
    res_nsf = evaluate_retry_eligibility(nsf_non_recurring)
    assert res_nsf.eligible is False

    nsf_recurring = _make_attempt(attempt_number=2, prior_response_code="51", is_recurring=True)
    res_nsf_rec = evaluate_retry_eligibility(nsf_recurring)
    assert res_nsf_rec.eligible is True


def test_boundary_fixtures_match_expected_retry_decisions() -> None:
    cases = json.loads((FIXTURES_DIR / "boundary_attempts.json").read_text(encoding="utf-8"))
    for case in cases:
        attempt = validate_attempt(case["payload"])
        res = evaluate_retry_eligibility(attempt, max_retries=2)
        assert res.eligible is case["expected_eligible"], f"Failed boundary case {case['case_name']}"
        if "expected_action" in case:
            assert res.recommended_action.value == case["expected_action"]
