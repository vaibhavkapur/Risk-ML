"""Contract validation and fixture verification tests (M1/M2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from payment_auth_engine.schemas import ContractValidationError, validate_attempt

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def test_valid_attempt_fixtures_pass_contract() -> None:
    raw = json.loads((FIXTURES_DIR / "valid_attempts.json").read_text(encoding="utf-8"))
    assert len(raw) >= 5
    for item in raw:
        attempt = validate_attempt(item)
        assert attempt.attempt_id == item["attempt_id"]
        assert attempt.timestamp_utc.tzinfo is not None


def test_invalid_attempt_fixtures_raise_typed_errors() -> None:
    cases = json.loads((FIXTURES_DIR / "invalid_attempts.json").read_text(encoding="utf-8"))
    assert len(cases) >= 8
    for case in cases:
        with pytest.raises(ContractValidationError) as exc_info:
            validate_attempt(case["payload"])
        assert exc_info.value.error_code == case["expected_error_code"], (
            f"Case {case['case_name']}: expected {case['expected_error_code']}, got {exc_info.value.error_code}"
        )
        assert len(exc_info.value.details) >= 1


def test_boundary_attempt_fixtures_validate_cleanly() -> None:
    cases = json.loads((FIXTURES_DIR / "boundary_attempts.json").read_text(encoding="utf-8"))
    assert len(cases) >= 6
    for case in cases:
        attempt = validate_attempt(case["payload"])
        assert attempt.attempt_id == case["payload"]["attempt_id"]


def test_extra_unknown_fields_rejected() -> None:
    raw = json.loads((FIXTURES_DIR / "valid_attempts.json").read_text(encoding="utf-8"))[0]
    raw["unexpected_extra_field"] = "forbidden"
    with pytest.raises(ContractValidationError):
        validate_attempt(raw)
