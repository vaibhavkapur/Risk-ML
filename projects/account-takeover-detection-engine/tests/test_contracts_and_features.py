"""Tests for data contracts, validation errors, reference oracle fixture, and strict causality."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from ato_engine.config import FeaturesConfig
from ato_engine.features import (
    FEATURE_NAMES,
    compute_features,
    extract_features_from_store,
    extract_stream_features,
)
from ato_engine.models import rule_oracle_score
from ato_engine.schemas import (
    ContractValidationError,
    LoginSessionEvent,
    UnsupportedInputError,
    parse_and_validate_event,
)
from ato_engine.store import InMemoryStateStore


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def test_invalid_unsupported_and_boundary_fixtures() -> None:
    data = json.loads((FIXTURES_DIR / "invalid_and_boundary_events.json").read_text(encoding="utf-8"))

    for case in data["invalid_contract_cases"]:
        with pytest.raises(ContractValidationError, match="."):
            parse_and_validate_event(case["payload"])

    for case in data["unsupported_input_cases"]:
        with pytest.raises(UnsupportedInputError, match="."):
            parse_and_validate_event(case["payload"])

    for case in data["boundary_valid_cases"]:
        ev = parse_and_validate_event(case["payload"])
        assert ev.event_id == case["payload"]["event_id"]
        assert ev.timestamp.tzinfo is not None


def test_reference_oracle_hand_calculated_features_and_ordering() -> None:
    fixture = json.loads((FIXTURES_DIR / "reference_oracle.json").read_text(encoding="utf-8"))
    events = [parse_and_validate_event(raw) for raw in fixture["events"]]

    store = InMemoryStateStore()
    fvs = extract_stream_features(events, store=store, config=FeaturesConfig())
    by_id = {fv.event_id: fv for fv in fvs}

    # Verify all 22 feature names exist
    for fv in fvs:
        assert tuple(fv.values.keys()) == FEATURE_NAMES

    # Exact hand-calculated checks
    exp = fixture["expected_hand_calculated_features"]
    for key, val in exp["ref_002"].items():
        assert by_id["ref_002"].values[key] == pytest.approx(val, abs=1e-4)

    for key, val in exp["ref_003_new_device"].items():
        assert by_id["ref_003_new_device"].values[key] == pytest.approx(val, abs=1e-4)

    comp = by_id["ref_004_compromised"].values
    for key, val in exp["ref_004_compromised"].items():
        if key == "geo_velocity_kmh_min":
            assert comp["geo_velocity_kmh"] >= val
        else:
            assert comp[key] == pytest.approx(val, abs=1e-4)

    # Verify hand-calculated oracle ordering: compromised >> new_device >= returning
    s_ret = rule_oracle_score(by_id["ref_002"].values)
    s_new = rule_oracle_score(by_id["ref_003_new_device"].values)
    s_comp = rule_oracle_score(by_id["ref_004_compromised"].values)
    min_gap = fixture["expected_oracle_ordering"]["min_gap_compromised_minus_new_device"]
    assert s_new >= s_ret
    assert s_comp - s_new >= min_gap


def test_no_future_events_in_features_store_and_guard() -> None:
    """Acceptance criterion: no future events may ever influence feature values."""
    t0 = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
    past_ev = LoginSessionEvent(
        event_id="c_001",
        timestamp=t0 - timedelta(hours=2),
        user_id="usr_causal",
        session_id="ses_c1",
        device_id="dev_1",
        device_trust_score=0.9,
        ip_address="192.0.2.10",
        ip_reputation_risk=0.05,
        geo_lat=40.7128,
        geo_lon=-74.0060,
    )
    target_ev = LoginSessionEvent(
        event_id="c_002",
        timestamp=t0,
        user_id="usr_causal",
        session_id="ses_c2",
        device_id="dev_1",
        device_trust_score=0.9,
        ip_address="192.0.2.10",
        ip_reputation_risk=0.05,
        geo_lat=40.7128,
        geo_lon=-74.0060,
    )
    future_ev = LoginSessionEvent(
        event_id="c_003_future",
        timestamp=t0 + timedelta(minutes=1),
        user_id="usr_causal",
        session_id="ses_c3",
        auth_outcome="failure",
        device_id="dev_attacker",
        device_trust_score=0.05,
        ip_address="203.0.113.99",
        ip_reputation_risk=0.99,
        geo_lat=35.6762,
        geo_lon=139.6503,
        sensitive_actions=["change_mfa", "add_payee"],
    )

    cfg = FeaturesConfig()
    store_past_only = InMemoryStateStore()
    store_past_only.record_event(past_ev)
    fv_clean = extract_features_from_store(target_ev, store_past_only, cfg)

    # Even if future events are ALREADY present in the store, causal query must ignore them!
    store_with_future = InMemoryStateStore()
    store_with_future.record_event(past_ev)
    store_with_future.record_event(target_ev)  # same timestamp must also be excluded
    store_with_future.record_event(future_ev)
    fv_with_future = extract_features_from_store(target_ev, store_with_future, cfg)

    assert fv_clean.values == fv_with_future.values
    assert fv_with_future.values["failed_attempts_1h"] == 0.0
    assert fv_with_future.values["user_prior_event_count_30d"] == 1.0

    # Passing a future event directly to compute_features must raise ContractValidationError
    with pytest.raises(ContractValidationError, match="Causal violation"):
        compute_features(
            event=target_ev,
            user_history=[past_ev, future_ev],
            ip_history=[],
            has_expired_history=False,
            config=cfg,
        )
