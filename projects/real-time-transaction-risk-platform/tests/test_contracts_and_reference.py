"""Contract validation, hand-calculated reference oracle, boundary cases, and operational error tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest

from risk_platform.config import load_config
from risk_platform.errors import (
    ArtifactIntegrityError,
    ConfigurationError,
    ContractValidationError,
    FeatureStoreError,
    SensitiveDataViolationError,
    UnsupportedCurrencyError,
    UnsupportedSchemaVersionError,
)
from risk_platform.features import FEATURE_NAMES, OfflineFeatureExtractor, check_feature_parity
from risk_platform.model import CalibratedLinearRiskModel
from risk_platform.policy import PolicyEngine
from risk_platform.schemas import TransactionEvent
from risk_platform.store import InMemoryFeatureStore, RedisFeatureStoreAdapter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"
DEFAULT_CFG_PATH = PROJECT_ROOT / "config" / "default.yaml"

ERROR_MAP = {
    "UnsupportedSchemaVersionError": UnsupportedSchemaVersionError,
    "UnsupportedCurrencyError": UnsupportedCurrencyError,
    "SensitiveDataViolationError": SensitiveDataViolationError,
    "ContractValidationError": ContractValidationError,
}


def test_hand_calculated_reference_oracle_offline_and_online() -> None:
    cfg = load_config(DEFAULT_CFG_PATH)
    fixture = json.loads((FIXTURES_DIR / "reference_transactions.json").read_text(encoding="utf-8"))
    events = [TransactionEvent.from_dict(raw) for raw in fixture["events"]]
    expected = fixture["expected_features"]

    offline_vectors = OfflineFeatureExtractor(fx_to_usd=cfg.schema.fx_to_usd).extract_batch(events)

    store = InMemoryFeatureStore(fx_to_usd=cfg.schema.fx_to_usd)
    online_vectors = {}
    for ev in events:
        online_vectors[ev.event_id] = store.get_features(ev)
        res = store.ingest(ev)
        assert res.status == "ACCEPTED"

    for ev in events:
        for fname in FEATURE_NAMES:
            exp_val = expected[ev.event_id][fname]
            off_val = offline_vectors[ev.event_id].values[fname]
            on_val = online_vectors[ev.event_id].values[fname]
            assert math.isclose(off_val, exp_val, rel_tol=0.0, abs_tol=1e-9), (
                f"Offline mismatch on {ev.event_id}.{fname}: got {off_val}, expected {exp_val}"
            )
            assert math.isclose(on_val, exp_val, rel_tol=0.0, abs_tol=1e-9), (
                f"Online mismatch on {ev.event_id}.{fname}: got {on_val}, expected {exp_val}"
            )

    parity = check_feature_parity(offline_vectors, online_vectors, atol=1e-12)
    assert parity.passed is True
    assert parity.events_compared == 4
    assert parity.max_abs_diff == 0.0


def test_invalid_transaction_fixtures_raise_explicit_errors() -> None:
    fixture = json.loads((FIXTURES_DIR / "invalid_transactions.json").read_text(encoding="utf-8"))
    for case in fixture["cases"]:
        expected_exc = ERROR_MAP[case["expected_error"]]
        with pytest.raises(expected_exc):
            TransactionEvent.from_dict(case["payload"])


def test_boundary_fixtures_window_edges_tie_break_and_unknown_inputs() -> None:
    cfg = load_config(DEFAULT_CFG_PATH)
    fixture = json.loads((FIXTURES_DIR / "boundary_transactions.json").read_text(encoding="utf-8"))
    cases = {c["case_name"]: TransactionEvent.from_dict(c["event"]) for c in fixture["cases"]}

    ev_min = cases["minimum_amount_1_cent"]
    ev_5m = cases["exact_5m_boundary_excluded"]
    ev_tie = cases["same_timestamp_lexicographic_tie_break"]
    ev_unknown = cases["unknown_mcc_conservative_input"]

    store = InMemoryFeatureStore(fx_to_usd=cfg.schema.fx_to_usd)
    store.ingest(ev_min)

    # Exactly 5 minutes later (300_000 ms): half-open (t - 5m, t] MUST exclude ev_min from 5m window,
    # while still including it in 1h and 24h windows!
    fv_5m = store.get_features(ev_5m)
    assert fv_5m.values["tx_count_5m"] == 0.0
    assert fv_5m.values["tx_count_1h"] == 1.0
    assert fv_5m.values["tx_count_24h"] == 1.0
    store.ingest(ev_5m)

    # Identical timestamp but lexicographically greater event_id ('evt_bnd_002b' > 'evt_bnd_002')
    fv_tie = store.get_features(ev_tie)
    assert fv_tie.values["tx_count_5m"] == 1.0
    assert fv_tie.values["tx_count_1h"] == 2.0

    # Unknown MCC + unknown country triggers conservative escalation even if model score is low
    fv_unk = store.get_features(ev_unknown)
    assert any(flag.startswith("UNKNOWN_MCC") for flag in fv_unk.unknown_input_flags)
    assert any(flag.startswith("UNKNOWN_COUNTRY") for flag in fv_unk.unknown_input_flags)

    policy = PolicyEngine(cfg.serving, fx_to_usd=cfg.schema.fx_to_usd)
    decision = policy.evaluate_model_score(
        event=ev_unknown,
        feature_vector=fv_unk,
        calibrated_score=0.10,
        raw_score=0.10,
        model_version="model.test.v1",
        attributions=[],
        latency_ms=0.2,
        decided_at_ms=ev_unknown.event_ts_ms,
    )
    assert decision.action == "REVIEW"
    assert decision.fallback_used is True
    assert decision.fallback_reason == "UNKNOWN_INPUT_CONSERVATIVE"
    assert "UNKNOWN_INPUT_CONSERVATIVE_ESCALATION" in decision.reason_codes


def test_operational_errors_missing_config_corrupt_artifact_and_redis_unavailable(
    tmp_path: Path,
) -> None:
    # Missing configuration file
    with pytest.raises(ConfigurationError):
        load_config(tmp_path / "does_not_exist.yaml")

    # Malformed thresholds in config
    bad_cfg = tmp_path / "bad.yaml"
    raw = DEFAULT_CFG_PATH.read_text(encoding="utf-8").replace(
        "review_threshold: 0.35", "review_threshold: 0.85"
    )
    bad_cfg.write_text(raw, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(bad_cfg)

    # Corrupted model artifact checksum
    model = CalibratedLinearRiskModel(
        model_version="model.test.v1",
        feature_version="feat.v1",
        feature_names=FEATURE_NAMES,
        scaler_mean=tuple(0.0 for _ in FEATURE_NAMES),
        scaler_scale=tuple(1.0 for _ in FEATURE_NAMES),
        weights=tuple(0.1 for _ in FEATURE_NAMES),
        intercept=-1.0,
        platt_a=1.0,
        platt_b=0.0,
        training_metadata={},
    )
    art_path = tmp_path / "model.json"
    model.save(art_path)
    tampered = json.loads(art_path.read_text(encoding="utf-8"))
    tampered["intercept"] = 9.99  # Tamper without updating artifact_sha256
    art_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        CalibratedLinearRiskModel.load(art_path)

    # Redis adapter fails explicitly when redis is not installed/running (no silent success)
    with pytest.raises(FeatureStoreError):
        RedisFeatureStoreAdapter("redis://127.0.0.1:6399/0")
