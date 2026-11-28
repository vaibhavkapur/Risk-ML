"""Integration, boundary, resilience, drift, fallback, and reproducibility tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ato_engine.audit import DecisionAuditLogger, compare_decision_replays
from ato_engine.config import ConfigurationError, EngineConfig, load_config
from ato_engine.engine import ATODetectionEngine
from ato_engine.features import extract_stream_features
from ato_engine.models import train_model_bundle
from ato_engine.schemas import FallbackReason, LoginSessionEvent, StepUpAction
from ato_engine.store import InMemoryStateStore
from ato_engine.synthetic import generate_synthetic_dataset


@pytest.fixture(scope="module")
def model_bundle_and_cfg():
    cfg = EngineConfig(seed=42)
    events = generate_synthetic_dataset(seed=cfg.seed, num_users=50, num_days=28, base_events_per_day=40)
    fvs = extract_stream_features(events, InMemoryStateStore(), cfg.features)
    bundle, _, _ = train_model_bundle(events, fvs, cfg)
    return cfg, bundle


def test_shared_network_scenario_dampens_false_blocks(model_bundle_and_cfg) -> None:
    cfg, bundle = model_bundle_and_cfg
    engine = ATODetectionEngine(config=cfg, model_bundle=bundle, audit_logger=DecisionAuditLogger())
    t0 = datetime(2026, 2, 1, 9, 0, 0, tzinfo=timezone.utc)
    corp_ip = "198.51.100.210"

    for i in range(6):
        engine.evaluate(
            LoginSessionEvent(
                event_id=f"sh_{i}",
                timestamp=t0 + timedelta(minutes=i * 4),
                user_id=f"usr_sh_{i}",
                session_id=f"ses_sh_{i}",
                auth_method="sso",
                device_id=f"dev_sh_{i}",
                device_trust_score=0.92,
                ip_address=corp_ip,
                ip_reputation_risk=0.08,
                is_known_shared_network=True,
                geo_lat=37.7749,
                geo_lon=-122.4194,
                country_code="US",
            )
        )

    dec = engine.evaluate(
        LoginSessionEvent(
            event_id="sh_returning",
            timestamp=t0 + timedelta(minutes=40),
            user_id="usr_sh_0",
            session_id="ses_sh_ret",
            auth_method="sso",
            device_id="dev_sh_0",
            device_trust_score=0.92,
            ip_address=corp_ip,
            ip_reputation_risk=0.08,
            is_known_shared_network=False,
            geo_lat=37.7750,
            geo_lon=-122.4192,
            country_code="US",
        )
    )
    assert dec.features_snapshot["is_shared_network_ip"] == 1.0
    assert "shared_network_dampening" in dec.threshold_adjustments
    assert dec.action == StepUpAction.ALLOW


def test_missing_data_triggers_fallback_and_never_fails_open(model_bundle_and_cfg) -> None:
    cfg, bundle = model_bundle_and_cfg
    engine = ATODetectionEngine(config=cfg, model_bundle=bundle, audit_logger=DecisionAuditLogger())
    t0 = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)

    dec = engine.evaluate(
        LoginSessionEvent(
            event_id="miss_01",
            timestamp=t0,
            user_id="usr_miss",
            session_id="ses_miss",
            device_id=None,
            device_trust_score=None,
            ip_address=None,
            ip_reputation_risk=None,
            geo_lat=None,
            geo_lon=None,
        )
    )
    assert dec.fallback_used is True
    assert dec.fallback_reason == FallbackReason.EXCESSIVE_MISSING_FEATURES
    assert dec.action != StepUpAction.ALLOW
    assert len(dec.evidence.missing_fields) == 5


def test_repeated_attempts_escalation_and_block(model_bundle_and_cfg) -> None:
    cfg, bundle = model_bundle_and_cfg
    engine = ATODetectionEngine(config=cfg, model_bundle=bundle, audit_logger=DecisionAuditLogger())
    t0 = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)

    for i in range(5):
        engine.evaluate(
            LoginSessionEvent(
                event_id=f"rep_f_{i}",
                timestamp=t0 + timedelta(minutes=i),
                user_id="usr_brute",
                session_id=f"ses_bf_{i}",
                auth_outcome="failure",
                auth_method="password",
                device_id="dev_b",
                device_trust_score=0.80,
                ip_address="192.0.2.77",
                ip_reputation_risk=0.15,
                geo_lat=40.7128,
                geo_lon=-74.0060,
            )
        )
    dec_5 = engine.evaluate(
        LoginSessionEvent(
            event_id="rep_after_5",
            timestamp=t0 + timedelta(minutes=6),
            user_id="usr_brute",
            session_id="ses_b5",
            auth_outcome="success",
            auth_method="password",
            device_id="dev_b",
            device_trust_score=0.85,
            ip_address="192.0.2.77",
            ip_reputation_risk=0.15,
            geo_lat=40.7128,
            geo_lon=-74.0060,
        )
    )
    assert dec_5.action in (StepUpAction.CHALLENGE_WEBAUTHN, StepUpAction.BLOCK_AND_RESET)
    assert "HARD_STEP_UP_REPEATED_ATTEMPTS_1H" in dec_5.evidence.triggered_rules

    for i in range(5, 11):
        engine.evaluate(
            LoginSessionEvent(
                event_id=f"rep_f_{i}",
                timestamp=t0 + timedelta(minutes=i + 2),
                user_id="usr_brute",
                session_id=f"ses_bf_{i}",
                auth_outcome="failure",
                auth_method="password",
                device_id="dev_b",
                device_trust_score=0.80,
                ip_address="192.0.2.77",
                ip_reputation_risk=0.15,
                geo_lat=40.7128,
                geo_lon=-74.0060,
            )
        )
    dec_10 = engine.evaluate(
        LoginSessionEvent(
            event_id="rep_after_10",
            timestamp=t0 + timedelta(minutes=20),
            user_id="usr_brute",
            session_id="ses_b10",
            auth_outcome="success",
            auth_method="password",
            device_id="dev_b",
            device_trust_score=0.90,
            ip_address="192.0.2.77",
            ip_reputation_risk=0.10,
            geo_lat=40.7128,
            geo_lon=-74.0060,
        )
    )
    assert dec_10.action == StepUpAction.BLOCK_AND_RESET
    assert "HARD_BLOCK_REPEATED_ATTEMPTS_1H" in dec_10.evidence.triggered_rules


def test_feature_expiry_past_ttl(model_bundle_and_cfg) -> None:
    cfg, bundle = model_bundle_and_cfg
    engine = ATODetectionEngine(config=cfg, model_bundle=bundle, audit_logger=DecisionAuditLogger())
    t0 = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)

    engine.evaluate(
        LoginSessionEvent(
            event_id="exp_01",
            timestamp=t0,
            user_id="usr_expired",
            session_id="ses_e1",
            auth_method="webauthn",
            device_id="dev_e1",
            device_trust_score=0.95,
            ip_address="192.0.2.88",
            ip_reputation_risk=0.05,
            geo_lat=40.7128,
            geo_lon=-74.0060,
        )
    )

    # 31 days later (> 30d TTL)
    dec_expired = engine.evaluate(
        LoginSessionEvent(
            event_id="exp_02",
            timestamp=t0 + timedelta(days=31),
            user_id="usr_expired",
            session_id="ses_e2",
            auth_method="webauthn",
            device_id="dev_e1",
            device_trust_score=0.95,
            ip_address="192.0.2.88",
            ip_reputation_risk=0.05,
            geo_lat=40.7128,
            geo_lon=-74.0060,
        )
    )
    assert dec_expired.evidence.profile_expired is True
    assert dec_expired.features_snapshot["profile_expired"] == 1.0
    assert dec_expired.features_snapshot["user_prior_event_count_30d"] == 0.0
    assert dec_expired.features_snapshot["is_new_device_for_user"] == 1.0
    assert dec_expired.action != StepUpAction.ALLOW


def test_drift_alert_store_outage_stale_model_and_audit_logging(
    model_bundle_and_cfg, tmp_path: Path
) -> None:
    cfg, bundle = model_bundle_and_cfg
    audit_path = tmp_path / "audit.jsonl"
    store = InMemoryStateStore()
    engine = ATODetectionEngine(
        config=cfg,
        model_bundle=bundle,
        store=store,
        audit_logger=DecisionAuditLogger(log_path=audit_path),
    )
    t0 = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Simulate store outage -> explicit fallback, still logged
    store.set_available(False)
    dec_outage = engine.evaluate(
        LoginSessionEvent(
            event_id="outage_01",
            timestamp=t0,
            user_id="usr_out",
            session_id="ses_out",
            device_id="dev_out",
            device_trust_score=0.90,
            ip_address="192.0.2.15",
            ip_reputation_risk=0.05,
            geo_lat=40.7128,
            geo_lon=-74.0060,
        )
    )
    assert dec_outage.fallback_used is True
    assert dec_outage.fallback_reason == FallbackReason.STORE_UNAVAILABLE
    assert dec_outage.action != StepUpAction.ALLOW
    store.set_available(True)

    # Simulate stale model (> 90 days after training timestamp)
    dec_stale = engine.evaluate(
        LoginSessionEvent(
            event_id="stale_01",
            timestamp=bundle.trained_at + timedelta(days=cfg.policy.stale_model_max_age_days + 5),
            user_id="usr_stale",
            session_id="ses_stale",
            device_id="dev_stale",
            device_trust_score=0.90,
            ip_address="192.0.2.16",
            ip_reputation_risk=0.05,
            geo_lat=40.7128,
            geo_lon=-74.0060,
        )
    )
    assert dec_stale.fallback_used is True
    assert dec_stale.fallback_reason == FallbackReason.MODEL_STALE

    # Trigger critical drift via high-risk attack wave
    for i in range(25):
        engine.evaluate(
            LoginSessionEvent(
                event_id=f"drift_ev_{i}",
                timestamp=t0 + timedelta(minutes=i + 1),
                user_id=f"usr_dr_{i}",
                session_id=f"ses_dr_{i}",
                device_id=f"dev_dr_{i}",
                device_trust_score=0.08,
                ip_address="203.0.113.91",
                ip_reputation_risk=0.96,
                geo_lat=51.5074,
                geo_lon=-0.1278,
                sensitive_actions=["change_mfa", "add_payee"],
                session_duration_seconds=20.0,
            )
        )
    drift_report = engine.check_drift()
    assert drift_report.critical_drift_detected is True
    assert engine.critical_drift_breaker is True

    # Every decision logged on disk and in memory
    persisted = engine.audit_logger.read_persisted_records()
    assert len(persisted) == engine.counters.total_decisions == 27


def test_reproducibility_and_missing_config_error(model_bundle_and_cfg, tmp_path: Path) -> None:
    cfg, bundle = model_bundle_and_cfg
    stream = generate_synthetic_dataset(seed=99, num_users=15, num_days=5, base_events_per_day=10)

    e1 = ATODetectionEngine(config=cfg, model_bundle=bundle, audit_logger=DecisionAuditLogger())
    e2 = ATODetectionEngine(config=cfg, model_bundle=bundle, audit_logger=DecisionAuditLogger())
    r1 = e1.evaluate_batch(stream)
    r2 = e2.evaluate_batch(stream)
    cmp = compare_decision_replays(r1, r2)
    assert cmp["matched"] is True
    assert cmp["checked"] == len(stream)

    # Missing config fails explicitly (no silent default)
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")

    # Malformed thresholds config fails explicitly
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("policy:\n  base_thresholds:\n    challenge_otp: 0.9\n    challenge_webauthn: 0.5\n    block_and_reset: 0.8\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="base_thresholds"):
        load_config(bad_cfg)
