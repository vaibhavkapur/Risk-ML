"""Domain unit tests for chronological training, label maturity, calibration, artifacts,
and New Device vs Compromised Session separation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ato_engine.audit import DecisionAuditLogger
from ato_engine.config import EngineConfig
from ato_engine.engine import ATODetectionEngine
from ato_engine.features import extract_stream_features
from ato_engine.models import (
    ATORiskModelBundle,
    chronological_split_with_maturity,
    train_model_bundle,
)
from ato_engine.schemas import LoginSessionEvent, ModelArtifactError, StepUpAction
from ato_engine.store import InMemoryStateStore
from ato_engine.synthetic import generate_synthetic_dataset


@pytest.fixture(scope="module")
def trained_artifacts():
    cfg = EngineConfig(seed=42)
    events = generate_synthetic_dataset(seed=cfg.seed, num_users=60, num_days=30, base_events_per_day=45)
    fvs = extract_stream_features(events, InMemoryStateStore(), cfg.features)
    bundle, split, eval_summary = train_model_bundle(events, fvs, cfg)
    return cfg, events, fvs, bundle, split, eval_summary


def test_chronological_split_enforces_delayed_label_maturity(trained_artifacts) -> None:
    _, events, _, _, split, _ = trained_artifacts
    assert len(split.excluded_immature_train_indices) > 0
    for idx in split.train_indices:
        ev = events[idx]
        assert ev.label is not None
        assert ev.label_available_at is not None
        assert ev.label_available_at <= split.train_cutoff

    for idx in split.excluded_immature_train_indices:
        ev = events[idx]
        if ev.label is not None and ev.label_available_at is not None:
            assert ev.label_available_at > split.train_cutoff

    # Strict temporal ordering between splits
    max_train_ts = max(events[i].timestamp for i in split.train_indices)
    min_cal_ts = min(events[i].timestamp for i in split.calibration_indices)
    max_cal_ts = max(events[i].timestamp for i in split.calibration_indices)
    min_test_ts = min(events[i].timestamp for i in split.test_indices)
    assert max_train_ts <= split.train_cutoff <= min_cal_ts
    assert max_cal_ts <= split.calibration_cutoff <= min_test_ts


def test_recall_at_fixed_fpr_and_calibration_reported(trained_artifacts) -> None:
    _, _, _, _, _, eval_summary = trained_artifacts
    metrics = eval_summary["test_metrics"]
    ens = metrics["calibrated_ensemble"]
    oracle = metrics["rule_oracle_baseline"]
    anom = metrics["anomaly_isolation_forest"]
    lr = metrics["logistic_regression"]

    for m in (oracle, anom, lr, ens):
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["ece"] <= 1.0
        assert 0.0 <= m["brier_score"] <= 1.0
        for fpr_key in ("1.00%", "2.00%", "5.00%"):
            assert fpr_key in m["recall_at_fpr"]
            assert m["recall_at_fpr"][fpr_key]["achieved_fpr"] <= m["recall_at_fpr"][fpr_key]["target_fpr"] + 1e-6

    # Calibrated ensemble must achieve strong discrimination and low calibration error on synthetic stream
    assert ens["roc_auc"] >= 0.90
    assert ens["ece"] <= 0.12
    assert ens["recall_at_fpr"]["2.00%"]["recall"] >= 0.70


def test_new_device_vs_compromised_session(trained_artifacts) -> None:
    """Required domain test: benign new device must be separated from compromised session."""
    cfg, _, _, bundle, _, _ = trained_artifacts
    engine = ATODetectionEngine(
        config=cfg,
        model_bundle=bundle,
        store=InMemoryStateStore(),
        audit_logger=DecisionAuditLogger(),
    )

    t0 = datetime(2026, 2, 10, 10, 0, 0, tzinfo=timezone.utc)
    for d in range(3):
        engine.evaluate(
            LoginSessionEvent(
                event_id=f"nd_hist_{d}",
                timestamp=t0 + timedelta(days=d),
                user_id="usr_domain_01",
                session_id=f"ses_h_{d}",
                auth_method="webauthn",
                device_id="dev_known_01",
                device_trust_score=0.95,
                ip_address="192.0.2.42",
                ip_reputation_risk=0.04,
                geo_lat=40.7128,
                geo_lon=-74.0060,
                country_code="US",
                session_duration_seconds=360.0,
            )
        )

    # Benign new device: same user/IP/city/hour/auth, high trust, zero sensitive burst
    dec_new_device = engine.evaluate(
        LoginSessionEvent(
            event_id="nd_benign_new_device",
            timestamp=t0 + timedelta(days=4, minutes=10),
            user_id="usr_domain_01",
            session_id="ses_new_dev",
            auth_method="webauthn",
            device_id="dev_brand_new_phone",
            device_trust_score=0.89,
            ip_address="192.0.2.42",
            ip_reputation_risk=0.05,
            geo_lat=40.7180,
            geo_lon=-74.0030,
            country_code="US",
            session_duration_seconds=300.0,
        )
    )

    # Compromised session: impossible travel (NYC -> London in 30 min), low trust, high IP risk,
    # auth downgrade (webauthn -> password), rapid sensitive account changes
    dec_compromised = engine.evaluate(
        LoginSessionEvent(
            event_id="nd_compromised_session",
            timestamp=t0 + timedelta(days=4, minutes=40),
            user_id="usr_domain_01",
            session_id="ses_compromised",
            event_type="session_activity",
            auth_method="password",
            device_id="dev_unknown_atk",
            device_trust_score=0.08,
            ip_address="203.0.113.90",
            ip_reputation_risk=0.95,
            geo_lat=51.5074,
            geo_lon=-0.1278,
            country_code="GB",
            sensitive_actions=["change_mfa", "change_email", "add_payee"],
            session_duration_seconds=25.0,
        )
    )

    assert dec_new_device.features_snapshot["is_new_device_for_user"] == 1.0
    assert dec_compromised.features_snapshot["is_impossible_travel"] == 1.0
    assert dec_compromised.calibrated_risk_score > dec_new_device.calibrated_risk_score + 0.35
    assert dec_new_device.action in (StepUpAction.ALLOW, StepUpAction.CHALLENGE_OTP)
    assert dec_compromised.action == StepUpAction.BLOCK_AND_RESET
    assert len(dec_compromised.evidence.top_factors) >= 1
    assert "IMPOSSIBLE_TRAVEL_DETECTED" in dec_compromised.evidence.triggered_rules


def test_model_artifact_persistence_and_tamper_detection(trained_artifacts, tmp_path: Path) -> None:
    _, _, fvs, bundle, _, _ = trained_artifacts
    artifact = tmp_path / "model.joblib"
    digest = bundle.save(artifact)
    assert len(digest) == 64

    loaded = ATORiskModelBundle.load(artifact, verify_manifest=True)
    assert loaded.predict_one(fvs[-1]) == bundle.predict_one(fvs[-1])

    # Tamper with artifact bytes and verify checksum failure
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")
    with pytest.raises(ModelArtifactError, match="checksum mismatch"):
        ATORiskModelBundle.load(artifact, verify_manifest=True)
