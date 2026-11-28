#!/usr/bin/env python3
"""Reproducible end-to-end demo walkthrough of the ATO Detection Engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ato_engine.audit import DecisionAuditLogger
from ato_engine.config import load_config
from ato_engine.engine import ATODetectionEngine
from ato_engine.features import extract_stream_features
from ato_engine.models import ATORiskModelBundle, train_model_bundle
from ato_engine.schemas import LoginSessionEvent
from ato_engine.store import InMemoryStateStore
from ato_engine.synthetic import generate_synthetic_dataset


def main() -> int:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    model_path = PROJECT_ROOT / "reports" / "model_bundle.joblib"
    if model_path.exists():
        bundle = ATORiskModelBundle.load(model_path)
    else:
        events = generate_synthetic_dataset(seed=cfg.seed)
        fvs = extract_stream_features(events, InMemoryStateStore(), cfg.features)
        bundle, _, _ = train_model_bundle(events, fvs, cfg)

    demo_audit = PROJECT_ROOT / "reports" / "demo_audit.jsonl"
    if demo_audit.exists():
        demo_audit.unlink()

    engine = ATODetectionEngine(
        config=cfg,
        model_bundle=bundle,
        store=InMemoryStateStore(),
        audit_logger=DecisionAuditLogger(log_path=demo_audit),
    )

    t0 = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
    demo_events = [
        ("1. Returning trusted login", LoginSessionEvent(
            event_id="demo_001", timestamp=t0, user_id="usr_demo", session_id="ses_d1",
            auth_method="webauthn", device_id="dev_demo_primary", device_trust_score=0.96,
            ip_address="192.0.2.21", ip_reputation_risk=0.04, geo_lat=40.7128, geo_lon=-74.0060,
            country_code="US", session_duration_seconds=360.0,
        )),
        ("2. Benign new device (same city/IP)", LoginSessionEvent(
            event_id="demo_002", timestamp=t0 + timedelta(days=1, minutes=20), user_id="usr_demo",
            session_id="ses_d2", auth_method="webauthn", device_id="dev_demo_new_tablet",
            device_trust_score=0.89, ip_address="192.0.2.21", ip_reputation_risk=0.05,
            geo_lat=40.7210, geo_lon=-74.0020, country_code="US", session_duration_seconds=280.0,
        )),
        ("3. Compromised session (impossible travel + MFA/payee changes)", LoginSessionEvent(
            event_id="demo_003", timestamp=t0 + timedelta(days=1, minutes=50), user_id="usr_demo",
            session_id="ses_d3", event_type="session_activity", auth_method="password",
            device_id="dev_attacker_x", device_trust_score=0.10, ip_address="203.0.113.55",
            ip_reputation_risk=0.95, geo_lat=51.5074, geo_lon=-0.1278, country_code="GB",
            sensitive_actions=["change_mfa", "change_email", "add_payee"],
            session_duration_seconds=25.0,
        )),
        ("4. Missing telemetry fallback", LoginSessionEvent(
            event_id="demo_004", timestamp=t0 + timedelta(days=2), user_id="usr_demo",
            session_id="ses_d4", device_id=None, device_trust_score=None, ip_address=None,
            ip_reputation_risk=None, geo_lat=None, geo_lon=None,
        )),
    ]

    print("=== Account Takeover Detection Engine — Demo Walkthrough ===")
    for label, ev in demo_events:
        dec = engine.evaluate(ev)
        print(f"\n[{label}]")
        print(json.dumps({
            "decision_id": dec.decision_id,
            "action": dec.action.value,
            "calibrated_risk_score": dec.calibrated_risk_score,
            "anomaly_score": dec.anomaly_score,
            "fallback_used": dec.fallback_used,
            "fallback_reason": dec.fallback_reason.value,
            "triggered_rules": dec.evidence.triggered_rules,
            "summary": dec.evidence.summary,
            "top_factor": dec.evidence.top_factors[0].model_dump() if dec.evidence.top_factors else None,
        }, indent=2))

    print(f"\nAudit log records written: {engine.audit_logger.count()} -> {demo_audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
