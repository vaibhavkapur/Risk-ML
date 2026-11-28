"""FastAPI service endpoint tests covering evaluation, batch, metrics, drift, and error codes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ato_engine.api import create_app
from ato_engine.audit import DecisionAuditLogger
from ato_engine.config import EngineConfig
from ato_engine.engine import ATODetectionEngine
from ato_engine.features import extract_stream_features
from ato_engine.models import train_model_bundle
from ato_engine.store import InMemoryStateStore
from ato_engine.synthetic import generate_synthetic_dataset


def _make_client() -> TestClient:
    cfg = EngineConfig(seed=42)
    events = generate_synthetic_dataset(seed=cfg.seed, num_users=30, num_days=20, base_events_per_day=30)
    fvs = extract_stream_features(events, InMemoryStateStore(), cfg.features)
    bundle, _, _ = train_model_bundle(events, fvs, cfg)
    engine = ATODetectionEngine(
        config=cfg,
        model_bundle=bundle,
        store=InMemoryStateStore(),
        audit_logger=DecisionAuditLogger(),
    )
    return TestClient(create_app(engine=engine))


def test_api_health_evaluate_metrics_and_errors() -> None:
    client = _make_client()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model_loaded"] is True

    valid_payload = {
        "event_id": "api_001",
        "schema_version": "1.0.0",
        "timestamp": "2026-02-01T10:00:00Z",
        "user_id": "usr_api_01",
        "session_id": "ses_api_01",
        "auth_method": "webauthn",
        "device_id": "dev_api_01",
        "device_trust_score": 0.94,
        "ip_address": "192.0.2.55",
        "ip_reputation_risk": 0.05,
        "geo_lat": 40.7128,
        "geo_lon": -74.0060,
        "country_code": "US",
    }
    resp = client.post("/v1/evaluate", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision_id"].startswith("dec_")
    assert body["action"] in ("ALLOW", "CHALLENGE_OTP", "CHALLENGE_WEBAUTHN", "BLOCK_AND_RESET")
    assert "evidence" in body

    # Contract violation -> 422
    bad_ts = dict(valid_payload, event_id="api_bad_ts", timestamp="2026-02-01T10:00:00")
    resp_422 = client.post("/v1/evaluate", json=bad_ts)
    assert resp_422.status_code == 422

    # Unsupported input -> 400
    bad_ver = dict(valid_payload, event_id="api_bad_ver", schema_version="9.9.9")
    resp_400 = client.post("/v1/evaluate", json=bad_ver)
    assert resp_400.status_code == 400

    # Batch endpoint
    batch_resp = client.post(
        "/v1/evaluate/batch",
        json={"events": [dict(valid_payload, event_id="api_b1", timestamp="2026-02-01T10:05:00Z")]},
    )
    assert batch_resp.status_code == 200
    assert batch_resp.json()["count"] == 1

    # Metrics reflect logged decisions and errors
    metrics = client.get("/v1/metrics")
    assert metrics.status_code == 200
    m = metrics.json()
    assert m["audit_decisions_logged"] == 2
    assert m["counters"]["validation_errors"] == 1
    assert m["counters"]["unsupported_input_errors"] == 1

    # Drift check endpoint
    drift = client.post("/v1/drift/check")
    assert drift.status_code == 200
    assert "score_psi" in drift.json()
