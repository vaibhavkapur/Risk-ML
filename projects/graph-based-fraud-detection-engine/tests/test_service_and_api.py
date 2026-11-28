"""API serving, graph-traced explanation audit, analyst review workflow, and error handling tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from graph_fraud_engine.api import create_app
from graph_fraud_engine.config import load_config
from graph_fraud_engine.pipeline import run_training_pipeline
from graph_fraud_engine.service import FraudDetectionService
from graph_fraud_engine.synthetic import generate_synthetic_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_test_client():
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    bundle = generate_synthetic_dataset(config)
    artifacts = run_training_pipeline(bundle, config)
    service = FraudDetectionService.from_artifacts(artifacts)
    app = create_app(service)
    return TestClient(app), bundle, artifacts


def test_api_score_graph_traced_evidence_and_analyst_review_workflow() -> None:
    client, bundle, artifacts = _build_test_client()

    # Health check
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"
    assert set(h.json()["models_loaded"]) >= {"logistic", "boosted", "gnn"}

    # Find a test-split coordinated fraud ring transaction
    labels_by_tx = {l.tx_id: l for l in bundle.labels}
    val_cutoff = artifacts.evaluation_report["splits"]["val_cutoff_utc"]
    ring_tx = next(
        tx
        for tx in reversed(bundle.transactions)
        if tx.tx_timestamp.isoformat() > val_cutoff
        and labels_by_tx[tx.tx_id].fraud_pattern == "coordinated_fraud_ring"
    )

    resp = client.post(
        "/v1/score",
        json={"transaction": ring_tx.model_dump(mode="json"), "model_name": "gnn"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] in ("review", "block")
    assert body["review_case_id"] is not None

    # Verify explanations trace strictly to graph nodes & past timestamps
    expl = body["explanation"]
    assert len(expl["evidence_paths"]) >= 1
    tx_ts = ring_tx.tx_timestamp
    for path in expl["evidence_paths"]:
        assert len(path["nodes"]) >= 2
        assert len(path["edges"]) >= 1
        for edge in path["edges"]:
            edge_ts = datetime.fromisoformat(edge["timestamp"].replace("Z", "+00:00"))
            if edge["relation"].startswith("PRIOR_"):
                assert edge_ts < tx_ts, f"Future prior edge leaked in explanation: {edge}"
            else:
                assert edge_ts == tx_ts

    # Verify review queue lists the created case
    case_id = body["review_case_id"]
    reviews = client.get("/v1/reviews?status=open")
    assert reviews.status_code == 200
    assert any(c["case_id"] == case_id for c in reviews.json())

    # Fetch individual case
    detail = client.get(f"/v1/reviews/{case_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "open"

    # Analyst submits confirmed_fraud disposition
    disp = client.post(
        f"/v1/reviews/{case_id}/disposition",
        json={
            "analyst_id": "analyst_qa_01",
            "disposition": "confirmed_fraud",
            "notes": "Confirmed shared-device fraud ring in API integration test.",
        },
    )
    assert disp.status_code == 200
    assert disp.json()["disposition"] == "confirmed_fraud"

    # Case is now resolved
    detail_after = client.get(f"/v1/reviews/{case_id}")
    assert detail_after.status_code == 200
    assert detail_after.json()["status"] == "resolved"
    assert len(detail_after.json()["dispositions"]) == 1

    # Metrics reflect scored transaction & disposition
    m = client.get("/metrics")
    assert m.status_code == 200
    assert m.json()["counters"]["total_scored"] == 1
    assert m.json()["counters"]["dispositions_recorded"] == 1
    assert m.json()["resolved_review_cases"] == 1


def test_api_explicit_error_handling_no_silent_success() -> None:
    # Uninitialized service returns 503
    uninit_client = TestClient(create_app(None))
    assert uninit_client.get("/health").json()["status"] == "degraded"
    assert uninit_client.get("/metrics").status_code == 503

    client, _, _ = _build_test_client()

    # Unknown review case -> 404
    assert client.get("/v1/reviews/case_doesnotexist").status_code == 404
    assert (
        client.post(
            "/v1/reviews/case_doesnotexist/disposition",
            json={"analyst_id": "analyst_1", "disposition": "false_positive"},
        ).status_code
        == 404
    )

    # Unsupported status filter -> 400
    assert client.get("/v1/reviews?status=archived").status_code == 400

    # Invalid payload (PII ID + negative amount) -> 422
    bad_resp = client.post(
        "/v1/score",
        json={
            "transaction": {
                "tx_id": "tx_1234567890ab",
                "account_id": "not-safe@example.com",
                "device_id": "dev_1234567890ab",
                "merchant_id": "mer_1234567890ab",
                "tx_timestamp": "2025-04-01T10:00:00Z",
                "amount_usd": -10.0,
            }
        },
    )
    assert bad_resp.status_code == 422
