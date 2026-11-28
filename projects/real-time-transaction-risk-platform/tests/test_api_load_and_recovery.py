"""FastAPI integration, audit replay, concurrent load, and checkpoint restart recovery tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from risk_platform.api import create_app
from risk_platform.audit import AuditLogger
from risk_platform.config import load_config
from risk_platform.features import check_feature_parity
from risk_platform.service import RiskScoringService
from risk_platform.store import InMemoryFeatureStore
from risk_platform.synthetic import generate_synthetic_dataset
from risk_platform.training import train_calibrated_baseline_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def _build_service(tmp_path: Path):
    cfg = load_config(DEFAULT_CFG_PATH)
    events, labels = generate_synthetic_dataset(
        num_events=1200,
        num_accounts=60,
        seed=cfg.training.seed,
        label_maturity_hours=cfg.training.label_maturity_hours,
        fx_to_usd=cfg.schema.fx_to_usd,
    )
    model, _ = train_calibrated_baseline_model(events, labels, cfg)
    store = InMemoryFeatureStore(
        allowed_lateness_ms=cfg.ingestion.allowed_lateness_ms,
        retention_window_ms=cfg.ingestion.retention_window_ms,
        max_events_per_account=cfg.ingestion.max_events_per_account,
        max_staleness_ms=cfg.serving.max_feature_staleness_ms,
        fx_to_usd=cfg.schema.fx_to_usd,
    )
    audit = AuditLogger(path=tmp_path / "audit.jsonl")
    service = RiskScoringService(config=cfg, store=store, model=model, audit_logger=audit)
    return cfg, events, model, store, service


def test_fastapi_endpoints_traceability_errors_and_audit_replay(tmp_path: Path) -> None:
    _, events, _, _, service = _build_service(tmp_path)
    app = create_app(service)
    client = TestClient(app)

    try:
        # Health & active model
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["model_version"] == "model.logreg_cal.v1"
        assert r_health.json()["feature_version"] == "feat.v1"

        r_model = client.get("/v1/models/active")
        assert r_model.status_code == 200
        assert len(r_model.json()["artifact_sha256"]) == 64

        # Ingest event + idempotent duplicate + conflicting duplicate (409)
        ev0 = events[0].to_dict()
        r_ing1 = client.post("/v1/events/ingest", json=ev0)
        assert r_ing1.status_code == 200
        assert r_ing1.json()["status"] == "ACCEPTED"

        r_ing2 = client.post("/v1/events/ingest", json=ev0)
        assert r_ing2.status_code == 200
        assert r_ing2.json()["status"] == "DUPLICATE_IGNORED"

        ev0_conflict = dict(ev0)
        ev0_conflict["amount_cents"] = ev0["amount_cents"] + 500
        r_conf = client.post("/v1/events/ingest", json=ev0_conflict)
        assert r_conf.status_code == 409
        assert r_conf.json()["detail"]["error_code"] == "DUPLICATE_CONFLICT"

        # Sensitive PAN rejected with 422
        ev_pan = dict(ev0)
        ev_pan["event_id"] = "evt_pan_test"
        ev_pan["card_token"] = "4111-1111-1111-1111"
        r_pan = client.post("/v1/events/ingest", json=ev_pan)
        assert r_pan.status_code == 422
        assert r_pan.json()["detail"]["error_code"] == "SENSITIVE_DATA_VIOLATION"

        # Score sequence with ingest_after_score=True and check traceability headers
        for ev in events[1:15]:
            r_score = client.post(
                "/v1/score",
                json={"event": ev.to_dict(), "ingest_after_score": True},
            )
            assert r_score.status_code == 200
            body = r_score.json()
            dec = body["decision"]
            assert dec["model_version"] == "model.logreg_cal.v1"
            assert dec["feature_version"] == "feat.v1"
            assert dec["policy_version"] == "policy.v1"
            assert dec["decision_id"].startswith("dec_")
            assert r_score.headers["X-Decision-Id"] == dec["decision_id"]
            assert r_score.headers["X-Model-Version"] == "model.logreg_cal.v1"
            assert r_score.headers["X-Feature-Version"] == "feat.v1"

        # Metrics endpoint
        r_metrics = client.get("/metrics")
        assert r_metrics.status_code == 200
        assert r_metrics.json()["service"]["counters"]["decisions_total"] == 14

        # Replay audit log via API
        r_replay = client.post("/v1/audit/replay")
        assert r_replay.status_code == 200
        rep_body = r_replay.json()
        assert rep_body["hash_chain"]["valid"] is True
        assert rep_body["hash_chain"]["records_checked"] == 14
        assert rep_body["replay"]["replayed"] == 14
        assert rep_body["replay"]["matched"] == 14
        assert rep_body["replay"]["exact_match"] is True
    finally:
        service.close()


def test_checkpoint_restart_recovery_preserves_state_and_decisions(tmp_path: Path) -> None:
    cfg, events, model, store, service = _build_service(tmp_path)
    ckpt_path = tmp_path / "store_checkpoint.json"

    try:
        # Ingest first 200 events and checkpoint
        for ev in events[:200]:
            service.ingest_event(ev)
        store.save_checkpoint(ckpt_path)

        probe_events = events[200:225]
        pre_features = {}
        pre_decisions = {}
        for ev in probe_events:
            dec, fv = service.score_transaction(ev, record_audit=False)
            pre_features[ev.event_id] = fv
            pre_decisions[ev.event_id] = dec
    finally:
        service.close()

    # Simulate process restart with fresh store restored from checkpoint
    restored_store = InMemoryFeatureStore.load_checkpoint(ckpt_path)
    restored_service = RiskScoringService(
        config=cfg,
        store=restored_store,
        model=model,
        audit_logger=AuditLogger(path=None),
    )
    try:
        # Duplicate event from before restart must STILL be deduplicated after restart
        dup_res = restored_service.ingest_event(events[0])
        assert dup_res.status == "DUPLICATE_IGNORED"

        post_features = {}
        for ev in probe_events:
            dec_after, fv_after = restored_service.score_transaction(ev, record_audit=False)
            post_features[ev.event_id] = fv_after
            dec_before = pre_decisions[ev.event_id]
            assert dec_after.decision_id == dec_before.decision_id
            assert dec_after.action == dec_before.action
            assert abs(dec_after.risk_score - dec_before.risk_score) <= 1e-12

        parity = check_feature_parity(pre_features, post_features, atol=0.0)
        assert parity.passed is True
        assert parity.max_abs_diff == 0.0
    finally:
        restored_service.close()


def test_concurrent_load_scoring_and_bounded_latency(tmp_path: Path) -> None:
    _, events, _, _, service = _build_service(tmp_path)
    try:
        for ev in events[:300]:
            service.ingest_event(ev)

        targets = events[300:700]

        def _score_one(ev):
            dec, _ = service.score_transaction(ev, record_audit=False)
            return dec

        with ThreadPoolExecutor(max_workers=4) as pool:
            decisions = list(pool.map(_score_one, targets))

        assert len(decisions) == 400
        assert all(d.action in ("APPROVE", "REVIEW", "DECLINE") for d in decisions)
        assert all(d.model_version == "model.logreg_cal.v1" for d in decisions)
        snap = service.metrics.snapshot()
        assert snap["counters"]["decisions_total"] == 400
        assert snap["scoring_latency"]["p99_ms"] < 25.0
    finally:
        service.close()
