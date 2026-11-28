"""Operational review tests: FastAPI endpoints, missing config, durable restart/replay, and optional Redis adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from smart_auth_engine.api import create_app
from smart_auth_engine.config import ConfigurationError, load_config
from smart_auth_engine.contracts import PaymentEvent
from smart_auth_engine.inference import AuthenticationEngine
from smart_auth_engine.policy import PolicyThresholds
from smart_auth_engine.store import JsonlDecisionStore, OptionalRedisDecisionStore
from smart_auth_engine.training import ReferenceOracleScorer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"


class _FakeRedisClient:
    """In-memory stub verifying OptionalRedisDecisionStore contract without a live Redis server."""

    def __init__(self) -> None:
        self._kv: Dict[str, str] = {}
        self._lists: Dict[str, List[str]] = {}

    def set(self, key: str, value: str) -> None:
        self._kv[key] = value

    def get(self, key: str) -> Optional[str]:
        return self._kv.get(key)

    def rpush(self, key: str, value: str) -> None:
        self._lists.setdefault(key, []).append(value)

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        items = self._lists.get(key, [])
        return items[start:] if end == -1 else items[start : end + 1]


def test_missing_or_invalid_config_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        load_config(tmp_path / "nonexistent.yaml")

    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("training:\n  train_end_day: 40\n  cal_end_day: 30\n  val_end_day: 20\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid configuration"):
        load_config(bad_cfg)


def test_fastapi_authenticate_step_up_replay_and_validation_errors(tmp_path: Path) -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    thresholds = PolicyThresholds(challenge_threshold=0.12, decline_threshold=0.55)
    store = JsonlDecisionStore(tmp_path / "decisions.jsonl")
    engine = AuthenticationEngine(cfg, thresholds, model=ReferenceOracleScorer(), store=store)
    client = TestClient(create_app(engine))

    # Health check
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    valid_payloads = json.loads((FIXTURES_DIR / "valid_payments.json").read_text(encoding="utf-8"))

    # Authenticate valid payment
    r_auth = client.post("/v1/authenticate", json={"event": valid_payloads[0]})
    assert r_auth.status_code == 200
    dec = r_auth.json()
    assert dec["action"] == "approve"
    decision_id = dec["decision_id"]

    # Fetch persisted decision
    r_get = client.get(f"/v1/decisions/{decision_id}")
    assert r_get.status_code == 200
    assert r_get.json()["decision"]["decision_id"] == decision_id

    # Replay decision and verify exact parity
    r_replay = client.post(f"/v1/replay/{decision_id}")
    assert r_replay.status_code == 200
    assert r_replay.json()["exact_match"] is True

    # Step-up simulation endpoint + outage simulation
    r_step = client.post(
        "/v1/step-up/simulate",
        json={"event": valid_payloads[1], "actual_is_fraud": False, "simulate_outage": True},
    )
    assert r_step.status_code == 200
    step_body = r_step.json()
    assert step_body["decision"]["fallback_used"] is True
    assert step_body["decision"]["action"] == "challenge"
    assert step_body["challenge_outcome"] is not None

    # Unsupported / invalid payload returns structured 422 ErrorResponse
    invalid_payloads = json.loads((FIXTURES_DIR / "invalid_payments.json").read_text(encoding="utf-8"))
    r_inv = client.post("/v1/authenticate", json={"event": invalid_payloads[0]["payload"]})
    assert r_inv.status_code == 422
    assert r_inv.json()["error_code"] == "VALIDATION_ERROR"

    # Unknown decision_id returns 404
    r_404 = client.get("/v1/decisions/dec_does_not_exist")
    assert r_404.status_code == 404

    # Diagnostic metrics endpoint
    r_metrics = client.get("/v1/metrics")
    assert r_metrics.status_code == 200
    assert r_metrics.json()["counters"]["total_requests"] >= 2

    # Simulated process restart: reload JsonlDecisionStore from disk and verify replay works
    restarted_store = JsonlDecisionStore(tmp_path / "decisions.jsonl")
    restarted_engine = AuthenticationEngine(cfg, thresholds, model=ReferenceOracleScorer(), store=restarted_store)
    replay_after_restart = restarted_engine.replay_decision(decision_id)
    assert replay_after_restart["exact_match"] is True


def test_optional_redis_store_adapter_with_fake_client() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    thresholds = PolicyThresholds(challenge_threshold=0.12, decline_threshold=0.55)
    redis_store = OptionalRedisDecisionStore(client=_FakeRedisClient())
    engine = AuthenticationEngine(cfg, thresholds, model=ReferenceOracleScorer(), store=redis_store)

    valid_payloads = json.loads((FIXTURES_DIR / "valid_payments.json").read_text(encoding="utf-8"))
    ev = PaymentEvent.model_validate(valid_payloads[0])
    dec = engine.evaluate(ev, record_store=True)

    loaded = redis_store.get_record(dec.decision_id)
    assert loaded is not None
    assert loaded[1].decision_id == dec.decision_id
    assert dec.decision_id in redis_store.list_decision_ids()
