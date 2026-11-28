"""FastAPI service and CLI interface tests."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from payment_auth_engine.api import create_app
from payment_auth_engine.cli import main as cli_main
from payment_auth_engine.config import load_config
from payment_auth_engine.engine import AuthorizationEngine
from payment_auth_engine.models import AuthorizationModelBundle
from payment_auth_engine.policy import ConstrainedRiskAwarePolicy
from payment_auth_engine.synthetic import generate_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "data" / "fixtures"


def _build_client() -> TestClient:
    config = load_config(ROOT / "config" / "default.yaml")
    dataset = generate_synthetic_dataset(config)
    bundle, _, cal_idx, _ = AuthorizationModelBundle.train(
        dataset.attempts, dataset.outcomes, config, dataset.as_of_utc
    )
    cal_attempts = [dataset.attempts[int(i)] for i in cal_idx]
    cal_preds = bundle.predict_batch_all_actions(cal_attempts)
    policy = ConstrainedRiskAwarePolicy(config)
    policy.fit_dual_multipliers(cal_attempts, cal_preds)
    engine = AuthorizationEngine(config=config, model_bundle=bundle, policy=policy)
    return TestClient(create_app(engine))


def test_fastapi_health_authorize_retry_and_outage_endpoints() -> None:
    client = _build_client()
    valid_raw = json.loads((FIXTURES_DIR / "valid_attempts.json").read_text(encoding="utf-8"))
    invalid_raw = json.loads((FIXTURES_DIR / "invalid_attempts.json").read_text(encoding="utf-8"))
    boundary_raw = json.loads((FIXTURES_DIR / "boundary_attempts.json").read_text(encoding="utf-8"))

    # Health check
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"
    assert r_health.json()["mode"] == "offline_synthetic_prototype"

    # Valid authorization
    r_auth = client.post("/v1/authorize", json=valid_raw[0])
    assert r_auth.status_code == 200
    body = r_auth.json()
    assert body["decision_id"].startswith("dec_")
    assert body["any_constraint_breach"] is False

    # Invalid payload -> structured 422 error
    r_inv = client.post("/v1/authorize", json=invalid_raw[1]["payload"])
    assert r_inv.status_code == 422
    assert r_inv.json()["error_code"] == "UNSUPPORTED_CURRENCY"

    # Retry eligibility endpoint on hard decline
    r_retry = client.post("/v1/retry-eligibility", json=boundary_raw[2]["payload"])
    assert r_retry.status_code == 200
    assert r_retry.json()["eligible"] is False
    assert r_retry.json()["recommended_action"] == "BLOCK_HARD_DECLINE"

    # Update processor health -> 200 for known, 404 for unknown
    r_outage = client.post(
        "/v1/processors/proc_a/health",
        json={"status": "OUTAGE", "reason": "test_outage"},
    )
    assert r_outage.status_code == 200
    assert r_outage.json()["status"] == "OUTAGE"

    r_unknown = client.post(
        "/v1/processors/nonexistent_proc/health",
        json={"status": "OUTAGE", "reason": "test"},
    )
    assert r_unknown.status_code == 404

    # Diagnostics endpoint reflects recorded activity
    r_diag = client.get("/v1/diagnostics")
    assert r_diag.status_code == 200
    assert r_diag.json()["diagnostics"]["counters"]["total_requests"] >= 1


def test_cli_demo_replay_and_missing_config_exit_codes(tmp_path: Path) -> None:
    cfg = str(ROOT / "config" / "default.yaml")
    out_dir = str(tmp_path / "reports")

    rc_demo = cli_main(["demo", "--config", cfg, "--output-dir", out_dir])
    assert rc_demo == 0
    assert (tmp_path / "reports" / "policy_comparison.json").exists()
    assert (tmp_path / "reports" / "audit_report.json").exists()

    rc_replay = cli_main(["replay", "--config", cfg])
    assert rc_replay == 0

    rc_missing = cli_main(["demo", "--config", str(tmp_path / "does_not_exist.yaml")])
    assert rc_missing == 2
