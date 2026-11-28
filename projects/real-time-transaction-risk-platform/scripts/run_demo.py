#!/usr/bin/env python3
"""Reproducible end-to-end demo, parity verification, skew check, recovery check, and latency benchmark.

Produces machine-readable raw outputs in reports/ for acceptance evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from risk_platform.audit import AuditLogger, replay_audit_decisions
from risk_platform.config import load_config
from risk_platform.features import OfflineFeatureExtractor, check_feature_parity
from risk_platform.metrics import MetricsRegistry
from risk_platform.model import CalibratedLinearRiskModel
from risk_platform.monitoring import detect_training_serving_skew
from risk_platform.schemas import FeatureVector, TransactionEvent
from risk_platform.service import RiskScoringService
from risk_platform.store import InMemoryFeatureStore
from risk_platform.synthetic import generate_synthetic_dataset
from risk_platform.training import train_calibrated_baseline_model


class _SlowModelBackend:
    """Injected slow model backend for deterministic timeout fallback verification."""

    def __init__(self, delay_sec: float = 0.08) -> None:
        self.delay_sec = delay_sec

    @property
    def model_version(self) -> str:
        return "model.slow_test.v1"

    @property
    def feature_version(self) -> str:
        return "feat.v1"

    def predict(self, feature_vector: FeatureVector):
        time.sleep(self.delay_sec)
        return 0.05, 0.05, []


def run_demo(config_path: Path, benchmark_requests: int = 2000) -> Dict[str, Any]:
    cfg = load_config(config_path)
    reports_dir = PROJECT_ROOT / "reports"
    artifacts_dir = PROJECT_ROOT / "data" / "artifacts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify hand-calculated reference fixture oracle
    ref_fixture = json.loads(
        (PROJECT_ROOT / "data" / "fixtures" / "reference_transactions.json").read_text(encoding="utf-8")
    )
    ref_events = [TransactionEvent.from_dict(e) for e in ref_fixture["events"]]
    offline_ext = OfflineFeatureExtractor(fx_to_usd=cfg.schema.fx_to_usd)
    ref_offline = offline_ext.extract_batch(ref_events)
    ref_store = InMemoryFeatureStore(
        allowed_lateness_ms=cfg.ingestion.allowed_lateness_ms,
        retention_window_ms=cfg.ingestion.retention_window_ms,
        max_events_per_account=cfg.ingestion.max_events_per_account,
        max_staleness_ms=cfg.serving.max_feature_staleness_ms,
        fx_to_usd=cfg.schema.fx_to_usd,
    )
    ref_online: Dict[str, FeatureVector] = {}
    for ev in ref_events:
        ref_online[ev.event_id] = ref_store.get_features(ev)
        ref_store.ingest(ev)
    ref_parity = check_feature_parity(ref_offline, ref_online, atol=1e-9)

    # 2. Generate deterministic synthetic dataset & train calibrated baseline model
    events, labels = generate_synthetic_dataset(
        num_events=cfg.training.num_synthetic_events,
        num_accounts=cfg.training.num_accounts,
        seed=cfg.training.seed,
        label_maturity_hours=cfg.training.label_maturity_hours,
        fx_to_usd=cfg.schema.fx_to_usd,
    )
    model, training_meta = train_calibrated_baseline_model(
        events=events,
        labels=labels,
        config=cfg,
        model_version="model.logreg_cal.v1",
    )
    model_path = PROJECT_ROOT / cfg.storage.model_artifact_path
    model.save(model_path)
    reloaded_model = CalibratedLinearRiskModel.load(model_path)

    # 3. Full offline/online parity test on the chronological held-out test stream
    test_start_idx = int(
        len(events) * (cfg.training.train_fraction + cfg.training.calibration_fraction)
    )
    online_store = InMemoryFeatureStore(
        allowed_lateness_ms=cfg.ingestion.allowed_lateness_ms,
        retention_window_ms=cfg.ingestion.retention_window_ms,
        max_events_per_account=cfg.ingestion.max_events_per_account,
        max_staleness_ms=cfg.serving.max_feature_staleness_ms,
        fx_to_usd=cfg.schema.fx_to_usd,
    )
    audit_path = PROJECT_ROOT / cfg.storage.audit_log_path
    if audit_path.exists():
        audit_path.unlink()
    audit_logger = AuditLogger(path=audit_path)
    service = RiskScoringService(
        config=cfg,
        store=online_store,
        model=reloaded_model,
        audit_logger=audit_logger,
    )

    # Warm up online store with pre-test events, then score & ingest test slice
    for ev in events[:test_start_idx]:
        service.ingest_event(ev)

    offline_all = offline_ext.extract_batch(events)
    offline_test = {ev.event_id: offline_all[ev.event_id] for ev in events[test_start_idx:]}
    online_test: Dict[str, FeatureVector] = {}
    sample_decisions: List[Dict[str, Any]] = []

    for ev in events[test_start_idx:]:
        decision, fv = service.score_transaction(ev, ingest_after_score=True, record_audit=True)
        online_test[ev.event_id] = fv
        if len(sample_decisions) < 5:
            sample_decisions.append(decision.to_dict())

    stream_parity = check_feature_parity(offline_test, online_test, atol=1e-9)

    # 4. Training-serving skew report (in-distribution test slice vs drifted serving slice)
    clean_skew = detect_training_serving_skew(
        model=reloaded_model,
        serving_vectors=list(online_test.values()),
        psi_warning=cfg.monitoring.skew_psi_warning,
        psi_critical=cfg.monitoring.skew_psi_critical,
        mean_shift_sigma_threshold=cfg.monitoring.skew_mean_shift_sigma,
    )
    drift_events, _ = generate_synthetic_dataset(
        num_events=800,
        num_accounts=60,
        seed=cfg.training.seed + 999,
        label_maturity_hours=cfg.training.label_maturity_hours,
        fx_to_usd=cfg.schema.fx_to_usd,
        drift_mode=True,
        id_prefix="drift",
    )
    drift_vectors = list(offline_ext.extract_batch(drift_events).values())
    drifted_skew = detect_training_serving_skew(
        model=reloaded_model,
        serving_vectors=drift_vectors,
        psi_warning=cfg.monitoring.skew_psi_warning,
        psi_critical=cfg.monitoring.skew_psi_critical,
        mean_shift_sigma_threshold=cfg.monitoring.skew_mean_shift_sigma,
    )

    # 5. Checkpoint & restart recovery verification + audit replay verification
    ckpt_path = PROJECT_ROOT / cfg.storage.checkpoint_path
    online_store.save_checkpoint(ckpt_path)
    restored_store = InMemoryFeatureStore.load_checkpoint(ckpt_path)
    probe_event = TransactionEvent(
        schema_version="tx.v1",
        event_id="evt_recovery_probe_001",
        account_id=events[-1].account_id,
        card_token=events[-1].card_token,
        merchant_id="merch_syn_777",
        merchant_category="electronics",
        amount_cents=42500,
        currency="USD",
        country_code="GB",
        channel="ecommerce",
        device_id="dev_probe_01",
        event_ts_ms=events[-1].event_ts_ms + 30_000,
        ip_risk_score=0.45,
    )
    pre_restart_fv = online_store.get_features(probe_event)
    post_restart_fv = restored_store.get_features(probe_event)
    recovery_parity = check_feature_parity(
        {probe_event.event_id: pre_restart_fv},
        {probe_event.event_id: post_restart_fv},
        atol=0.0,
    )

    chain_verify = audit_logger.verify_hash_chain()
    replay_verify = replay_audit_decisions(
        audit_logger.read_records(), reloaded_model, service.policy
    )

    # 6. Model timeout fallback verification
    timeout_service = RiskScoringService(
        config=cfg,
        store=restored_store,
        model=_SlowModelBackend(delay_sec=0.06),
        audit_logger=AuditLogger(path=None),
    )
    timeout_decision, _ = timeout_service.score_transaction(probe_event, record_audit=False)
    timeout_service.close()

    # 7. Measured throughput & p50/p99 latency benchmark (in-process scoring path, no file I/O)
    bench_metrics = MetricsRegistry(max_latency_samples=benchmark_requests + 100)
    bench_service = RiskScoringService(
        config=cfg,
        store=restored_store,
        model=reloaded_model,
        metrics=bench_metrics,
        audit_logger=AuditLogger(path=None),
        enforce_async_timeout=True,
    )
    bench_events = events[-min(len(events), 400) :]
    latencies_ms: List[float] = []
    t_wall_start = time.perf_counter()
    for i in range(benchmark_requests):
        base_ev = bench_events[i % len(bench_events)]
        t0 = time.perf_counter()
        bench_service.score_transaction(base_ev, record_audit=False)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    wall_sec = time.perf_counter() - t_wall_start
    bench_service.close()
    service.close()

    lat_arr = np.array(latencies_ms, dtype=float)
    benchmark_report = {
        "measurement_type": "measured_local_in_process",
        "requests": benchmark_requests,
        "wall_time_sec": round(wall_sec, 4),
        "throughput_rps": round(benchmark_requests / max(wall_sec, 1e-9), 2),
        "p50_ms": round(float(np.percentile(lat_arr, 50)), 4),
        "p95_ms": round(float(np.percentile(lat_arr, 95)), 4),
        "p99_ms": round(float(np.percentile(lat_arr, 99)), 4),
        "mean_ms": round(float(np.mean(lat_arr)), 4),
        "max_ms": round(float(np.max(lat_arr)), 4),
        "inference_timeout_budget_ms": cfg.serving.inference_timeout_ms,
        "model_version": reloaded_model.model_version,
        "feature_version": reloaded_model.feature_version,
    }

    parity_report = {
        "reference_fixture_parity": ref_parity.to_dict(),
        "test_stream_parity": stream_parity.to_dict(),
        "restart_recovery_parity": recovery_parity.to_dict(),
    }

    skew_report = {
        "in_distribution_test_slice": clean_skew,
        "simulated_drifted_serving_slice": drifted_skew,
    }

    runtime_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "seed": cfg.training.seed,
        "model_version": reloaded_model.model_version,
        "model_artifact_sha256": reloaded_model.artifact_sha256,
        "feature_version": reloaded_model.feature_version,
        "policy_version": cfg.schema.policy_version,
        "optional_external_brokers": {
            "kafka_or_redpanda": "not_run_offline_prototype",
            "redis": "not_run_offline_prototype",
        },
    }

    acceptance_summary = {
        "runtime_metadata": runtime_metadata,
        "training_evaluation": training_meta["evaluation"],
        "training_splits": training_meta["splits"],
        "parity": {
            "reference_passed": ref_parity.passed,
            "stream_passed": stream_parity.passed,
            "stream_events_compared": stream_parity.events_compared,
            "stream_max_abs_diff": stream_parity.max_abs_diff,
            "recovery_passed": recovery_parity.passed,
        },
        "skew": {
            "clean_status": clean_skew["status"],
            "clean_max_psi": clean_skew["max_psi"],
            "drifted_status": drifted_skew["status"],
            "drifted_max_psi": drifted_skew["max_psi"],
            "drifted_alert_triggered": drifted_skew["alert_triggered"],
        },
        "timeout_fallback": {
            "fallback_used": timeout_decision.fallback_used,
            "fallback_reason": timeout_decision.fallback_reason,
            "model_version": timeout_decision.model_version,
            "action": timeout_decision.action,
            "latency_ms": round(timeout_decision.latency_ms, 4),
        },
        "audit_and_replay": {
            "hash_chain_valid": chain_verify["valid"],
            "records_checked": chain_verify["records_checked"],
            "replayed": replay_verify["replayed"],
            "matched": replay_verify["matched"],
            "exact_match": replay_verify["exact_match"],
        },
        "benchmark": benchmark_report,
        "sample_decisions": sample_decisions,
    }

    (reports_dir / "parity_report.json").write_text(
        json.dumps(parity_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (reports_dir / "skew_report.json").write_text(
        json.dumps(skew_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (reports_dir / "benchmark.json").write_text(
        json.dumps(benchmark_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (reports_dir / "runtime_metadata.json").write_text(
        json.dumps(runtime_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    (reports_dir / "acceptance_summary.json").write_text(
        json.dumps(acceptance_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    return acceptance_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Real-Time Transaction Risk Platform demo & benchmarks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--benchmark-requests",
        type=int,
        default=2000,
        help="Number of scoring requests in the local latency benchmark.",
    )
    args = parser.parse_args()
    summary = run_demo(config_path=args.config, benchmark_requests=args.benchmark_requests)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
