"""End-to-end chronological training, baseline comparison, scenario suite, and report writer."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import sklearn

from . import FEATURE_VERSION, POLICY_VERSION, SCHEMA_VERSION, __version__
from .audit import DecisionAuditLogger, compare_decision_replays
from .config import EngineConfig
from .engine import ATODetectionEngine
from .features import extract_stream_features
from .models import ATORiskModelBundle, rule_oracle_score, train_model_bundle
from .schemas import FallbackReason, LoginSessionEvent, StepUpAction
from .store import InMemoryStateStore
from .synthetic import generate_synthetic_dataset


def runtime_metadata(config: EngineConfig) -> Dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "policy_version": POLICY_VERSION,
        "config_version": config.config_version,
        "seed": config.seed,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
    }


def evaluate_policy_on_split(
    events: List[LoginSessionEvent],
    bundle: ATORiskModelBundle,
    config: EngineConfig,
    audit_path: Path | None = None,
) -> Tuple[Dict[str, Any], List[Any]]:
    """Replay events sequentially through ATODetectionEngine and summarize step-up rates."""
    engine = ATODetectionEngine(
        config=config,
        model_bundle=bundle,
        store=InMemoryStateStore(),
        audit_logger=DecisionAuditLogger(log_path=audit_path),
    )
    decisions = engine.evaluate_batch(events, record_to_store=True)

    benign_actions = Counter(
        d.action.value for ev, d in zip(events, decisions) if ev.label == 0
    )
    ato_actions = Counter(
        d.action.value for ev, d in zip(events, decisions) if ev.label == 1
    )
    n_benign = sum(benign_actions.values())
    n_ato = sum(ato_actions.values())

    benign_friction_rate = (
        (n_benign - benign_actions.get(StepUpAction.ALLOW.value, 0)) / max(n_benign, 1)
    )
    benign_block_rate = benign_actions.get(StepUpAction.BLOCK_AND_RESET.value, 0) / max(n_benign, 1)
    ato_intervened_rate = (
        (n_ato - ato_actions.get(StepUpAction.ALLOW.value, 0)) / max(n_ato, 1)
    )
    ato_strong_or_block_rate = (
        (
            ato_actions.get(StepUpAction.CHALLENGE_WEBAUTHN.value, 0)
            + ato_actions.get(StepUpAction.BLOCK_AND_RESET.value, 0)
        )
        / max(n_ato, 1)
    )

    summary = {
        "total_events": len(events),
        "decisions_logged": engine.audit_logger.count(),
        "every_decision_logged": engine.audit_logger.count() == len(events),
        "benign_count": n_benign,
        "ato_count": n_ato,
        "benign_actions": dict(benign_actions),
        "ato_actions": dict(ato_actions),
        "benign_step_up_or_block_rate": round(benign_friction_rate, 6),
        "benign_block_rate": round(benign_block_rate, 6),
        "ato_intervened_rate": round(ato_intervened_rate, 6),
        "ato_strong_challenge_or_block_rate": round(ato_strong_or_block_rate, 6),
        "counters": engine.counters.snapshot(),
    }
    return summary, decisions


def run_required_scenarios(
    bundle: ATORiskModelBundle,
    config: EngineConfig,
) -> Dict[str, Any]:
    """Run all required domain, integration, boundary, drift, and fallback scenarios."""
    results: Dict[str, Any] = {}

    # 1. Domain correctness: New device vs compromised session
    eng1 = ATODetectionEngine(config=config, model_bundle=bundle, audit_logger=DecisionAuditLogger())
    base_ts = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    hist = [
        LoginSessionEvent(
            event_id=f"scn_nd_h{i}",
            timestamp=base_ts + timedelta(days=i),
            user_id="usr_scn_01",
            session_id=f"ses_nd_h{i}",
            auth_method="webauthn",
            device_id="dev_scn_primary",
            device_trust_score=0.95,
            ip_address="192.0.2.50",
            ip_reputation_risk=0.05,
            geo_lat=40.7128,
            geo_lon=-74.0060,
            country_code="US",
            session_duration_seconds=300.0,
        )
        for i in range(3)
    ]
    for ev in hist:
        eng1.evaluate(ev)

    benign_new_dev = LoginSessionEvent(
        event_id="scn_nd_newdev",
        timestamp=base_ts + timedelta(days=4, minutes=15),
        user_id="usr_scn_01",
        session_id="ses_nd_newdev",
        auth_method="webauthn",
        device_id="dev_scn_new_laptop",
        device_trust_score=0.88,
        ip_address="192.0.2.50",
        ip_reputation_risk=0.06,
        geo_lat=40.7200,
        geo_lon=-74.0010,
        country_code="US",
        session_duration_seconds=260.0,
    )
    d_new_dev = eng1.evaluate(benign_new_dev)

    compromised = LoginSessionEvent(
        event_id="scn_nd_compromised",
        timestamp=base_ts + timedelta(days=4, minutes=45),
        user_id="usr_scn_01",
        session_id="ses_nd_compromised",
        event_type="session_activity",
        auth_method="password",
        device_id="dev_scn_attacker",
        device_trust_score=0.10,
        ip_address="203.0.113.88",
        ip_reputation_risk=0.94,
        geo_lat=51.5074,
        geo_lon=-0.1278,
        country_code="GB",
        sensitive_actions=["change_mfa", "change_email", "add_payee"],
        session_duration_seconds=30.0,
    )
    d_comp = eng1.evaluate(compromised)
    results["new_device_vs_compromised_session"] = {
        "new_device_score": d_new_dev.calibrated_risk_score,
        "new_device_action": d_new_dev.action.value,
        "compromised_score": d_comp.calibrated_risk_score,
        "compromised_action": d_comp.action.value,
        "passed": (
            d_comp.calibrated_risk_score > d_new_dev.calibrated_risk_score + 0.30
            and d_comp.action == StepUpAction.BLOCK_AND_RESET
            and d_new_dev.action != StepUpAction.BLOCK_AND_RESET
        ),
    }

    # 2. Shared network (corporate/campus NAT)
    eng2 = ATODetectionEngine(config=config, model_bundle=bundle, audit_logger=DecisionAuditLogger())
    shared_ip = "198.51.100.222"
    for i in range(6):
        eng2.evaluate(
            LoginSessionEvent(
                event_id=f"scn_sh_{i}",
                timestamp=base_ts + timedelta(minutes=i * 5),
                user_id=f"usr_corp_{i}",
                session_id=f"ses_corp_{i}",
                auth_method="sso",
                device_id=f"dev_corp_{i}",
                device_trust_score=0.90,
                ip_address=shared_ip,
                ip_reputation_risk=0.10,
                is_known_shared_network=True,
                geo_lat=37.7749,
                geo_lon=-122.4194,
                country_code="US",
                session_duration_seconds=400.0,
            )
        )
    d_shared = eng2.evaluate(
        LoginSessionEvent(
            event_id="scn_sh_target",
            timestamp=base_ts + timedelta(minutes=40),
            user_id="usr_corp_0",
            session_id="ses_corp_target",
            auth_method="sso",
            device_id="dev_corp_0",
            device_trust_score=0.91,
            ip_address=shared_ip,
            ip_reputation_risk=0.10,
            is_known_shared_network=False,  # inferred from prior peaceful multi-user IP history!
            geo_lat=37.7750,
            geo_lon=-122.4190,
            country_code="US",
            session_duration_seconds=350.0,
        )
    )
    results["shared_network"] = {
        "inferred_shared_network": d_shared.features_snapshot["is_shared_network_ip"],
        "threshold_adjustments": d_shared.threshold_adjustments,
        "score": d_shared.calibrated_risk_score,
        "action": d_shared.action.value,
        "passed": (
            d_shared.features_snapshot["is_shared_network_ip"] == 1.0
            and d_shared.action == StepUpAction.ALLOW
        ),
    }

    # 3. Missing data triggers explicit fallback policy (never silent fail-open)
    eng3 = ATODetectionEngine(config=config, model_bundle=bundle, audit_logger=DecisionAuditLogger())
    d_missing = eng3.evaluate(
        LoginSessionEvent(
            event_id="scn_missing_01",
            timestamp=base_ts,
            user_id="usr_missing_01",
            session_id="ses_missing_01",
            device_id=None,
            device_trust_score=None,
            ip_address=None,
            ip_reputation_risk=None,
            geo_lat=None,
            geo_lon=None,
        )
    )
    results["missing_data_fallback"] = {
        "missing_fields": d_missing.evidence.missing_fields,
        "fallback_used": d_missing.fallback_used,
        "fallback_reason": d_missing.fallback_reason.value,
        "action": d_missing.action.value,
        "passed": (
            d_missing.fallback_used
            and d_missing.fallback_reason == FallbackReason.EXCESSIVE_MISSING_FEATURES
            and d_missing.action != StepUpAction.ALLOW
        ),
    }

    # 4. Repeated attempts burst -> hard escalation
    eng4 = ATODetectionEngine(config=config, model_bundle=bundle, audit_logger=DecisionAuditLogger())
    for i in range(10):
        eng4.evaluate(
            LoginSessionEvent(
                event_id=f"scn_rep_{i}",
                timestamp=base_ts + timedelta(minutes=i * 2),
                user_id="usr_rep_01",
                session_id=f"ses_rep_{i}",
                auth_outcome="failure",
                auth_method="password",
                device_id="dev_rep_01",
                device_trust_score=0.50,
                ip_address="203.0.113.45",
                ip_reputation_risk=0.60,
                geo_lat=40.7128,
                geo_lon=-74.0060,
            )
        )
    d_rep = eng4.evaluate(
        LoginSessionEvent(
            event_id="scn_rep_final",
            timestamp=base_ts + timedelta(minutes=25),
            user_id="usr_rep_01",
            session_id="ses_rep_final",
            auth_outcome="success",
            auth_method="password",
            device_id="dev_rep_01",
            device_trust_score=0.70,
            ip_address="203.0.113.45",
            ip_reputation_risk=0.30,
            geo_lat=40.7128,
            geo_lon=-74.0060,
        )
    )
    results["repeated_attempts"] = {
        "failed_attempts_1h": d_rep.features_snapshot["failed_attempts_1h"],
        "action": d_rep.action.value,
        "triggered_rules": d_rep.evidence.triggered_rules,
        "passed": (
            d_rep.features_snapshot["failed_attempts_1h"] == 10.0
            and d_rep.action == StepUpAction.BLOCK_AND_RESET
        ),
    }

    # 5. Feature expiry past TTL
    eng5 = ATODetectionEngine(config=config, model_bundle=bundle, audit_logger=DecisionAuditLogger())
    eng5.evaluate(
        LoginSessionEvent(
            event_id="scn_exp_old",
            timestamp=base_ts,
            user_id="usr_exp_01",
            session_id="ses_exp_old",
            device_id="dev_exp_01",
            device_trust_score=0.95,
            ip_address="192.0.2.99",
            ip_reputation_risk=0.05,
            geo_lat=40.7128,
            geo_lon=-74.0060,
        )
    )
    d_exp = eng5.evaluate(
        LoginSessionEvent(
            event_id="scn_exp_after_ttl",
            timestamp=base_ts + timedelta(days=45),  # > 30-day TTL
            user_id="usr_exp_01",
            session_id="ses_exp_after_ttl",
            device_id="dev_exp_01",
            device_trust_score=0.95,
            ip_address="192.0.2.99",
            ip_reputation_risk=0.05,
            geo_lat=40.7128,
            geo_lon=-74.0060,
        )
    )
    results["feature_expiry"] = {
        "profile_expired": d_exp.evidence.profile_expired,
        "user_prior_event_count_30d": d_exp.features_snapshot["user_prior_event_count_30d"],
        "action": d_exp.action.value,
        "passed": (
            d_exp.evidence.profile_expired
            and d_exp.features_snapshot["user_prior_event_count_30d"] == 0.0
            and d_exp.action != StepUpAction.ALLOW
        ),
    }

    # 6. Drift detection & circuit breaker
    eng6 = ATODetectionEngine(config=config, model_bundle=bundle, audit_logger=DecisionAuditLogger())
    for i in range(30):
        eng6.evaluate(
            LoginSessionEvent(
                event_id=f"scn_drift_{i}",
                timestamp=base_ts + timedelta(minutes=i),
                user_id=f"usr_drift_{i}",
                session_id=f"ses_drift_{i}",
                auth_method="password",
                device_id=f"dev_drift_{i}",
                device_trust_score=0.10,
                ip_address="203.0.113.99",
                ip_reputation_risk=0.95,
                geo_lat=51.5074,
                geo_lon=-0.1278,
                sensitive_actions=["change_mfa", "add_payee"],
                session_duration_seconds=20.0,
            )
        )
    drift_report = eng6.check_drift()
    results["drift_alerts"] = {
        "score_psi": drift_report.score_psi,
        "alert_count": len(drift_report.alerts),
        "critical_drift_detected": drift_report.critical_drift_detected,
        "circuit_breaker_engaged": eng6.critical_drift_breaker,
        "passed": drift_report.critical_drift_detected and eng6.critical_drift_breaker,
    }

    return results


def run_full_evaluation(
    config: EngineConfig,
    output_dir: Path,
    model_artifact_path: Path | None = None,
) -> Dict[str, Any]:
    """Execute full reproducible pipeline and write raw artifacts to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    events = generate_synthetic_dataset(seed=config.seed)

    # Strictly causal feature extraction across full stream
    feat_store = InMemoryStateStore()
    feature_vectors = extract_stream_features(
        events=events,
        store=feat_store,
        config=config.features,
        feature_version=config.versions.feature_version,
        record_events=True,
    )

    bundle, split, model_eval = train_model_bundle(
        events=events,
        feature_vectors=feature_vectors,
        config=config,
    )

    artifact_digest = None
    if model_artifact_path is not None:
        artifact_digest = bundle.save(model_artifact_path)

    # Evaluate adaptive policy on chronological test split & verify replay reproducibility
    test_events = [events[i] for i in split.test_indices]
    audit_file = output_dir / "decision_audit.jsonl"
    if audit_file.exists():
        audit_file.unlink()
    policy_summary, run_a = evaluate_policy_on_split(
        events=test_events,
        bundle=bundle,
        config=config,
        audit_path=audit_file,
    )
    _, run_b = evaluate_policy_on_split(
        events=test_events,
        bundle=bundle,
        config=config,
        audit_path=None,
    )
    replay_check = compare_decision_replays(run_a, run_b)

    scenarios = run_required_scenarios(bundle=bundle, config=config)

    # Reference oracle vs calibrated ensemble comparison on test split
    oracle_m = model_eval["test_metrics"]["rule_oracle_baseline"]
    ens_m = model_eval["test_metrics"]["calibrated_ensemble"]
    uncal_m = model_eval["test_metrics"]["supervised_uncalibrated_blend"]
    baseline_comparison = {
        "calibrated_vs_rule_oracle": {
            "roc_auc_delta": round(ens_m["roc_auc"] - oracle_m["roc_auc"], 6),
            "pr_auc_delta": round(ens_m["pr_auc"] - oracle_m["pr_auc"], 6),
            "brier_delta": round(ens_m["brier_score"] - oracle_m["brier_score"], 6),
            "ece_delta": round(ens_m["ece"] - oracle_m["ece"], 6),
            "recall_at_1pct_fpr_delta": round(
                ens_m["recall_at_fpr"]["1.00%"]["recall"]
                - oracle_m["recall_at_fpr"]["1.00%"]["recall"],
                6,
            ),
        },
        "calibrated_vs_uncalibrated_blend": {
            "brier_delta": round(ens_m["brier_score"] - uncal_m["brier_score"], 6),
            "ece_delta": round(ens_m["ece"] - uncal_m["ece"], 6),
        },
    }

    report = {
        "runtime": runtime_metadata(config),
        "config": config.to_dict(),
        "dataset": {
            "source": "synthetic_deterministic_generator",
            "total_events": len(events),
            "positive_events": sum(1 for e in events if e.label == 1),
            "benign_events": sum(1 for e in events if e.label == 0),
            "time_range_utc": [events[0].timestamp.isoformat(), events[-1].timestamp.isoformat()],
        },
        "model_artifact_sha256": artifact_digest,
        "chronological_evaluation": model_eval,
        "baseline_comparison": baseline_comparison,
        "adaptive_policy_test_summary": policy_summary,
        "replay_reproducibility": replay_check,
        "required_scenarios": scenarios,
    }

    (output_dir / "evaluation_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
