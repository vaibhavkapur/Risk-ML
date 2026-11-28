#!/usr/bin/env python3
"""End-to-end reproducible demo for Smart Payment Authentication Engine."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fastapi
import numpy as np
import pydantic
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smart_auth_engine.config import load_config
from smart_auth_engine.data_generator import generate_synthetic_dataset
from smart_auth_engine.evaluation import evaluate_chronologically
from smart_auth_engine.features import extract_feature_matrix
from smart_auth_engine.inference import AuthenticationEngine
from smart_auth_engine.policy import optimize_policy_thresholds
from smart_auth_engine.store import JsonlDecisionStore
from smart_auth_engine.training import ReferenceOracleScorer, train_calibrated_risk_model


def main() -> None:
    cfg_path = PROJECT_ROOT / "config" / "default.yaml"
    cfg = load_config(cfg_path)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = PROJECT_ROOT / "artifacts" / "models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Generating synthetic dataset (n={cfg.data_generation.n_samples}, seed={cfg.seed})...")
    events, labels, true_probs = generate_synthetic_dataset(cfg.data_generation, seed=cfg.seed)

    print("[2/6] Training calibrated risk model with temporal splits & delayed-label maturity filtering...")
    train_report = train_calibrated_risk_model(events, labels, cfg)
    model_path, manifest_path = train_report.model.save(artifacts_dir)

    print(
        f"      Mature train={train_report.model.manifest.n_train_mature}, "
        f"Excluded immature={train_report.model.manifest.n_train_excluded_immature}, "
        f"Calibrated Val AUC={train_report.calibrated_val_metrics.roc_auc:.4f}, "
        f"Brier={train_report.calibrated_val_metrics.brier_score:.5f}, "
        f"ECE={train_report.calibrated_val_metrics.ece:.5f}"
    )

    print("[3/6] Optimizing constrained authentication policy on chronological validation split...")
    X_val = extract_feature_matrix(train_report.splits.val_events)
    val_cal_probs, _ = train_report.model.predict_proba_matrix(X_val)
    opt_res = optimize_policy_thresholds(
        events=train_report.splits.val_events,
        labels=train_report.splits.val_labels,
        calibrated_probs=val_cal_probs,
        cfg=cfg,
    )
    print(
        f"      Selected thresholds: challenge>={opt_res.selected_thresholds.challenge_threshold:.4f}, "
        f"decline>={opt_res.selected_thresholds.decline_threshold:.4f}, "
        f"constraints_satisfied={opt_res.constraints_satisfied}"
    )

    print("[4/6] Running chronological holdout test evaluation (calibrated policy vs reference oracle)...")
    decision_log_path = reports_dir / "decision_log.jsonl"
    if decision_log_path.exists():
        decision_log_path.unlink()
    store = JsonlDecisionStore(decision_log_path)

    cal_engine = AuthenticationEngine(
        cfg=cfg,
        thresholds=opt_res.selected_thresholds,
        model=train_report.model,
        store=store,
    )
    cal_eval_full = evaluate_chronologically(
        events=train_report.splits.test_events,
        labels=train_report.splits.test_labels,
        engine=cal_engine,
        cfg=cfg,
        record_store=True,
    )

    # Partial maturity snapshot (mid-delay) to demonstrate label-delay accounting
    partial_as_of = train_report.splits.test_events[-1].timestamp + timedelta(days=5)
    cal_engine_partial = AuthenticationEngine(
        cfg=cfg,
        thresholds=opt_res.selected_thresholds,
        model=train_report.model,
    )
    cal_eval_partial = evaluate_chronologically(
        events=train_report.splits.test_events,
        labels=train_report.splits.test_labels,
        engine=cal_engine_partial,
        cfg=cfg,
        evaluation_as_of=partial_as_of,
    )

    # Reference oracle baseline evaluated on the exact same chronological test split
    oracle_engine = AuthenticationEngine(
        cfg=cfg,
        thresholds=opt_res.selected_thresholds,
        model=ReferenceOracleScorer(),
    )
    oracle_eval_full = evaluate_chronologically(
        events=train_report.splits.test_events,
        labels=train_report.splits.test_labels,
        engine=oracle_engine,
        cfg=cfg,
    )

    # Outage fallback drill on the same test split
    outage_engine = AuthenticationEngine(
        cfg=cfg,
        thresholds=opt_res.selected_thresholds,
        model=train_report.model,
        model_available=False,
    )
    outage_eval = evaluate_chronologically(
        events=train_report.splits.test_events,
        labels=train_report.splits.test_labels,
        engine=outage_engine,
        cfg=cfg,
    )

    print("[5/6] Verifying deterministic replay on recorded decisions...")
    sample_ids = store.list_decision_ids()[:25]
    replays = [cal_engine.replay_decision(did) for did in sample_ids]
    replay_all_match = all(r["exact_match"] for r in replays)
    assert replay_all_match, "Replay parity check failed!"

    print("[6/6] Writing versioned machine-readable reports to reports/...")
    # 1. Baseline vs calibrated metrics
    baseline_vs_cal = {
        "schema_version": "1.0.0",
        "model_version": train_report.model.model_version,
        "feature_version": train_report.model.feature_version,
        "manifest": train_report.model.manifest.model_dump(mode="json"),
        "validation_metrics": {
            "raw_gbm": train_report.raw_val_metrics.to_dict(),
            "calibrated_gbm": train_report.calibrated_val_metrics.to_dict(),
            "reference_oracle": train_report.oracle_val_metrics.to_dict(),
            "comparison_by_calibrator": train_report.comparison_by_calibrator,
        },
        "dataset_summary": {
            "n_total": len(events),
            "empirical_fraud_rate": round(float(np.mean([1 if l.is_fraud else 0 for l in labels])), 6),
            "mean_true_prob": round(float(true_probs.mean()), 6),
            "n_train_mature": len(train_report.splits.train_events),
            "n_train_excluded_immature": train_report.splits.train_excluded_immature,
            "n_cal_mature": len(train_report.splits.cal_events),
            "n_val": len(train_report.splits.val_events),
            "n_test": len(train_report.splits.test_events),
        },
    }
    (reports_dir / "baseline_vs_calibrated_metrics.json").write_text(
        json.dumps(baseline_vs_cal, indent=2, sort_keys=True), encoding="utf-8"
    )

    # 2. Policy tradeoff frontier
    policy_report = {
        "schema_version": "1.0.0",
        "constraints": {
            "max_challenge_rate": cfg.policy_optimization.max_challenge_rate,
            "max_fraud_loss_rate": cfg.policy_optimization.max_fraud_loss_rate,
        },
        "constraints_satisfied": opt_res.constraints_satisfied,
        "evaluated_candidates": opt_res.evaluated_candidates,
        "selected_policy": opt_res.selected_point.to_dict(),
        "pareto_frontier": [pt.to_dict() for pt in opt_res.pareto_frontier],
    }
    (reports_dir / "policy_tradeoff_frontier.json").write_text(
        json.dumps(policy_report, indent=2, sort_keys=True), encoding="utf-8"
    )

    # 3. Chronological evaluation report
    chrono_report = {
        "schema_version": "1.0.0",
        "calibrated_policy_full_maturity": cal_eval_full.to_dict(),
        "calibrated_policy_partial_maturity_plus5d": cal_eval_partial.to_dict(),
        "reference_oracle_full_maturity": oracle_eval_full.to_dict(),
        "model_outage_fallback_full_maturity": outage_eval.to_dict(),
    }
    (reports_dir / "chronological_evaluation.json").write_text(
        json.dumps(chrono_report, indent=2, sort_keys=True), encoding="utf-8"
    )

    # 4. Step-up simulation & replay summary
    step_up_summary = {
        "schema_version": "1.0.0",
        "calibrated_engine_counters": cal_engine.counters.to_dict(),
        "outage_engine_counters": outage_engine.counters.to_dict(),
        "replay_checked_count": len(sample_ids),
        "replay_all_exact_match": replay_all_match,
        "decision_log_path": str(decision_log_path.relative_to(PROJECT_ROOT)),
    }
    (reports_dir / "step_up_simulation_summary.json").write_text(
        json.dumps(step_up_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    # 5. Runtime metadata
    run_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg_path.relative_to(PROJECT_ROOT)),
        "config_hash": cfg.config_hash(),
        "seed": cfg.seed,
        "model_artifact": str(model_path.relative_to(PROJECT_ROOT)),
        "model_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "artifact_sha256": train_report.model.manifest.artifact_sha256,
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "fastapi": fastapi.__version__,
            "pydantic": pydantic.__version__,
        },
        "data_provenance": "100% synthetic data generated deterministically via fixed seed; zero real transactions or PII.",
    }
    (reports_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\n=== Demo Summary (Holdout Chronological Test Split) ===")
    print(f"Test events:                 {cal_eval_full.total_events}")
    print(f"Calibrated fraud loss rate:  {cal_eval_full.matured_fraud_loss_rate:.4%}")
    print(f"Calibrated acceptance rate:  {cal_eval_full.matured_genuine_acceptance_rate:.4%}")
    print(f"Calibrated challenge rate:   {cal_eval_full.challenge_rate:.4%}")
    print(f"Calibrated decline rate:     {cal_eval_full.decline_rate:.4%}")
    print(f"Mandatory challenge bypass:  {cal_eval_full.mandatory_challenge_bypass_count}")
    print(f"High-risk override bypass:   {cal_eval_full.override_bypass_count}")
    print(f"Oracle fraud loss rate:      {oracle_eval_full.matured_fraud_loss_rate:.4%}")
    print(f"Outage fallback decisions:   {outage_eval.fallback_count} (100% deterministic)")
    print(f"Replay parity verified:      {replay_all_match} ({len(sample_ids)} sampled decisions)")


if __name__ == "__main__":
    main()
