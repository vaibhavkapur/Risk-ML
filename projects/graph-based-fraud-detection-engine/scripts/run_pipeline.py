#!/usr/bin/env python3
"""Run the full reproducible training, calibration, leakage audit, and reporting pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_fraud_engine.config import load_config  # noqa: E402
from graph_fraud_engine.pipeline import run_training_pipeline, save_artifacts  # noqa: E402
from graph_fraud_engine.service import FraudDetectionService  # noqa: E402
from graph_fraud_engine.schemas import AnalystDispositionRequest, RiskScoreRequest  # noqa: E402
from graph_fraud_engine.synthetic import generate_synthetic_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Graph-Based Fraud Detection Engine pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Directory to write versioned evaluation reports.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save serialized joblib model bundle under artifacts/.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = generate_synthetic_dataset(config)
    artifacts = run_training_pipeline(bundle, config)

    # Reproducibility check: second deterministic run on identical config/data
    bundle_repeat = generate_synthetic_dataset(config)
    artifacts_repeat = run_training_pipeline(bundle_repeat, config)
    repro_match = (
        artifacts.evaluation_report["comparison_vs_baseline"]
        == artifacts_repeat.evaluation_report["comparison_vs_baseline"]
        and artifacts.run_id == artifacts_repeat.run_id
    )

    # Score sample test transactions (1 fraud ring tx, 1 legitimate shared device tx) for demo evidence
    service = FraudDetectionService.from_artifacts(artifacts)
    labels_by_tx = {l.tx_id: l for l in bundle.labels}
    test_cutoff_start = artifacts.evaluation_report["splits"]["val_cutoff_utc"]
    sample_ring_tx = next(
        tx
        for tx in reversed(bundle.transactions)
        if tx.tx_timestamp.isoformat() > test_cutoff_start
        and labels_by_tx[tx.tx_id].fraud_pattern == "coordinated_fraud_ring"
    )
    sample_shared_legit_tx = next(
        tx
        for tx in reversed(bundle.transactions)
        if tx.tx_timestamp.isoformat() > test_cutoff_start
        and labels_by_tx[tx.tx_id].fraud_pattern == "legitimate_shared_device"
    )

    ring_score = service.score_transaction(
        RiskScoreRequest(transaction=sample_ring_tx, model_name=config.serving.default_model)
    )
    shared_legit_score = service.score_transaction(
        RiskScoreRequest(transaction=sample_shared_legit_tx, model_name=config.serving.default_model)
    )

    disposition_record = None
    if ring_score.review_case_id is not None:
        disposition_record = service.submit_disposition(
            ring_score.review_case_id,
            AnalystDispositionRequest(
                analyst_id="analyst_demo_01",
                disposition="confirmed_fraud",
                notes="Verified coordinated shared-device cash-out ring via graph evidence paths.",
            ),
            recorded_at=sample_ring_tx.tx_timestamp,
        )

    reports_dir = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "evaluation_report.json").write_text(
        json.dumps(artifacts.evaluation_report, indent=2), encoding="utf-8"
    )
    (reports_dir / "runtime_metadata.json").write_text(
        json.dumps(artifacts.runtime_metadata, indent=2), encoding="utf-8"
    )
    (reports_dir / "reproducibility_check.json").write_text(
        json.dumps(
            {
                "run_id_first": artifacts.run_id,
                "run_id_second": artifacts_repeat.run_id,
                "exact_logical_match": repro_match,
                "seed": config.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (reports_dir / "sample_explanations.json").write_text(
        json.dumps(
            {
                "fraud_ring_example": ring_score.model_dump(mode="json"),
                "legitimate_shared_device_example": shared_legit_score.model_dump(mode="json"),
                "analyst_disposition_example": (
                    disposition_record.model_dump(mode="json") if disposition_record else None
                ),
                "service_diagnostics": service.get_diagnostics(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.save_artifacts:
        bundle_path = save_artifacts(artifacts, PROJECT_ROOT / "artifacts")
        print(f"Saved model bundle: {bundle_path}")

    print(f"Run ID: {artifacts.run_id}")
    print(f"Leakage audit: {artifacts.evaluation_report['leakage_audit']['status']}")
    print(f"Exact reproducibility: {repro_match}")
    print("\nModel comparison (test calibrated):")
    for m_name, row in artifacts.evaluation_report["comparison_vs_baseline"].items():
        print(
            f"  {m_name:26s} | PR-AUC={row['test_pr_auc']:.4f} "
            f"| Rec@1%FPR={row['recall_at_1pct_fpr']:.4f} "
            f"| Rec@5%FPR={row['recall_at_5pct_fpr']:.4f} "
            f"| ECE={row['calibrated_ece']:.4f} "
            f"| Brier={row['calibrated_brier']:.4f} "
            f"| RingDet={row['fraud_ring_detection_rate']:.4f} "
            f"| LegitSharedFPR={row['legitimate_shared_device_fpr']:.4f}"
        )


if __name__ == "__main__":
    main()
