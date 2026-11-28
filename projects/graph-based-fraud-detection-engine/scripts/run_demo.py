#!/usr/bin/env python3
"""Interactive-free end-to-end demo showcasing scoring, graph evidence, and analyst review."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_fraud_engine.config import load_config  # noqa: E402
from graph_fraud_engine.pipeline import run_training_pipeline  # noqa: E402
from graph_fraud_engine.schemas import AnalystDispositionRequest, RiskScoreRequest  # noqa: E402
from graph_fraud_engine.service import FraudDetectionService  # noqa: E402
from graph_fraud_engine.synthetic import generate_synthetic_dataset  # noqa: E402


def main() -> None:
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    bundle = generate_synthetic_dataset(config)
    artifacts = run_training_pipeline(bundle, config)
    service = FraudDetectionService.from_artifacts(artifacts)

    labels_by_tx = {l.tx_id: l for l in bundle.labels}
    val_cutoff = artifacts.evaluation_report["splits"]["val_cutoff_utc"]

    ring_tx = next(
        tx
        for tx in reversed(bundle.transactions)
        if tx.tx_timestamp.isoformat() > val_cutoff
        and labels_by_tx[tx.tx_id].fraud_pattern == "coordinated_fraud_ring"
    )
    shared_tx = next(
        tx
        for tx in reversed(bundle.transactions)
        if tx.tx_timestamp.isoformat() > val_cutoff
        and labels_by_tx[tx.tx_id].fraud_pattern == "legitimate_shared_device"
    )

    print("=== 1. Coordinated Fraud Ring Transaction ===")
    r1 = service.score_transaction(RiskScoreRequest(transaction=ring_tx, model_name="gnn"))
    print(
        json.dumps(
            {
                "tx_id": r1.tx_id,
                "decision_id": r1.decision_id,
                "calibrated_probability": r1.calibrated_probability,
                "decision": r1.decision,
                "review_case_id": r1.review_case_id,
                "reason_codes": r1.explanation.reason_codes,
                "evidence_paths": [p.summary for p in r1.explanation.evidence_paths],
            },
            indent=2,
        )
    )

    print("\n=== 2. Legitimate Shared-Device Transaction ===")
    r2 = service.score_transaction(RiskScoreRequest(transaction=shared_tx, model_name="gnn"))
    print(
        json.dumps(
            {
                "tx_id": r2.tx_id,
                "decision_id": r2.decision_id,
                "calibrated_probability": r2.calibrated_probability,
                "decision": r2.decision,
                "reason_codes": r2.explanation.reason_codes,
                "legitimate_shared_device_mitigant": r2.explanation.legitimate_shared_device_mitigant,
            },
            indent=2,
        )
    )

    if r1.review_case_id:
        print("\n=== 3. Analyst Review Disposition ===")
        disp = service.submit_disposition(
            r1.review_case_id,
            AnalystDispositionRequest(
                analyst_id="analyst_01",
                disposition="confirmed_fraud",
                notes="Coordinated mule ring confirmed via shared emulator device and gift-card triangulation.",
            ),
            recorded_at=ring_tx.tx_timestamp,
        )
        print(disp.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
