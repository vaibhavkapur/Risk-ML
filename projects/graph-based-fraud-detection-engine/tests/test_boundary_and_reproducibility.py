"""Boundary, failure, delayed-label maturity, and deterministic reproducibility tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from graph_fraud_engine.config import load_config
from graph_fraud_engine.metrics import build_temporal_splits
from graph_fraud_engine.pipeline import load_artifacts, run_training_pipeline, save_artifacts
from graph_fraud_engine.schemas import RiskScoreRequest, TransactionRecord
from graph_fraud_engine.service import FraudDetectionService
from graph_fraud_engine.synthetic import generate_synthetic_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unseen_cold_start_nodes_handled_explicitly() -> None:
    """Scoring completely unseen account/device/merchant succeeds with cold-start flags and valid evidence."""
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    bundle = generate_synthetic_dataset(config)
    artifacts = run_training_pipeline(bundle, config)
    service = FraudDetectionService.from_artifacts(artifacts)

    unseen_tx = TransactionRecord(
        tx_id="tx_aaaaaaaa1111",
        account_id="acc_aaaaaaaa1111",
        device_id="dev_aaaaaaaa1111",
        merchant_id="mer_aaaaaaaa1111",
        tx_timestamp=datetime(2025, 4, 2, 10, 0, tzinfo=timezone.utc),
        amount_usd=125.0,
        currency="USD",
        channel="web",
    )

    for model_name in ("logistic", "boosted", "gnn"):
        resp = service.score_transaction(
            RiskScoreRequest(transaction=unseen_tx, model_name=model_name, record_for_review=False)
        )
        assert 0.0 <= resp.calibrated_probability <= 1.0
        assert resp.cold_start_flags == {
            "account_unseen": True,
            "device_unseen": True,
            "merchant_unseen": True,
        }
        assert "COLD_START_UNSEEN_ACCOUNT" in resp.explanation.reason_codes
        assert "COLD_START_UNSEEN_DEVICE" in resp.explanation.reason_codes
        assert len(resp.explanation.evidence_paths) >= 1
        assert resp.explanation.graph_audit_max_edge_timestamp is None


def test_delayed_labels_excluded_before_maturity_cutoff() -> None:
    """Verify every training and validation transaction has label_available_at <= split cutoff."""
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    bundle = generate_synthetic_dataset(config)
    splits = build_temporal_splits(bundle, config)
    labels_by_tx = {l.tx_id: l for l in bundle.labels}

    for tx in splits.train_txs:
        assert labels_by_tx[tx.tx_id].label_available_at <= splits.train_cutoff
    for tx in splits.val_txs:
        assert labels_by_tx[tx.tx_id].label_available_at <= splits.val_cutoff

    # Confirm toggling enforce_label_maturity changes counts as expected (proving active filtering)
    cfg_no_mat = config.model_copy(
        update={
            "temporal_split": config.temporal_split.model_copy(
                update={"enforce_label_maturity": False}
            )
        }
    )
    splits_no_mat = build_temporal_splits(bundle, cfg_no_mat)
    assert len(splits_no_mat.train_txs) == len(splits.train_txs) + splits.train_immature_excluded
    assert splits.train_immature_excluded > 0


def test_exact_reproducibility_and_artifact_roundtrip(tmp_path: Path) -> None:
    """Running twice with identical versioned config & seed produces identical metrics and decision IDs."""
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    b1 = generate_synthetic_dataset(config)
    b2 = generate_synthetic_dataset(config)
    assert [t.tx_id for t in b1.transactions] == [t.tx_id for t in b2.transactions]

    a1 = run_training_pipeline(b1, config)
    a2 = run_training_pipeline(b2, config)
    assert a1.run_id == a2.run_id
    assert a1.evaluation_report["comparison_vs_baseline"] == a2.evaluation_report["comparison_vs_baseline"]

    # Save & reload artifact bundle and confirm identical inference output
    saved_path = save_artifacts(a1, tmp_path)
    loaded = load_artifacts(saved_path)

    svc1 = FraudDetectionService.from_artifacts(a1)
    svc_reloaded = FraudDetectionService(
        config=config,
        graph=loaded["graph"],
        models=loaded["models"],
        run_id=loaded["run_id"],
    )
    sample_tx = b1.transactions[-1]
    r1 = svc1.score_transaction(RiskScoreRequest(transaction=sample_tx, model_name="gnn"))
    r2 = svc_reloaded.score_transaction(RiskScoreRequest(transaction=sample_tx, model_name="gnn"))
    assert r1.decision_id == r2.decision_id
    assert r1.calibrated_probability == r2.calibrated_probability
    assert r1.decision == r2.decision
