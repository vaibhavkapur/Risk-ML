"""Domain correctness tests: hand-calculated oracle, fraud rings vs legitimate shared devices."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_fraud_engine.config import load_config
from graph_fraud_engine.explainer import build_graph_explanation
from graph_fraud_engine.features import extract_transaction_features
from graph_fraud_engine.graph import TemporalEntityGraph
from graph_fraud_engine.pipeline import run_training_pipeline
from graph_fraud_engine.schemas import RiskScoreRequest
from graph_fraud_engine.service import FraudDetectionService
from graph_fraud_engine.synthetic import (
    build_hand_calculated_reference_bundle,
    generate_synthetic_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_hand_calculated_reference_oracle_exact_match() -> None:
    """Verify graph features against exact hand-calculated expectations (including delayed labels & future tx)."""
    bundle, expected = build_hand_calculated_reference_bundle()
    graph = TemporalEntityGraph.from_dataset(bundle)
    tx_by_id = {tx.tx_id: tx for tx in bundle.transactions}
    ids = expected["ids"]

    # T5: Ring account A3 on shared device D_RING at M_GIFT (Day 5)
    res_t5 = extract_transaction_features(graph, tx_by_id[ids["T5"]])
    for feat_name, exp_val in expected["T5_expected"].items():
        assert res_t5.features[feat_name] == pytest.approx(exp_val, abs=1e-9), (
            f"T5 feature {feat_name}: got {res_t5.features[feat_name]}, expected {exp_val}"
        )
    # Audit timestamps strictly obey causality
    assert res_t5.max_edge_timestamp_used is not None
    assert res_t5.max_edge_timestamp_used < tx_by_id[ids["T5"]].tx_timestamp
    assert res_t5.max_label_available_at_used is not None
    assert res_t5.max_label_available_at_used <= tx_by_id[ids["T5"]].tx_timestamp

    # T7: Legitimate shared household device D_HOME at M_GROC
    res_t7 = extract_transaction_features(graph, tx_by_id[ids["T7"]])
    for feat_name, exp_val in expected["T7_expected"].items():
        assert res_t7.features[feat_name] == pytest.approx(exp_val, abs=1e-9)

    expl_t7 = build_graph_explanation(graph, tx_by_id[ids["T7"]], res_t7, {})
    assert expl_t7.legitimate_shared_device_mitigant is True
    assert "LEGITIMATE_SHARED_DEVICE_MITIGANT" in expl_t7.reason_codes

    # T8: Pure cold-start unseen account & unseen device
    res_t8 = extract_transaction_features(graph, tx_by_id[ids["T8"]])
    for feat_name, exp_val in expected["T8_expected"].items():
        assert res_t8.features[feat_name] == pytest.approx(exp_val, abs=1e-9)
    assert res_t8.cold_start_flags["account_unseen"] is True
    assert res_t8.cold_start_flags["device_unseen"] is True


def test_fraud_rings_vs_legitimate_shared_devices_separation() -> None:
    """Models must score coordinated fraud rings much higher than legitimate shared-device households."""
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    bundle = generate_synthetic_dataset(config)
    artifacts = run_training_pipeline(bundle, config)
    service = FraudDetectionService.from_artifacts(artifacts)

    labels_by_tx = {l.tx_id: l for l in bundle.labels}
    val_cutoff = artifacts.evaluation_report["splits"]["val_cutoff_utc"]

    ring_scores = []
    legit_shared_scores = []
    for tx in bundle.transactions:
        if tx.tx_timestamp.isoformat() <= val_cutoff:
            continue
        pat = labels_by_tx[tx.tx_id].fraud_pattern
        if pat == "coordinated_fraud_ring" and len(ring_scores) < 25:
            resp = service.score_transaction(
                RiskScoreRequest(transaction=tx, model_name="gnn", record_for_review=False)
            )
            ring_scores.append(resp.calibrated_probability)
        elif pat == "legitimate_shared_device" and len(legit_shared_scores) < 40:
            resp = service.score_transaction(
                RiskScoreRequest(transaction=tx, model_name="gnn", record_for_review=False)
            )
            legit_shared_scores.append(resp.calibrated_probability)

    assert len(ring_scores) >= 10
    assert len(legit_shared_scores) >= 20
    mean_ring = sum(ring_scores) / len(ring_scores)
    mean_shared = sum(legit_shared_scores) / len(legit_shared_scores)

    assert mean_ring > 0.55, f"Expected high average risk on fraud rings, got {mean_ring:.4f}"
    assert mean_shared < 0.15, f"Expected low average risk on legitimate shared devices, got {mean_shared:.4f}"
    assert mean_ring - mean_shared > 0.45

    # Verify reported segment metrics across all graph models keep shared-device FPR very low
    for model_name in ("logistic", "boosted", "gnn"):
        seg = artifacts.evaluation_report["models"][model_name]["test_segments"]
        assert seg["fraud_ring_detection_rate"] >= 0.70, f"{model_name} ring detection too low: {seg}"
        assert seg["legitimate_shared_device_fpr"] <= 0.08, f"{model_name} shared-device FPR too high: {seg}"
