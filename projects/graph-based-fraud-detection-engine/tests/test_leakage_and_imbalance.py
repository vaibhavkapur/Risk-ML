"""Integration & resilience tests: adversarial future-edge leakage, delayed labels, and class imbalance."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np

from graph_fraud_engine.config import load_config
from graph_fraud_engine.features import extract_feature_matrix, extract_transaction_features
from graph_fraud_engine.graph import TemporalEntityGraph
from graph_fraud_engine.metrics import build_temporal_splits
from graph_fraud_engine.models import build_causal_tx_adjacency
from graph_fraud_engine.pipeline import run_training_pipeline
from graph_fraud_engine.schemas import FraudLabelRecord, TransactionRecord
from graph_fraud_engine.synthetic import (
    build_hand_calculated_reference_bundle,
    generate_synthetic_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_adversarial_future_edge_and_future_label_injection_zero_leakage() -> None:
    """Injecting massive future fraud edges/labels must leave all earlier features bit-for-bit identical."""
    bundle, expected = build_hand_calculated_reference_bundle()
    ids = expected["ids"]
    tx_by_id = {tx.tx_id: tx for tx in bundle.transactions}
    target_tx = tx_by_id[ids["T5"]]

    graph_clean = TemporalEntityGraph.from_dataset(bundle)
    feat_clean = extract_transaction_features(graph_clean, target_tx)

    # Adversarially inject 10 future fraud transactions on the same device/merchant right after T5
    future_txs = []
    future_labels = []
    for k in range(10):
        f_ts = target_tx.tx_timestamp + timedelta(seconds=1 + k)
        f_id = f"tx_ff00000000{k:02d}"
        future_txs.append(
            TransactionRecord(
                tx_id=f_id,
                account_id=ids["H1"],  # Cross-link household account in the future!
                device_id=target_tx.device_id,
                merchant_id=target_tx.merchant_id,
                tx_timestamp=f_ts,
                amount_usd=9999.0,
                channel="mobile",
            )
        )
        future_labels.append(
            FraudLabelRecord(
                tx_id=f_id,
                is_fraud=1,
                tx_timestamp=f_ts,
                label_available_at=f_ts,
                label_source="rule_confirmed",
            )
        )

    augmented = bundle.model_copy(
        update={
            "transactions": bundle.transactions + future_txs,
            "labels": bundle.labels + future_labels,
        }
    )
    graph_aug = TemporalEntityGraph.from_dataset(augmented)
    feat_aug = extract_transaction_features(graph_aug, target_tx)

    assert feat_clean.features == feat_aug.features
    assert feat_aug.max_edge_timestamp_used == feat_clean.max_edge_timestamp_used
    assert feat_aug.max_label_available_at_used == feat_clean.max_label_available_at_used


def test_gnn_causal_adjacency_is_strictly_lower_triangular() -> None:
    """Verify A and A^2 have zero entries on or above the diagonal (no same-time or future leakage)."""
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    bundle = generate_synthetic_dataset(config)
    txs = sorted(bundle.transactions[:400], key=lambda t: (t.tx_timestamp, t.tx_id))
    A, stats = build_causal_tx_adjacency(txs, neighbor_window_days=30)

    assert stats["n_causal_edges"] > 0
    # Upper triangle including diagonal must be identically 0
    assert float(np.max(np.abs(np.triu(A, k=0)))) == 0.0
    # 2-hop propagation A @ A must ALSO have zero upper triangle including diagonal
    A2 = A @ A
    assert float(np.max(np.abs(np.triu(A2, k=0)))) == 0.0


def test_class_imbalance_evaluation_and_graph_lift_vs_tabular_baseline() -> None:
    """Verify imbalanced dataset handling, PR-AUC lift from graph features, calibration, and zero leakage."""
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    bundle = generate_synthetic_dataset(config)
    splits = build_temporal_splits(bundle, config)

    # Confirm realistic class imbalance (< 10% fraud) and delayed labels excluded from train/val
    assert 0.01 < float(np.mean(splits.train_y)) < 0.12
    assert splits.train_immature_excluded > 0
    assert splits.val_immature_excluded > 0

    artifacts = run_training_pipeline(bundle, config)
    report = artifacts.evaluation_report
    assert report["leakage_audit"]["status"] == "PASS"
    assert report["leakage_audit"]["future_edge_or_immature_label_violations"] == 0

    comp = report["comparison_vs_baseline"]
    tab_pr_auc = comp["logistic_tabular_baseline"]["test_pr_auc"]
    for graph_model in ("logistic", "boosted", "gnn"):
        row = comp[graph_model]
        # Graph-aware models must beat tabular-only baseline on PR-AUC
        assert row["test_pr_auc"] > tab_pr_auc, (
            f"{graph_model} PR-AUC ({row['test_pr_auc']}) did not exceed tabular baseline ({tab_pr_auc})"
        )
        assert row["recall_at_5pct_fpr"] >= 0.70
        # Platt calibration improves or maintains low ECE on imbalanced data
        assert row["calibrated_ece"] <= 0.05
        assert row["calibrated_ece"] <= row["raw_ece"] + 0.01
