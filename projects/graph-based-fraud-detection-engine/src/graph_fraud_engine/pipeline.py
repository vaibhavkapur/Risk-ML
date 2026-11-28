"""End-to-end training, temporal validation, calibration, and evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import networkx as nx
import numpy as np
import sklearn

from graph_fraud_engine import FEATURE_VERSION, SCHEMA_VERSION, __version__
from graph_fraud_engine.config import EngineConfig, config_to_dict
from graph_fraud_engine.features import (
    FEATURE_NAMES,
    TABULAR_ONLY_FEATURE_NAMES,
    extract_feature_matrix,
)
from graph_fraud_engine.graph import TemporalEntityGraph
from graph_fraud_engine.metrics import build_temporal_splits, compute_classification_metrics
from graph_fraud_engine.models import (
    BoostedFraudBaseline,
    LogisticFraudBaseline,
    TemporalGraphSAGEModel,
    is_pyg_available,
)
from graph_fraud_engine.schemas import DatasetBundle
from graph_fraud_engine.validation import validate_dataset_bundle


@dataclass
class TrainedArtifacts:
    config: EngineConfig
    run_id: str
    graph: TemporalEntityGraph
    models: Dict[str, Any]
    evaluation_report: Dict[str, Any]
    runtime_metadata: Dict[str, Any]


def compute_run_id(config: EngineConfig, dataset_id: str, n_tx: int) -> str:
    """Stable deterministic run identifier derived from versioned config and dataset."""
    payload = json.dumps(
        {
            "config": config_to_dict(config),
            "dataset_id": dataset_id,
            "n_tx": n_tx,
            "schema_version": SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION,
        },
        sort_keys=True,
    )
    return f"run_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def collect_runtime_metadata(config: EngineConfig, run_id: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "engine_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_version": config.model_version,
        "config_version": config.config_version,
        "seed": config.seed,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "networkx": nx.__version__,
            "torch_geometric_available": is_pyg_available(),
        },
        "gnn_backend": "numpy_temporal_graphsage",
        "pyg_note": (
            "PyTorch Geometric is installed."
            if is_pyg_available()
            else "Optional PyTorch Geometric is not installed in this environment; "
            "using deterministic native NumPy 2-layer TemporalGraphSAGE."
        ),
    }


def _segment_metrics(
    bundle: DatasetBundle,
    test_txs,
    y_test: np.ndarray,
    probs: np.ndarray,
    review_threshold: float,
) -> Dict[str, Any]:
    """Measure detection rate on synthetic fraud rings vs false-positive rate on legitimate shared devices."""
    labels_by_tx = {l.tx_id: l for l in bundle.labels}
    ring_mask = np.array(
        [labels_by_tx[tx.tx_id].fraud_pattern == "coordinated_fraud_ring" for tx in test_txs],
        dtype=bool,
    )
    shared_legit_mask = np.array(
        [labels_by_tx[tx.tx_id].fraud_pattern == "legitimate_shared_device" for tx in test_txs],
        dtype=bool,
    )
    organic_legit_mask = np.array(
        [labels_by_tx[tx.tx_id].fraud_pattern == "organic_legitimate" for tx in test_txs],
        dtype=bool,
    )

    ring_recall = (
        float(np.mean(probs[ring_mask] >= review_threshold)) if np.any(ring_mask) else 0.0
    )
    shared_legit_fpr = (
        float(np.mean(probs[shared_legit_mask] >= review_threshold))
        if np.any(shared_legit_mask)
        else 0.0
    )
    organic_legit_fpr = (
        float(np.mean(probs[organic_legit_mask] >= review_threshold))
        if np.any(organic_legit_mask)
        else 0.0
    )

    return {
        "threshold_used": review_threshold,
        "fraud_ring_test_count": int(np.sum(ring_mask)),
        "fraud_ring_detection_rate": round(ring_recall, 6),
        "legitimate_shared_device_test_count": int(np.sum(shared_legit_mask)),
        "legitimate_shared_device_fpr": round(shared_legit_fpr, 6),
        "organic_legitimate_test_count": int(np.sum(organic_legit_mask)),
        "organic_legitimate_fpr": round(organic_legit_fpr, 6),
    }


def run_training_pipeline(
    bundle: DatasetBundle,
    config: EngineConfig,
) -> TrainedArtifacts:
    """Validate data, build temporal graph, extract causal features, train, calibrate, and evaluate."""
    validate_dataset_bundle(bundle)
    run_id = compute_run_id(config, bundle.dataset_id, len(bundle.transactions))
    graph = TemporalEntityGraph.from_dataset(bundle)

    # Sort all transactions chronologically and extract strictly causal features
    all_txs = sorted(bundle.transactions, key=lambda t: (t.tx_timestamp, t.tx_id))
    tx_to_idx = {tx.tx_id: idx for idx, tx in enumerate(all_txs)}
    all_X, all_audits = extract_feature_matrix(graph, all_txs, config.features)

    # Verify zero future edges and zero immature labels across every extracted feature vector
    leakage_violations = 0
    for tx, audit in zip(all_txs, all_audits):
        if audit.max_edge_timestamp_used is not None and audit.max_edge_timestamp_used >= tx.tx_timestamp:
            leakage_violations += 1
        if audit.max_label_available_at_used is not None and audit.max_label_available_at_used > tx.tx_timestamp:
            leakage_violations += 1

    # Temporal splits with label maturity enforcement
    splits = build_temporal_splits(bundle, config)
    train_idx = np.array([tx_to_idx[tx.tx_id] for tx in splits.train_txs], dtype=np.int64)
    val_idx = np.array([tx_to_idx[tx.tx_id] for tx in splits.val_txs], dtype=np.int64)
    test_idx = np.array([tx_to_idx[tx.tx_id] for tx in splits.test_txs], dtype=np.int64)

    X_train, y_train = all_X[train_idx], splits.train_y
    X_val, y_val = all_X[val_idx], splits.val_y
    X_test, y_test = all_X[test_idx], splits.test_y

    cal_method = config.calibration.method
    fpr_targets = config.policy.fixed_fpr_targets
    n_bins = config.calibration.n_bins
    review_thresh = config.policy.review_threshold

    models: Dict[str, Any] = {}
    model_reports: Dict[str, Any] = {}

    # 1. Tabular-only logistic reference baseline (ablation without multi-hop graph features)
    tab_log = LogisticFraudBaseline(
        config=config.models.logistic,
        feature_names=TABULAR_ONLY_FEATURE_NAMES,
        seed=config.seed,
    ).fit(X_train, y_train)
    tab_log.fit_calibrator(X_val, y_val, method=cal_method)
    models["logistic_tabular_baseline"] = tab_log

    # 2. Logistic regression on full temporal graph features
    graph_log = LogisticFraudBaseline(
        config=config.models.logistic,
        feature_names=FEATURE_NAMES,
        seed=config.seed,
    ).fit(X_train, y_train)
    graph_log.fit_calibrator(X_val, y_val, method=cal_method)
    models["logistic"] = graph_log

    # 3. Boosted tree baseline on full temporal graph features
    boosted = BoostedFraudBaseline(
        config=config.models.boosted,
        feature_names=FEATURE_NAMES,
        seed=config.seed,
    ).fit(X_train, y_train)
    boosted.fit_calibrator(X_val, y_val, method=cal_method)
    models["boosted"] = boosted

    # Evaluate tabular & tree models
    for name, mdl in (
        ("logistic_tabular_baseline", tab_log),
        ("logistic", graph_log),
        ("boosted", boosted),
    ):
        raw_val = mdl.predict_raw(X_val)
        cal_val = mdl.predict_calibrated(X_val)
        raw_test = mdl.predict_raw(X_test)
        cal_test = mdl.predict_calibrated(X_test)
        model_reports[name] = {
            "feature_count": len(mdl.feature_names),
            "validation_raw": compute_classification_metrics(y_val, raw_val, fpr_targets, n_bins),
            "validation_calibrated": compute_classification_metrics(y_val, cal_val, fpr_targets, n_bins),
            "test_raw": compute_classification_metrics(y_test, raw_test, fpr_targets, n_bins),
            "test_calibrated": compute_classification_metrics(y_test, cal_test, fpr_targets, n_bins),
            "test_segments": _segment_metrics(bundle, splits.test_txs, y_test, cal_test, review_thresh),
        }

    # 4. Temporal GraphSAGE GNN trained on strictly causal temporal splits
    if config.models.gnn.enabled:
        gnn = TemporalGraphSAGEModel(
            config=config.models.gnn,
            feature_names=FEATURE_NAMES,
            seed=config.seed,
        ).fit_temporal(all_txs, all_X, train_idx, y_train)

        all_raw_gnn = gnn.predict_raw_temporal(all_txs, all_X)
        gnn.fit_calibrator(all_raw_gnn[val_idx], y_val, method=cal_method)
        all_cal_gnn = gnn.calibrator.transform(all_raw_gnn)

        models["gnn"] = gnn
        model_reports["gnn"] = {
            "feature_count": len(FEATURE_NAMES),
            "backend": gnn.backend,
            "training_history": gnn.training_history,
            "validation_raw": compute_classification_metrics(y_val, all_raw_gnn[val_idx], fpr_targets, n_bins),
            "validation_calibrated": compute_classification_metrics(y_val, all_cal_gnn[val_idx], fpr_targets, n_bins),
            "test_raw": compute_classification_metrics(y_test, all_raw_gnn[test_idx], fpr_targets, n_bins),
            "test_calibrated": compute_classification_metrics(y_test, all_cal_gnn[test_idx], fpr_targets, n_bins),
            "test_segments": _segment_metrics(bundle, splits.test_txs, y_test, all_cal_gnn[test_idx], review_thresh),
        }

    # Compute baseline deltas (graph models vs tabular-only logistic baseline)
    base_pr_auc = model_reports["logistic_tabular_baseline"]["test_calibrated"]["pr_auc"]
    base_brier = model_reports["logistic_tabular_baseline"]["test_calibrated"]["brier_score"]
    base_ece = model_reports["logistic_tabular_baseline"]["test_calibrated"]["ece"]

    comparison_vs_baseline: Dict[str, Any] = {}
    for m_name, m_rep in model_reports.items():
        tc = m_rep["test_calibrated"]
        tr = m_rep["test_raw"]
        comparison_vs_baseline[m_name] = {
            "test_pr_auc": tc["pr_auc"],
            "pr_auc_delta_vs_tabular_baseline": round(tc["pr_auc"] - base_pr_auc, 6),
            "recall_at_1pct_fpr": tc["recall_at_fixed_fpr"]["fpr_01pct"]["recall"],
            "recall_at_5pct_fpr": tc["recall_at_fixed_fpr"]["fpr_05pct"]["recall"],
            "raw_ece": tr["ece"],
            "calibrated_ece": tc["ece"],
            "ece_improvement_from_calibration": round(tr["ece"] - tc["ece"], 6),
            "calibrated_brier": tc["brier_score"],
            "brier_delta_vs_tabular_baseline": round(tc["brier_score"] - base_brier, 6),
            "ece_delta_vs_tabular_baseline": round(tc["ece"] - base_ece, 6),
            "fraud_ring_detection_rate": m_rep["test_segments"]["fraud_ring_detection_rate"],
            "legitimate_shared_device_fpr": m_rep["test_segments"]["legitimate_shared_device_fpr"],
        }

    runtime_metadata = collect_runtime_metadata(config, run_id)
    evaluation_report: Dict[str, Any] = {
        "run_id": run_id,
        "dataset_id": bundle.dataset_id,
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_version": config.model_version,
        "seed": config.seed,
        "calibration_method": cal_method,
        "leakage_audit": {
            "transactions_audited": len(all_txs),
            "future_edge_or_immature_label_violations": leakage_violations,
            "status": "PASS" if leakage_violations == 0 else "FAIL",
        },
        "splits": {
            "train_cutoff_utc": splits.train_cutoff.isoformat(),
            "val_cutoff_utc": splits.val_cutoff.isoformat(),
            "test_cutoff_utc": splits.test_cutoff.isoformat(),
            "train_matured_count": int(len(splits.train_txs)),
            "train_positives": int(np.sum(y_train)),
            "train_immature_labels_excluded": splits.train_immature_excluded,
            "val_matured_count": int(len(splits.val_txs)),
            "val_positives": int(np.sum(y_val)),
            "val_immature_labels_excluded": splits.val_immature_excluded,
            "test_count": int(len(splits.test_txs)),
            "test_positives": int(np.sum(y_test)),
        },
        "models": model_reports,
        "comparison_vs_baseline": comparison_vs_baseline,
    }

    return TrainedArtifacts(
        config=config,
        run_id=run_id,
        graph=graph,
        models=models,
        evaluation_report=evaluation_report,
        runtime_metadata=runtime_metadata,
    )


def save_artifacts(artifacts: TrainedArtifacts, output_dir: str | Path) -> Path:
    """Persist trained models and graph state with version metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle_path = out / f"model_bundle_{artifacts.run_id}.joblib"
    joblib.dump(
        {
            "run_id": artifacts.run_id,
            "config": config_to_dict(artifacts.config),
            "models": artifacts.models,
            "graph": artifacts.graph,
            "runtime_metadata": artifacts.runtime_metadata,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        bundle_path,
    )
    return bundle_path


def load_artifacts(bundle_path: str | Path) -> Dict[str, Any]:
    """Load persisted model bundle from disk."""
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    return joblib.load(path)
