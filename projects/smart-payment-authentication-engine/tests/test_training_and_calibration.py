"""M2 Training & Calibration tests: Delayed-label maturity filtering, temporal splits, and artifact persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smart_auth_engine.config import load_config
from smart_auth_engine.data_generator import generate_synthetic_dataset
from smart_auth_engine.features import extract_feature_matrix
from smart_auth_engine.training import (
    CalibratedRiskModel,
    ModelOutageError,
    build_temporal_splits,
    train_calibrated_risk_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_delayed_label_maturity_filtering_prevents_future_leakage() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    events, labels, _ = generate_synthetic_dataset(cfg.data_generation, seed=cfg.seed)
    splits = build_temporal_splits(events, labels, cfg)

    # Immature examples near train_end MUST be excluded
    assert splits.train_excluded_immature > 0
    assert len(splits.train_events) > 100

    # Strict anti-leakage assertion: every training label matured at or before train_as_of
    for ev, lbl in zip(splits.train_events, splits.train_labels):
        assert ev.timestamp <= splits.train_as_of
        assert lbl.label_available_at <= splits.train_as_of

    # Strict temporal ordering across train < cal < val < test
    max_train_ts = max(ev.timestamp for ev in splits.train_events)
    min_cal_ts = min(ev.timestamp for ev in splits.cal_events)
    max_cal_ts = max(ev.timestamp for ev in splits.cal_events)
    min_val_ts = min(ev.timestamp for ev in splits.val_events)
    max_val_ts = max(ev.timestamp for ev in splits.val_events)
    min_test_ts = min(ev.timestamp for ev in splits.test_events)
    assert max_train_ts < min_cal_ts
    assert max_cal_ts < min_val_ts
    assert max_val_ts < min_test_ts


def test_calibrated_model_quality_and_artifact_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    events, labels, _ = generate_synthetic_dataset(cfg.data_generation, seed=cfg.seed)
    report = train_calibrated_risk_model(events, labels, cfg)

    # Calibration metrics within acceptable prototype quality bounds
    assert report.calibrated_val_metrics.roc_auc >= 0.80
    assert report.calibrated_val_metrics.ece <= cfg.evaluation.max_acceptable_ece
    assert report.calibrated_val_metrics.brier_score <= cfg.evaluation.max_acceptable_brier
    assert set(report.comparison_by_calibrator.keys()) == {"none", "sigmoid", "isotonic"}

    # Artifact save/load roundtrip with SHA-256 verification
    model_path, manifest_path = report.model.save(tmp_path)
    loaded = CalibratedRiskModel.load(model_path, manifest_path)
    X_sample = extract_feature_matrix(report.splits.test_events[:25])
    orig_cal, orig_raw = report.model.predict_proba_matrix(X_sample)
    load_cal, load_raw = loaded.predict_proba_matrix(X_sample)
    assert np.allclose(orig_cal, load_cal, atol=cfg.evaluation.reproducibility_tolerance)
    assert np.allclose(orig_raw, load_raw, atol=cfg.evaluation.reproducibility_tolerance)

    # Tampered artifact must fail integrity check explicitly (never silent success)
    model_path.write_bytes(model_path.read_bytes() + b"corrupted")
    with pytest.raises(ModelOutageError, match="SHA-256 mismatch"):
        CalibratedRiskModel.load(model_path, manifest_path)
