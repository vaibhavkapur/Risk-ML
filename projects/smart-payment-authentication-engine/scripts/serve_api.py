#!/usr/bin/env python3
"""Start the local FastAPI authentication service using the calibrated model."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smart_auth_engine.api import create_app
from smart_auth_engine.config import load_config
from smart_auth_engine.data_generator import generate_synthetic_dataset
from smart_auth_engine.features import extract_feature_matrix
from smart_auth_engine.inference import AuthenticationEngine
from smart_auth_engine.policy import optimize_policy_thresholds
from smart_auth_engine.store import JsonlDecisionStore
from smart_auth_engine.training import train_calibrated_risk_model


def build_default_app():
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    events, labels, _ = generate_synthetic_dataset(cfg.data_generation, seed=cfg.seed)
    train_report = train_calibrated_risk_model(events, labels, cfg)
    X_val = extract_feature_matrix(train_report.splits.val_events)
    val_probs, _ = train_report.model.predict_proba_matrix(X_val)
    opt_res = optimize_policy_thresholds(
        train_report.splits.val_events, train_report.splits.val_labels, val_probs, cfg
    )
    store = JsonlDecisionStore(PROJECT_ROOT / cfg.storage.jsonl_path)
    engine = AuthenticationEngine(
        cfg=cfg,
        thresholds=opt_res.selected_thresholds,
        model=train_report.model,
        store=store,
    )
    return create_app(engine)


if __name__ == "__main__":
    app = build_default_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)
