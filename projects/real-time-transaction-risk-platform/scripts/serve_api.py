#!/usr/bin/env python3
"""Launch the FastAPI real-time transaction risk scoring service locally."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from risk_platform.api import create_app
from risk_platform.audit import AuditLogger
from risk_platform.config import load_config
from risk_platform.model import CalibratedLinearRiskModel
from risk_platform.service import RiskScoringService
from risk_platform.store import InMemoryFeatureStore
from risk_platform.synthetic import generate_synthetic_dataset
from risk_platform.training import train_calibrated_baseline_model


def build_app(config_path: Path):
    cfg = load_config(config_path)
    model_path = PROJECT_ROOT / cfg.storage.model_artifact_path
    if model_path.exists():
        model = CalibratedLinearRiskModel.load(model_path)
    else:
        events, labels = generate_synthetic_dataset(
            num_events=cfg.training.num_synthetic_events,
            num_accounts=cfg.training.num_accounts,
            seed=cfg.training.seed,
            label_maturity_hours=cfg.training.label_maturity_hours,
            fx_to_usd=cfg.schema.fx_to_usd,
        )
        model, _ = train_calibrated_baseline_model(events, labels, cfg)
        model.save(model_path)

    ckpt_path = PROJECT_ROOT / cfg.storage.checkpoint_path
    if ckpt_path.exists():
        store = InMemoryFeatureStore.load_checkpoint(ckpt_path)
    else:
        store = InMemoryFeatureStore(
            allowed_lateness_ms=cfg.ingestion.allowed_lateness_ms,
            retention_window_ms=cfg.ingestion.retention_window_ms,
            max_events_per_account=cfg.ingestion.max_events_per_account,
            max_staleness_ms=cfg.serving.max_feature_staleness_ms,
            fx_to_usd=cfg.schema.fx_to_usd,
        )

    audit = AuditLogger(path=PROJECT_ROOT / cfg.storage.audit_log_path)
    service = RiskScoringService(config=cfg, store=store, model=model, audit_logger=audit)
    return create_app(service)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FastAPI scoring server.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "default.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = build_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
