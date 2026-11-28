#!/usr/bin/env python3
"""Start local FastAPI server for scoring and analyst review workflows (offline/synthetic only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_fraud_engine.api import create_app  # noqa: E402
from graph_fraud_engine.config import load_config  # noqa: E402
from graph_fraud_engine.pipeline import run_training_pipeline  # noqa: E402
from graph_fraud_engine.service import FraudDetectionService  # noqa: E402
from graph_fraud_engine.synthetic import generate_synthetic_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Graph-Based Fraud Detection Engine HTTP API.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
    )
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = generate_synthetic_dataset(config)
    artifacts = run_training_pipeline(bundle, config)
    service = FraudDetectionService.from_artifacts(artifacts)
    app = create_app(service)

    uvicorn.run(
        app,
        host=args.host or config.serving.host,
        port=args.port or config.serving.port,
    )


if __name__ == "__main__":
    main()
