#!/usr/bin/env python3
"""Run temporal rule-vs-ML evaluation and persist model artifact + report to reports/."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aml_engine.cli import main  # noqa: E402


if __name__ == "__main__":
    report_path = PROJECT_ROOT / "reports" / "baseline_vs_ml_evaluation.json"
    artifact_path = PROJECT_ROOT / "reports" / "ml_prioritizer_artifact.json"
    sys.exit(
        main(
            [
                "--config",
                str(PROJECT_ROOT / "config" / "default_config.json"),
                "evaluate",
                "--output",
                str(report_path),
                "--artifact-out",
                str(artifact_path),
            ]
        )
    )
