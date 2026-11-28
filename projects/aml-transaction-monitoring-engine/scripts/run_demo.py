#!/usr/bin/env python3
"""Reproducible end-to-end demo script writing machine-readable output to reports/."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aml_engine.cli import main  # noqa: E402


if __name__ == "__main__":
    output_path = PROJECT_ROOT / "reports" / "demo_run_output.json"
    sys.exit(
        main(
            [
                "--config",
                str(PROJECT_ROOT / "config" / "default_config.json"),
                "demo",
                "--output",
                str(output_path),
            ]
        )
    )
