#!/usr/bin/env python3
"""Run the end-to-end Credit Risk Scoring Platform demo."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from credit_risk.cli import main  # noqa: E402

if __name__ == "__main__":
    config_arg = sys.argv[1:] if len(sys.argv) > 1 else ["--config", str(PROJECT_ROOT / "config" / "default.yaml")]
    raise SystemExit(main(["demo", *config_arg]))
