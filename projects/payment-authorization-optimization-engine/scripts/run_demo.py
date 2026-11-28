#!/usr/bin/env python3
"""Run the reproducible end-to-end demo and write machine-readable reports."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from payment_auth_engine.cli import main  # noqa: E402


if __name__ == "__main__":
    default_args = [
        "demo",
        "--config",
        str(ROOT / "config" / "default.yaml"),
        "--output-dir",
        str(ROOT / "reports"),
    ]
    raise SystemExit(main(sys.argv[1:] or default_args))
