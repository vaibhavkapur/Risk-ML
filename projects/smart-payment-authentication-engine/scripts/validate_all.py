#!/usr/bin/env python3
"""Run full validation suite: pytest + reproducible end-to-end demo verification."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("==> Running pytest suite...")
    subprocess.run([sys.executable, "-m", "pytest"], cwd=PROJECT_ROOT, check=True)

    print("\n==> Running reproducible demo (Pass 1)...")
    subprocess.run([sys.executable, "scripts/run_demo.py"], cwd=PROJECT_ROOT, check=True)
    snap1 = json.loads((PROJECT_ROOT / "reports" / "chronological_evaluation.json").read_text(encoding="utf-8"))

    print("\n==> Re-running demo (Pass 2) to verify bit-for-bit logical reproducibility...")
    subprocess.run([sys.executable, "scripts/run_demo.py"], cwd=PROJECT_ROOT, check=True)
    snap2 = json.loads((PROJECT_ROOT / "reports" / "chronological_evaluation.json").read_text(encoding="utf-8"))

    assert snap1 == snap2, "Reproducibility check failed: Pass 1 and Pass 2 chronological reports differed!"
    print("\nALL VALIDATION CHECKS PASSED (tests + reproducible demo).")


if __name__ == "__main__":
    main()
