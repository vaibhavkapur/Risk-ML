#!/usr/bin/env python3
"""Run full validation: pytest suite + reproducible demo + replay identity check."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    import os

    env = dict(os.environ)
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
    print(f"\n>>> Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    python = sys.executable
    _run([python, "-m", "pytest"])
    _run([python, str(ROOT / "scripts" / "run_demo.py")])
    _run([python, "-m", "payment_auth_engine.cli", "replay", "--config", str(ROOT / "config" / "default.yaml")])
    print("\nAll validation gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
