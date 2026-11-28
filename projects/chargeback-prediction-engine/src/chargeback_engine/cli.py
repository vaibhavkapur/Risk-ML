"""Command-line interface for Chargeback Prediction Engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from chargeback_engine.config import load_config
from chargeback_engine.contracts import ChargebackEngineError
from chargeback_engine.pipeline import (
    collect_runtime_metadata,
    run_pipeline,
    verify_reference_fixture,
)
from chargeback_engine.synthetic import generate_synthetic_dataset


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chargeback-engine",
        description="Chargeback Prediction Engine offline CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ref_cmd = sub.add_parser(
        "verify-reference", help="Verify deterministic hand-calculated reference fixture"
    )
    ref_cmd.add_argument(
        "--fixtures-dir",
        default="data/fixtures",
        help="Path to fixtures directory (default: data/fixtures)",
    )

    demo_cmd = sub.add_parser(
        "demo", help="Run end-to-end synthetic chargeback prediction demo"
    )
    demo_cmd.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML configuration file",
    )
    demo_cmd.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for machine-readable JSON outputs",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "verify-reference":
            result = verify_reference_fixture(args.fixtures_dir)
            print(json.dumps(result, indent=2))
            return 0

        if args.command == "demo":
            cfg = load_config(args.config)
            merchants, transactions, disputes = generate_synthetic_dataset(
                cfg.synthetic, seed=cfg.seed
            )
            result = run_pipeline(cfg, merchants, transactions, disputes)
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "demo_summary.json").write_text(
                json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
            (out_dir / "runtime_metadata.json").write_text(
                json.dumps(collect_runtime_metadata(cfg), indent=2) + "\n",
                encoding="utf-8",
            )
            cal = result.metrics_primary_calibrated
            print(
                f"Run {result.run_id} completed | PR-AUC={cal['pr_auc']:.4f} | "
                f"Brier={cal['brier_score']:.5f} | ECE={cal['ece']:.5f} | "
                f"ExpectedLoss=${cal['total_expected_loss_usd']:.2f} "
                f"(Realized=${cal['total_realized_loss_usd']:.2f}) | "
                f"Precision@Budget={cal['precision_at_review_budget_by_value']:.4f}"
            )
            return 0
    except ChargebackEngineError as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
