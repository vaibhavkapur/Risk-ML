#!/usr/bin/env python3
"""Run reproducible end-to-end Chargeback Prediction Engine demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chargeback_engine.config import load_config
from chargeback_engine.pipeline import (
    collect_runtime_metadata,
    run_pipeline,
    verify_reference_fixture,
)
from chargeback_engine.synthetic import generate_synthetic_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Chargeback Prediction Engine demo")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "default.yaml"),
        help="Path to YAML configuration",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports"),
        help="Directory to write versioned JSON reports",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify hand-calculated reference fixture first
    ref_summary = verify_reference_fixture(PROJECT_ROOT / "data" / "fixtures")

    # 2. Load config and run end-to-end synthetic scenario
    cfg = load_config(args.config)
    merchants, transactions, disputes = generate_synthetic_dataset(
        cfg.synthetic, seed=cfg.seed
    )
    result = run_pipeline(cfg, merchants, transactions, disputes)

    payload = result.to_dict()
    payload["reference_fixture_verification"] = ref_summary

    (out_dir / "demo_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "runtime_metadata.json").write_text(
        json.dumps(collect_runtime_metadata(cfg), indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "review_simulation.json").write_text(
        json.dumps(
            {
                "run_id": result.run_id,
                "config_hash": result.config_hash,
                "policy_comparison": result.policy_comparison,
                "delayed_outcome_monitoring": result.delayed_outcome_monitoring,
                "sample_explanations": result.sample_explanations,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cal = result.metrics_primary_calibrated
    base = result.metrics_baseline_logistic
    print(f"Reference fixture check: {ref_summary['status']} ({ref_summary['checked_transactions']} txns)")
    print(f"Demo Run ID: {result.run_id} (config_hash={result.config_hash})")
    print(
        f"Maturity: mature_pos={result.maturity_diagnostics.mature_positive_count}, "
        f"mature_neg={result.maturity_diagnostics.mature_negative_count}, "
        f"immature_pos={result.maturity_diagnostics.immature_positive_count}, "
        f"immature_neg_excluded={result.maturity_diagnostics.immature_negative_count}"
    )
    print(
        f"Primary Calibrated | PR-AUC={cal['pr_auc']:.4f} | ROC-AUC={cal['roc_auc']:.4f} | "
        f"Brier={cal['brier_score']:.5f} | ECE={cal['ece']:.5f} | "
        f"ExpLoss=${cal['total_expected_loss_usd']:.2f} vs Realized=${cal['total_realized_loss_usd']:.2f} | "
        f"Precision@Budget={cal['precision_at_review_budget_by_value']:.4f}"
    )
    print(
        f"Baseline Logistic  | PR-AUC={base['pr_auc']:.4f} | ROC-AUC={base['roc_auc']:.4f} | "
        f"Brier={base['brier_score']:.5f} | ECE={base['ece']:.5f} | "
        f"ExpLoss=${base['total_expected_loss_usd']:.2f} | "
        f"Precision@Budget={base['precision_at_review_budget_by_value']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
