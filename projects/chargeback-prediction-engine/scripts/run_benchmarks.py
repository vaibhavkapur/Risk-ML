#!/usr/bin/env python3
"""Run reproducibility verification, baseline vs calibrated comparison, and stress scenarios."""

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
from chargeback_engine.pipeline import run_pipeline
from chargeback_engine.synthetic import generate_synthetic_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Chargeback Engine benchmarks")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "default.yaml"),
        help="Primary default YAML config",
    )
    parser.add_argument(
        "--cold-start-config",
        default=str(PROJECT_ROOT / "config" / "cold_start.yaml"),
        help="Imbalance + cold-start scenario config",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports"),
        help="Directory to write benchmark artifacts",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Default scenario run twice for bit-for-bit reproducibility check
    cfg = load_config(args.config)
    m1, t1, d1 = generate_synthetic_dataset(cfg.synthetic, seed=cfg.seed)
    res_a = run_pipeline(cfg, m1, t1, d1)

    m2, t2, d2 = generate_synthetic_dataset(cfg.synthetic, seed=cfg.seed)
    res_b = run_pipeline(cfg, m2, t2, d2)

    exact_match = res_a.to_dict() == res_b.to_dict()
    repro_payload = {
        "exact_match": exact_match,
        "run_id_a": res_a.run_id,
        "run_id_b": res_b.run_id,
        "config_hash": res_a.config_hash,
        "seed": cfg.seed,
        "pr_auc_a": res_a.metrics_primary_calibrated["pr_auc"],
        "pr_auc_b": res_b.metrics_primary_calibrated["pr_auc"],
        "expected_loss_a": res_a.metrics_primary_calibrated["total_expected_loss_usd"],
        "expected_loss_b": res_b.metrics_primary_calibrated["total_expected_loss_usd"],
        "precision_at_budget_a": res_a.metrics_primary_calibrated[
            "precision_at_review_budget_by_value"
        ],
        "precision_at_budget_b": res_b.metrics_primary_calibrated[
            "precision_at_review_budget_by_value"
        ],
    }
    (out_dir / "reproducibility_check.json").write_text(
        json.dumps(repro_payload, indent=2) + "\n", encoding="utf-8"
    )
    if not exact_match:
        print("ERROR: Reproducibility check failed!", file=sys.stderr)
        return 1

    # 2. Cold-start + severe class imbalance scenario
    cold_cfg = load_config(args.cold_start_config)
    mc, tc, dc = generate_synthetic_dataset(cold_cfg.synthetic, seed=cold_cfg.seed)
    res_cold = run_pipeline(cold_cfg, mc, tc, dc)

    comparison_payload = {
        "default_scenario": {
            "run_id": res_a.run_id,
            "config_hash": res_a.config_hash,
            "baseline_logistic_calibrated": res_a.metrics_baseline_logistic,
            "primary_uncalibrated": res_a.metrics_primary_uncalibrated,
            "primary_calibrated": res_a.metrics_primary_calibrated,
            "calibration_brier_improvement": round(
                res_a.metrics_primary_uncalibrated["brier_score"]
                - res_a.metrics_primary_calibrated["brier_score"],
                6,
            ),
            "calibration_ece_improvement": round(
                res_a.metrics_primary_uncalibrated["ece"]
                - res_a.metrics_primary_calibrated["ece"],
                6,
            ),
            "cold_start_slices": res_a.cold_start_slice_metrics,
        },
        "imbalance_cold_start_scenario": {
            "run_id": res_cold.run_id,
            "config_hash": res_cold.config_hash,
            "maturity_diagnostics": res_cold.maturity_diagnostics.to_dict(),
            "baseline_logistic_calibrated": res_cold.metrics_baseline_logistic,
            "primary_uncalibrated": res_cold.metrics_primary_uncalibrated,
            "primary_calibrated": res_cold.metrics_primary_calibrated,
            "cold_start_slices": res_cold.cold_start_slice_metrics,
            "policy_comparison": res_cold.policy_comparison,
        },
    }
    (out_dir / "baseline_vs_calibrated.json").write_text(
        json.dumps(comparison_payload, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Reproducibility check: exact_match={exact_match} (run_id={res_a.run_id})"
    )
    print(
        f"Default Brier: uncal={res_a.metrics_primary_uncalibrated['brier_score']:.5f} -> "
        f"cal={res_a.metrics_primary_calibrated['brier_score']:.5f} | "
        f"ECE: uncal={res_a.metrics_primary_uncalibrated['ece']:.5f} -> "
        f"cal={res_a.metrics_primary_calibrated['ece']:.5f}"
    )
    print(
        f"Cold-start/imbalance scenario | rate={res_cold.metrics_primary_calibrated['observed_dispute_rate']:.4f} | "
        f"PR-AUC={res_cold.metrics_primary_calibrated['pr_auc']:.4f} | "
        f"ECE={res_cold.metrics_primary_calibrated['ece']:.5f} | "
        f"Precision@Budget={res_cold.metrics_primary_calibrated['precision_at_review_budget_by_value']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
