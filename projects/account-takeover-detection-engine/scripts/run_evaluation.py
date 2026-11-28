#!/usr/bin/env python3
"""Run chronological training, calibration, evaluation, and scenario checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ato_engine.config import load_config
from ato_engine.evaluation import run_full_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ATO Detection Engine evaluation suite.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
        help="Path to engine YAML/JSON configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Directory for machine-readable reports and artifacts.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = run_full_evaluation(
        config=cfg,
        output_dir=args.output_dir,
        model_artifact_path=args.output_dir / "model_bundle.joblib",
    )

    ens = report["chronological_evaluation"]["test_metrics"]["calibrated_ensemble"]
    oracle = report["chronological_evaluation"]["test_metrics"]["rule_oracle_baseline"]
    policy = report["adaptive_policy_test_summary"]

    print("=== Account Takeover Detection Engine — Evaluation Summary ===")
    print(f"Dataset events      : {report['dataset']['total_events']} (ATO={report['dataset']['positive_events']})")
    print(f"Train/Cal/Test rows : {report['chronological_evaluation']['split']['train_rows']}/"
          f"{report['chronological_evaluation']['split']['calibration_rows']}/"
          f"{report['chronological_evaluation']['split']['test_rows']} "
          f"(excluded immature train={report['chronological_evaluation']['split']['excluded_immature_train_rows']})")
    print(f"Rule Oracle Test    : ROC-AUC={oracle['roc_auc']:.4f} | Brier={oracle['brier_score']:.4f} | ECE={oracle['ece']:.4f} | Recall@1%FPR={oracle['recall_at_fpr']['1.00%']['recall']:.4f}")
    print(f"Calibrated Ensemble : ROC-AUC={ens['roc_auc']:.4f} | Brier={ens['brier_score']:.4f} | ECE={ens['ece']:.4f} | Recall@1%FPR={ens['recall_at_fpr']['1.00%']['recall']:.4f} | Recall@2%FPR={ens['recall_at_fpr']['2.00%']['recall']:.4f} | Recall@5%FPR={ens['recall_at_fpr']['5.00%']['recall']:.4f}")
    print(f"Audit Log Complete  : {policy['every_decision_logged']} ({policy['decisions_logged']}/{policy['total_events']})")
    print(f"Replay Match        : {report['replay_reproducibility']['matched']} ({report['replay_reproducibility']['checked']} events)")
    for name, sc in report["required_scenarios"].items():
        print(f"Scenario [{name}]: {'PASS' if sc['passed'] else 'FAIL'}")
    print(f"Raw report saved to : {args.output_dir / 'evaluation_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
