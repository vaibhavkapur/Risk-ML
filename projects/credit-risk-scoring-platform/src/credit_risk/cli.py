"""Command-line interface for running the demo, scoring fixtures, and launching manual review."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List, Optional

from credit_risk.config import load_config
from credit_risk.data import load_json_fixture
from credit_risk.inference import CreditRiskScorer
from credit_risk.pipeline import run_from_config_path
from credit_risk.review import ManualReviewStore, create_review_http_server


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="credit-risk",
        description="Credit Risk Scoring Platform (Educational Prototype — Manual Review Only)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run full reproducible train/calibrate/evaluate/report pipeline.")
    p_demo.add_argument(
        "--config",
        default=str(_default_project_root() / "config" / "default.yaml"),
        help="Path to YAML configuration file.",
    )

    p_score = sub.add_parser("score", help="Score applicant JSON fixture using saved model bundle.")
    p_score.add_argument("--bundle", required=True, help="Path to scorer_bundle.joblib.")
    p_score.add_argument("--input", required=True, help="Path to applicant JSON file (object or list).")

    p_ui = sub.add_parser("review-ui", help="Serve local HTML manual review workbench.")
    p_ui.add_argument(
        "--store",
        default=str(_default_project_root() / "reports" / "manual_review_queue.json"),
        help="Path to manual review queue JSON file.",
    )
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8088)

    p_list = sub.add_parser("review-list", help="Print manual review queue summary to stdout.")
    p_list.add_argument(
        "--store",
        default=str(_default_project_root() / "reports" / "manual_review_queue.json"),
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.command == "demo":
        res = run_from_config_path(args.config)
        er = res["evaluation_report"]
        lr = er["models"]["logistic_regression"]["calibrated_test"]
        gb = er["models"]["boosted_trees"]["calibrated_test"]
        print("=== Credit Risk Scoring Platform — End-to-End Demo Summary ===")
        print(f"Primary model: {er['primary_model']} | Calibration: {er['selected_calibration_method']}")
        print(f"Logistic Calibrated  -> AUROC={lr['auroc']:.4f}, PR-AUC={lr['pr_auc']:.4f}, Brier={lr['brier_score']:.4f}, ECE={lr['ece']:.4f}")
        print(f"Boosted  Calibrated  -> AUROC={gb['auroc']:.4f}, PR-AUC={gb['pr_auc']:.4f}, Brier={gb['brier_score']:.4f}, ECE={gb['ece']:.4f}")
        print(f"Score PSI (train->test): {er['stability']['score_psi_train_vs_test']['psi']} ({er['stability']['score_psi_train_vs_test']['status']})")
        print(f"Baseline comparison passed: {er['baseline_comparison']['all_checks_passed']}")
        print(f"Replay verification match:  {er['replay_verification']['decision_id_match']}")
        print("Generated artifacts:")
        for k, v in res["outputs"].items():
            print(f"  - {k}: {v}")
        return 0

    if args.command == "score":
        scorer = CreditRiskScorer.load_bundle(args.bundle)
        raw = load_json_fixture(args.input)
        payloads = raw if isinstance(raw, list) else [raw]
        decisions = [d.to_dict() for d in scorer.score_batch(payloads)]
        print(json.dumps({"automated_decision_allowed": False, "decisions": decisions}, indent=2))
        return 0

    if args.command == "review-list":
        store = ManualReviewStore(args.store)
        cases = store.list_cases()
        print(f"Manual Review Queue ({len(cases)} cases, automated_decision_allowed=False):")
        for c in cases:
            s = c["scoring_output"]
            print(
                f"  - {c['decision_id']} | {c['application_id']} | PD={s['calibrated_pd']:.2%} "
                f"| {s['risk_band']} | status={c['review_status']}"
            )
        return 0

    if args.command == "review-ui":
        store = ManualReviewStore(args.store)
        server = create_review_http_server(store, host=args.host, port=args.port)
        print(f"Serving Manual Review Workbench at http://{args.host}:{args.port} (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
