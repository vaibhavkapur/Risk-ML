"""Command-line entry point for demo, evaluation, and replay checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from payment_auth_engine.config import ConfigurationError, load_config
from payment_auth_engine.engine import AuthorizationEngine, run_full_experiment
from payment_auth_engine.models import AuthorizationModelBundle
from payment_auth_engine.policy import ConstrainedRiskAwarePolicy
from payment_auth_engine.synthetic import generate_synthetic_dataset


def _format_summary_table(result: dict) -> str:
    lines = [
        "=== Payment Authorization Optimization Engine — Policy Comparison (Synthetic Test Fold) ===",
        f"Model Version: {result['model_summary']['model_version']} | Config Hash: {result['runtime_metadata']['config_hash']} | Seed: {result['runtime_metadata']['seed']}",
        f"{'Policy':<25} {'DR Accept':>10} {'95% CI':>21} {'DR Loss bps':>12} {'95% CI bps':>19} {'P95 Lat ms':>11} {'Audit':>8}",
        "-" * 112,
    ]
    for name, metrics in result["policy_comparison"].items():
        acc = metrics["net_acceptance_bounds"]
        loss = metrics["expected_loss_bps_bounds"]
        audit_ok = "PASS" if result["policy_audit"][name]["passed"] else "BREACH"
        lines.append(
            f"{name:<25} {metrics['dr_net_acceptance_rate']:>10.4f} "
            f"[{acc['lower']:.4f}, {acc['upper']:.4f}] "
            f"{metrics['dr_expected_loss_bps']:>12.2f} "
            f"[{loss['lower']:>6.2f}, {loss['upper']:>6.2f}] "
            f"{metrics['p95_latency_ms']:>11.1f} {audit_ok:>8}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="payment-auth-engine",
        description="Payment Authorization Optimization Engine CLI (offline synthetic prototype).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo_p = sub.add_parser("demo", help="Run full reproducible training, calibration, counterfactual comparison, and audit.")
    demo_p.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    demo_p.add_argument("--output-dir", default="reports", help="Directory for machine-readable output reports.")
    demo_p.add_argument("--json", action="store_true", help="Print full JSON summary to stdout.")

    replay_p = sub.add_parser("replay", help="Verify deterministic replay identity on synthetic test fold.")
    replay_p.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")

    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigurationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.command == "demo":
        out_dir = Path(args.output_dir) if args.output_dir else None
        result = run_full_experiment(config, output_dir=out_dir)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(_format_summary_table(result))
            if out_dir is not None:
                print(f"\nReports written to: {out_dir.resolve()}")
        return 0

    if args.command == "replay":
        dataset = generate_synthetic_dataset(config)
        bundle, _, cal_idx, test_idx = AuthorizationModelBundle.train(
            dataset.attempts, dataset.outcomes, config, dataset.as_of_utc
        )
        cal_attempts = [dataset.attempts[int(i)] for i in cal_idx]
        cal_preds = bundle.predict_batch_all_actions(cal_attempts)
        test_attempts = [dataset.attempts[int(i)] for i in test_idx]

        pol = ConstrainedRiskAwarePolicy(config)
        pol.fit_dual_multipliers(cal_attempts, cal_preds)
        engine = AuthorizationEngine(config=config, model_bundle=bundle, policy=pol)
        run1 = engine.decide_batch(test_attempts)
        run2 = engine.replay(test_attempts)

        identical = all(
            a.decision_id == b.decision_id and a.action == b.action and a.expected_net_value_cents == b.expected_net_value_cents
            for a, b in zip(run1, run2)
        )
        print(json.dumps({"replay_verified": identical, "n_decisions": len(run1), "model_version": bundle.model_version}))
        return 0 if identical else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
