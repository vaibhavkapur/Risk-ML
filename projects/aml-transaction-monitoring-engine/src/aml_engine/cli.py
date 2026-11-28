"""Command-line interface for the AML Transaction Monitoring Engine."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List, Optional

from aml_engine.api.http_server import create_http_server
from aml_engine.api.service import AMLMonitoringService
from aml_engine.case_management.workflow import (
    transition_case,
    verify_audit_chain,
    verify_evidence_provenance,
)
from aml_engine.config import default_config_path, load_config
from aml_engine.evaluation import evaluate_rule_vs_ml_prioritization
from aml_engine.ml.synthetic_data import generate_synthetic_dataset
from aml_engine.models import CaseStatus, ReviewPriority
from aml_engine.pipeline import MonitoringPipeline
from aml_engine.storage.repository import SQLiteCaseRepository


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def cmd_demo(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dataset = generate_synthetic_dataset(seed=config.seed, num_days=90, num_suspicious_episodes=45)
    prioritizer, eval_report = evaluate_rule_vs_ml_prioritization(dataset, config)

    db_path = args.db or ":memory:"
    repo = SQLiteCaseRepository(db_path)
    pipeline = MonitoringPipeline(config=config, repository=repo, prioritizer=prioritizer)

    first_run = pipeline.run_batch(dataset)
    replay_run = pipeline.run_batch(dataset)

    # Demonstrate audited analyst review + reviewer override on top P1_HIGH case
    reviewed_case_summary = None
    if first_run.cases:
        p1_candidates = [c for c in first_run.cases if c.system_priority == ReviewPriority.P1_HIGH]
        selected = p1_candidates[0] if p1_candidates else first_run.cases[0]
        top_case = repo.get_case(selected.case_id)
        if top_case is not None:
            transition_case(
                case=top_case,
                reviewer_id="ANALYST_DEMO_01",
                new_status=CaseStatus.IN_REVIEW,
                rationale="Assigned case for initial synthetic alert triage and evidence verification.",
                timestamp_iso="2026-04-02T10:00:00Z",
            )
            transition_case(
                case=top_case,
                reviewer_id="SENIOR_REVIEWER_02",
                new_status=CaseStatus.CLOSED_BENIGN,
                new_priority=ReviewPriority.P3_LOW,
                override_reason_code="VERIFIED_BUSINESS_ACTIVITY",
                rationale=(
                    "Synthetic demo override: corroborated invoice documentation confirmed "
                    "benign business settlement; preserving human analyst judgment."
                ),
                timestamp_iso="2026-04-02T11:30:00Z",
            )
            repo.update_case(top_case)
            chain_ok, chain_err = verify_audit_chain(top_case)
            reviewed_case_summary = {
                "case_id": top_case.case_id,
                "status": top_case.status.value,
                "disposition": top_case.disposition.value,
                "system_priority": top_case.system_priority.value,
                "current_priority": top_case.current_priority.value,
                "override_count": top_case.override_count,
                "audit_events": len(top_case.audit_trail),
                "audit_chain_verified": chain_ok,
                "audit_chain_error": chain_err,
            }

    all_provenance_ok = all(verify_evidence_provenance(a) for a in first_run.alerts)
    summary = {
        "run_diagnostics_initial": first_run.diagnostics.to_dict(),
        "run_diagnostics_replay": replay_run.diagnostics.to_dict(),
        "replay_idempotent": (
            first_run.diagnostics.alerts_generated == replay_run.diagnostics.alerts_generated
            and replay_run.diagnostics.cases_created == 0
            and replay_run.diagnostics.cases_reused_on_replay == first_run.diagnostics.cases_created
        ),
        "all_evidence_provenance_verified": all_provenance_ok,
        "sample_alerts": [a.to_dict() for a in first_run.alerts[:3]],
        "reviewed_case_demo": reviewed_case_summary,
        "evaluation_snapshot": eval_report["out_of_time_test_metrics"],
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    repo.close()
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dataset = generate_synthetic_dataset(seed=config.seed, num_days=90, num_suspicious_episodes=45)
    prioritizer, report = evaluate_rule_vs_ml_prioritization(dataset, config)

    if args.artifact_out:
        prioritizer.save_artifact(args.artifact_out)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_monitor_file(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    repo = SQLiteCaseRepository(args.db or ":memory:")
    pipeline = MonitoringPipeline(config=config, repository=repo, prioritizer=None)
    res = pipeline.run_batch(payload, strict_validation=args.strict)
    output = {
        "diagnostics": res.diagnostics.to_dict(),
        "alerts": [a.to_dict() for a in res.alerts],
        "cases": [c.to_dict() for c in res.cases],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    repo.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    repo = SQLiteCaseRepository(args.db or ":memory:")
    service = AMLMonitoringService(config=config, repository=repo)
    server = create_http_server(service, host=args.host, port=args.port)
    print(f"Serving AML Transaction Monitoring Engine on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        repo.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aml-monitor",
        description="AML Transaction Monitoring Engine (Educational Synthetic Prototype)",
    )
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help="Path to JSON configuration file (default: config/default_config.json)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable structured INFO logs")

    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run reproducible end-to-end synthetic demo")
    p_demo.add_argument("--db", default=None, help="Optional SQLite DB path")
    p_demo.add_argument("--output", default=None, help="Optional JSON report output path")
    p_demo.set_defaults(func=cmd_demo)

    p_eval = sub.add_parser("evaluate", help="Run temporal rule-vs-ML prioritization evaluation")
    p_eval.add_argument("--output", default=None, help="Optional JSON evaluation output path")
    p_eval.add_argument("--artifact-out", default=None, help="Optional model artifact JSON path")
    p_eval.set_defaults(func=cmd_evaluate)

    p_mon = sub.add_parser("monitor", help="Run monitoring batch against a JSON fixture file")
    p_mon.add_argument("--input", required=True, help="Path to JSON transaction batch file")
    p_mon.add_argument("--db", default=None, help="Optional SQLite DB path")
    p_mon.add_argument("--strict", action="store_true", help="Fail fast on first invalid event")
    p_mon.set_defaults(func=cmd_monitor_file)

    p_srv = sub.add_parser("serve", help="Start zero-dependency local HTTP API server")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--db", default=None)
    p_srv.set_defaults(func=cmd_serve)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
