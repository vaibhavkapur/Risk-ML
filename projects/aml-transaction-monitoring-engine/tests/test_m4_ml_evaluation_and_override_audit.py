"""M4 tests: temporal ML evaluation, calibration, workload reduction, reviewer override audit, reproducibility, and no automatic accusation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from aml_engine.case_management.workflow import (
    transition_case,
    verify_audit_chain,
)
from aml_engine.config import load_config
from aml_engine.evaluation import evaluate_rule_vs_ml_prioritization
from aml_engine.ml.prioritizer import MLPrioritizer
from aml_engine.ml.synthetic_data import generate_synthetic_dataset
from aml_engine.models import (
    ADVISORY_NOTICE,
    AuditEvent,
    CaseDisposition,
    CaseStatus,
    ReviewPriority,
)
from aml_engine.pipeline import MonitoringPipeline
from aml_engine.validation import CaseWorkflowError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.json"


class TestM4MLEvaluationAndOverrideAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)
        self.dataset = generate_synthetic_dataset(
            seed=self.config.seed,
            num_days=90,
            num_suspicious_episodes=45,
        )

    def test_temporal_ml_evaluation_and_artifact_roundtrip(self) -> None:
        prioritizer, report = evaluate_rule_vs_ml_prioritization(self.dataset, self.config)

        # Temporal split integrity
        split = report["temporal_split"]
        self.assertGreater(split["train_count"], 0)
        self.assertGreater(split["test_count"], 0)
        self.assertEqual(split["label_maturity_days"], 7)

        # Out-of-time metrics must be computed and hybrid prioritization should achieve strong precision/recall
        metrics = report["out_of_time_test_metrics"]
        rules_m = metrics["rules_only_baseline"]
        hybrid_p1 = metrics["hybrid_rule_plus_ml_p1_high_queue"]
        ml_only = metrics["ml_calibrated_only"]

        self.assertGreaterEqual(ml_only["pr_auc"], rules_m["pr_auc"] - 0.05)
        self.assertGreaterEqual(hybrid_p1["precision"], rules_m["precision"])
        self.assertGreaterEqual(hybrid_p1["workload_reduction_vs_rules_only"], 0.0)
        self.assertGreaterEqual(report["dataset_summary"]["synthetic_suspicious_tx_coverage_recall"], 0.85)

        # Save and reload model artifact -> identical predictions
        sample_features = {
            "rule_score": 0.88,
            "num_scenarios_triggered": 1.0,
            "is_layering_or_cycle": 1.0,
            "is_cash_or_wire_ratio": 1.0,
            "log_total_involved_usd": 10.5,
            "entity_risk_tier_num": 1.0,
            "is_high_risk_jurisdiction": 1.0,
            "structuring_band_count_24h": 0.0,
            "passthrough_ratio_24h": 0.95,
            "velocity_ratio_vs_baseline": 12.0,
            "fan_in_fan_out_ratio": 1.0,
            "ego_density": 0.3,
            "dormancy_days_log": 0.0,
        }
        orig_pred = prioritizer.predict_and_explain(sample_features)

        with tempfile.TemporaryDirectory() as tmpdir:
            art_path = Path(tmpdir) / "model_artifact.json"
            prioritizer.save_artifact(art_path)
            reloaded = MLPrioritizer.load_artifact(art_path)
            reloaded_pred = reloaded.predict_and_explain(sample_features)
            self.assertEqual(orig_pred[0], reloaded_pred[0])
            self.assertEqual(orig_pred[1], reloaded_pred[1])
            self.assertEqual(prioritizer.model_version, reloaded.model_version)

    def test_reviewer_override_audit_boundary_and_tamper_detection(self) -> None:
        fixture = json.loads(
            (PROJECT_ROOT / "data" / "fixtures" / "reference_hand_calc.json").read_text(encoding="utf-8")
        )
        out = MonitoringPipeline(config=self.config).run_batch(fixture)
        case = out.cases[0]
        self.assertEqual(case.system_priority, ReviewPriority.P1_HIGH)

        # 1. Missing reviewer_id rejected
        with self.assertRaises(CaseWorkflowError):
            transition_case(case, "", CaseStatus.IN_REVIEW, "Taking case.")

        # 2. SYSTEM_MONITOR cannot act as human reviewer
        with self.assertRaises(CaseWorkflowError):
            transition_case(case, "SYSTEM_MONITOR", CaseStatus.IN_REVIEW, "Automated close.")

        # 3. Illegal transition OPEN -> REOPENED rejected
        with self.assertRaises(CaseWorkflowError):
            transition_case(case, "ANALYST_01", CaseStatus.REOPENED, "Invalid reopen.")

        # 4. Override (closing P1_HIGH as benign) without valid override_reason_code rejected
        with self.assertRaises(CaseWorkflowError):
            transition_case(
                case,
                reviewer_id="ANALYST_01",
                new_status=CaseStatus.CLOSED_BENIGN,
                rationale="Customer provided valid source of funds invoice.",
                override_reason_code=None,
            )

        # 5. Override with too-short rationale rejected
        with self.assertRaises(CaseWorkflowError):
            transition_case(
                case,
                reviewer_id="ANALYST_01",
                new_status=CaseStatus.CLOSED_BENIGN,
                rationale="fine",
                override_reason_code="VERIFIED_BUSINESS_ACTIVITY",
            )

        # 6. Valid override succeeds and records immutable hash-chained audit event
        ev = transition_case(
            case,
            reviewer_id="SENIOR_ANALYST_09",
            new_status=CaseStatus.CLOSED_BENIGN,
            new_priority=ReviewPriority.P3_LOW,
            override_reason_code="VERIFIED_BUSINESS_ACTIVITY",
            rationale="Verified supporting commercial invoice and counterparty KYC documents.",
            timestamp_iso="2026-01-12T12:00:00Z",
        )
        self.assertTrue(ev.is_override)
        self.assertEqual(ev.override_reason_code, "VERIFIED_BUSINESS_ACTIVITY")
        self.assertEqual(case.override_count, 1)
        self.assertEqual(case.disposition, CaseDisposition.CLEARED_EXPLAINED_ACTIVITY)
        ok, err = verify_audit_chain(case)
        self.assertTrue(ok, msg=f"Expected intact audit chain, got: {err}")

        # 7. Tampering with audit event rationale breaks cryptographic hash verification
        tampered = AuditEvent(
            event_id=ev.event_id,
            case_id=ev.case_id,
            sequence_num=ev.sequence_num,
            timestamp_iso=ev.timestamp_iso,
            actor_id=ev.actor_id,
            action=ev.action,
            previous_status=ev.previous_status,
            new_status=ev.new_status,
            previous_priority=ev.previous_priority,
            new_priority=ev.new_priority,
            is_override=ev.is_override,
            override_reason_code=ev.override_reason_code,
            rationale="TAMPERED RATIONALE TEXT",
            prev_hash=ev.prev_hash,
            event_hash=ev.event_hash,
        )
        case.audit_trail[-1] = tampered
        ok_after, err_after = verify_audit_chain(case)
        self.assertFalse(ok_after)
        self.assertIn("Tampered", err_after or "")

    def test_strict_reproducibility_and_no_automatic_accusation(self) -> None:
        p1, r1 = evaluate_rule_vs_ml_prioritization(self.dataset, self.config)
        p2, r2 = evaluate_rule_vs_ml_prioritization(self.dataset, self.config)

        self.assertEqual(p1.model_version, p2.model_version)
        self.assertEqual(r1["out_of_time_test_metrics"], r2["out_of_time_test_metrics"])

        run1 = MonitoringPipeline(config=self.config, prioritizer=p1).run_batch(self.dataset)
        run2 = MonitoringPipeline(config=self.config, prioritizer=p2).run_batch(self.dataset)

        self.assertEqual(run1.diagnostics.run_id, run2.diagnostics.run_id)
        self.assertEqual([a.to_dict() for a in run1.alerts], [a.to_dict() for a in run2.alerts])

        # Verify no automatic accusation across all alerts and cases
        forbidden_accusations = ("guilty", "criminal", "convicted", "auto-sar-filed")
        for alert in run1.alerts:
            self.assertEqual(alert.advisory_notice, ADVISORY_NOTICE)
            serialized = json.dumps(alert.to_dict()).lower()
            for term in forbidden_accusations:
                self.assertNotIn(term, serialized)
        for case in run1.cases:
            self.assertEqual(case.status, CaseStatus.OPEN)
            self.assertEqual(case.disposition, CaseDisposition.PENDING_ANALYST_REVIEW)
            self.assertEqual(case.advisory_notice, ADVISORY_NOTICE)


if __name__ == "__main__":
    unittest.main()
