"""M3 tests: duplicate events, missing entity mapping, provenance, replay, persistence, and HTTP API."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from aml_engine.api.http_server import dispatch_http_request
from aml_engine.api.service import AMLMonitoringService
from aml_engine.case_management.workflow import (
    verify_audit_chain,
    verify_evidence_provenance,
)
from aml_engine.config import load_config
from aml_engine.models import CaseStatus, EntityMappingStatus, EvidenceRecord
from aml_engine.pipeline import MonitoringPipeline
from aml_engine.storage.repository import SQLiteCaseRepository


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"
CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.json"


class TestM3IntegrationAndCaseWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_duplicate_events_and_missing_entity_mapping_diagnostics(self) -> None:
        fixture = json.loads((FIXTURES_DIR / "boundary_cases.json").read_text(encoding="utf-8"))
        pipeline = MonitoringPipeline(config=self.config)
        out = pipeline.run_batch(fixture, strict_validation=True)

        # Fixture has 6 raw tx, 1 duplicate TX-BND-DUP-1, and 1 tx with missing account + missing entity
        self.assertEqual(out.diagnostics.total_raw_events, 6)
        self.assertEqual(out.diagnostics.duplicate_events_skipped, 1)
        self.assertEqual(out.diagnostics.valid_normalized_events, 5)
        self.assertEqual(out.diagnostics.missing_entity_mappings, 1)
        self.assertTrue(any("TX-BND-MISSING-ENT" in w for w in out.diagnostics.warnings))

        missing_tx = next(t for t in out.normalized_transactions if t.tx_id == "TX-BND-MISSING-ENT")
        self.assertEqual(missing_tx.entity_mapping_status, EntityMappingStatus.PARTIAL_OR_UNMAPPED)
        self.assertIsNone(missing_tx.source_entity_id)
        self.assertIsNone(missing_tx.target_entity_id)

    def test_replay_idempotency_and_sqlite_restart_recovery(self) -> None:
        fixture = json.loads((FIXTURES_DIR / "reference_hand_calc.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "aml_cases.sqlite3"
            repo1 = SQLiteCaseRepository(db_path)
            pipe1 = MonitoringPipeline(config=self.config, repository=repo1)
            run1 = pipe1.run_batch(fixture)
            self.assertEqual(run1.diagnostics.cases_created, len(run1.alerts))
            self.assertEqual(run1.diagnostics.cases_reused_on_replay, 0)
            alert_ids_1 = [a.alert_id for a in run1.alerts]
            case_ids_1 = [c.case_id for c in run1.cases]
            repo1.close()

            # Reopen SQLite DB (simulating service restart) and replay identical batch
            repo2 = SQLiteCaseRepository(db_path)
            pipe2 = MonitoringPipeline(config=self.config, repository=repo2)
            run2 = pipe2.run_batch(fixture)
            self.assertEqual(run2.diagnostics.cases_created, 0)
            self.assertEqual(run2.diagnostics.cases_reused_on_replay, len(run1.alerts))
            self.assertEqual([a.alert_id for a in run2.alerts], alert_ids_1)
            self.assertEqual([c.case_id for c in run2.cases], case_ids_1)
            self.assertEqual(len(repo2.list_cases()), len(run1.cases))
            self.assertEqual(len(repo2.list_runs()), 1)  # Same deterministic run_id upserted cleanly

            for alert in repo2.list_alerts():
                self.assertTrue(verify_evidence_provenance(alert))
            for case in repo2.list_cases():
                ok, err = verify_audit_chain(case)
                self.assertTrue(ok, msg=f"Audit chain failed: {err}")
            repo2.close()

    def test_provenance_tamper_detection(self) -> None:
        fixture = json.loads((FIXTURES_DIR / "reference_hand_calc.json").read_text(encoding="utf-8"))
        out = MonitoringPipeline(config=self.config).run_batch(fixture)
        alert = out.alerts[0]
        self.assertTrue(verify_evidence_provenance(alert))

        # Tamper with evidence source_tx_hashes while keeping original provenance_hash
        tampered_ev = EvidenceRecord(
            evidence_id=alert.evidence.evidence_id,
            alert_id=alert.evidence.alert_id,
            config_hash=alert.evidence.config_hash,
            feature_version=alert.evidence.feature_version,
            policy_version=alert.evidence.policy_version,
            model_version=alert.evidence.model_version,
            source_tx_hashes=("tampered_hash",),
            involved_tx_ids=alert.evidence.involved_tx_ids,
            graph_path=alert.evidence.graph_path,
            entity_mapping_warnings=alert.evidence.entity_mapping_warnings,
            provenance_hash=alert.evidence.provenance_hash,
        )
        tampered_alert = type(alert)(
            alert_id=alert.alert_id,
            subject_account_id=alert.subject_account_id,
            subject_entity_id=alert.subject_entity_id,
            scenarios=alert.scenarios,
            trigger_timestamp_iso=alert.trigger_timestamp_iso,
            involved_tx_ids=alert.involved_tx_ids,
            prioritization=alert.prioritization,
            reasons=alert.reasons,
            evidence=tampered_ev,
            advisory_notice=alert.advisory_notice,
        )
        self.assertFalse(verify_evidence_provenance(tampered_alert))

    def test_service_and_http_api_end_to_end(self) -> None:
        fixture = json.loads((FIXTURES_DIR / "reference_hand_calc.json").read_text(encoding="utf-8"))
        repo = SQLiteCaseRepository(":memory:")
        service = AMLMonitoringService(config=self.config, repository=repo)

        try:
            # GET /health
            status, health = dispatch_http_request(service, "GET", "/health")
            self.assertEqual(status, 200)
            self.assertEqual(health["status"], "ok")

            # POST /monitor
            status, mon = dispatch_http_request(
                service, "POST", "/monitor", json.dumps(fixture).encode("utf-8")
            )
            self.assertEqual(status, 200)
            self.assertGreaterEqual(len(mon["alerts"]), 2)
            case_id = mon["cases"][0]["case_id"]
            alert_id = mon["alerts"][0]["alert_id"]

            # GET /alerts/<id> with provenance verification
            status, prov = dispatch_http_request(service, "GET", f"/alerts/{alert_id}")
            self.assertEqual(status, 200)
            self.assertTrue(prov["provenance_verified"])

            # POST /cases/<id>/review valid transition
            review_payload = json.dumps(
                {
                    "reviewer_id": "ANALYST_HTTP_01",
                    "new_status": CaseStatus.IN_REVIEW.value,
                    "rationale": "HTTP integration test taking ownership of case.",
                }
            ).encode("utf-8")
            status, reviewed = dispatch_http_request(
                service, "POST", f"/cases/{case_id}/review", review_payload
            )
            self.assertEqual(status, 200)
            self.assertEqual(reviewed["case"]["status"], "IN_REVIEW")
            self.assertTrue(reviewed["audit_chain_verified"])

            # POST /cases/<id>/review invalid override must return HTTP 400
            bad_override = json.dumps(
                {
                    "reviewer_id": "ANALYST_HTTP_01",
                    "new_status": "CLOSED_BENIGN",
                    "new_priority": "P3_LOW",
                    "rationale": "ok",
                }
            ).encode("utf-8")
            status, err_body = dispatch_http_request(
                service, "POST", f"/cases/{case_id}/review", bad_override
            )
            self.assertEqual(status, 400)
            self.assertEqual(err_body["error_type"], "CaseWorkflowError")

            # GET /runs returns run diagnostics
            status, runs_body = dispatch_http_request(service, "GET", "/runs")
            self.assertEqual(status, 200)
            self.assertEqual(len(runs_body["runs"]), 1)

            # GET unknown route returns 404
            status, _ = dispatch_http_request(service, "GET", "/does-not-exist")
            self.assertEqual(status, 404)
        finally:
            repo.close()


if __name__ == "__main__":
    unittest.main()
