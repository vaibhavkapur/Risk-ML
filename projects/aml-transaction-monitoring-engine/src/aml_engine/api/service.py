"""Framework-agnostic typed service facade for AML transaction monitoring and case review."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aml_engine.case_management.workflow import (
    transition_case,
    verify_audit_chain,
    verify_evidence_provenance,
)
from aml_engine.config import EngineConfig
from aml_engine.ml.prioritizer import MLPrioritizer
from aml_engine.models import CaseStatus, ReviewPriority
from aml_engine.pipeline import MonitoringPipeline
from aml_engine.storage.repository import CaseRepository, SQLiteCaseRepository
from aml_engine.validation import CaseWorkflowError, ValidationError


class AMLMonitoringService:
    """Service layer backing both the stdlib HTTP server and the optional FastAPI app."""

    def __init__(
        self,
        config: EngineConfig,
        repository: Optional[CaseRepository] = None,
        prioritizer: Optional[MLPrioritizer] = None,
    ) -> None:
        self.config = config
        self.repository: CaseRepository = repository or SQLiteCaseRepository(":memory:")
        self.prioritizer = prioritizer
        self.pipeline = MonitoringPipeline(
            config=self.config,
            repository=self.repository,
            prioritizer=self.prioritizer,
        )

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "config_version": self.config.config_version,
            "config_hash": self.config.config_hash,
            "feature_version": self.config.feature_version,
            "policy_version": self.config.policy_version,
            "model_version": self.prioritizer.model_version if self.prioritizer else None,
        }

    def monitor_batch(
        self,
        payload: Dict[str, Any],
        strict_validation: bool = False,
    ) -> Dict[str, Any]:
        result = self.pipeline.run_batch(payload, strict_validation=strict_validation)
        return {
            "diagnostics": result.diagnostics.to_dict(),
            "alerts": [a.to_dict() for a in result.alerts],
            "cases": [c.to_dict() for c in result.cases],
        }

    def list_alerts(self, priority: Optional[str] = None) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.repository.list_alerts(priority=priority)]

    def get_alert_with_provenance(self, alert_id: str) -> Optional[Dict[str, Any]]:
        alert = self.repository.get_alert(alert_id)
        if alert is None:
            return None
        return {
            "alert": alert.to_dict(),
            "provenance_verified": verify_evidence_provenance(alert),
        }

    def list_cases(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.repository.list_cases(status=status)]

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        case = self.repository.get_case(case_id)
        if case is None:
            return None
        ok, err = verify_audit_chain(case)
        return {
            "case": case.to_dict(),
            "audit_chain_verified": ok,
            "audit_chain_error": err,
        }

    def review_case(
        self,
        case_id: str,
        reviewer_id: str,
        new_status: str,
        rationale: str,
        new_priority: Optional[str] = None,
        override_reason_code: Optional[str] = None,
        timestamp_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        case = self.repository.get_case(case_id)
        if case is None:
            raise CaseWorkflowError(f"Case not found: {case_id}")
        try:
            status_enum = CaseStatus(new_status.upper())
        except ValueError as exc:
            raise ValidationError(f"Unsupported case status: {new_status!r}") from exc

        priority_enum: Optional[ReviewPriority] = None
        if new_priority is not None:
            try:
                priority_enum = ReviewPriority(new_priority.upper())
            except ValueError as exc:
                raise ValidationError(f"Unsupported review priority: {new_priority!r}") from exc

        event = transition_case(
            case=case,
            reviewer_id=reviewer_id,
            new_status=status_enum,
            rationale=rationale,
            new_priority=priority_enum,
            override_reason_code=override_reason_code,
            timestamp_iso=timestamp_iso,
        )
        self.repository.update_case(case)
        ok, err = verify_audit_chain(case)
        return {
            "case": case.to_dict(),
            "recorded_event": event.to_dict(),
            "audit_chain_verified": ok,
            "audit_chain_error": err,
        }

    def list_runs(self) -> List[Dict[str, Any]]:
        return self.repository.list_runs()
