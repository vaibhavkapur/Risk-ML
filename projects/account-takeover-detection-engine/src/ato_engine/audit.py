"""Structured decision audit logging, diagnostic counters, and replay verification."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .schemas import PolicyDecision


class DecisionAuditLogger:
    """Append-only structured JSONL audit log guaranteeing every decision is recorded."""

    def __init__(self, log_path: Optional[str | Path] = None) -> None:
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict[str, Any]] = []

    def log_decision(self, decision: PolicyDecision, latency_ms: float) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "decision_id": decision.decision_id,
            "event_id": decision.event_id,
            "user_id": decision.user_id,
            "session_id": decision.session_id,
            "event_timestamp": decision.timestamp.isoformat(),
            "schema_version": decision.schema_version,
            "feature_version": decision.feature_version,
            "model_version": decision.model_version,
            "policy_version": decision.policy_version,
            "action": decision.action.value,
            "calibrated_risk_score": decision.calibrated_risk_score,
            "anomaly_score": decision.anomaly_score,
            "supervised_probability": decision.supervised_probability,
            "thresholds_applied": decision.thresholds_applied,
            "threshold_adjustments": decision.threshold_adjustments,
            "fallback_used": decision.fallback_used,
            "fallback_reason": decision.fallback_reason.value,
            "drift_alert_active": decision.drift_alert_active,
            "triggered_rules": decision.evidence.triggered_rules,
            "missing_fields": decision.evidence.missing_fields,
            "profile_expired": decision.evidence.profile_expired,
            "latency_ms": round(latency_ms, 3),
        }
        self._records.append(record)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def count(self) -> int:
        return len(self._records)

    def read_persisted_records(self) -> List[Dict[str, Any]]:
        if self.log_path is None or not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


@dataclass
class DiagnosticCounters:
    """Operational diagnostic counters for monitoring and runbook inspection."""

    total_decisions: int = 0
    decisions_by_action: Counter[str] = field(default_factory=Counter)
    fallbacks_by_reason: Counter[str] = field(default_factory=Counter)
    validation_errors: int = 0
    unsupported_input_errors: int = 0
    drift_checks_run: int = 0
    critical_drift_events: int = 0
    total_latency_ms: float = 0.0

    def record_decision(self, decision: PolicyDecision, latency_ms: float) -> None:
        self.total_decisions += 1
        self.decisions_by_action[decision.action.value] += 1
        if decision.fallback_used:
            self.fallbacks_by_reason[decision.fallback_reason.value] += 1
        self.total_latency_ms += latency_ms

    def snapshot(self) -> Dict[str, Any]:
        avg_latency = (
            round(self.total_latency_ms / self.total_decisions, 3)
            if self.total_decisions > 0
            else 0.0
        )
        return {
            "total_decisions": self.total_decisions,
            "decisions_by_action": dict(self.decisions_by_action),
            "fallbacks_by_reason": dict(self.fallbacks_by_reason),
            "validation_errors": self.validation_errors,
            "unsupported_input_errors": self.unsupported_input_errors,
            "drift_checks_run": self.drift_checks_run,
            "critical_drift_events": self.critical_drift_events,
            "avg_latency_ms": avg_latency,
        }


def compare_decision_replays(
    run_a: Sequence[PolicyDecision],
    run_b: Sequence[PolicyDecision],
    score_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Verify two replay runs produce identical logical outputs within tolerance."""
    if len(run_a) != len(run_b):
        return {
            "matched": False,
            "reason": f"Length mismatch: {len(run_a)} vs {len(run_b)}",
            "checked": 0,
            "mismatches": [],
        }
    mismatches: List[Dict[str, Any]] = []
    for d1, d2 in zip(run_a, run_b):
        if (
            d1.decision_id != d2.decision_id
            or d1.action != d2.action
            or d1.fallback_reason != d2.fallback_reason
            or abs(d1.calibrated_risk_score - d2.calibrated_risk_score) > score_tolerance
        ):
            mismatches.append(
                {
                    "event_id": d1.event_id,
                    "decision_id_a": d1.decision_id,
                    "decision_id_b": d2.decision_id,
                    "action_a": d1.action.value,
                    "action_b": d2.action.value,
                    "score_a": d1.calibrated_risk_score,
                    "score_b": d2.calibrated_risk_score,
                }
            )
    return {
        "matched": len(mismatches) == 0,
        "checked": len(run_a),
        "mismatches": mismatches,
    }
