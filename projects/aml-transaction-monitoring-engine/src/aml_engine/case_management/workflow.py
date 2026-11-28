"""Case review workflow, deterministic provenance IDs, and reviewer override audit.

Key invariants:
1. No automatic accusation: all cases start in OPEN / PENDING_ANALYST_REVIEW with
   an explicit advisory notice preserving human analyst judgment.
2. Stable deterministic IDs: alert_id, evidence_id, and case_id are SHA-256 derived
   from canonical logical inputs so replays produce identical identifiers.
3. Reviewer override audit: any priority override or contrarian closure/escalation
   requires reviewer_id, a valid override_reason_code, and a substantive rationale,
   recorded in an append-only hash-chained audit trail verifiable for tampering.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence, Set, Tuple

from aml_engine.config import EngineConfig
from aml_engine.models import (
    ADVISORY_NOTICE,
    Alert,
    AuditEvent,
    Case,
    CaseDisposition,
    CaseStatus,
    EntityMappingStatus,
    EvidenceRecord,
    NormalizedTransaction,
    PrioritizationResult,
    ReasonContribution,
    ReviewPriority,
    ScenarioHit,
    utc_now_iso,
)
from aml_engine.validation import CaseWorkflowError


ALLOWED_STATUS_TRANSITIONS: Dict[CaseStatus, Set[CaseStatus]] = {
    CaseStatus.OPEN: {
        CaseStatus.IN_REVIEW,
        CaseStatus.ESCALATED_FOR_REVIEW,
        CaseStatus.CLOSED_BENIGN,
        CaseStatus.CLOSED_DUPLICATE,
    },
    CaseStatus.IN_REVIEW: {
        CaseStatus.ESCALATED_FOR_REVIEW,
        CaseStatus.CLOSED_BENIGN,
        CaseStatus.CLOSED_DUPLICATE,
    },
    CaseStatus.ESCALATED_FOR_REVIEW: {
        CaseStatus.IN_REVIEW,
        CaseStatus.CLOSED_BENIGN,
        CaseStatus.CLOSED_DUPLICATE,
    },
    CaseStatus.CLOSED_BENIGN: {CaseStatus.REOPENED},
    CaseStatus.CLOSED_DUPLICATE: {CaseStatus.REOPENED},
    CaseStatus.REOPENED: {
        CaseStatus.IN_REVIEW,
        CaseStatus.ESCALATED_FOR_REVIEW,
        CaseStatus.CLOSED_BENIGN,
        CaseStatus.CLOSED_DUPLICATE,
    },
}

VALID_OVERRIDE_REASON_CODES: Set[str] = {
    "VERIFIED_BUSINESS_ACTIVITY",
    "CORROBORATED_SOURCE_OF_FUNDS",
    "KNOWN_PAYROLL_OR_TREASURY_SWEEP",
    "MANUAL_INTELLIGENCE_ESCALATION",
    "RELATED_CASE_CONSOLIDATION",
    "FALSE_POSITIVE_RULE_ARTIFACT",
}

MIN_OVERRIDE_RATIONALE_LENGTH = 10


def _stable_hash(prefix: str, payload: Dict[str, object], length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:length].upper()
    return f"{prefix}-{digest}"


def build_alert_and_evidence(
    subject_account_id: str,
    hits: Sequence[ScenarioHit],
    tx_by_id: Dict[str, NormalizedTransaction],
    prioritization: PrioritizationResult,
    config: EngineConfig,
) -> Alert:
    """Create a deterministic, explainable Alert with cryptographic provenance."""
    scenarios = tuple(sorted({h.scenario_name for h in hits}))
    involved_tx_ids = tuple(sorted({tid for h in hits for tid in h.involved_tx_ids}))
    trigger_ts = max(h.trigger_timestamp_iso for h in hits)
    subject_entity_id = next((h.subject_entity_id for h in hits if h.subject_entity_id), None)

    graph_path: Tuple[str, ...] = ()
    for h in hits:
        if len(h.graph_path) > len(graph_path):
            graph_path = h.graph_path

    source_tx_hashes = tuple(
        tx_by_id[tid].canonical_event_hash for tid in involved_tx_ids if tid in tx_by_id
    )
    mapping_warnings: List[str] = []
    for tid in involved_tx_ids:
        tx = tx_by_id.get(tid)
        if tx and tx.entity_mapping_status != EntityMappingStatus.MAPPED:
            mapping_warnings.append(f"{tid}:{tx.entity_mapping_status.value}")

    alert_id = _stable_hash(
        "ALT",
        {
            "subject_account_id": subject_account_id,
            "scenarios": list(scenarios),
            "involved_tx_ids": list(involved_tx_ids),
            "policy_version": config.policy_version,
        },
    )

    provenance_payload = {
        "alert_id": alert_id,
        "config_hash": config.config_hash,
        "feature_version": config.feature_version,
        "policy_version": config.policy_version,
        "model_version": prioritization.model_version,
        "source_tx_hashes": list(source_tx_hashes),
        "graph_path": list(graph_path),
    }
    provenance_hash = hashlib.sha256(
        json.dumps(provenance_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    evidence_id = f"EVD-{provenance_hash[:16].upper()}"

    evidence = EvidenceRecord(
        evidence_id=evidence_id,
        alert_id=alert_id,
        config_hash=config.config_hash,
        feature_version=config.feature_version,
        policy_version=config.policy_version,
        model_version=prioritization.model_version,
        source_tx_hashes=source_tx_hashes,
        involved_tx_ids=involved_tx_ids,
        graph_path=graph_path,
        entity_mapping_warnings=tuple(sorted(set(mapping_warnings))),
        provenance_hash=provenance_hash,
    )

    reasons: List[ReasonContribution] = []
    for h in hits:
        reasons.extend(h.reasons)
    reasons.extend(prioritization.feature_contributions)
    if mapping_warnings:
        reasons.append(
            ReasonContribution(
                code="DATA_QUALITY_ENTITY_MAPPING_INCOMPLETE",
                category="DATA_QUALITY",
                weight=0.0,
                description=(
                    f"Entity mapping incomplete on {len(mapping_warnings)} involved transaction(s); "
                    "analyst should verify KYC entity linkage."
                ),
                evidence_metrics={"warnings": list(evidence.entity_mapping_warnings)},
            )
        )

    return Alert(
        alert_id=alert_id,
        subject_account_id=subject_account_id,
        subject_entity_id=subject_entity_id,
        scenarios=scenarios,
        trigger_timestamp_iso=trigger_ts,
        involved_tx_ids=involved_tx_ids,
        prioritization=prioritization,
        reasons=tuple(reasons),
        evidence=evidence,
        advisory_notice=ADVISORY_NOTICE,
    )


def verify_evidence_provenance(alert: Alert) -> bool:
    """Recompute and verify the SHA-256 provenance hash for an Alert's evidence."""
    ev = alert.evidence
    payload = {
        "alert_id": alert.alert_id,
        "config_hash": ev.config_hash,
        "feature_version": ev.feature_version,
        "policy_version": ev.policy_version,
        "model_version": ev.model_version,
        "source_tx_hashes": list(ev.source_tx_hashes),
        "graph_path": list(ev.graph_path),
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ev.provenance_hash == expected


def _compute_audit_event_hash(
    case_id: str,
    sequence_num: int,
    timestamp_iso: str,
    actor_id: str,
    action: str,
    previous_status: Optional[str],
    new_status: str,
    previous_priority: Optional[str],
    new_priority: str,
    is_override: bool,
    override_reason_code: Optional[str],
    rationale: str,
    prev_hash: str,
) -> str:
    payload = {
        "case_id": case_id,
        "sequence_num": sequence_num,
        "timestamp_iso": timestamp_iso,
        "actor_id": actor_id,
        "action": action,
        "previous_status": previous_status,
        "new_status": new_status,
        "previous_priority": previous_priority,
        "new_priority": new_priority,
        "is_override": is_override,
        "override_reason_code": override_reason_code,
        "rationale": rationale,
        "prev_hash": prev_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def create_case_for_alert(alert: Alert, created_at_iso: Optional[str] = None) -> Case:
    """Create a deterministic Case initialized in OPEN / PENDING_ANALYST_REVIEW."""
    ts = created_at_iso or alert.trigger_timestamp_iso
    case_id = _stable_hash("CASE", {"alert_id": alert.alert_id})
    genesis_prev = "0" * 64
    rationale = (
        f"System opened advisory review case for alert {alert.alert_id} "
        f"(scenarios={','.join(alert.scenarios)}, priority={alert.prioritization.priority.value})."
    )
    event_hash = _compute_audit_event_hash(
        case_id=case_id,
        sequence_num=1,
        timestamp_iso=ts,
        actor_id="SYSTEM_MONITOR",
        action="CASE_OPENED",
        previous_status=None,
        new_status=CaseStatus.OPEN.value,
        previous_priority=None,
        new_priority=alert.prioritization.priority.value,
        is_override=False,
        override_reason_code=None,
        rationale=rationale,
        prev_hash=genesis_prev,
    )
    init_event = AuditEvent(
        event_id=f"AUD-{event_hash[:16].upper()}",
        case_id=case_id,
        sequence_num=1,
        timestamp_iso=ts,
        actor_id="SYSTEM_MONITOR",
        action="CASE_OPENED",
        previous_status=None,
        new_status=CaseStatus.OPEN.value,
        previous_priority=None,
        new_priority=alert.prioritization.priority.value,
        is_override=False,
        override_reason_code=None,
        rationale=rationale,
        prev_hash=genesis_prev,
        event_hash=event_hash,
    )
    return Case(
        case_id=case_id,
        alert_id=alert.alert_id,
        subject_account_id=alert.subject_account_id,
        subject_entity_id=alert.subject_entity_id,
        status=CaseStatus.OPEN,
        disposition=CaseDisposition.PENDING_ANALYST_REVIEW,
        system_priority=alert.prioritization.priority,
        current_priority=alert.prioritization.priority,
        created_at_iso=ts,
        updated_at_iso=ts,
        assigned_analyst=None,
        override_count=0,
        advisory_notice=ADVISORY_NOTICE,
        audit_trail=[init_event],
    )


def _is_reviewer_override(
    case: Case,
    new_status: CaseStatus,
    new_priority: ReviewPriority,
) -> bool:
    if new_priority != case.system_priority:
        return True
    if case.system_priority == ReviewPriority.P1_HIGH and new_status == CaseStatus.CLOSED_BENIGN:
        return True
    if case.system_priority == ReviewPriority.P3_LOW and new_status == CaseStatus.ESCALATED_FOR_REVIEW:
        return True
    return False


def transition_case(
    case: Case,
    reviewer_id: str,
    new_status: CaseStatus,
    rationale: str,
    new_priority: Optional[ReviewPriority] = None,
    override_reason_code: Optional[str] = None,
    timestamp_iso: Optional[str] = None,
) -> AuditEvent:
    """Apply an audited analyst transition or reviewer override to a Case.

    Raises CaseWorkflowError on invalid state transition, missing reviewer_id,
    or missing/invalid override audit metadata.
    """
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise CaseWorkflowError("A non-empty 'reviewer_id' is required for all case actions.")
    reviewer = reviewer_id.strip()
    if reviewer.upper() == "SYSTEM_MONITOR":
        raise CaseWorkflowError("Case review decisions must be recorded by a human reviewer_id, not SYSTEM_MONITOR.")

    allowed = ALLOWED_STATUS_TRANSITIONS.get(case.status, set())
    if new_status != case.status and new_status not in allowed:
        raise CaseWorkflowError(
            f"Invalid case status transition from {case.status.value} to {new_status.value}."
        )

    target_priority = new_priority or case.current_priority
    is_override = _is_reviewer_override(case, new_status, target_priority)

    clean_rationale = (rationale or "").strip()
    if not clean_rationale:
        raise CaseWorkflowError("A non-empty 'rationale' is required for case review transitions.")

    clean_code: Optional[str] = None
    if is_override:
        if not override_reason_code or override_reason_code.strip().upper() not in VALID_OVERRIDE_REASON_CODES:
            raise CaseWorkflowError(
                f"Reviewer override requires a valid 'override_reason_code' from "
                f"{sorted(VALID_OVERRIDE_REASON_CODES)}; got {override_reason_code!r}."
            )
        if len(clean_rationale) < MIN_OVERRIDE_RATIONALE_LENGTH:
            raise CaseWorkflowError(
                f"Reviewer override requires a substantive rationale of at least "
                f"{MIN_OVERRIDE_RATIONALE_LENGTH} characters."
            )
        clean_code = override_reason_code.strip().upper()
    elif override_reason_code:
        clean_code = override_reason_code.strip().upper()

    ts = timestamp_iso or utc_now_iso()
    prev_event = case.audit_trail[-1]
    seq = prev_event.sequence_num + 1
    action = "REVIEWER_OVERRIDE" if is_override else f"STATUS_{new_status.value}"

    event_hash = _compute_audit_event_hash(
        case_id=case.case_id,
        sequence_num=seq,
        timestamp_iso=ts,
        actor_id=reviewer,
        action=action,
        previous_status=case.status.value,
        new_status=new_status.value,
        previous_priority=case.current_priority.value,
        new_priority=target_priority.value,
        is_override=is_override,
        override_reason_code=clean_code,
        rationale=clean_rationale,
        prev_hash=prev_event.event_hash,
    )
    event = AuditEvent(
        event_id=f"AUD-{event_hash[:16].upper()}",
        case_id=case.case_id,
        sequence_num=seq,
        timestamp_iso=ts,
        actor_id=reviewer,
        action=action,
        previous_status=case.status.value,
        new_status=new_status.value,
        previous_priority=case.current_priority.value,
        new_priority=target_priority.value,
        is_override=is_override,
        override_reason_code=clean_code,
        rationale=clean_rationale,
        prev_hash=prev_event.event_hash,
        event_hash=event_hash,
    )

    case.status = new_status
    case.current_priority = target_priority
    case.assigned_analyst = reviewer
    case.updated_at_iso = ts
    if new_status == CaseStatus.ESCALATED_FOR_REVIEW:
        case.disposition = CaseDisposition.REQUIRES_FURTHER_INVESTIGATION
    elif new_status == CaseStatus.CLOSED_BENIGN:
        case.disposition = CaseDisposition.CLEARED_EXPLAINED_ACTIVITY
    elif new_status == CaseStatus.CLOSED_DUPLICATE:
        case.disposition = CaseDisposition.DUPLICATE_ALERT
    else:
        case.disposition = CaseDisposition.PENDING_ANALYST_REVIEW

    if is_override:
        case.override_count += 1
    case.audit_trail.append(event)
    return event


def verify_audit_chain(case: Case) -> Tuple[bool, Optional[str]]:
    """Verify integrity of a Case's append-only hash-chained audit trail."""
    if not case.audit_trail:
        return False, "Audit trail is empty."
    expected_prev = "0" * 64
    for idx, ev in enumerate(case.audit_trail, start=1):
        if ev.sequence_num != idx:
            return False, f"Sequence mismatch at index {idx}: got {ev.sequence_num}."
        if ev.prev_hash != expected_prev:
            return False, f"Broken hash link at sequence {idx}."
        recomputed = _compute_audit_event_hash(
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
            rationale=ev.rationale,
            prev_hash=ev.prev_hash,
        )
        if ev.event_hash != recomputed:
            return False, f"Tampered event payload detected at sequence {idx} ({ev.event_id})."
        expected_prev = ev.event_hash
    return True, None
