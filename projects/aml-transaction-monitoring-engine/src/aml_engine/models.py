"""Canonical typed domain models for the AML Transaction Monitoring Engine.

All alert and case terminology is intentionally non-accusatory: the engine
produces advisory investigative alerts for human analyst review, never
automated compliance determinations or accusations of wrongdoing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


ADVISORY_NOTICE = (
    "Advisory transaction-monitoring indicator for human analyst review only. "
    "This output is not an automated accusation, legal finding, or regulatory filing."
)


class EntityMappingStatus(str, Enum):
    MAPPED = "MAPPED"
    MISSING_SOURCE_ACCOUNT = "MISSING_SOURCE_ACCOUNT"
    MISSING_TARGET_ACCOUNT = "MISSING_TARGET_ACCOUNT"
    MISSING_SOURCE_ENTITY = "MISSING_SOURCE_ENTITY"
    MISSING_TARGET_ENTITY = "MISSING_TARGET_ENTITY"
    PARTIAL_OR_UNMAPPED = "PARTIAL_OR_UNMAPPED"


class ReviewPriority(str, Enum):
    P1_HIGH = "P1_HIGH"
    P2_MEDIUM = "P2_MEDIUM"
    P3_LOW = "P3_LOW"
    SUPPRESSED_BELOW_THRESHOLD = "SUPPRESSED_BELOW_THRESHOLD"


class CaseStatus(str, Enum):
    OPEN = "OPEN"
    IN_REVIEW = "IN_REVIEW"
    ESCALATED_FOR_REVIEW = "ESCALATED_FOR_REVIEW"
    CLOSED_BENIGN = "CLOSED_BENIGN"
    CLOSED_DUPLICATE = "CLOSED_DUPLICATE"
    REOPENED = "REOPENED"


class CaseDisposition(str, Enum):
    PENDING_ANALYST_REVIEW = "PENDING_ANALYST_REVIEW"
    REQUIRES_FURTHER_INVESTIGATION = "REQUIRES_FURTHER_INVESTIGATION"
    CLEARED_EXPLAINED_ACTIVITY = "CLEARED_EXPLAINED_ACTIVITY"
    DUPLICATE_ALERT = "DUPLICATE_ALERT"


@dataclass(frozen=True)
class Entity:
    entity_id: str
    name: str
    risk_tier: str = "LOW"
    jurisdiction: str = "US"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Account:
    account_id: str
    entity_id: str
    account_type: str = "RETAIL"
    opened_at: str = "2025-01-01T00:00:00Z"
    baseline_monthly_usd: float = 5000.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawTransaction:
    tx_id: str
    source_account_id: str
    target_account_id: str
    amount: float
    currency: str
    timestamp: str
    channel: str
    scenario_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedTransaction:
    tx_id: str
    canonical_event_hash: str
    source_account_id: str
    target_account_id: str
    source_entity_id: Optional[str]
    target_entity_id: Optional[str]
    source_account_type: str
    target_account_type: str
    source_risk_tier: str
    target_risk_tier: str
    original_amount: float
    original_currency: str
    fx_rate_to_usd: float
    amount_usd: float
    timestamp_utc: datetime
    timestamp_iso: str
    channel: str
    entity_mapping_status: EntityMappingStatus
    scenario_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "canonical_event_hash": self.canonical_event_hash,
            "source_account_id": self.source_account_id,
            "target_account_id": self.target_account_id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "source_account_type": self.source_account_type,
            "target_account_type": self.target_account_type,
            "source_risk_tier": self.source_risk_tier,
            "target_risk_tier": self.target_risk_tier,
            "original_amount": self.original_amount,
            "original_currency": self.original_currency,
            "fx_rate_to_usd": self.fx_rate_to_usd,
            "amount_usd": round(self.amount_usd, 6),
            "timestamp_iso": self.timestamp_iso,
            "channel": self.channel,
            "entity_mapping_status": self.entity_mapping_status.value,
            "scenario_label": self.scenario_label,
        }


@dataclass(frozen=True)
class RollingFeatureVector:
    tx_id: str
    account_id: str
    role: str  # "SOURCE" or "TARGET"
    as_of_iso: str
    feature_version: str
    inbound_count_1h: int
    inbound_sum_1h_usd: float
    inbound_count_24h: int
    inbound_sum_24h_usd: float
    inbound_count_7d: int
    inbound_sum_7d_usd: float
    inbound_sum_30d_usd: float
    outbound_count_1h: int
    outbound_sum_1h_usd: float
    outbound_count_24h: int
    outbound_sum_24h_usd: float
    outbound_count_7d: int
    outbound_sum_7d_usd: float
    outbound_sum_30d_usd: float
    structuring_band_count_24h: int
    structuring_band_sum_24h_usd: float
    unique_inbound_senders_24h: int
    unique_outbound_recipients_24h: int
    passthrough_ratio_24h: float
    velocity_ratio_vs_baseline: float
    dormancy_days_prior: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, float):
                data[key] = round(value, 6)
        return data


@dataclass(frozen=True)
class GraphFeatureVector:
    account_id: str
    in_degree: int
    out_degree: int
    unique_in_neighbors: int
    unique_out_neighbors: int
    fan_in_fan_out_ratio: float
    total_inbound_usd: float
    total_outbound_usd: float
    net_flow_ratio: float
    ego_density: float
    shared_entity_counterparties: int
    in_layering_chain: bool
    in_round_trip_cycle: bool

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, float):
                data[key] = round(value, 6)
        return data


@dataclass(frozen=True)
class ReasonContribution:
    code: str
    category: str  # "RULE", "GRAPH", "ML_FEATURE", "DATA_QUALITY"
    weight: float
    description: str
    evidence_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "category": self.category,
            "weight": round(self.weight, 6),
            "description": self.description,
            "evidence_metrics": self.evidence_metrics,
        }


@dataclass(frozen=True)
class ScenarioHit:
    scenario_name: str
    subject_account_id: str
    subject_entity_id: Optional[str]
    trigger_timestamp_iso: str
    rule_score: float
    involved_tx_ids: Tuple[str, ...]
    involved_accounts: Tuple[str, ...]
    graph_path: Tuple[str, ...]
    reasons: Tuple[ReasonContribution, ...]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "subject_account_id": self.subject_account_id,
            "subject_entity_id": self.subject_entity_id,
            "trigger_timestamp_iso": self.trigger_timestamp_iso,
            "rule_score": round(self.rule_score, 6),
            "involved_tx_ids": list(self.involved_tx_ids),
            "involved_accounts": list(self.involved_accounts),
            "graph_path": list(self.graph_path),
            "reasons": [r.to_dict() for r in self.reasons],
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class PrioritizationResult:
    rule_score: float
    ml_raw_score: Optional[float]
    ml_calibrated_probability: Optional[float]
    combined_score: float
    priority: ReviewPriority
    policy_version: str
    model_version: Optional[str]
    feature_contributions: Tuple[ReasonContribution, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_score": round(self.rule_score, 6),
            "ml_raw_score": round(self.ml_raw_score, 6) if self.ml_raw_score is not None else None,
            "ml_calibrated_probability": (
                round(self.ml_calibrated_probability, 6)
                if self.ml_calibrated_probability is not None
                else None
            ),
            "combined_score": round(self.combined_score, 6),
            "priority": self.priority.value,
            "policy_version": self.policy_version,
            "model_version": self.model_version,
            "feature_contributions": [c.to_dict() for c in self.feature_contributions],
        }


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    alert_id: str
    config_hash: str
    feature_version: str
    policy_version: str
    model_version: Optional[str]
    source_tx_hashes: Tuple[str, ...]
    involved_tx_ids: Tuple[str, ...]
    graph_path: Tuple[str, ...]
    entity_mapping_warnings: Tuple[str, ...]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "alert_id": self.alert_id,
            "config_hash": self.config_hash,
            "feature_version": self.feature_version,
            "policy_version": self.policy_version,
            "model_version": self.model_version,
            "source_tx_hashes": list(self.source_tx_hashes),
            "involved_tx_ids": list(self.involved_tx_ids),
            "graph_path": list(self.graph_path),
            "entity_mapping_warnings": list(self.entity_mapping_warnings),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class Alert:
    alert_id: str
    subject_account_id: str
    subject_entity_id: Optional[str]
    scenarios: Tuple[str, ...]
    trigger_timestamp_iso: str
    involved_tx_ids: Tuple[str, ...]
    prioritization: PrioritizationResult
    reasons: Tuple[ReasonContribution, ...]
    evidence: EvidenceRecord
    advisory_notice: str = ADVISORY_NOTICE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "subject_account_id": self.subject_account_id,
            "subject_entity_id": self.subject_entity_id,
            "scenarios": list(self.scenarios),
            "trigger_timestamp_iso": self.trigger_timestamp_iso,
            "involved_tx_ids": list(self.involved_tx_ids),
            "prioritization": self.prioritization.to_dict(),
            "reasons": [r.to_dict() for r in self.reasons],
            "evidence": self.evidence.to_dict(),
            "advisory_notice": self.advisory_notice,
        }


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    case_id: str
    sequence_num: int
    timestamp_iso: str
    actor_id: str
    action: str
    previous_status: Optional[str]
    new_status: str
    previous_priority: Optional[str]
    new_priority: str
    is_override: bool
    override_reason_code: Optional[str]
    rationale: str
    prev_hash: str
    event_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Case:
    case_id: str
    alert_id: str
    subject_account_id: str
    subject_entity_id: Optional[str]
    status: CaseStatus
    disposition: CaseDisposition
    system_priority: ReviewPriority
    current_priority: ReviewPriority
    created_at_iso: str
    updated_at_iso: str
    assigned_analyst: Optional[str]
    override_count: int = 0
    advisory_notice: str = ADVISORY_NOTICE
    audit_trail: List[AuditEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "alert_id": self.alert_id,
            "subject_account_id": self.subject_account_id,
            "subject_entity_id": self.subject_entity_id,
            "status": self.status.value,
            "disposition": self.disposition.value,
            "system_priority": self.system_priority.value,
            "current_priority": self.current_priority.value,
            "created_at_iso": self.created_at_iso,
            "updated_at_iso": self.updated_at_iso,
            "assigned_analyst": self.assigned_analyst,
            "override_count": self.override_count,
            "advisory_notice": self.advisory_notice,
            "audit_trail": [e.to_dict() for e in self.audit_trail],
        }


@dataclass
class RunDiagnostics:
    run_id: str
    config_version: str
    config_hash: str
    feature_version: str
    policy_version: str
    model_version: Optional[str]
    total_raw_events: int = 0
    valid_normalized_events: int = 0
    invalid_events_rejected: int = 0
    duplicate_events_skipped: int = 0
    missing_entity_mappings: int = 0
    scenario_hits_count: int = 0
    alerts_generated: int = 0
    cases_created: int = 0
    cases_reused_on_replay: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
