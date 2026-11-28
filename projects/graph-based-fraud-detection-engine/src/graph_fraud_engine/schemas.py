"""Canonical data contracts, typed I/O schemas, and domain errors."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from graph_fraud_engine import FEATURE_VERSION, SCHEMA_VERSION
from graph_fraud_engine.privacy import assert_privacy_safe_id

SUPPORTED_CURRENCIES = {"USD", "EUR", "GBP", "CAD"}
SUPPORTED_CHANNELS = {"web", "mobile", "pos", "api"}
SUPPORTED_DEVICE_TYPES = {"mobile_ios", "mobile_android", "desktop_web", "kiosk", "emulator"}
SUPPORTED_MCC_CATEGORIES = {
    "groceries",
    "utilities",
    "retail",
    "travel",
    "digital_goods",
    "crypto_onramp",
    "gift_cards",
    "electronics",
    "restaurant",
}
DECISION_ACTIONS = Literal["approve", "review", "block"]
DISPOSITION_VALUES = Literal["confirmed_fraud", "false_positive", "escalated", "needs_info"]


class DomainError(Exception):
    """Base class for structured domain exceptions."""


class SchemaValidationError(DomainError):
    """Raised when input entities, transactions, or labels violate the data contract."""

    def __init__(self, message: str, errors: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.errors: List[str] = errors or [message]


class TemporalLeakageError(DomainError):
    """Raised when a future edge or immature label would leak into historical state."""


class UnsupportedOperationError(DomainError):
    """Raised for explicitly unsupported models, currencies, channels, or operations."""


class ModelNotReadyError(DomainError):
    """Raised when scoring is requested before models are trained or loaded."""


def ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC; reject naive timestamps."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("Timestamps must include explicit UTC/timezone offset (no naive timestamps)")
    return dt.astimezone(timezone.utc)


class AccountRecord(BaseModel):
    account_id: str
    created_at: datetime
    account_age_days_at_start: float = Field(ge=0.0)
    kyc_tier: Literal["basic", "standard", "enhanced"] = "standard"
    home_region: str = Field(min_length=2, max_length=16)
    is_synthetic_ring_member: bool = False
    is_shared_household_member: bool = False

    @field_validator("account_id")
    @classmethod
    def _check_account_id(cls, v: str) -> str:
        return assert_privacy_safe_id(v, "acc")

    @field_validator("created_at")
    @classmethod
    def _check_created_at(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class DeviceRecord(BaseModel):
    device_id: str
    first_seen_at: datetime
    device_type: str
    is_shared_legitimate: bool = False
    device_risk_hint: float = Field(default=0.1, ge=0.0, le=1.0)

    @field_validator("device_id")
    @classmethod
    def _check_device_id(cls, v: str) -> str:
        return assert_privacy_safe_id(v, "dev")

    @field_validator("first_seen_at")
    @classmethod
    def _check_first_seen_at(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @field_validator("device_type")
    @classmethod
    def _check_device_type(cls, v: str) -> str:
        if v not in SUPPORTED_DEVICE_TYPES:
            raise ValueError(f"Unsupported device_type '{v}'. Allowed: {sorted(SUPPORTED_DEVICE_TYPES)}")
        return v


class MerchantRecord(BaseModel):
    merchant_id: str
    onboarded_at: datetime
    category: str
    country: str = Field(min_length=2, max_length=8)
    base_chargeback_rate: float = Field(default=0.01, ge=0.0, le=1.0)

    @field_validator("merchant_id")
    @classmethod
    def _check_merchant_id(cls, v: str) -> str:
        return assert_privacy_safe_id(v, "mer")

    @field_validator("onboarded_at")
    @classmethod
    def _check_onboarded_at(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @field_validator("category")
    @classmethod
    def _check_category(cls, v: str) -> str:
        if v not in SUPPORTED_MCC_CATEGORIES:
            raise ValueError(f"Unsupported merchant category '{v}'")
        return v


class TransactionRecord(BaseModel):
    tx_id: str
    account_id: str
    device_id: str
    merchant_id: str
    tx_timestamp: datetime
    amount_usd: float = Field(gt=0.0, le=1_000_000.0)
    currency: str = "USD"
    channel: str = "web"
    ip_prefix: str = Field(default="10.0.0", min_length=3, max_length=32)

    @field_validator("tx_id")
    @classmethod
    def _check_tx_id(cls, v: str) -> str:
        return assert_privacy_safe_id(v, "tx")

    @field_validator("account_id")
    @classmethod
    def _check_acc_id(cls, v: str) -> str:
        return assert_privacy_safe_id(v, "acc")

    @field_validator("device_id")
    @classmethod
    def _check_dev_id(cls, v: str) -> str:
        return assert_privacy_safe_id(v, "dev")

    @field_validator("merchant_id")
    @classmethod
    def _check_mer_id(cls, v: str) -> str:
        return assert_privacy_safe_id(v, "mer")

    @field_validator("tx_timestamp")
    @classmethod
    def _check_tx_ts(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @field_validator("currency")
    @classmethod
    def _check_currency(cls, v: str) -> str:
        if v not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency '{v}'. Supported: {sorted(SUPPORTED_CURRENCIES)}")
        return v

    @field_validator("channel")
    @classmethod
    def _check_channel(cls, v: str) -> str:
        if v not in SUPPORTED_CHANNELS:
            raise ValueError(f"Unsupported channel '{v}'. Supported: {sorted(SUPPORTED_CHANNELS)}")
        return v


class FraudLabelRecord(BaseModel):
    tx_id: str
    is_fraud: int = Field(ge=0, le=1)
    tx_timestamp: datetime
    label_available_at: datetime
    label_source: Literal["chargeback", "analyst_review", "rule_confirmed", "matured_clean"] = "chargeback"
    fraud_pattern: Optional[str] = None

    @field_validator("tx_id")
    @classmethod
    def _check_tx_id(cls, v: str) -> str:
        return assert_privacy_safe_id(v, "tx")

    @field_validator("tx_timestamp", "label_available_at")
    @classmethod
    def _check_timestamps(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @model_validator(mode="after")
    def _check_maturity_order(self) -> "FraudLabelRecord":
        if self.label_available_at < self.tx_timestamp:
            raise ValueError("label_available_at cannot be earlier than tx_timestamp")
        return self


class DatasetBundle(BaseModel):
    schema_version: str = SCHEMA_VERSION
    dataset_id: str = "synthetic-v1"
    seed: int = 42
    accounts: List[AccountRecord]
    devices: List[DeviceRecord]
    merchants: List[MerchantRecord]
    transactions: List[TransactionRecord]
    labels: List[FraudLabelRecord]


class EvidenceEdge(BaseModel):
    source: str
    target: str
    relation: str
    timestamp: datetime
    tx_id: Optional[str] = None


class EvidencePath(BaseModel):
    path_id: str
    summary: str
    nodes: List[str]
    edges: List[EvidenceEdge]
    contains_known_fraud: bool = False


class LinkedEntitySummary(BaseModel):
    entity_id: str
    entity_type: Literal["account", "device", "merchant"]
    prior_tx_count: int
    prior_known_fraud_count: int
    unseen_cold_start: bool


class GraphExplanation(BaseModel):
    as_of_timestamp: datetime
    reason_codes: List[str]
    top_feature_contributions: Dict[str, float]
    linked_entities: List[LinkedEntitySummary]
    evidence_paths: List[EvidencePath]
    ring_cluster_size: int
    shared_device_accounts_prior: int
    legitimate_shared_device_mitigant: bool
    graph_audit_max_edge_timestamp: Optional[datetime] = None


class RiskScoreRequest(BaseModel):
    transaction: TransactionRecord
    model_name: Optional[Literal["logistic", "boosted", "gnn"]] = None
    record_for_review: bool = True


class RiskScoreResponse(BaseModel):
    decision_id: str
    schema_version: str = SCHEMA_VERSION
    feature_version: str = FEATURE_VERSION
    model_name: str
    model_version: str
    tx_id: str
    raw_score: float = Field(ge=0.0, le=1.0)
    calibrated_probability: float = Field(ge=0.0, le=1.0)
    decision: DECISION_ACTIONS
    cold_start_flags: Dict[str, bool]
    explanation: GraphExplanation
    review_case_id: Optional[str] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class AnalystDispositionRequest(BaseModel):
    analyst_id: str = Field(min_length=3, max_length=64)
    disposition: DISPOSITION_VALUES
    notes: str = Field(default="", max_length=1000)


class AnalystDispositionRecord(BaseModel):
    disposition_id: str
    case_id: str
    tx_id: str
    analyst_id: str
    disposition: DISPOSITION_VALUES
    notes: str
    recorded_at: datetime


class ReviewCase(BaseModel):
    case_id: str
    decision_id: str
    tx_id: str
    account_id: str
    device_id: str
    merchant_id: str
    tx_timestamp: datetime
    calibrated_probability: float
    decision: DECISION_ACTIONS
    status: Literal["open", "resolved"] = "open"
    explanation: GraphExplanation
    dispositions: List[AnalystDispositionRecord] = Field(default_factory=list)
