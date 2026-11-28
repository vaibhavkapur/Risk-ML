"""Canonical data contracts, validation rules, and output schemas.

All data contracts are synthetic-only; real customer PII, credentials,
or production device fingerprints are explicitly excluded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import ipaddress
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import FEATURE_VERSION, POLICY_VERSION, SCHEMA_VERSION


SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})
SUPPORTED_EVENT_TYPES = frozenset(
    {"login_attempt", "session_activity", "mfa_challenge", "password_reset"}
)
SUPPORTED_AUTH_OUTCOMES = frozenset(
    {"success", "failure", "challenge_passed", "challenge_failed"}
)
SUPPORTED_AUTH_METHODS = frozenset({"password", "otp", "webauthn", "sso"})
SUPPORTED_ACCOUNT_TIERS = frozenset({"standard", "high_value"})
SUPPORTED_SENSITIVE_ACTIONS = frozenset(
    {
        "change_mfa",
        "change_email",
        "change_password",
        "add_payee",
        "wire_transfer_setup",
        "export_data",
        "disable_alerts",
        "api_key_create",
    }
)
AUTH_METHOD_STRENGTH: Dict[str, int] = {
    "password": 1,
    "otp": 2,
    "sso": 2,
    "webauthn": 3,
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_COUNTRY_RE = re.compile(r"^[A-Z]{2}$")


class ATOEngineError(Exception):
    """Base error for the Account Takeover Detection Engine."""


class ContractValidationError(ATOEngineError):
    """Raised when an input event violates the data contract."""


class UnsupportedInputError(ATOEngineError):
    """Raised when an input uses an unsupported schema version, event type, or field value."""


class StoreUnavailableError(ATOEngineError):
    """Raised when the feature state store cannot be reached."""


class ModelArtifactError(ATOEngineError):
    """Raised when a model artifact is missing, corrupt, or incompatible."""


class StepUpAction(str, Enum):
    ALLOW = "ALLOW"
    CHALLENGE_OTP = "CHALLENGE_OTP"
    CHALLENGE_WEBAUTHN = "CHALLENGE_WEBAUTHN"
    BLOCK_AND_RESET = "BLOCK_AND_RESET"


class FallbackReason(str, Enum):
    NONE = "none"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_STALE = "model_stale"
    STORE_UNAVAILABLE = "store_unavailable"
    EXCESSIVE_MISSING_FEATURES = "excessive_missing_features"
    CRITICAL_DRIFT_CIRCUIT_BREAKER = "critical_drift_circuit_breaker"


class LoginSessionEvent(BaseModel):
    """Canonical synthetic login, device, network, and session event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(..., min_length=1, max_length=128)
    schema_version: str = Field(default=SCHEMA_VERSION)
    timestamp: datetime = Field(..., description="UTC timezone-aware event timestamp (ISO-8601)")
    user_id: str = Field(..., min_length=1, max_length=128)
    session_id: str = Field(..., min_length=1, max_length=128)
    event_type: str = Field(default="login_attempt")
    auth_outcome: str = Field(default="success")
    auth_method: str = Field(default="password")
    account_tier: str = Field(default="standard")

    device_id: Optional[str] = Field(default=None, max_length=128)
    device_trust_score: Optional[float] = Field(default=None)

    ip_address: Optional[str] = Field(default=None)
    ip_reputation_risk: Optional[float] = Field(default=None)
    is_known_shared_network: bool = Field(default=False)

    geo_lat: Optional[float] = Field(default=None)
    geo_lon: Optional[float] = Field(default=None)
    country_code: Optional[str] = Field(default=None)

    sensitive_actions: List[str] = Field(default_factory=list)
    session_duration_seconds: float = Field(default=0.0)

    label: Optional[int] = Field(default=None, description="0=benign, 1=account_takeover")
    label_available_at: Optional[datetime] = Field(
        default=None, description="UTC timestamp when delayed label became mature"
    )

    @field_validator("event_id", "user_id", "session_id")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        if not _IDENTIFIER_RE.match(value):
            raise ValueError(f"Invalid identifier format: {value!r}")
        return value

    @field_validator("device_id")
    @classmethod
    def _validate_device_id(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not _IDENTIFIER_RE.match(value):
            raise ValueError(f"Invalid device_id format: {value!r}")
        return value

    @field_validator("timestamp", "label_available_at")
    @classmethod
    def _validate_utc_timestamp(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timestamp must be timezone-aware (ISO-8601 UTC)")
        return value.astimezone(timezone.utc)

    @field_validator("device_trust_score", "ip_reputation_risk")
    @classmethod
    def _validate_unit_interval(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError(f"Score must be in [0.0, 1.0], got {value}")
        return value

    @field_validator("ip_address")
    @classmethod
    def _validate_ip_address(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            try:
                ipaddress.ip_address(value)
            except ValueError as exc:
                raise ValueError(f"Invalid IP address: {value!r}") from exc
        return value

    @field_validator("geo_lat")
    @classmethod
    def _validate_lat(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not (-90.0 <= value <= 90.0):
            raise ValueError(f"geo_lat must be in [-90, 90], got {value}")
        return value

    @field_validator("geo_lon")
    @classmethod
    def _validate_lon(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not (-180.0 <= value <= 180.0):
            raise ValueError(f"geo_lon must be in [-180, 180], got {value}")
        return value

    @field_validator("country_code")
    @classmethod
    def _validate_country_code(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not _COUNTRY_RE.match(value):
            raise ValueError(f"country_code must be ISO-2 uppercase, got {value!r}")
        return value

    @field_validator("session_duration_seconds")
    @classmethod
    def _validate_duration(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("session_duration_seconds must be >= 0.0")
        return value

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {value}")
        return value

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> "LoginSessionEvent":
        if (self.geo_lat is None) != (self.geo_lon is None):
            raise ValueError("geo_lat and geo_lon must both be provided or both be null")
        if self.label_available_at is not None and self.label_available_at < self.timestamp:
            raise ValueError("label_available_at cannot precede event timestamp")
        if self.label is None and self.label_available_at is not None:
            raise ValueError("label_available_at cannot be set when label is null")
        return self


def parse_and_validate_event(payload: Dict[str, Any] | LoginSessionEvent) -> LoginSessionEvent:
    """Validate an event payload and distinguish unsupported values from contract violations."""
    if isinstance(payload, LoginSessionEvent):
        event = payload
    else:
        if not isinstance(payload, dict):
            raise ContractValidationError("Event payload must be a JSON object / dict")
        # Check unsupported enum/version fields first so callers get explicit UnsupportedInputError
        schema_ver = payload.get("schema_version", SCHEMA_VERSION)
        if isinstance(schema_ver, str) and schema_ver not in SUPPORTED_SCHEMA_VERSIONS:
            raise UnsupportedInputError(
                f"Unsupported schema_version {schema_ver!r}; supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        event_type = payload.get("event_type", "login_attempt")
        if isinstance(event_type, str) and event_type not in SUPPORTED_EVENT_TYPES:
            raise UnsupportedInputError(
                f"Unsupported event_type {event_type!r}; supported: {sorted(SUPPORTED_EVENT_TYPES)}"
            )
        auth_outcome = payload.get("auth_outcome", "success")
        if isinstance(auth_outcome, str) and auth_outcome not in SUPPORTED_AUTH_OUTCOMES:
            raise UnsupportedInputError(
                f"Unsupported auth_outcome {auth_outcome!r}; supported: {sorted(SUPPORTED_AUTH_OUTCOMES)}"
            )
        auth_method = payload.get("auth_method", "password")
        if isinstance(auth_method, str) and auth_method not in SUPPORTED_AUTH_METHODS:
            raise UnsupportedInputError(
                f"Unsupported auth_method {auth_method!r}; supported: {sorted(SUPPORTED_AUTH_METHODS)}"
            )
        account_tier = payload.get("account_tier", "standard")
        if isinstance(account_tier, str) and account_tier not in SUPPORTED_ACCOUNT_TIERS:
            raise UnsupportedInputError(
                f"Unsupported account_tier {account_tier!r}; supported: {sorted(SUPPORTED_ACCOUNT_TIERS)}"
            )
        sensitive = payload.get("sensitive_actions", [])
        if isinstance(sensitive, list):
            unknown_actions = [
                a for a in sensitive if isinstance(a, str) and a not in SUPPORTED_SENSITIVE_ACTIONS
            ]
            if unknown_actions:
                raise UnsupportedInputError(
                    f"Unsupported sensitive_actions {unknown_actions}; supported: {sorted(SUPPORTED_SENSITIVE_ACTIONS)}"
                )
        try:
            event = LoginSessionEvent.model_validate(payload)
        except Exception as exc:
            raise ContractValidationError(f"Event contract validation failed: {exc}") from exc

    # Re-verify supported enums when a LoginSessionEvent instance was constructed directly
    if event.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedInputError(f"Unsupported schema_version {event.schema_version!r}")
    if event.event_type not in SUPPORTED_EVENT_TYPES:
        raise UnsupportedInputError(f"Unsupported event_type {event.event_type!r}")
    if event.auth_outcome not in SUPPORTED_AUTH_OUTCOMES:
        raise UnsupportedInputError(f"Unsupported auth_outcome {event.auth_outcome!r}")
    if event.auth_method not in SUPPORTED_AUTH_METHODS:
        raise UnsupportedInputError(f"Unsupported auth_method {event.auth_method!r}")
    if event.account_tier not in SUPPORTED_ACCOUNT_TIERS:
        raise UnsupportedInputError(f"Unsupported account_tier {event.account_tier!r}")
    unknown_actions = [a for a in event.sensitive_actions if a not in SUPPORTED_SENSITIVE_ACTIONS]
    if unknown_actions:
        raise UnsupportedInputError(f"Unsupported sensitive_actions {unknown_actions}")

    return event


class FeatureVector(BaseModel):
    """Versioned strictly causal feature vector extracted for an event."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    user_id: str
    timestamp: datetime
    feature_version: str = FEATURE_VERSION
    values: Dict[str, float]
    missing_fields: List[str] = Field(default_factory=list)
    profile_expired: bool = False
    history_events_considered: int = 0


class ModelScore(BaseModel):
    """Output of the anomaly + supervised + calibration scoring layer."""

    model_config = ConfigDict(frozen=True)

    calibrated_risk_score: float = Field(..., ge=0.0, le=1.0)
    supervised_probability: float = Field(..., ge=0.0, le=1.0)
    logistic_probability: float = Field(..., ge=0.0, le=1.0)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class EvidenceItem(BaseModel):
    """Single ranked risk or mitigating factor for analyst review."""

    model_config = ConfigDict(frozen=True)

    factor: str
    direction: str  # "risk_increasing" | "risk_mitigating" | "context"
    contribution: float
    observed_value: float
    explanation: str


class AnalystEvidence(BaseModel):
    """Structured explanation bundle surfaced to fraud / security analysts."""

    model_config = ConfigDict(frozen=True)

    summary: str
    top_factors: List[EvidenceItem]
    triggered_rules: List[str]
    missing_fields: List[str]
    profile_expired: bool
    recommended_playbook: str


class DriftAlert(BaseModel):
    """Alert generated when live score or feature distributions drift from baseline."""

    model_config = ConfigDict(frozen=True)

    metric: str
    severity: str  # "warning" | "critical"
    observed_value: float
    threshold: float
    message: str


class DriftReport(BaseModel):
    """Summary of drift diagnostics for a batch/window compared to reference baseline."""

    model_config = ConfigDict(frozen=True)

    sample_size: int
    score_psi: float
    feature_psi: Dict[str, float]
    missingness_rate: float
    baseline_missingness_rate: float
    expiry_rate: float
    alerts: List[DriftAlert]
    critical_drift_detected: bool


class PolicyDecision(BaseModel):
    """Canonical decision returned by the ATO detection engine and persisted to audit logs."""

    model_config = ConfigDict(frozen=True)

    decision_id: str
    event_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    schema_version: str = SCHEMA_VERSION
    feature_version: str = FEATURE_VERSION
    model_version: str
    policy_version: str = POLICY_VERSION

    action: StepUpAction
    calibrated_risk_score: float = Field(..., ge=0.0, le=1.0)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    supervised_probability: float = Field(..., ge=0.0, le=1.0)

    thresholds_applied: Dict[str, float]
    threshold_adjustments: Dict[str, float]

    fallback_used: bool = False
    fallback_reason: FallbackReason = FallbackReason.NONE
    drift_alert_active: bool = False

    evidence: AnalystEvidence
    features_snapshot: Dict[str, float]


def make_decision_id(
    event_id: str,
    timestamp: datetime,
    model_version: str,
    feature_version: str,
    policy_version: str,
) -> str:
    """Create a deterministic, stable decision identifier from versioned inputs."""
    ts_iso = timestamp.astimezone(timezone.utc).isoformat()
    material = f"{event_id}|{ts_iso}|{model_version}|{feature_version}|{policy_version}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"dec_{digest}"
