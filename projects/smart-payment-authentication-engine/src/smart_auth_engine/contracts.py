"""Canonical data, risk, action, and audit contracts for Smart Payment Authentication Engine."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"

SUPPORTED_CURRENCIES = frozenset({"USD", "EUR", "GBP", "CAD", "AUD", "JPY"})
ZERO_DECIMAL_CURRENCIES = frozenset({"JPY"})
SUPPORTED_MERCHANT_CATEGORIES = frozenset(
    {
        "retail",
        "digital_goods",
        "travel",
        "crypto_exchange",
        "gambling",
        "groceries",
        "utilities",
        "luxury",
    }
)
MAX_AMOUNT_MINOR = 100_000_000  # 1,000,000.00 in 2-decimal currencies


class AuthAction(str, Enum):
    """Canonical authentication action contract."""

    APPROVE = "approve"
    CHALLENGE = "challenge"
    DECLINE = "decline"
    ABSTAIN = "abstain"


class ChallengeMethod(str, Enum):
    """Simulated step-up authentication challenge methods."""

    NONE = "none"
    OTP_SMS = "otp_sms"
    BIOMETRIC_APP = "biometric_app"
    HARDWARE_TOKEN = "hardware_token"


class ChallengeStatus(str, Enum):
    """Simulated challenge completion status."""

    NOT_REQUIRED = "not_required"
    PASSED = "passed"
    FAILED = "failed"
    ABANDONED = "abandoned"
    TIMED_OUT = "timed_out"


class FinalAuthOutcome(str, Enum):
    """Final post-step-up authentication outcome."""

    AUTHORIZED = "authorized"
    DECLINED = "declined"
    ABANDONED = "abandoned"
    HELD_FOR_REVIEW = "held_for_review"


def _ensure_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be a timezone-aware ISO-8601 timestamp (UTC required).")
    return value.astimezone(timezone.utc)


def _validate_iso2(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 2 or not value.isalpha() or not value.isupper():
        raise ValueError(f"{field_name} must be a 2-letter uppercase ISO country code, got {value!r}.")
    return value


class DeviceSignals(BaseModel):
    """Device and network telemetry attached to a payment authentication request."""

    model_config = ConfigDict(extra="forbid")

    device_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    device_age_days: Optional[float] = Field(default=None, ge=0.0, le=36500.0)
    is_emulator: Optional[bool] = Field(default=None)
    is_tor_or_vpn: Optional[bool] = Field(default=None)
    ip_country: Optional[str] = Field(default=None)
    ip_velocity_1h: Optional[int] = Field(default=None, ge=0, le=100_000)
    ip_velocity_24h: Optional[int] = Field(default=None, ge=0, le=1_000_000)
    device_trust_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    biometric_enrolled: Optional[bool] = Field(default=None)
    browser_fingerprint_mismatch: Optional[bool] = Field(default=None)

    @field_validator("ip_country")
    @classmethod
    def check_ip_country(cls, v: Optional[str]) -> Optional[str]:
        return _validate_iso2(v, "ip_country")

    @model_validator(mode="after")
    def check_velocity_consistency(self) -> "DeviceSignals":
        if (
            self.ip_velocity_1h is not None
            and self.ip_velocity_24h is not None
            and self.ip_velocity_24h < self.ip_velocity_1h
        ):
            raise ValueError("ip_velocity_24h cannot be less than ip_velocity_1h.")
        return self


class PaymentEvent(BaseModel):
    """Canonical synthetic payment authentication event."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION)
    payment_id: str = Field(..., min_length=1, max_length=128)
    account_id: str = Field(..., min_length=1, max_length=128)
    merchant_id: str = Field(..., min_length=1, max_length=128)
    merchant_category: str = Field(...)
    amount_minor: int = Field(..., gt=0, le=MAX_AMOUNT_MINOR)
    currency: str = Field(...)
    timestamp: datetime = Field(...)
    billing_country: str = Field(...)
    card_bin_country: str = Field(...)
    is_recurring: bool = Field(default=False)
    is_cross_border: Optional[bool] = Field(default=None)
    account_age_days: Optional[float] = Field(default=None, ge=0.0, le=36500.0)
    prior_chargebacks_90d: Optional[int] = Field(default=None, ge=0, le=1000)
    tx_count_24h: Optional[int] = Field(default=None, ge=0, le=100_000)
    tx_amount_24h_minor: Optional[int] = Field(default=None, ge=0, le=MAX_AMOUNT_MINOR * 100)
    device: DeviceSignals = Field(default_factory=DeviceSignals)

    @field_validator("schema_version")
    @classmethod
    def check_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version {v!r}; expected {SCHEMA_VERSION!r}.")
        return v

    @field_validator("currency")
    @classmethod
    def check_currency(cls, v: str) -> str:
        if v not in SUPPORTED_CURRENCIES:
            raise ValueError(
                f"Unsupported currency {v!r}; supported currencies: {sorted(SUPPORTED_CURRENCIES)}."
            )
        return v

    @field_validator("merchant_category")
    @classmethod
    def check_merchant_category(cls, v: str) -> str:
        if v not in SUPPORTED_MERCHANT_CATEGORIES:
            raise ValueError(
                f"Unsupported merchant_category {v!r}; supported: {sorted(SUPPORTED_MERCHANT_CATEGORIES)}."
            )
        return v

    @field_validator("timestamp")
    @classmethod
    def check_timestamp(cls, v: datetime) -> datetime:
        return _ensure_utc(v, "timestamp")

    @field_validator("billing_country", "card_bin_country")
    @classmethod
    def check_countries(cls, v: str, info: Any) -> str:
        checked = _validate_iso2(v, info.field_name)
        assert checked is not None
        return checked

    @model_validator(mode="after")
    def derive_cross_border(self) -> "PaymentEvent":
        if self.is_cross_border is None:
            ip_mismatch = (
                self.device.ip_country is not None
                and self.device.ip_country != self.billing_country
            )
            bin_mismatch = self.card_bin_country != self.billing_country
            self.is_cross_border = bool(ip_mismatch or bin_mismatch)
        return self

    @property
    def amount_major(self) -> float:
        """Convert minor units to major currency units."""
        if self.currency in ZERO_DECIMAL_CURRENCIES:
            return float(self.amount_minor)
        return self.amount_minor / 100.0


class RiskLabel(BaseModel):
    """Delayed fraud/non-fraud ground truth label with explicit label maturity timestamp."""

    model_config = ConfigDict(extra="forbid")

    payment_id: str = Field(..., min_length=1, max_length=128)
    event_timestamp: datetime = Field(...)
    is_fraud: bool = Field(...)
    label_available_at: datetime = Field(...)
    label_source: str = Field(...)
    fraud_loss_minor: int = Field(..., ge=0)

    @field_validator("event_timestamp", "label_available_at")
    @classmethod
    def check_timestamps(cls, v: datetime, info: Any) -> datetime:
        return _ensure_utc(v, info.field_name)

    @field_validator("label_source")
    @classmethod
    def check_label_source(cls, v: str) -> str:
        allowed = {"chargeback", "issuer_confirmed_fraud", "matured_non_fraud"}
        if v not in allowed:
            raise ValueError(f"Unsupported label_source {v!r}; allowed: {sorted(allowed)}.")
        return v

    @model_validator(mode="after")
    def check_label_causality(self) -> "RiskLabel":
        if self.label_available_at < self.event_timestamp:
            raise ValueError("label_available_at cannot precede event_timestamp.")
        if not self.is_fraud and self.fraud_loss_minor != 0:
            raise ValueError("Non-fraud labels must have fraud_loss_minor == 0.")
        return self

    def is_mature_at(self, as_of: datetime) -> bool:
        """Return True iff this label was available at or before `as_of`."""
        as_of_utc = _ensure_utc(as_of, "as_of")
        return self.label_available_at <= as_of_utc


class ChallengeOutcome(BaseModel):
    """Outcome of a simulated step-up challenge."""

    model_config = ConfigDict(extra="forbid")

    payment_id: str = Field(..., min_length=1, max_length=128)
    decision_id: str = Field(..., min_length=1, max_length=128)
    challenge_method: ChallengeMethod = Field(...)
    status: ChallengeStatus = Field(...)
    latency_ms: int = Field(..., ge=0, le=600_000)
    completed_at: datetime = Field(...)

    @field_validator("completed_at")
    @classmethod
    def check_completed_at(cls, v: datetime) -> datetime:
        return _ensure_utc(v, "completed_at")


class FeatureContribution(BaseModel):
    """Per-feature explanation item recorded alongside a decision."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    value: Optional[float] = None
    contribution: float


class PolicyDecision(BaseModel):
    """Output contract for authentication policy decisions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION)
    decision_id: str = Field(..., min_length=8, max_length=64)
    payment_id: str = Field(..., min_length=1, max_length=128)
    timestamp: datetime = Field(...)
    action: AuthAction = Field(...)
    calibrated_fraud_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    raw_fraud_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    challenge_method: ChallengeMethod = Field(default=ChallengeMethod.NONE)
    mandatory_rule_triggered: bool = Field(default=False)
    override_triggered: bool = Field(default=False)
    fallback_used: bool = Field(default=False)
    reason_codes: List[str] = Field(default_factory=list)
    explanations: List[FeatureContribution] = Field(default_factory=list)
    missing_features: List[str] = Field(default_factory=list)
    missing_feature_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    model_version: str = Field(...)
    feature_version: str = Field(...)
    policy_version: str = Field(...)

    @field_validator("timestamp")
    @classmethod
    def check_timestamp(cls, v: datetime) -> datetime:
        return _ensure_utc(v, "timestamp")


class StepUpExecutionResult(BaseModel):
    """Combined decision and simulated step-up execution record."""

    model_config = ConfigDict(extra="forbid")

    decision: PolicyDecision
    challenge_outcome: Optional[ChallengeOutcome] = None
    final_outcome: FinalAuthOutcome
    expected_fraud_loss_minor: float = Field(..., ge=0.0)
    friction_cost_minor: float = Field(..., ge=0.0)
    audit_timestamp: datetime

    @field_validator("audit_timestamp")
    @classmethod
    def check_audit_timestamp(cls, v: datetime) -> datetime:
        return _ensure_utc(v, "audit_timestamp")


class ModelArtifactManifest(BaseModel):
    """Versioned model artifact metadata for reproducibility and auditing."""

    model_config = ConfigDict(extra="forbid")

    model_version: str
    feature_version: str
    feature_names: List[str]
    calibration_method: str
    trained_as_of: datetime
    label_maturity_days: int
    n_train_total: int
    n_train_mature: int
    n_train_excluded_immature: int
    n_calibration_mature: int
    seed: int
    artifact_sha256: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Structured error payload for API and CLI validation failures."""

    model_config = ConfigDict(extra="forbid")

    error_code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


def build_stable_decision_id(
    payment_id: str,
    timestamp: datetime,
    model_version: str,
    feature_version: str,
    policy_version: str,
) -> str:
    """Create a deterministic, replay-stable decision identifier."""
    payload = json.dumps(
        {
            "payment_id": payment_id,
            "timestamp": _ensure_utc(timestamp, "timestamp").isoformat(),
            "model_version": model_version,
            "feature_version": feature_version,
            "policy_version": policy_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:20]
    return f"dec_{digest}"
