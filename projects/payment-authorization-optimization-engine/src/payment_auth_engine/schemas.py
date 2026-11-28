"""Canonical input/output data contracts and validation helpers."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from payment_auth_engine import FEATURE_SCHEMA_VERSION, SCHEMA_VERSION

SUPPORTED_CURRENCIES = frozenset({"USD", "EUR", "GBP", "CAD", "AUD", "JPY"})
MCC_PATTERN = re.compile(r"^[0-9]{4}$")
COUNTRY_PATTERN = re.compile(r"^[A-Z]{2}$")
MAX_AMOUNT_CENTS = 100_000_000  # $1,000,000.00 ceiling for synthetic research scope


class ContractValidationError(Exception):
    """Typed contract validation exception with explicit machine-readable error code."""

    def __init__(self, error_code: str, message: str, details: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__(f"[{error_code}] {message}")
        self.error_code = error_code
        self.message = message
        self.details = details or []


class PolicyAction(str, Enum):
    """All possible authorization engine decision actions."""

    APPROVE_PRIMARY = "APPROVE_PRIMARY"
    ROUTE_SECONDARY = "ROUTE_SECONDARY"
    STEP_UP_3DS = "STEP_UP_3DS"
    DECLINE_POLICY = "DECLINE_POLICY"
    BLOCK_HARD_DECLINE = "BLOCK_HARD_DECLINE"
    HOLD_RECONCILE = "HOLD_RECONCILE"
    DEDUPLICATE_RETURN_PRIOR = "DEDUPLICATE_RETURN_PRIOR"
    BLOCK_MAX_RETRIES = "BLOCK_MAX_RETRIES"
    BLOCK_PROCESSOR_OUTAGE = "BLOCK_PROCESSOR_OUTAGE"


EVALUABLE_ACTIONS = (
    PolicyAction.APPROVE_PRIMARY,
    PolicyAction.ROUTE_SECONDARY,
    PolicyAction.STEP_UP_3DS,
    PolicyAction.DECLINE_POLICY,
)


class ProcessorHealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    OUTAGE = "OUTAGE"


class ProcessorHealthState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processor_id: str
    status: ProcessorHealthStatus = ProcessorHealthStatus.HEALTHY
    latency_multiplier: float = Field(default=1.0, ge=1.0, le=20.0)
    approval_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = "nominal"


class AuthorizationAttempt(BaseModel):
    """Canonical authorization attempt input contract (v1.0.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION)
    attempt_id: str = Field(..., min_length=1, max_length=128)
    transaction_id: str = Field(..., min_length=1, max_length=128)
    idempotency_key: str = Field(..., min_length=1, max_length=128)
    timestamp_utc: datetime
    amount_cents: int = Field(..., ge=1, le=MAX_AMOUNT_CENTS)
    currency: str
    merchant_id: str = Field(..., min_length=1, max_length=64)
    merchant_category_code: str
    card_network: Literal["visa", "mastercard", "amex", "discover"]
    issuer_country: str
    merchant_country: str
    entry_mode: Literal["chip", "contactless", "ecommerce", "recurring", "moto"]
    is_recurring: bool = False
    has_3ds: bool = False
    has_network_token: bool = False
    attempt_number: int = Field(default=1, ge=1, le=5)
    prior_response_code: Optional[str] = None
    prior_outcome_known: bool = True
    risk_score: Optional[float] = None
    velocity_1h: Optional[int] = Field(default=None, ge=0, le=10_000)
    velocity_24h: Optional[int] = Field(default=None, ge=0, le=100_000)
    account_age_days: Optional[int] = Field(default=None, ge=0, le=36_500)
    bin_approval_rate_30d: Optional[float] = None
    processor_id: str = Field(default="proc_a", min_length=1, max_length=64)

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(f"UNSUPPORTED_SCHEMA_VERSION: expected {SCHEMA_VERSION}, got {value}")
        return value

    @field_validator("timestamp_utc")
    @classmethod
    def _check_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("MISSING_TIMEZONE: timestamp_utc must be timezone-aware (UTC ISO-8601)")
        return value.astimezone(timezone.utc)

    @field_validator("currency")
    @classmethod
    def _check_currency(cls, value: str) -> str:
        upper = value.strip().upper()
        if upper not in SUPPORTED_CURRENCIES:
            raise ValueError(f"UNSUPPORTED_CURRENCY: {value} not in {sorted(SUPPORTED_CURRENCIES)}")
        return upper

    @field_validator("merchant_category_code")
    @classmethod
    def _check_mcc(cls, value: str) -> str:
        if not MCC_PATTERN.match(value):
            raise ValueError(f"INVALID_MCC: expected 4-digit MCC, got {value!r}")
        return value

    @field_validator("issuer_country", "merchant_country")
    @classmethod
    def _check_country(cls, value: str) -> str:
        upper = value.strip().upper()
        if not COUNTRY_PATTERN.match(upper):
            raise ValueError(f"INVALID_COUNTRY_CODE: expected ISO-3166 alpha-2, got {value!r}")
        return upper

    @field_validator("risk_score")
    @classmethod
    def _check_risk_score(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"INVALID_RISK_SCORE: must be finite in [0.0, 1.0], got {value}")
        return float(value)

    @field_validator("bin_approval_rate_30d")
    @classmethod
    def _check_bin_rate(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"INVALID_BIN_APPROVAL_RATE: must be finite in [0.0, 1.0], got {value}")
        return float(value)

    @model_validator(mode="after")
    def _check_retry_consistency(self) -> "AuthorizationAttempt":
        if self.attempt_number > 1 and self.prior_outcome_known and not self.prior_response_code:
            raise ValueError(
                "MISSING_PRIOR_RESPONSE_CODE: attempt_number > 1 with prior_outcome_known=True "
                "requires prior_response_code"
            )
        if self.attempt_number == 1 and self.prior_response_code is not None:
            raise ValueError(
                "UNEXPECTED_PRIOR_RESPONSE_CODE: attempt_number == 1 must not specify prior_response_code"
            )
        return self


class ProcessorResponse(BaseModel):
    """Canonical processor response record."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str = Field(..., min_length=1)
    processor_id: str = Field(..., min_length=1)
    response_code: str = Field(..., min_length=1)
    latency_ms: float = Field(..., ge=0.0, le=120_000.0)
    outcome_certain: bool = True
    timestamp_utc: datetime

    @field_validator("timestamp_utc")
    @classmethod
    def _check_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("MISSING_TIMEZONE: timestamp_utc must be timezone-aware")
        return value.astimezone(timezone.utc)


class HistoricalOutcome(BaseModel):
    """Historical logged authorization outcome with maturity and propensity metadata."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str = Field(..., min_length=1)
    transaction_id: str = Field(..., min_length=1)
    attempt_timestamp_utc: datetime
    logged_action: PolicyAction
    propensity_score: float = Field(..., gt=0.0, le=1.0)
    authorized: bool
    response_code: str
    latency_ms: float = Field(..., ge=0.0)
    fraud_confirmed: Optional[bool] = None
    loss_cents: int = Field(default=0, ge=0)
    label_maturity_days: int = Field(..., ge=0)
    matured_at_utc: Optional[datetime] = None


class RetryEligibilityResult(BaseModel):
    """Deterministic retry eligibility evaluation output."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    transaction_id: str
    eligible: bool
    recommended_action: PolicyAction
    response_category: Optional[str] = None
    reason_code: str
    explanation: str
    requires_reconciliation: bool = False
    is_hard_stop: bool = False


class ConstraintCheckResult(BaseModel):
    """Per-constraint audit result attached to every authorization decision."""

    model_config = ConfigDict(extra="forbid")

    constraint_name: str
    limit_value: float
    observed_value: float
    passed: bool
    message: str


class DecisionExplanation(BaseModel):
    """Structured explanation for model- and rule-driven decisions."""

    model_config = ConfigDict(extra="forbid")

    reason_codes: List[str]
    missing_features: List[str]
    top_contributing_factors: Dict[str, float]
    policy_notes: str


class DecisionRecord(BaseModel):
    """Canonical authorization decision output contract (v1.0.0)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION)
    decision_id: str
    attempt_id: str
    transaction_id: str
    idempotency_key: str
    policy_name: str
    model_version: str
    feature_schema_version: str = Field(default=FEATURE_SCHEMA_VERSION)
    config_hash: str
    action: PolicyAction
    processor_selected: Optional[str] = None
    predicted_acceptance_prob: float = Field(..., ge=0.0, le=1.0)
    predicted_loss_prob: float = Field(..., ge=0.0, le=1.0)
    expected_loss_cents: float = Field(..., ge=0.0)
    expected_net_value_cents: float
    estimated_latency_ms: float = Field(..., ge=0.0)
    constraint_checks: List[ConstraintCheckResult]
    any_constraint_breach: bool
    explanation: DecisionExplanation


def validate_attempt(payload: Dict[str, Any]) -> AuthorizationAttempt:
    """Validate raw dictionary payload into an AuthorizationAttempt or raise ContractValidationError."""
    try:
        return AuthorizationAttempt.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors()
        error_code = "CONTRACT_VALIDATION_ERROR"
        for err in errors:
            msg = str(err.get("msg", ""))
            for known in (
                "UNSUPPORTED_SCHEMA_VERSION",
                "MISSING_TIMEZONE",
                "UNSUPPORTED_CURRENCY",
                "INVALID_MCC",
                "INVALID_COUNTRY_CODE",
                "INVALID_RISK_SCORE",
                "INVALID_BIN_APPROVAL_RATE",
                "MISSING_PRIOR_RESPONSE_CODE",
                "UNEXPECTED_PRIOR_RESPONSE_CODE",
            ):
                if known in msg:
                    error_code = known
                    break
            if error_code != "CONTRACT_VALIDATION_ERROR":
                break
        raise ContractValidationError(
            error_code=error_code,
            message=f"AuthorizationAttempt validation failed ({len(errors)} error(s))",
            details=[{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")} for e in errors],
        ) from exc
