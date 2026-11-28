"""Canonical domain schemas, timestamp utilities, and contract validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from risk_platform.errors import (
    ContractValidationError,
    SensitiveDataViolationError,
    UnsupportedCurrencyError,
    UnsupportedSchemaVersionError,
)

SUPPORTED_SCHEMA_VERSIONS = ("tx.v1",)
SUPPORTED_CURRENCIES = ("USD", "EUR", "GBP")
DEFAULT_FX_TO_USD: Dict[str, float] = {"USD": 1.0, "EUR": 1.08, "GBP": 1.26}
ALLOWED_CHANNELS = ("card_present", "ecommerce", "in_app", "recurring")
KNOWN_MERCHANT_CATEGORIES = (
    "grocery",
    "electronics",
    "travel",
    "gaming",
    "digital_goods",
    "utilities",
    "restaurant",
    "crypto_exchange",
    "unknown",
)
MAX_AMOUNT_CENTS_DEFAULT = 100_000_000

# Matches raw 13-19 digit PANs with optional spaces or hyphens
_RAW_PAN_PATTERN = re.compile(r"^(?:\d[ -]?){13,19}$")
_COUNTRY_PATTERN = re.compile(r"^[A-Z]{2}$")


def parse_utc_timestamp_ms(value: Any, field_name: str = "event_ts") -> int:
    """Parse an ISO-8601 timezone-aware timestamp string or UTC datetime to epoch milliseconds."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ContractValidationError(f"'{field_name}' must be timezone-aware UTC")
        return int(round(value.astimezone(timezone.utc).timestamp() * 1000))

    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"'{field_name}' must be a non-empty ISO-8601 UTC string")

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ContractValidationError(f"Invalid ISO-8601 timestamp for '{field_name}': {value}") from exc

    if dt.tzinfo is None:
        raise ContractValidationError(
            f"Naive timestamp without timezone offset rejected for '{field_name}': {value}"
        )
    return int(round(dt.astimezone(timezone.utc).timestamp() * 1000))


def format_utc_ms(ts_ms: int) -> str:
    """Format epoch milliseconds into canonical ISO-8601 UTC string with 'Z' suffix."""
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    if ts_ms % 1000 == 0:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _assert_not_raw_pan(value: str, field_name: str) -> None:
    digits_only = re.sub(r"[ -]", "", value.strip())
    if _RAW_PAN_PATTERN.match(value.strip()) and 13 <= len(digits_only) <= 19:
        raise SensitiveDataViolationError(
            f"Raw card number (PAN) pattern detected in '{field_name}'. Use synthetic tokens only."
        )


def _require_non_empty_str(payload: Dict[str, Any], key: str) -> str:
    val = payload.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ContractValidationError(f"Field '{key}' must be a non-empty string")
    return val.strip()


@dataclass(frozen=True)
class TransactionEvent:
    """Canonical immutable transaction event contract (tx.v1)."""

    schema_version: str
    event_id: str
    account_id: str
    card_token: str
    merchant_id: str
    merchant_category: str
    amount_cents: int
    currency: str
    country_code: str
    channel: str
    device_id: str
    event_ts_ms: int
    ip_risk_score: float
    ingest_ts_ms: Optional[int] = None

    @property
    def event_ts(self) -> str:
        return format_utc_ms(self.event_ts_ms)

    @property
    def ingest_ts(self) -> Optional[str]:
        return format_utc_ms(self.ingest_ts_ms) if self.ingest_ts_ms is not None else None

    def amount_usd(self, fx_to_usd: Optional[Dict[str, float]] = None) -> float:
        rates = fx_to_usd or DEFAULT_FX_TO_USD
        if self.currency not in rates:
            raise UnsupportedCurrencyError(f"Unsupported currency: '{self.currency}'")
        return (self.amount_cents / 100.0) * rates[self.currency]

    def canonical_payload_dict(self) -> Dict[str, Any]:
        """Canonical representation used for idempotency hashing and serialization."""
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "account_id": self.account_id,
            "card_token": self.card_token,
            "merchant_id": self.merchant_id,
            "merchant_category": self.merchant_category,
            "amount_cents": self.amount_cents,
            "currency": self.currency,
            "country_code": self.country_code,
            "channel": self.channel,
            "device_id": self.device_id,
            "event_ts": self.event_ts,
            "ip_risk_score": round(self.ip_risk_score, 6),
        }

    def payload_hash(self) -> str:
        encoded = json.dumps(self.canonical_payload_dict(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = self.canonical_payload_dict()
        if self.ingest_ts is not None:
            d["ingest_ts"] = self.ingest_ts
        return d

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        supported_versions: Sequence[str] = SUPPORTED_SCHEMA_VERSIONS,
        supported_currencies: Sequence[str] = SUPPORTED_CURRENCIES,
        max_amount_cents: int = MAX_AMOUNT_CENTS_DEFAULT,
    ) -> "TransactionEvent":
        if not isinstance(payload, dict):
            raise ContractValidationError("Transaction payload must be a JSON object")

        schema_version = _require_non_empty_str(payload, "schema_version")
        if schema_version not in supported_versions:
            raise UnsupportedSchemaVersionError(
                f"Unsupported schema_version '{schema_version}'. Supported: {list(supported_versions)}"
            )

        event_id = _require_non_empty_str(payload, "event_id")
        account_id = _require_non_empty_str(payload, "account_id")
        card_token = _require_non_empty_str(payload, "card_token")
        merchant_id = _require_non_empty_str(payload, "merchant_id")
        merchant_category = _require_non_empty_str(payload, "merchant_category").lower()
        device_id = _require_non_empty_str(payload, "device_id")

        _assert_not_raw_pan(card_token, "card_token")
        _assert_not_raw_pan(account_id, "account_id")

        amount_raw = payload.get("amount_cents")
        if isinstance(amount_raw, bool) or not isinstance(amount_raw, int):
            raise ContractValidationError("Field 'amount_cents' must be an integer in minor currency units")
        if amount_raw <= 0 or amount_raw > max_amount_cents:
            raise ContractValidationError(
                f"Field 'amount_cents' must be in [1, {max_amount_cents}], got {amount_raw}"
            )

        currency = _require_non_empty_str(payload, "currency").upper()
        if currency not in supported_currencies:
            raise UnsupportedCurrencyError(
                f"Unsupported currency '{currency}'. Supported: {list(supported_currencies)}"
            )

        country_code = _require_non_empty_str(payload, "country_code").upper()
        if not _COUNTRY_PATTERN.match(country_code):
            raise ContractValidationError(
                f"Field 'country_code' must be ISO-3166 alpha-2 uppercase, got '{country_code}'"
            )

        channel = _require_non_empty_str(payload, "channel").lower()
        if channel not in ALLOWED_CHANNELS:
            raise ContractValidationError(
                f"Unsupported channel '{channel}'. Allowed: {list(ALLOWED_CHANNELS)}"
            )

        if "event_ts_ms" in payload and isinstance(payload["event_ts_ms"], int) and not isinstance(
            payload["event_ts_ms"], bool
        ):
            event_ts_ms = payload["event_ts_ms"]
        else:
            event_ts_ms = parse_utc_timestamp_ms(payload.get("event_ts"), "event_ts")

        ingest_ts_ms: Optional[int] = None
        if payload.get("ingest_ts_ms") is not None:
            raw_ing = payload["ingest_ts_ms"]
            if isinstance(raw_ing, bool) or not isinstance(raw_ing, int):
                raise ContractValidationError("Field 'ingest_ts_ms' must be an integer")
            ingest_ts_ms = raw_ing
        elif payload.get("ingest_ts") is not None:
            ingest_ts_ms = parse_utc_timestamp_ms(payload.get("ingest_ts"), "ingest_ts")

        ip_raw = payload.get("ip_risk_score")
        if isinstance(ip_raw, bool) or not isinstance(ip_raw, (int, float)):
            raise ContractValidationError("Field 'ip_risk_score' must be a float in [0.0, 1.0]")
        ip_risk_score = float(ip_raw)
        if not (0.0 <= ip_risk_score <= 1.0):
            raise ContractValidationError(
                f"Field 'ip_risk_score' must be in [0.0, 1.0], got {ip_risk_score}"
            )

        return cls(
            schema_version=schema_version,
            event_id=event_id,
            account_id=account_id,
            card_token=card_token,
            merchant_id=merchant_id,
            merchant_category=merchant_category,
            amount_cents=amount_raw,
            currency=currency,
            country_code=country_code,
            channel=channel,
            device_id=device_id,
            event_ts_ms=event_ts_ms,
            ip_risk_score=ip_risk_score,
            ingest_ts_ms=ingest_ts_ms,
        )


@dataclass(frozen=True)
class OutcomeLabel:
    """Matured outcome label for temporal training and evaluation."""

    event_id: str
    is_fraud: int
    label_ts_ms: int
    label_source: str

    @property
    def label_ts(self) -> str:
        return format_utc_ms(self.label_ts_ms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "is_fraud": self.is_fraud,
            "label_ts": self.label_ts,
            "label_source": self.label_source,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OutcomeLabel":
        event_id = _require_non_empty_str(payload, "event_id")
        is_fraud = payload.get("is_fraud")
        if isinstance(is_fraud, bool) or is_fraud not in (0, 1):
            raise ContractValidationError(f"Field 'is_fraud' must be 0 or 1, got {is_fraud}")
        label_ts_ms = parse_utc_timestamp_ms(payload.get("label_ts"), "label_ts")
        label_source = _require_non_empty_str(payload, "label_source")
        return cls(
            event_id=event_id,
            is_fraud=int(is_fraud),
            label_ts_ms=label_ts_ms,
            label_source=label_source,
        )


@dataclass(frozen=True)
class FeatureVector:
    """Versioned point-in-time feature vector for a transaction."""

    event_id: str
    account_id: str
    feature_version: str
    as_of_ts_ms: int
    values: Dict[str, float]
    feature_freshness_ms: Optional[int] = None
    feature_status: str = "FRESH"
    unknown_input_flags: List[str] = field(default_factory=list)

    def ordered_values(self, feature_names: Sequence[str]) -> List[float]:
        return [float(self.values[name]) for name in feature_names]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "account_id": self.account_id,
            "feature_version": self.feature_version,
            "as_of_ts": format_utc_ms(self.as_of_ts_ms),
            "feature_freshness_ms": self.feature_freshness_ms,
            "feature_status": self.feature_status,
            "unknown_input_flags": list(self.unknown_input_flags),
            "values": {k: float(v) for k, v in self.values.items()},
        }


@dataclass(frozen=True)
class IngestionResult:
    """Result of idempotent event ingestion into the streaming store."""

    event_id: str
    status: str  # ACCEPTED | DUPLICATE_IGNORED | LATE_ACCEPTED | LATE_DROPPED
    watermark_ms: int
    account_event_count: int
    payload_hash: str
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "status": self.status,
            "watermark_ts": format_utc_ms(self.watermark_ms),
            "watermark_ms": self.watermark_ms,
            "account_event_count": self.account_event_count,
            "payload_hash": self.payload_hash,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FeatureAttribution:
    """Linear/model contribution of a single feature to a decision."""

    feature: str
    value: float
    contribution: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "value": round(self.value, 6),
            "contribution": round(self.contribution, 6),
        }


@dataclass(frozen=True)
class RiskDecision:
    """Versioned, traceable risk scoring decision."""

    decision_id: str
    event_id: str
    account_id: str
    action: str  # APPROVE | REVIEW | DECLINE
    risk_score: float
    raw_score: Optional[float]
    model_version: str
    feature_version: str
    policy_version: str
    feature_freshness_ms: Optional[int]
    feature_status: str  # FRESH | COLD_START | STALE
    fallback_used: bool
    fallback_reason: Optional[str]
    reason_codes: List[str]
    top_attributions: List[FeatureAttribution]
    latency_ms: float
    decided_at_ms: int

    @property
    def decided_at(self) -> str:
        return format_utc_ms(self.decided_at_ms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "event_id": self.event_id,
            "account_id": self.account_id,
            "action": self.action,
            "risk_score": round(self.risk_score, 6),
            "raw_score": round(self.raw_score, 6) if self.raw_score is not None else None,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "policy_version": self.policy_version,
            "feature_freshness_ms": self.feature_freshness_ms,
            "feature_status": self.feature_status,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "reason_codes": list(self.reason_codes),
            "top_attributions": [a.to_dict() for a in self.top_attributions],
            "latency_ms": round(self.latency_ms, 4),
            "decided_at": self.decided_at,
        }


def compute_decision_id(
    event_id: str,
    model_version: str,
    feature_version: str,
    policy_version: str,
    as_of_ts_ms: int,
) -> str:
    """Deterministic stable decision identifier for audit traceability and replay verification."""
    raw = f"{event_id}|{model_version}|{feature_version}|{policy_version}|{as_of_ts_ms}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"dec_{digest}"
