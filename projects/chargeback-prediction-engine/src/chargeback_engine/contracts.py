"""Canonical input/output schemas, enums, validation rules, and error types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "1.0.0"

# Fixed offline synthetic FX rates to USD for deterministic multi-currency normalization.
SUPPORTED_CURRENCIES: Dict[str, float] = {
    "USD": 1.00,
    "EUR": 1.08,
    "GBP": 1.27,
    "CAD": 0.74,
}

SUPPORTED_MCC_GROUPS = {
    "digital_goods",
    "travel",
    "retail",
    "subscription",
    "hospitality",
    "services",
}


class ChargebackEngineError(Exception):
    """Base error for all chargeback prediction engine failures."""


class ConfigError(ChargebackEngineError):
    """Raised when configuration is missing, malformed, or violates policy invariants."""


class ContractValidationError(ChargebackEngineError):
    """Raised when input records violate canonical data contracts."""


class UnsupportedInputError(ChargebackEngineError):
    """Raised when inputs contain unsupported currencies, channels, or schema versions."""


class DuplicateDisputeError(ContractValidationError):
    """Raised when multiple dispute records reference the same transaction_id under strict mode."""

    def __init__(self, duplicate_transaction_ids: Sequence[str]) -> None:
        self.duplicate_transaction_ids = tuple(sorted(set(duplicate_transaction_ids)))
        super().__init__(
            "Duplicate dispute records detected for transaction_id(s): "
            + ", ".join(self.duplicate_transaction_ids)
        )


class TemporalLeakageError(ChargebackEngineError):
    """Raised when split boundaries or feature timestamps violate strict causality."""


class PaymentChannel(str, Enum):
    CARD_NOT_PRESENT = "card_not_present"
    CARD_PRESENT = "card_present"
    RECURRING = "recurring"
    DIGITAL_WALLET = "digital_wallet"


class ReasonCategory(str, Enum):
    FRAUD = "fraud"
    CONSUMER_DISPUTE = "consumer_dispute"
    AUTHORIZATION = "authorization"
    PROCESSING_ERROR = "processing_error"
    UNKNOWN = "unknown"


class LabelStatus(str, Enum):
    MATURE_POSITIVE = "mature_positive"
    MATURE_NEGATIVE = "mature_negative"
    IMMATURE_POSITIVE = "immature_positive"
    IMMATURE_NEGATIVE = "immature_negative"


# Synthetic network-style reason code taxonomy mapped to canonical categories.
REASON_CODE_TO_CATEGORY: Dict[str, ReasonCategory] = {
    "10.4": ReasonCategory.FRAUD,
    "10.5": ReasonCategory.FRAUD,
    "4837": ReasonCategory.FRAUD,
    "13.1": ReasonCategory.CONSUMER_DISPUTE,
    "13.3": ReasonCategory.CONSUMER_DISPUTE,
    "4853": ReasonCategory.CONSUMER_DISPUTE,
    "11.1": ReasonCategory.AUTHORIZATION,
    "4808": ReasonCategory.AUTHORIZATION,
    "12.5": ReasonCategory.PROCESSING_ERROR,
    "4834": ReasonCategory.PROCESSING_ERROR,
}


def parse_utc_timestamp(value: Any, field_name: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp string or datetime into a timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ContractValidationError(
                f"Invalid ISO-8601 timestamp for '{field_name}': {value!r}"
            ) from exc
    else:
        raise ContractValidationError(
            f"Missing or non-string timestamp for '{field_name}': {value!r}"
        )

    if dt.tzinfo is None:
        raise ContractValidationError(
            f"Timestamp for '{field_name}' must include explicit UTC timezone: {value!r}"
        )
    return dt.astimezone(timezone.utc)


def normalize_reason_code(
    reason_code: Optional[str], missing_label: str = "UNKNOWN"
) -> Tuple[str, ReasonCategory, bool, bool]:
    """Normalize a dispute reason code.

    Returns:
        (normalized_code, category, is_missing, is_unrecognized)
    """
    if reason_code is None:
        return missing_label, ReasonCategory.UNKNOWN, True, False
    cleaned = str(reason_code).strip()
    if not cleaned or cleaned.upper() in {"NONE", "NULL", "N/A", "UNKNOWN"}:
        return missing_label, ReasonCategory.UNKNOWN, True, False
    category = REASON_CODE_TO_CATEGORY.get(cleaned)
    if category is None:
        return cleaned, ReasonCategory.UNKNOWN, False, True
    return cleaned, category, False, False


@dataclass(frozen=True)
class MerchantRecord:
    merchant_id: str
    mcc_group: str
    onboarding_ts: datetime
    country: str
    avg_fulfillment_days: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MerchantRecord":
        if not isinstance(data, dict):
            raise ContractValidationError("Merchant record must be a mapping")
        merchant_id = str(data.get("merchant_id", "")).strip()
        if not merchant_id:
            raise ContractValidationError("merchant_id must be non-empty")
        mcc_group = str(data.get("mcc_group", "")).strip().lower()
        if mcc_group not in SUPPORTED_MCC_GROUPS:
            raise UnsupportedInputError(
                f"Unsupported mcc_group {mcc_group!r} for merchant {merchant_id!r}"
            )
        onboarding_ts = parse_utc_timestamp(data.get("onboarding_ts"), "onboarding_ts")
        country = str(data.get("country", "")).strip().upper()
        if len(country) != 2 or not country.isalpha():
            raise ContractValidationError(
                f"country must be 2-letter ISO code, got {country!r}"
            )
        try:
            avg_fulfillment_days = float(data["avg_fulfillment_days"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractValidationError(
                f"Invalid avg_fulfillment_days for merchant {merchant_id!r}"
            ) from exc
        if not math.isfinite(avg_fulfillment_days) or avg_fulfillment_days < 0:
            raise ContractValidationError(
                f"avg_fulfillment_days must be finite and >= 0, got {avg_fulfillment_days}"
            )
        return cls(
            merchant_id=merchant_id,
            mcc_group=mcc_group,
            onboarding_ts=onboarding_ts,
            country=country,
            avg_fulfillment_days=avg_fulfillment_days,
        )


@dataclass(frozen=True)
class TransactionRecord:
    transaction_id: str
    merchant_id: str
    transaction_ts: datetime
    settlement_ts: datetime
    amount: float
    currency: str
    amount_usd: float
    channel: PaymentChannel
    is_cross_border: bool
    has_3ds: bool
    has_avs_match: bool
    customer_tenure_days: int
    explicit_maturity_date: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransactionRecord":
        if not isinstance(data, dict):
            raise ContractValidationError("Transaction record must be a mapping")
        transaction_id = str(data.get("transaction_id", "")).strip()
        merchant_id = str(data.get("merchant_id", "")).strip()
        if not transaction_id or not merchant_id:
            raise ContractValidationError(
                "transaction_id and merchant_id must be non-empty"
            )
        transaction_ts = parse_utc_timestamp(
            data.get("transaction_ts"), "transaction_ts"
        )
        settlement_raw = data.get("settlement_ts") or data.get("transaction_ts")
        settlement_ts = parse_utc_timestamp(settlement_raw, "settlement_ts")
        if settlement_ts < transaction_ts:
            raise ContractValidationError(
                f"settlement_ts ({settlement_ts.isoformat()}) precedes transaction_ts "
                f"({transaction_ts.isoformat()}) for {transaction_id}"
            )

        try:
            amount = float(data["amount"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractValidationError(
                f"Missing or non-numeric amount for {transaction_id}"
            ) from exc
        if not math.isfinite(amount) or amount <= 0.0:
            raise ContractValidationError(
                f"Transaction amount must be finite and > 0, got {amount} for {transaction_id}"
            )

        currency = str(data.get("currency", "")).strip().upper()
        if currency not in SUPPORTED_CURRENCIES:
            raise UnsupportedInputError(
                f"Unsupported currency {currency!r} for transaction {transaction_id}. "
                f"Supported: {sorted(SUPPORTED_CURRENCIES)}"
            )
        amount_usd = round(amount * SUPPORTED_CURRENCIES[currency], 4)

        channel_raw = str(data.get("channel", "")).strip().lower()
        try:
            channel = PaymentChannel(channel_raw)
        except ValueError as exc:
            raise UnsupportedInputError(
                f"Unsupported payment channel {channel_raw!r} for {transaction_id}"
            ) from exc

        try:
            customer_tenure_days = int(data.get("customer_tenure_days", 0))
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(
                f"customer_tenure_days must be an integer for {transaction_id}"
            ) from exc
        if customer_tenure_days < 0:
            raise ContractValidationError(
                f"customer_tenure_days must be >= 0, got {customer_tenure_days} for {transaction_id}"
            )

        explicit_maturity_date: Optional[datetime] = None
        if data.get("maturity_date") is not None:
            explicit_maturity_date = parse_utc_timestamp(
                data["maturity_date"], "maturity_date"
            )
            if explicit_maturity_date <= settlement_ts:
                raise ContractValidationError(
                    f"maturity_date ({explicit_maturity_date.isoformat()}) must be after "
                    f"settlement_ts ({settlement_ts.isoformat()}) for {transaction_id}"
                )

        return cls(
            transaction_id=transaction_id,
            merchant_id=merchant_id,
            transaction_ts=transaction_ts,
            settlement_ts=settlement_ts,
            amount=amount,
            currency=currency,
            amount_usd=amount_usd,
            channel=channel,
            is_cross_border=bool(data.get("is_cross_border", False)),
            has_3ds=bool(data.get("has_3ds", False)),
            has_avs_match=bool(data.get("has_avs_match", True)),
            customer_tenure_days=customer_tenure_days,
            explicit_maturity_date=explicit_maturity_date,
        )


@dataclass(frozen=True)
class DisputeRecord:
    dispute_id: str
    transaction_id: str
    dispute_ts: datetime
    reason_code: Optional[str]
    dispute_amount: float
    currency: str
    dispute_amount_usd: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisputeRecord":
        if not isinstance(data, dict):
            raise ContractValidationError("Dispute record must be a mapping")
        dispute_id = str(data.get("dispute_id", "")).strip()
        transaction_id = str(data.get("transaction_id", "")).strip()
        if not dispute_id or not transaction_id:
            raise ContractValidationError(
                "dispute_id and transaction_id must be non-empty"
            )
        dispute_ts = parse_utc_timestamp(data.get("dispute_ts"), "dispute_ts")

        try:
            dispute_amount = float(data["dispute_amount"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractValidationError(
                f"Missing or non-numeric dispute_amount for {dispute_id}"
            ) from exc
        if not math.isfinite(dispute_amount) or dispute_amount <= 0.0:
            raise ContractValidationError(
                f"dispute_amount must be finite and > 0, got {dispute_amount} for {dispute_id}"
            )

        currency = str(data.get("currency", "USD")).strip().upper()
        if currency not in SUPPORTED_CURRENCIES:
            raise UnsupportedInputError(
                f"Unsupported currency {currency!r} for dispute {dispute_id}"
            )
        dispute_amount_usd = round(dispute_amount * SUPPORTED_CURRENCIES[currency], 4)

        raw_code = data.get("reason_code")
        reason_code = None if raw_code is None else str(raw_code)

        return cls(
            dispute_id=dispute_id,
            transaction_id=transaction_id,
            dispute_ts=dispute_ts,
            reason_code=reason_code,
            dispute_amount=dispute_amount,
            currency=currency,
            dispute_amount_usd=dispute_amount_usd,
        )


def validate_dataset_contracts(
    merchants: Iterable[Dict[str, Any]],
    transactions: Iterable[Dict[str, Any]],
    disputes: Iterable[Dict[str, Any]],
    *,
    require_known_merchant: bool = True,
) -> Tuple[List[MerchantRecord], List[TransactionRecord], List[DisputeRecord]]:
    """Validate raw dictionaries against canonical contracts and cross-entity invariants."""
    merchant_records: List[MerchantRecord] = []
    seen_merchants: set[str] = set()
    for row in merchants:
        rec = MerchantRecord.from_dict(row)
        if rec.merchant_id in seen_merchants:
            raise ContractValidationError(
                f"Duplicate merchant_id detected: {rec.merchant_id}"
            )
        seen_merchants.add(rec.merchant_id)
        merchant_records.append(rec)

    tx_records: List[TransactionRecord] = []
    tx_by_id: Dict[str, TransactionRecord] = {}
    for row in transactions:
        rec = TransactionRecord.from_dict(row)
        if rec.transaction_id in tx_by_id:
            raise ContractValidationError(
                f"Duplicate transaction_id detected: {rec.transaction_id}"
            )
        if require_known_merchant and rec.merchant_id not in seen_merchants:
            raise ContractValidationError(
                f"Transaction {rec.transaction_id} references unknown merchant_id "
                f"{rec.merchant_id!r}"
            )
        tx_by_id[rec.transaction_id] = rec
        tx_records.append(rec)

    dispute_records: List[DisputeRecord] = []
    seen_dispute_ids: set[str] = set()
    for row in disputes:
        rec = DisputeRecord.from_dict(row)
        if rec.dispute_id in seen_dispute_ids:
            raise ContractValidationError(
                f"Duplicate dispute_id detected: {rec.dispute_id}"
            )
        seen_dispute_ids.add(rec.dispute_id)
        parent_tx = tx_by_id.get(rec.transaction_id)
        if parent_tx is None:
            raise ContractValidationError(
                f"Dispute {rec.dispute_id} references unknown transaction_id "
                f"{rec.transaction_id!r}"
            )
        if rec.dispute_ts < parent_tx.transaction_ts:
            raise ContractValidationError(
                f"Dispute {rec.dispute_id} timestamp ({rec.dispute_ts.isoformat()}) "
                f"precedes transaction_ts ({parent_tx.transaction_ts.isoformat()})"
            )
        if rec.dispute_amount_usd > parent_tx.amount_usd * 1.05 + 1e-6:
            raise ContractValidationError(
                f"Dispute {rec.dispute_id} amount_usd ({rec.dispute_amount_usd}) exceeds "
                f"transaction amount_usd ({parent_tx.amount_usd}) beyond 5% FX tolerance"
            )
        dispute_records.append(rec)

    return merchant_records, tx_records, dispute_records
