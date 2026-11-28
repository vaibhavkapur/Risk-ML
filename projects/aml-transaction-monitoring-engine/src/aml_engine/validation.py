"""Strict input validation, timestamp parsing, and domain exception types."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, Set

from aml_engine.models import Account, Entity, RawTransaction


class AMLEngineError(Exception):
    """Base exception for the AML Transaction Monitoring Engine."""


class ConfigurationError(AMLEngineError):
    """Raised when required configuration is missing, unreadable, or invalid."""


class ValidationError(AMLEngineError):
    """Raised when an input record violates the data contract schema."""


class UnsupportedInputError(AMLEngineError):
    """Raised when an input uses an unsupported currency, channel, or schema version."""


class CaseWorkflowError(AMLEngineError):
    """Raised on invalid case state transition or missing reviewer override audit fields."""


def parse_utc_timestamp(value: str, field_name: str = "timestamp") -> datetime:
    """Parse an ISO-8601 timestamp string into a timezone-aware UTC datetime.

    Python 3.9's datetime.fromisoformat does not natively accept trailing 'Z',
    so we normalize 'Z' -> '+00:00' and reject naive timestamps explicitly.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"Field '{field_name}' must be a non-empty ISO-8601 UTC string.")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValidationError(
            f"Field '{field_name}' has invalid ISO-8601 format: {value!r}"
        ) from exc
    if dt.tzinfo is None:
        raise ValidationError(
            f"Field '{field_name}' must include timezone info (UTC 'Z' or offset): {value!r}"
        )
    return dt.astimezone(timezone.utc)


def format_utc_iso(dt: datetime) -> str:
    """Format a timezone-aware datetime as canonical ISO-8601 UTC string."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_entity_dict(data: Dict[str, Any]) -> Entity:
    if not isinstance(data, dict):
        raise ValidationError("Entity record must be a dictionary.")
    entity_id = data.get("entity_id")
    name = data.get("name")
    if not isinstance(entity_id, str) or not entity_id.strip():
        raise ValidationError("Entity 'entity_id' must be a non-empty string.")
    if not isinstance(name, str) or not name.strip():
        raise ValidationError(f"Entity '{entity_id}' must have a non-empty 'name'.")
    risk_tier = str(data.get("risk_tier", "LOW")).upper()
    if risk_tier not in {"LOW", "MEDIUM", "HIGH"}:
        raise ValidationError(
            f"Entity '{entity_id}' has invalid risk_tier={risk_tier!r}; expected LOW, MEDIUM, or HIGH."
        )
    jurisdiction = str(data.get("jurisdiction", "US")).upper()
    return Entity(
        entity_id=entity_id.strip(),
        name=name.strip(),
        risk_tier=risk_tier,
        jurisdiction=jurisdiction,
    )


def validate_account_dict(data: Dict[str, Any]) -> Account:
    if not isinstance(data, dict):
        raise ValidationError("Account record must be a dictionary.")
    account_id = data.get("account_id")
    entity_id = data.get("entity_id")
    if not isinstance(account_id, str) or not account_id.strip():
        raise ValidationError("Account 'account_id' must be a non-empty string.")
    if not isinstance(entity_id, str) or not entity_id.strip():
        raise ValidationError(f"Account '{account_id}' must have a non-empty 'entity_id'.")
    opened_at = data.get("opened_at", "2025-01-01T00:00:00Z")
    dt_opened = parse_utc_timestamp(opened_at, field_name="opened_at")
    baseline = data.get("baseline_monthly_usd", 5000.0)
    if not isinstance(baseline, (int, float)) or isinstance(baseline, bool) or not math.isfinite(baseline) or baseline < 0:
        raise ValidationError(
            f"Account '{account_id}' baseline_monthly_usd must be a finite non-negative number."
        )
    account_type = str(data.get("account_type", "RETAIL")).upper()
    return Account(
        account_id=account_id.strip(),
        entity_id=entity_id.strip(),
        account_type=account_type,
        opened_at=format_utc_iso(dt_opened),
        baseline_monthly_usd=float(baseline),
    )


def validate_transaction_dict(
    data: Dict[str, Any],
    supported_currencies: Set[str],
    supported_channels: Set[str],
) -> RawTransaction:
    if not isinstance(data, dict):
        raise ValidationError("Transaction record must be a dictionary.")

    required_fields = (
        "tx_id",
        "source_account_id",
        "target_account_id",
        "amount",
        "currency",
        "timestamp",
        "channel",
    )
    for req in required_fields:
        if req not in data or data[req] is None:
            raise ValidationError(f"Transaction missing required field '{req}'.")

    tx_id = str(data["tx_id"]).strip()
    src = str(data["source_account_id"]).strip()
    tgt = str(data["target_account_id"]).strip()
    if not tx_id:
        raise ValidationError("Transaction 'tx_id' must be non-empty.")
    if not src or not tgt:
        raise ValidationError(f"Transaction '{tx_id}' requires non-empty source and target account IDs.")
    if src == tgt:
        raise ValidationError(
            f"Transaction '{tx_id}' cannot have identical source and target account_id ({src!r})."
        )

    amount = data["amount"]
    if not isinstance(amount, (int, float)) or isinstance(amount, bool) or not math.isfinite(amount):
        raise ValidationError(f"Transaction '{tx_id}' amount must be a finite number.")
    if amount <= 0.0:
        raise ValidationError(f"Transaction '{tx_id}' amount must be strictly positive; got {amount}.")

    currency = str(data["currency"]).strip().upper()
    if currency not in supported_currencies:
        raise UnsupportedInputError(
            f"Transaction '{tx_id}' uses unsupported currency {currency!r}; "
            f"supported={sorted(supported_currencies)}."
        )

    channel = str(data["channel"]).strip().upper()
    if channel not in supported_channels:
        raise UnsupportedInputError(
            f"Transaction '{tx_id}' uses unsupported channel {channel!r}; "
            f"supported={sorted(supported_channels)}."
        )

    # Validate timestamp format and UTC timezone presence
    dt = parse_utc_timestamp(str(data["timestamp"]), field_name="timestamp")

    scenario_label = data.get("scenario_label")
    if scenario_label is not None:
        scenario_label = str(scenario_label).strip().upper() or None

    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValidationError(f"Transaction '{tx_id}' metadata must be a dictionary if provided.")

    return RawTransaction(
        tx_id=tx_id,
        source_account_id=src,
        target_account_id=tgt,
        amount=float(amount),
        currency=currency,
        timestamp=format_utc_iso(dt),
        channel=channel,
        scenario_label=scenario_label,
        metadata=dict(metadata),
    )
