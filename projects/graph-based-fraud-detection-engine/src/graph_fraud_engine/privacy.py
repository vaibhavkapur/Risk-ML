"""Privacy-safe entity pseudonymization and PII boundary guards.

All graph nodes and transactions use deterministic HMAC-SHA256 pseudonyms with
entity-type prefixes (acc_, dev_, mer_, tx_) so raw synthetic identifiers never
enter the entity graph, model features, or analyst-facing evidence logs.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Literal

EntityPrefix = Literal["acc", "dev", "mer", "tx"]

PRIVACY_SAFE_ID_PATTERNS = {
    "acc": re.compile(r"^acc_[0-9a-f]{8,32}$"),
    "dev": re.compile(r"^dev_[0-9a-f]{8,32}$"),
    "mer": re.compile(r"^mer_[0-9a-f]{8,32}$"),
    "tx": re.compile(r"^tx_[0-9a-f]{8,32}$"),
}

# Patterns that strongly resemble raw PII (email, SSN, payment card PAN).
_PII_PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
]


class PrivacyViolationError(ValueError):
    """Raised when an entity ID violates privacy-safe pseudonym rules or appears to contain PII."""


def contains_pii_like_pattern(value: str) -> bool:
    """Return True if the string matches common email, SSN, or card-PAN patterns."""
    return any(pattern.search(value) for pattern in _PII_PATTERNS)


def pseudonymize_id(
    raw_id: str,
    prefix: EntityPrefix,
    salt: str,
    digest_length: int = 12,
) -> str:
    """Generate a deterministic, privacy-safe pseudonymized ID for a raw synthetic identifier."""
    if not raw_id or not isinstance(raw_id, str):
        raise PrivacyViolationError("raw_id must be a non-empty string")
    if contains_pii_like_pattern(raw_id):
        raise PrivacyViolationError(
            f"Refusing to ingest PII-like raw identifier for prefix '{prefix}'"
        )
    if prefix not in PRIVACY_SAFE_ID_PATTERNS:
        raise PrivacyViolationError(f"Unsupported entity prefix: {prefix}")
    if not (8 <= digest_length <= 32):
        raise PrivacyViolationError("digest_length must be between 8 and 32")

    mac = hmac.new(
        key=salt.encode("utf-8"),
        msg=f"{prefix}:{raw_id}".encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()[:digest_length]
    return f"{prefix}_{mac}"


def is_privacy_safe_id(value: str, prefix: EntityPrefix) -> bool:
    """Check whether value conforms to the privacy-safe pseudonym format for the given prefix."""
    if not isinstance(value, str) or contains_pii_like_pattern(value):
        return False
    pattern = PRIVACY_SAFE_ID_PATTERNS.get(prefix)
    if pattern is None:
        return False
    return bool(pattern.match(value))


def assert_privacy_safe_id(value: str, prefix: EntityPrefix) -> str:
    """Validate that value is a privacy-safe ID with the expected prefix."""
    if not is_privacy_safe_id(value, prefix):
        raise PrivacyViolationError(
            f"Identifier '{value}' does not match privacy-safe format '{prefix}_<8..32 hex chars>'"
        )
    return value
