"""Explicit exception hierarchy for the Real-Time Transaction Risk Platform."""

from __future__ import annotations


class RiskPlatformError(Exception):
    """Base exception for all platform errors."""


class ConfigurationError(RiskPlatformError):
    """Raised when configuration is missing, malformed, or invalid."""


class ContractValidationError(RiskPlatformError):
    """Raised when an input payload violates the canonical data contract."""


class UnsupportedSchemaVersionError(ContractValidationError):
    """Raised when an event specifies an unsupported schema_version."""


class UnsupportedCurrencyError(ContractValidationError):
    """Raised when an event specifies an unsupported currency code."""


class SensitiveDataViolationError(ContractValidationError):
    """Raised when raw card numbers (PANs) or prohibited sensitive patterns are detected."""


class DuplicateConflictError(RiskPlatformError):
    """Raised when a duplicate event_id arrives with a conflicting payload hash."""


class ArtifactIntegrityError(RiskPlatformError):
    """Raised when a model artifact fails checksum or schema compatibility validation."""


class ModelTimeoutError(RiskPlatformError):
    """Raised when model inference exceeds its configured latency budget."""


class FeatureStoreError(RiskPlatformError):
    """Raised when the feature store adapter cannot complete an operation."""
