"""Domain-specific exception hierarchy for the Credit Risk Scoring Platform."""

from __future__ import annotations

from typing import Any, Dict, Optional


class CreditRiskError(Exception):
    """Base exception for all domain errors in the credit risk platform."""

    error_code: str = "CREDIT_RISK_ERROR"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(CreditRiskError):
    """Raised when configuration is missing, malformed, or violates invariants."""

    error_code = "CONFIGURATION_ERROR"


class ContractValidationError(CreditRiskError):
    """Raised when an input applicant or label record violates the data contract."""

    error_code = "CONTRACT_VALIDATION_ERROR"


class PointInTimeLeakageError(CreditRiskError):
    """Raised when post-origination fields, future timestamps, or immature labels leak."""

    error_code = "POINT_IN_TIME_LEAKAGE_ERROR"


class InsufficientSubgroupSampleError(CreditRiskError):
    """Raised when subgroup evaluation encounters a group smaller than minimum N."""

    error_code = "INSUFFICIENT_SUBGROUP_SAMPLE_ERROR"


class UnsupportedInputError(CreditRiskError):
    """Raised for unsupported backends, calibration methods, or disallowed policies."""

    error_code = "UNSUPPORTED_INPUT_ERROR"


class CalibrationMonotonicityError(CreditRiskError):
    """Raised when a calibration mapping violates monotonicity beyond tolerance."""

    error_code = "CALIBRATION_MONOTONICITY_ERROR"
