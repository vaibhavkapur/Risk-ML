"""Chargeback Prediction Engine.

Offline educational/research prototype for delayed dispute risk prediction,
calibrated expected loss estimation, and review prioritization with
delayed-outcome monitoring.
"""

from chargeback_engine.config import EngineConfig, load_config
from chargeback_engine.contracts import (
    SCHEMA_VERSION,
    ConfigError,
    ContractValidationError,
    DuplicateDisputeError,
    LabelStatus,
    PaymentChannel,
    ReasonCategory,
    TemporalLeakageError,
    UnsupportedInputError,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "EngineConfig",
    "load_config",
    "ConfigError",
    "ContractValidationError",
    "DuplicateDisputeError",
    "TemporalLeakageError",
    "UnsupportedInputError",
    "LabelStatus",
    "PaymentChannel",
    "ReasonCategory",
]
