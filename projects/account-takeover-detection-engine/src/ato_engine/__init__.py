"""Account Takeover Detection Engine — research prototype package.

Synthetic login, device, network, and session behavioral detection with
strictly causal features, calibrated risk scoring, adaptive step-up
recommendations, analyst evidence, drift alerts, and fallback policy.
"""

from __future__ import annotations

__version__ = "0.1.0"
SCHEMA_VERSION = "1.0.0"
FEATURE_VERSION = "1.0.0"
POLICY_VERSION = "1.0.0"

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "FEATURE_VERSION",
    "POLICY_VERSION",
]
