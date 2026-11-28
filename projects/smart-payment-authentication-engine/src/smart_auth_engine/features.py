"""Versioned feature engineering with explicit missing-feature tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from smart_auth_engine.contracts import PaymentEvent

FEATURE_VERSION = "1.0.0"

_CATEGORY_RISK_ENCODING: Dict[str, float] = {
    "utilities": 0.0,
    "groceries": 0.1,
    "retail": 0.3,
    "travel": 0.5,
    "digital_goods": 0.7,
    "luxury": 0.75,
    "gambling": 0.9,
    "crypto_exchange": 1.0,
}

# Canonical ordered feature schema for v1.0.0
FEATURE_NAMES: List[str] = [
    "log_amount_major",
    "merchant_category_risk",
    "is_recurring",
    "is_cross_border",
    "log_account_age_months",
    "prior_chargebacks_90d",
    "tx_count_24h",
    "log_tx_amount_24h_major",
    "log_device_age_months",
    "is_emulator",
    "is_tor_or_vpn",
    "ip_velocity_1h",
    "ip_velocity_24h",
    "device_trust_score",
    "biometric_enrolled",
    "browser_fingerprint_mismatch",
]

# Tracked raw optional telemetry fields (used to compute missing_feature_ratio)
TRACKED_TELEMETRY_FIELDS: List[str] = [
    "account_age_days",
    "prior_chargebacks_90d",
    "tx_count_24h",
    "tx_amount_24h_minor",
    "device_age_days",
    "is_emulator",
    "is_tor_or_vpn",
    "ip_country",
    "ip_velocity_1h",
    "ip_velocity_24h",
    "device_trust_score",
    "biometric_enrolled",
    "browser_fingerprint_mismatch",
]

# Conservative neutral defaults used only after missingness checks pass
_DEFAULT_IMPUTATIONS: Dict[str, float] = {
    "log_account_age_months": math.log1p(3.0),
    "prior_chargebacks_90d": 0.0,
    "tx_count_24h": 1.0,
    "log_tx_amount_24h_major": math.log1p(50.0),
    "log_device_age_months": math.log1p(2.0),
    "is_emulator": 0.0,
    "is_tor_or_vpn": 0.0,
    "ip_velocity_1h": 2.0,
    "ip_velocity_24h": 4.0,
    "device_trust_score": 0.5,
    "biometric_enrolled": 0.0,
    "browser_fingerprint_mismatch": 0.0,
}


@dataclass(frozen=True)
class ExtractedFeatures:
    """Versioned feature vector with explicit missing telemetry metadata."""

    feature_version: str
    feature_names: List[str]
    values: np.ndarray  # shape (len(FEATURE_NAMES),)
    raw_map: Dict[str, Optional[float]]
    missing_fields: List[str]
    missing_ratio: float


def extract_features(event: PaymentEvent) -> ExtractedFeatures:
    """Extract deterministic versioned features from a single PaymentEvent."""
    dev = event.device
    raw_optional: Dict[str, Optional[object]] = {
        "account_age_days": event.account_age_days,
        "prior_chargebacks_90d": event.prior_chargebacks_90d,
        "tx_count_24h": event.tx_count_24h,
        "tx_amount_24h_minor": event.tx_amount_24h_minor,
        "device_age_days": dev.device_age_days,
        "is_emulator": dev.is_emulator,
        "is_tor_or_vpn": dev.is_tor_or_vpn,
        "ip_country": dev.ip_country,
        "ip_velocity_1h": dev.ip_velocity_1h,
        "ip_velocity_24h": dev.ip_velocity_24h,
        "device_trust_score": dev.device_trust_score,
        "biometric_enrolled": dev.biometric_enrolled,
        "browser_fingerprint_mismatch": dev.browser_fingerprint_mismatch,
    }
    missing_fields = [k for k in TRACKED_TELEMETRY_FIELDS if raw_optional[k] is None]
    missing_ratio = len(missing_fields) / float(len(TRACKED_TELEMETRY_FIELDS))

    amt_major = event.amount_major
    tx_amt_24h_major = (
        None
        if event.tx_amount_24h_minor is None
        else (float(event.tx_amount_24h_minor) if event.currency == "JPY" else event.tx_amount_24h_minor / 100.0)
    )

    raw_map: Dict[str, Optional[float]] = {
        "log_amount_major": math.log1p(amt_major),
        "merchant_category_risk": _CATEGORY_RISK_ENCODING[event.merchant_category],
        "is_recurring": 1.0 if event.is_recurring else 0.0,
        "is_cross_border": 1.0 if event.is_cross_border else 0.0,
        "log_account_age_months": (
            None if event.account_age_days is None else math.log1p(event.account_age_days / 30.0)
        ),
        "prior_chargebacks_90d": (
            None if event.prior_chargebacks_90d is None else float(min(event.prior_chargebacks_90d, 10))
        ),
        "tx_count_24h": None if event.tx_count_24h is None else float(min(event.tx_count_24h, 100)),
        "log_tx_amount_24h_major": (
            None if tx_amt_24h_major is None else math.log1p(tx_amt_24h_major)
        ),
        "log_device_age_months": (
            None if dev.device_age_days is None else math.log1p(dev.device_age_days / 30.0)
        ),
        "is_emulator": None if dev.is_emulator is None else (1.0 if dev.is_emulator else 0.0),
        "is_tor_or_vpn": None if dev.is_tor_or_vpn is None else (1.0 if dev.is_tor_or_vpn else 0.0),
        "ip_velocity_1h": None if dev.ip_velocity_1h is None else float(min(dev.ip_velocity_1h, 100)),
        "ip_velocity_24h": None if dev.ip_velocity_24h is None else float(min(dev.ip_velocity_24h, 500)),
        "device_trust_score": None if dev.device_trust_score is None else float(dev.device_trust_score),
        "biometric_enrolled": (
            None if dev.biometric_enrolled is None else (1.0 if dev.biometric_enrolled else 0.0)
        ),
        "browser_fingerprint_mismatch": (
            None
            if dev.browser_fingerprint_mismatch is None
            else (1.0 if dev.browser_fingerprint_mismatch else 0.0)
        ),
    }

    vec = np.array(
        [
            raw_map[name] if raw_map[name] is not None else _DEFAULT_IMPUTATIONS[name]
            for name in FEATURE_NAMES
        ],
        dtype=np.float64,
    )
    return ExtractedFeatures(
        feature_version=FEATURE_VERSION,
        feature_names=list(FEATURE_NAMES),
        values=vec,
        raw_map=raw_map,
        missing_fields=missing_fields,
        missing_ratio=missing_ratio,
    )


def extract_feature_matrix(events: Sequence[PaymentEvent]) -> np.ndarray:
    """Extract a 2D feature matrix of shape (n_events, n_features)."""
    if not events:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([extract_features(ev).values for ev in events])
