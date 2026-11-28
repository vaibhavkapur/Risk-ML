"""Deterministic synthetic generator for payments, device signals, and delayed risk labels."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np

from smart_auth_engine.config import DataGenerationConfig
from smart_auth_engine.contracts import DeviceSignals, PaymentEvent, RiskLabel

_COUNTRIES = ["US", "GB", "DE", "CA", "AU", "FR", "SG", "JP"]
_COUNTRY_WEIGHTS = [0.42, 0.14, 0.11, 0.09, 0.07, 0.07, 0.05, 0.05]
_COUNTRY_TO_CURRENCY = {
    "US": "USD",
    "GB": "GBP",
    "DE": "EUR",
    "CA": "CAD",
    "AU": "AUD",
    "FR": "EUR",
    "SG": "USD",
    "JP": "JPY",
}
_CATEGORIES = [
    "retail",
    "digital_goods",
    "travel",
    "crypto_exchange",
    "gambling",
    "groceries",
    "utilities",
    "luxury",
]
_CATEGORY_WEIGHTS = [0.28, 0.18, 0.12, 0.05, 0.04, 0.17, 0.10, 0.06]
_CATEGORY_RISK_LOGIT = {
    "groceries": -0.9,
    "utilities": -1.1,
    "retail": -0.2,
    "travel": 0.25,
    "digital_goods": 0.65,
    "luxury": 0.75,
    "gambling": 1.10,
    "crypto_exchange": 1.35,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def generate_synthetic_dataset(
    cfg: DataGenerationConfig,
    seed: int = 42,
) -> Tuple[List[PaymentEvent], List[RiskLabel], np.ndarray]:
    """Generate chronologically sorted synthetic payments, delayed labels, and true risk probabilities.

    Returns:
        payments: Chronologically sorted list of PaymentEvent instances.
        labels: Matching list of RiskLabel instances with realistic label maturity delays.
        true_probs: True underlying data-generating fraud probabilities (oracle reference).
    """
    rng = np.random.default_rng(seed)
    start_dt = datetime.fromisoformat(cfg.start_date.replace("Z", "+00:00")).astimezone(timezone.utc)
    n = cfg.n_samples

    # Event timestamps spread across the horizon and sorted chronologically
    offsets_seconds = np.sort(rng.uniform(0.0, cfg.horizon_days * 86400.0, size=n))

    n_accounts = max(200, n // 5)
    n_merchants = max(60, n // 40)

    account_ids = rng.integers(1, n_accounts + 1, size=n)
    merchant_ids = rng.integers(1, n_merchants + 1, size=n)
    account_base_age = rng.gamma(shape=2.0, scale=180.0, size=n_accounts + 1)

    billing_countries = rng.choice(_COUNTRIES, size=n, p=_COUNTRY_WEIGHTS)
    categories = rng.choice(_CATEGORIES, size=n, p=_CATEGORY_WEIGHTS)

    # Generate correlated device and behavioral signals
    is_emulator = rng.random(n) < 0.03
    is_tor_or_vpn = rng.random(n) < 0.075
    browser_mismatch = rng.random(n) < 0.09
    biometric_enrolled = rng.random(n) < 0.62
    is_recurring = rng.random(n) < 0.18

    cross_border_draw = rng.random(n) < 0.16
    ip_countries = []
    bin_countries = []
    for i in range(n):
        bc = str(billing_countries[i])
        if cross_border_draw[i]:
            alt_country = str(rng.choice([c for c in _COUNTRIES if c != bc]))
            ip_countries.append(alt_country)
            bin_countries.append(bc if rng.random() < 0.65 else alt_country)
        else:
            ip_countries.append(bc)
            bin_countries.append(bc)

    # Device trust score negatively correlated with emulator/TOR/mismatch
    base_trust = rng.beta(a=5.0, b=1.6, size=n)
    trust_penalty = (
        0.35 * is_emulator.astype(float)
        + 0.22 * is_tor_or_vpn.astype(float)
        + 0.18 * browser_mismatch.astype(float)
    )
    device_trust = np.clip(base_trust - trust_penalty, 0.02, 0.99)

    device_age_days = np.clip(rng.exponential(scale=120.0, size=n) * (1.0 - 0.7 * is_emulator), 0.1, 2000.0)
    account_age_days = np.clip(
        account_base_age[account_ids] + offsets_seconds / 86400.0,
        0.1,
        5000.0,
    )

    # Velocity signals
    ip_velocity_1h = rng.poisson(lam=1.4 + 4.5 * is_tor_or_vpn + 3.0 * is_emulator).astype(int)
    # Inject occasional velocity bursts
    burst_mask = rng.random(n) < 0.015
    ip_velocity_1h = np.where(burst_mask, ip_velocity_1h + rng.integers(10, 26, size=n), ip_velocity_1h)
    ip_velocity_24h = ip_velocity_1h + rng.poisson(lam=3.0, size=n).astype(int)

    prior_chargebacks_90d = rng.choice([0, 1, 2, 3], size=n, p=[0.935, 0.045, 0.015, 0.005]).astype(int)
    tx_count_24h = rng.poisson(lam=1.8, size=n).astype(int) + 1

    # Payment amount in minor units (log-normal distribution, heavier for travel/luxury/crypto)
    cat_scale = np.array(
        [
            8.9 if c in ("luxury", "travel", "crypto_exchange") else 7.8
            for c in categories
        ]
    )
    amount_minor = np.clip(
        np.round(np.exp(rng.normal(loc=cat_scale, scale=1.05, size=n))).astype(int),
        100,
        600_000,
    )
    tx_amount_24h_minor = (amount_minor * np.maximum(1, tx_count_24h)).astype(int)

    # True log-odds model for synthetic fraud generation
    cat_logit = np.array([_CATEGORY_RISK_LOGIT[str(c)] for c in categories])
    is_cross_border = np.array(
        [ip_countries[i] != str(billing_countries[i]) or bin_countries[i] != str(billing_countries[i]) for i in range(n)],
        dtype=float,
    )
    log_amount = np.log1p(amount_minor / 100.0)
    raw_logit = (
        -3.55
        + 0.42 * (log_amount - 4.0)
        + cat_logit
        - 3.4 * (device_trust - 0.5)
        + 1.65 * is_emulator.astype(float)
        + 1.15 * is_tor_or_vpn.astype(float)
        + 0.95 * browser_mismatch.astype(float)
        + 0.75 * is_cross_border
        + 0.14 * np.minimum(ip_velocity_1h, 25)
        + 1.25 * np.minimum(prior_chargebacks_90d, 3)
        - 0.45 * np.log1p(account_age_days / 30.0)
        - 0.35 * np.log1p(device_age_days / 30.0)
        - 0.65 * is_recurring.astype(float)
        - 0.35 * biometric_enrolled.astype(float)
    )

    # Shift intercept so empirical average matches configured base_fraud_rate
    shift = 0.0
    for _ in range(30):
        mean_p = float(_sigmoid(raw_logit + shift).mean())
        shift += float(np.log(cfg.base_fraud_rate / max(mean_p, 1e-6))) * 0.6
    true_probs = _sigmoid(raw_logit + shift)
    is_fraud_arr = rng.random(n) < true_probs

    # Inject small rate of realistic missing optional telemetry (never corrupting payment_id/amount)
    missing_trust = rng.random(n) < 0.025
    missing_acct_age = rng.random(n) < 0.02

    payments: List[PaymentEvent] = []
    labels: List[RiskLabel] = []
    for i in range(n):
        ts = start_dt + timedelta(seconds=float(offsets_seconds[i]))
        pid = f"pay_{i + 1:06d}"
        bc = str(billing_countries[i])
        currency = _COUNTRY_TO_CURRENCY[bc]
        amt_minor = int(amount_minor[i])
        device = DeviceSignals(
            device_id=f"dev_{(account_ids[i] * 7 + i % 3):05d}",
            device_age_days=round(float(device_age_days[i]), 2),
            is_emulator=bool(is_emulator[i]),
            is_tor_or_vpn=bool(is_tor_or_vpn[i]),
            ip_country=ip_countries[i],
            ip_velocity_1h=int(ip_velocity_1h[i]),
            ip_velocity_24h=int(ip_velocity_24h[i]),
            device_trust_score=None if missing_trust[i] else round(float(device_trust[i]), 4),
            biometric_enrolled=bool(biometric_enrolled[i]),
            browser_fingerprint_mismatch=bool(browser_mismatch[i]),
        )
        event = PaymentEvent(
            payment_id=pid,
            account_id=f"acct_{int(account_ids[i]):05d}",
            merchant_id=f"merch_{int(merchant_ids[i]):04d}",
            merchant_category=str(categories[i]),
            amount_minor=amt_minor,
            currency=currency,
            timestamp=ts,
            billing_country=bc,
            card_bin_country=bin_countries[i],
            is_recurring=bool(is_recurring[i]),
            is_cross_border=bool(is_cross_border[i]),
            account_age_days=None if missing_acct_age[i] else round(float(account_age_days[i]), 2),
            prior_chargebacks_90d=int(prior_chargebacks_90d[i]),
            tx_count_24h=int(tx_count_24h[i]),
            tx_amount_24h_minor=int(tx_amount_24h_minor[i]),
            device=device,
        )
        payments.append(event)

        is_fraud = bool(is_fraud_arr[i])
        if is_fraud:
            delay_days = float(
                rng.uniform(cfg.fraud_label_delay_days_min, cfg.fraud_label_delay_days_max)
            )
            label_source = "chargeback" if delay_days >= 7.0 else "issuer_confirmed_fraud"
            fraud_loss = amt_minor
        else:
            delay_days = float(cfg.non_fraud_maturity_days)
            label_source = "matured_non_fraud"
            fraud_loss = 0

        label = RiskLabel(
            payment_id=pid,
            event_timestamp=ts,
            is_fraud=is_fraud,
            label_available_at=ts + timedelta(days=delay_days),
            label_source=label_source,
            fraud_loss_minor=fraud_loss,
        )
        labels.append(label)

    return payments, labels, true_probs


def index_labels_by_payment(labels: List[RiskLabel]) -> Dict[str, RiskLabel]:
    """Map payment_id -> RiskLabel for O(1) lookup."""
    return {lbl.payment_id: lbl for lbl in labels}
