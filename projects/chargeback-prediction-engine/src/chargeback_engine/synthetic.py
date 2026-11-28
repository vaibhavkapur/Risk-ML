"""Deterministic synthetic merchant, transaction, and delayed-dispute generator."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

from chargeback_engine.config import SyntheticConfig
from chargeback_engine.contracts import (
    SUPPORTED_CURRENCIES,
    SUPPORTED_MCC_GROUPS,
    PaymentChannel,
    parse_utc_timestamp,
)

_MCC_LIST = sorted(SUPPORTED_MCC_GROUPS)
_CHANNEL_LIST = [c.value for c in PaymentChannel]
_CURRENCIES = ["USD", "USD", "USD", "USD", "EUR", "GBP", "CAD"]
_COUNTRIES = ["US", "US", "US", "GB", "DE", "CA"]

_MCC_BASE_LOGIT = {
    "digital_goods": 0.65,
    "travel": 0.45,
    "subscription": 0.25,
    "hospitality": 0.10,
    "services": -0.15,
    "retail": -0.30,
}

_CHANNEL_LOGIT = {
    "card_not_present": 0.55,
    "recurring": 0.20,
    "digital_wallet": -0.15,
    "card_present": -0.85,
}

_REASON_CODES_NON_MISSING = [
    ("10.4", 0.34),
    ("4837", 0.16),
    ("13.1", 0.24),
    ("4853", 0.12),
    ("11.1", 0.08),
    ("12.5", 0.06),
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def generate_synthetic_dataset(
    cfg: SyntheticConfig,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate deterministic synthetic merchants, transactions, and delayed disputes.

    Explicitly models:
      - Cold-start merchants that onboard late in the observation window.
      - Heterogenous merchant risk profiles (latent risk + MCC + fulfillment delay).
      - Delayed dispute reporting (log-normal/gamma-like delay up to 110 days).
      - Missing reason codes at `cfg.missing_reason_code_rate`.
      - Right-censored disputes occurring after `cfg.as_of_date` so maturity logic
        can be verified against ground truth.
    """
    rng = np.random.default_rng(seed)
    start_dt = parse_utc_timestamp(cfg.start_date, "synthetic.start_date")

    n_cold = int(round(cfg.n_merchants * cfg.cold_start_merchant_fraction))
    merchants: List[Dict[str, Any]] = []
    merchant_latent_risk: Dict[str, float] = {}
    merchant_start_day: Dict[str, float] = {}

    for i in range(cfg.n_merchants):
        mid = f"M_{i + 1:04d}"
        is_cold = i >= (cfg.n_merchants - n_cold)
        mcc = _MCC_LIST[int(rng.integers(0, len(_MCC_LIST)))]
        country = _COUNTRIES[int(rng.integers(0, len(_COUNTRIES)))]
        avg_fulfillment = float(round(rng.uniform(0.5, 14.0), 2))

        if is_cold:
            # Cold-start merchants onboard during the second half of the span
            onboard_offset_days = float(
                rng.uniform(cfg.span_days * 0.55, cfg.span_days * 0.85)
            )
            onboard_ts = start_dt + timedelta(days=onboard_offset_days)
            merchant_start_day[mid] = onboard_offset_days
        else:
            onboard_ts = start_dt - timedelta(days=float(rng.uniform(90, 900)))
            merchant_start_day[mid] = 0.0

        # Mixture: most merchants low-to-moderate risk, ~18% elevated dispute propensity
        if rng.random() < 0.18:
            latent = float(rng.normal(1.65, 0.35))
        else:
            latent = float(rng.normal(-0.45, 0.40))

        merchant_latent_risk[mid] = latent
        merchants.append(
            {
                "merchant_id": mid,
                "mcc_group": mcc,
                "onboarding_ts": onboard_ts.astimezone(timezone.utc).isoformat(),
                "country": country,
                "avg_fulfillment_days": avg_fulfillment,
            }
        )

    # Calibrate intercept to hit approximate target base_dispute_rate
    base_logit = math.log(cfg.base_dispute_rate / (1.0 - cfg.base_dispute_rate)) - 0.95

    # Merchant transaction share weights (established get higher volume than cold-start)
    weights = np.array(
        [
            0.35 if merchant_start_day[m["merchant_id"]] > 0.0 else 1.0
            for m in merchants
        ],
        dtype=np.float64,
    )
    weights = weights * rng.lognormal(mean=0.0, sigma=0.45, size=len(merchants))
    weights /= weights.sum()

    reason_codes = [rc for rc, _ in _REASON_CODES_NON_MISSING]
    reason_probs = np.array([p for _, p in _REASON_CODES_NON_MISSING], dtype=np.float64)
    reason_probs /= reason_probs.sum()

    transactions: List[Dict[str, Any]] = []
    disputes: List[Dict[str, Any]] = []
    dispute_counter = 0

    chosen_merchant_indices = rng.choice(
        len(merchants), size=cfg.n_transactions, p=weights
    )

    for idx in range(cfg.n_transactions):
        m = merchants[int(chosen_merchant_indices[idx])]
        mid = m["merchant_id"]
        min_day = merchant_start_day[mid]
        tx_day = float(rng.uniform(min_day, cfg.span_days))
        tx_ts = start_dt + timedelta(days=tx_day)
        settle_delay_hours = float(rng.uniform(4.0, 48.0))
        settlement_ts = tx_ts + timedelta(hours=settle_delay_hours)

        currency = _CURRENCIES[int(rng.integers(0, len(_CURRENCIES)))]
        amount_base = float(np.exp(rng.normal(4.3, 0.95)))  # ~ $20 - $1500+
        amount = round(max(5.0, min(5000.0, amount_base)), 2)
        amount_usd = amount * SUPPORTED_CURRENCIES[currency]

        channel = _CHANNEL_LIST[int(rng.choice(len(_CHANNEL_LIST), p=[0.52, 0.20, 0.16, 0.12]))]
        is_cross_border = bool(rng.random() < (0.28 if channel == "card_not_present" else 0.12))
        has_3ds = bool(rng.random() < (0.45 if channel == "card_not_present" else 0.70))
        has_avs_match = bool(rng.random() < 0.84)
        customer_tenure_days = int(max(0, round(rng.exponential(160.0))))

        # Structural dispute risk model (synthetic ground truth with realistic interactions)
        high_risk_cnp_interaction = (
            1.35
            if (channel == "card_not_present" and is_cross_border and not has_3ds and amount_usd >= 150.0)
            else 0.0
        )
        new_cust_no_avs_interaction = (
            1.10 if (customer_tenure_days < 30 and not has_avs_match) else 0.0
        )
        logit = (
            base_logit
            + merchant_latent_risk[mid]
            + _MCC_BASE_LOGIT[m["mcc_group"]]
            + _CHANNEL_LOGIT[channel]
            + 0.55 * math.log1p(amount_usd / 100.0)
            + (0.85 if is_cross_border else 0.0)
            + (-0.85 if has_3ds else 0.40)
            + (-0.80 if has_avs_match else 0.95)
            + (-0.45 * math.log1p(customer_tenure_days / 30.0))
            + 0.06 * m["avg_fulfillment_days"]
            + high_risk_cnp_interaction
            + new_cust_no_avs_interaction
        )
        true_prob = _sigmoid(logit)

        tx_id = f"TX_{idx + 1:06d}"
        transactions.append(
            {
                "transaction_id": tx_id,
                "merchant_id": mid,
                "transaction_ts": tx_ts.astimezone(timezone.utc).isoformat(),
                "settlement_ts": settlement_ts.astimezone(timezone.utc).isoformat(),
                "amount": amount,
                "currency": currency,
                "channel": channel,
                "is_cross_border": is_cross_border,
                "has_3ds": has_3ds,
                "has_avs_match": has_avs_match,
                "customer_tenure_days": customer_tenure_days,
            }
        )

        if rng.random() < true_prob:
            dispute_counter += 1
            # Delayed reporting: most arrive between 7 and 75 days after settlement,
            # with a tail up to 110 days.
            delay_days = float(
                np.clip(
                    rng.gamma(shape=2.6, scale=11.5) + m["avg_fulfillment_days"] * 0.5,
                    2.0,
                    110.0,
                )
            )
            dispute_ts = settlement_ts + timedelta(days=delay_days)
            if rng.random() < cfg.missing_reason_code_rate:
                reason_code = None
            else:
                reason_code = str(rng.choice(reason_codes, p=reason_probs))
            # Mostly full amount disputes, occasionally partial
            dispute_fraction = 1.0 if rng.random() < 0.85 else float(rng.uniform(0.4, 0.95))
            dispute_amount = round(max(1.0, amount * dispute_fraction), 2)
            disputes.append(
                {
                    "dispute_id": f"DSP_{dispute_counter:06d}",
                    "transaction_id": tx_id,
                    "dispute_ts": dispute_ts.astimezone(timezone.utc).isoformat(),
                    "reason_code": reason_code,
                    "dispute_amount": dispute_amount,
                    "currency": currency,
                }
            )

    transactions.sort(key=lambda r: (r["transaction_ts"], r["transaction_id"]))
    disputes.sort(key=lambda r: (r["dispute_ts"], r["dispute_id"]))
    return merchants, transactions, disputes
