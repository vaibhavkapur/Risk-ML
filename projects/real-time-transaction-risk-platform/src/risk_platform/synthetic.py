"""Deterministic synthetic transaction stream and matured outcome label generator.

Generates realistic synthetic card-tokenized events (no real PANs or customer PII)
with controllable random seed and optional distribution-drift mode for skew testing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from risk_platform.features import OfflineFeatureExtractor
from risk_platform.schemas import (
    DEFAULT_FX_TO_USD,
    OutcomeLabel,
    TransactionEvent,
    parse_utc_timestamp_ms,
)

_DEFAULT_START_ISO = "2026-01-01T00:00:00Z"
_NORMAL_MCCS = ["grocery", "restaurant", "utilities", "travel", "electronics", "digital_goods"]
_NORMAL_MCC_PROBS = [0.30, 0.25, 0.15, 0.10, 0.12, 0.08]
_FRAUD_MCCS = ["crypto_exchange", "digital_goods", "electronics", "gaming", "unknown"]
_FRAUD_MCC_PROBS = [0.32, 0.28, 0.20, 0.14, 0.06]
_CURRENCIES = ["USD", "EUR", "GBP"]
_HOME_COUNTRIES = ["US", "GB", "DE", "CA", "FR"]
_FOREIGN_COUNTRIES = ["SG", "BR", "NG", "NL", "AU"]


def generate_synthetic_dataset(
    num_events: int = 4000,
    num_accounts: int = 160,
    seed: int = 42,
    label_maturity_hours: int = 24,
    start_iso: str = _DEFAULT_START_ISO,
    fx_to_usd: Dict[str, float] = DEFAULT_FX_TO_USD,
    drift_mode: bool = False,
    id_prefix: str = "syn",
) -> Tuple[List[TransactionEvent], List[OutcomeLabel]]:
    """Generate a deterministic synthetic transaction stream and matured outcome labels."""
    rng = np.random.default_rng(seed)
    start_ms = parse_utc_timestamp_ms(start_iso)

    accounts = [f"acct_syn_{i:04d}" for i in range(1, num_accounts + 1)]
    account_home_country = {
        acct: str(rng.choice(_HOME_COUNTRIES, p=[0.55, 0.15, 0.12, 0.10, 0.08]))
        for acct in accounts
    }
    account_base_currency = {
        acct: ("GBP" if c == "GB" else ("EUR" if c in ("DE", "FR") else "USD"))
        for acct, c in account_home_country.items()
    }
    account_primary_device = {acct: f"dev_{acct[-4:]}_01" for acct in accounts}
    account_typical_spend = {
        acct: float(np.exp(rng.normal(3.6, 0.45))) for acct in accounts  # ~$25-$90 typical
    }

    events: List[TransactionEvent] = []
    curr_ts_ms = start_ms
    idx = 0

    while len(events) < num_events:
        # Advance time between 45s and 360s on average (~5-10 days across 2000-4000 events)
        step_ms = int(rng.integers(45_000, 360_000))
        curr_ts_ms += step_ms

        acct = str(rng.choice(accounts))
        is_attack_burst = (rng.random() < (0.09 if drift_mode else 0.04)) and (
            len(events) + 3 <= num_events
        )

        if is_attack_burst:
            burst_len = int(rng.integers(2, 5))
            burst_len = min(burst_len, num_events - len(events))
            attack_country = str(rng.choice(_FOREIGN_COUNTRIES))
            attack_device = f"dev_atk_{int(rng.integers(100, 999))}"
            for b in range(burst_len):
                idx += 1
                b_ts = curr_ts_ms + b * int(rng.integers(15_000, 90_000))
                mcc = str(rng.choice(_FRAUD_MCCS, p=_FRAUD_MCC_PROBS))
                amt_usd = float(np.exp(rng.normal(5.8 if not drift_mode else 6.4, 0.55)))
                curr = str(rng.choice(_CURRENCIES, p=[0.65, 0.20, 0.15]))
                amt_cents = max(100, min(9_500_000, int(round((amt_usd / fx_to_usd[curr]) * 100))))
                channel = str(rng.choice(["ecommerce", "in_app"], p=[0.65, 0.35]))
                ip_risk = float(np.clip(rng.beta(5.0, 2.0) + (0.12 if drift_mode else 0.0), 0.0, 1.0))
                events.append(
                    TransactionEvent(
                        schema_version="tx.v1",
                        event_id=f"evt_{id_prefix}_{idx:06d}",
                        account_id=acct,
                        card_token=f"card_tok_{acct[-4:]}",
                        merchant_id=f"merch_syn_{int(rng.integers(1, 250)):03d}",
                        merchant_category=mcc,
                        amount_cents=amt_cents,
                        currency=curr,
                        country_code=attack_country,
                        channel=channel,
                        device_id=attack_device,
                        event_ts_ms=b_ts,
                        ip_risk_score=round(ip_risk, 4),
                    )
                )
            curr_ts_ms += burst_len * 30_000
        else:
            idx += 1
            base_amt = account_typical_spend[acct]
            if drift_mode:
                # Simulated serving drift: higher ticket sizes, more e-commerce, higher IP risk
                amt_usd = float(max(2.0, rng.lognormal(np.log(base_amt * 2.4), 0.65)))
                mcc = str(rng.choice(_NORMAL_MCCS + ["crypto_exchange"], p=[0.18, 0.15, 0.10, 0.12, 0.18, 0.15, 0.12]))
                channel = str(rng.choice(["card_present", "ecommerce", "in_app", "recurring"], p=[0.20, 0.45, 0.25, 0.10]))
                ip_risk = float(np.clip(rng.beta(2.8, 3.2), 0.0, 1.0))
            else:
                amt_usd = float(max(1.0, rng.lognormal(np.log(base_amt), 0.55)))
                mcc = str(rng.choice(_NORMAL_MCCS, p=_NORMAL_MCC_PROBS))
                channel = str(rng.choice(["card_present", "ecommerce", "in_app", "recurring"], p=[0.48, 0.26, 0.14, 0.12]))
                ip_risk = float(np.clip(rng.beta(1.3, 6.5), 0.0, 1.0))

            curr = account_base_currency[acct]
            country = account_home_country[acct]
            if rng.random() < (0.12 if drift_mode else 0.04):
                country = str(rng.choice(_FOREIGN_COUNTRIES))
            device = account_primary_device[acct]
            if rng.random() < 0.07:
                device = f"dev_{acct[-4:]}_02"

            amt_cents = max(100, min(9_500_000, int(round((amt_usd / fx_to_usd[curr]) * 100))))
            events.append(
                TransactionEvent(
                    schema_version="tx.v1",
                    event_id=f"evt_{id_prefix}_{idx:06d}",
                    account_id=acct,
                    card_token=f"card_tok_{acct[-4:]}",
                    merchant_id=f"merch_syn_{int(rng.integers(1, 250)):03d}",
                    merchant_category=mcc,
                    amount_cents=amt_cents,
                    currency=curr,
                    country_code=country,
                    channel=channel,
                    device_id=device,
                    event_ts_ms=curr_ts_ms,
                    ip_risk_score=round(ip_risk, 4),
                )
            )

    # Sort canonically and generate probabilistic fraud labels from true point-in-time risk signals
    events.sort(key=lambda e: (e.event_ts_ms, e.event_id))
    extractor = OfflineFeatureExtractor(fx_to_usd=fx_to_usd)
    feature_map = extractor.extract_batch(events)

    maturity_ms = int(label_maturity_hours) * 3_600_000
    labels: List[OutcomeLabel] = []
    for ev in events:
        fv = feature_map[ev.event_id].values
        # Latent ground-truth log-odds combining velocity, amount spike, geo jump, and context risk
        logit = (
            -5.6
            + 0.48 * fv["amount_usd_log1p"]
            + 0.72 * min(fv["tx_count_5m"], 5.0)
            + 0.22 * min(fv["tx_count_1h"], 8.0)
            + 0.26 * min(fv["amount_ratio_to_24h_avg"], 10.0)
            + 1.35 * fv["country_changed_from_prev"]
            + 2.40 * fv["high_risk_context_score"]
            - 0.16 * fv["seconds_since_prev_tx_log1p"]
            + 0.35 * max(0.0, fv["distinct_devices_24h"] - 1.0)
        )
        prob = 1.0 / (1.0 + float(np.exp(-np.clip(logit, -20.0, 20.0))))
        is_fraud = int(rng.random() < prob)
        # Fraud chargebacks take slightly longer to mature than confirmed legit, within maturity window
        extra_delay_ms = int(rng.integers(0, max(1, maturity_ms // 2))) if is_fraud else 0
        label_ts_ms = ev.event_ts_ms + maturity_ms + extra_delay_ms
        labels.append(
            OutcomeLabel(
                event_id=ev.event_id,
                is_fraud=is_fraud,
                label_ts_ms=label_ts_ms,
                label_source="synthetic_chargeback" if is_fraud else "synthetic_confirmed_legit",
            )
        )

    return events, labels
