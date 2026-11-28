"""Deterministic synthetic transaction graph generator and hand-calculated reference fixtures.

Generates synthetic accounts, devices, merchants, transactions, and delayed fraud labels
with distinct behavioral regimes:
1. Organic accounts (low fraud, personal devices, routine merchants).
2. Legitimate shared-device groups (households/kiosks: multiple accounts sharing a device
   with steady routine spend and very low fraud).
3. Coordinated fraud rings (clusters of accounts sharing mule/emulator devices, transacting
   in synchronized bursts at high-risk cash-out merchants).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set, Tuple

import numpy as np

from graph_fraud_engine import SCHEMA_VERSION
from graph_fraud_engine.config import EngineConfig
from graph_fraud_engine.privacy import pseudonymize_id
from graph_fraud_engine.schemas import (
    AccountRecord,
    DatasetBundle,
    DeviceRecord,
    FraudLabelRecord,
    MerchantRecord,
    TransactionRecord,
)

ROUTINE_CATEGORIES = ["groceries", "utilities", "retail", "restaurant", "travel"]
HIGH_RISK_CATEGORIES = ["gift_cards", "crypto_onramp", "digital_goods", "electronics"]


def _parse_utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc)


def generate_synthetic_dataset(config: EngineConfig) -> DatasetBundle:
    """Generate a reproducible synthetic temporal transaction graph dataset."""
    rng = np.random.default_rng(config.seed)
    sc = config.synthetic_data
    salt = config.privacy.salt
    dlen = config.privacy.digest_length
    start_dt = _parse_utc(sc.start_time_utc)

    def pid(raw: str, prefix: str) -> str:
        return pseudonymize_id(raw, prefix=prefix, salt=salt, digest_length=dlen)  # type: ignore[arg-type]

    # 1. Merchants: mix of routine and high-risk cash-out merchants
    merchants: List[MerchantRecord] = []
    routine_merchants: List[str] = []
    high_risk_merchants: List[str] = []
    n_high_risk_mer = max(4, int(sc.n_merchants * 0.25))
    for i in range(sc.n_merchants):
        is_hr = i < n_high_risk_mer
        category = (
            HIGH_RISK_CATEGORIES[i % len(HIGH_RISK_CATEGORIES)]
            if is_hr
            else ROUTINE_CATEGORIES[i % len(ROUTINE_CATEGORIES)]
        )
        mid = pid(f"synthetic_merchant_{i:04d}", "mer")
        merchants.append(
            MerchantRecord(
                merchant_id=mid,
                onboarded_at=start_dt - timedelta(days=int(rng.integers(30, 500))),
                category=category,
                country=str(rng.choice(["US", "CA", "GB", "DE"])),
                base_chargeback_rate=float(rng.uniform(0.03, 0.08) if is_hr else rng.uniform(0.002, 0.012)),
            )
        )
        if is_hr:
            high_risk_merchants.append(mid)
        else:
            routine_merchants.append(mid)

    # 2. Accounts and Devices allocation across 3 regimes:
    #    - Fraud rings
    #    - Legitimate shared device groups (households/kiosks)
    #    - Organic individual accounts
    accounts: List[AccountRecord] = []
    devices: List[DeviceRecord] = []

    ring_specs: List[Tuple[List[str], List[str], List[str]]] = []
    shared_group_specs: List[Tuple[List[str], str]] = []
    organic_specs: List[Tuple[str, List[str]]] = []

    acc_idx = 0
    dev_idx = 0

    # 2a. Fraud rings
    for r in range(sc.fraud_ring_count):
        ring_accs: List[str] = []
        ring_devs: List[str] = []
        for _ in range(sc.accounts_per_ring):
            aid = pid(f"synthetic_account_{acc_idx:05d}", "acc")
            acc_idx += 1
            age_days = float(rng.uniform(1.0, 25.0))
            accounts.append(
                AccountRecord(
                    account_id=aid,
                    created_at=start_dt - timedelta(days=age_days),
                    account_age_days_at_start=age_days,
                    kyc_tier="basic",
                    home_region=str(rng.choice(["US-EAST", "US-WEST", "EU-CENTRAL"])),
                    is_synthetic_ring_member=True,
                    is_shared_household_member=False,
                )
            )
            ring_accs.append(aid)

        for d_i in range(sc.devices_per_ring):
            did = pid(f"synthetic_device_{dev_idx:05d}", "dev")
            dev_idx += 1
            # Only half of rings use obvious emulators; others use standard mobile devices
            # so tabular device_type alone cannot detect coordinated rings.
            ring_dev_type = "emulator" if (d_i == 0 and r % 2 == 0) else "mobile_android"
            devices.append(
                DeviceRecord(
                    device_id=did,
                    first_seen_at=start_dt - timedelta(days=int(rng.integers(0, 15))),
                    device_type=ring_dev_type,
                    is_shared_legitimate=False,
                    device_risk_hint=float(rng.uniform(0.65, 0.92)),
                )
            )
            ring_devs.append(did)

        target_mers = list(
            rng.choice(high_risk_merchants, size=min(3, len(high_risk_merchants)), replace=False)
        )
        ring_specs.append((ring_accs, ring_devs, target_mers))

    # 2b. Legitimate shared-device groups (households / shared kiosks)
    for g in range(sc.legitimate_shared_device_groups):
        did = pid(f"synthetic_device_{dev_idx:05d}", "dev")
        dev_idx += 1
        devices.append(
            DeviceRecord(
                device_id=did,
                first_seen_at=start_dt - timedelta(days=int(rng.integers(120, 600))),
                device_type="kiosk" if g % 3 == 0 else "desktop_web",
                is_shared_legitimate=True,
                device_risk_hint=float(rng.uniform(0.05, 0.18)),
            )
        )
        grp_accs: List[str] = []
        for _ in range(sc.accounts_per_shared_group):
            aid = pid(f"synthetic_account_{acc_idx:05d}", "acc")
            acc_idx += 1
            age_days = float(rng.uniform(120.0, 900.0))
            accounts.append(
                AccountRecord(
                    account_id=aid,
                    created_at=start_dt - timedelta(days=age_days),
                    account_age_days_at_start=age_days,
                    kyc_tier=str(rng.choice(["standard", "enhanced"])),
                    home_region=str(rng.choice(["US-EAST", "US-WEST", "US-MID"])),
                    is_synthetic_ring_member=False,
                    is_shared_household_member=True,
                )
            )
            grp_accs.append(aid)
        shared_group_specs.append((grp_accs, did))

    # 2c. Remaining organic individual accounts & devices
    remaining_accounts = max(10, sc.n_accounts - acc_idx)
    remaining_devices = max(remaining_accounts, sc.n_devices - dev_idx)
    organic_device_pool: List[str] = []
    for _ in range(remaining_devices):
        did = pid(f"synthetic_device_{dev_idx:05d}", "dev")
        dev_idx += 1
        devices.append(
            DeviceRecord(
                device_id=did,
                first_seen_at=start_dt - timedelta(days=int(rng.integers(30, 800))),
                device_type=str(rng.choice(["mobile_ios", "mobile_android", "desktop_web"])),
                is_shared_legitimate=False,
                device_risk_hint=float(rng.uniform(0.03, 0.20)),
            )
        )
        organic_device_pool.append(did)

    for i in range(remaining_accounts):
        aid = pid(f"synthetic_account_{acc_idx:05d}", "acc")
        acc_idx += 1
        age_days = float(rng.uniform(30.0, 1200.0))
        accounts.append(
            AccountRecord(
                account_id=aid,
                created_at=start_dt - timedelta(days=age_days),
                account_age_days_at_start=age_days,
                kyc_tier=str(rng.choice(["basic", "standard", "enhanced"], p=[0.2, 0.55, 0.25])),
                home_region=str(rng.choice(["US-EAST", "US-WEST", "US-MID", "CA-ON", "GB-LON"])),
                is_synthetic_ring_member=False,
                is_shared_household_member=False,
            )
        )
        primary_dev = organic_device_pool[i % len(organic_device_pool)]
        acc_devs = [primary_dev]
        if rng.random() < 0.25 and len(organic_device_pool) > 1:
            sec_dev = organic_device_pool[(i + 1) % len(organic_device_pool)]
            if sec_dev != primary_dev:
                acc_devs.append(sec_dev)
        organic_specs.append((aid, acc_devs))

    # 3. Generate transactions & delayed labels across the horizon
    raw_events: List[Tuple[datetime, str, str, str, float, str, str, int, str]] = []
    # Tuple fields: (ts, acc_id, dev_id, mer_id, amount, channel, ip_prefix, is_fraud, pattern)

    n_target_fraud = max(20, int(sc.n_transactions * sc.target_fraud_rate))
    # Ring bursts account for ~85% of all fraud; ~15% is isolated noise fraud
    n_ring_fraud_tx = int(n_target_fraud * 0.85)
    n_shared_legit_tx = int(sc.n_transactions * 0.18)
    n_organic_tx = max(50, sc.n_transactions - n_ring_fraud_tx - n_shared_legit_tx)

    # 3a. Coordinated fraud ring bursts distributed across train, val, and test windows
    # Each ring performs multiple synchronized waves across the horizon.
    waves_per_ring = 8
    tx_per_ring = max(1, n_ring_fraud_tx // len(ring_specs))
    for r_idx, (ring_accs, ring_devs, target_mers) in enumerate(ring_specs):
        # Spread ring activity across the full horizon so train, val, and test all contain coordinated rings
        base_days = np.linspace(4 + r_idx * 1.2, sc.horizon_days - 3, waves_per_ring)
        ring_ip = f"198.51.{100 + r_idx}"
        for t_i in range(tx_per_ring):
            wave_day = float(base_days[t_i % waves_per_ring])
            # Synchronized burst within a 3-hour window
            burst_offset_minutes = float(rng.uniform(0.0, 180.0))
            ts = start_dt + timedelta(days=wave_day, minutes=burst_offset_minutes)
            aid = ring_accs[t_i % len(ring_accs)]
            did = ring_devs[0] if rng.random() < 0.72 else ring_devs[-1]
            # 85% high-risk cash-out merchants, 15% cover spend at routine merchants
            if rng.random() < 0.85:
                mid = str(rng.choice(target_mers))
                amount = round(float(rng.uniform(140.0, 780.0)), 2)
                is_fraud = 1 if rng.random() < 0.90 else 0
                pattern = "coordinated_fraud_ring" if is_fraud else "ring_cover_benign"
            else:
                mid = str(rng.choice(routine_merchants))
                amount = round(float(rng.uniform(15.0, 85.0)), 2)
                is_fraud = 0
                pattern = "ring_cover_benign"
            raw_events.append((ts, aid, did, mid, amount, "mobile", ring_ip, is_fraud, pattern))

    # 3b. Legitimate shared-device transactions (steady routine spend, multi-account on same device)
    tx_per_group = max(1, n_shared_legit_tx // len(shared_group_specs))
    for g_idx, (grp_accs, shared_dev) in enumerate(shared_group_specs):
        grp_ip = f"10.20.{10 + g_idx}"
        # Preferred neighborhood merchants for this household/kiosk
        pref_routine = list(rng.choice(routine_merchants, size=min(6, len(routine_merchants)), replace=False))
        for _ in range(tx_per_group):
            day_offset = float(rng.uniform(0.0, sc.horizon_days - 0.1))
            ts = start_dt + timedelta(days=day_offset)
            aid = str(rng.choice(grp_accs))
            # 93% routine merchants, 7% occasional electronics/gift card
            if rng.random() < 0.93:
                mid = str(rng.choice(pref_routine))
                amount = round(float(rng.lognormal(mean=3.5, sigma=0.55)), 2)
            else:
                mid = str(rng.choice(high_risk_merchants))
                amount = round(float(rng.uniform(45.0, 260.0)), 2)
            amount = float(np.clip(amount, 5.0, 650.0))
            is_fraud = 1 if rng.random() < 0.004 else 0
            pattern = "isolated_shared_device_fraud" if is_fraud else "legitimate_shared_device"
            raw_events.append((ts, aid, shared_dev, mid, amount, "web", grp_ip, is_fraud, pattern))

    # 3c. Organic individual transactions
    for _ in range(n_organic_tx):
        spec_idx = int(rng.integers(0, len(organic_specs)))
        aid, acc_devs = organic_specs[spec_idx]
        did = acc_devs[0] if len(acc_devs) == 1 or rng.random() < 0.8 else acc_devs[1]
        day_offset = float(rng.uniform(0.0, sc.horizon_days - 0.1))
        ts = start_dt + timedelta(days=day_offset)
        if rng.random() < 0.82:
            mid = str(rng.choice(routine_merchants))
            amount = round(float(np.clip(rng.lognormal(mean=3.7, sigma=0.7), 4.0, 900.0)), 2)
            base_p = 0.002
        else:
            # Legitimate organic spend at electronics/gift_cards/digital_goods overlaps in amount
            # with fraud rings so tabular-only models face realistic ambiguity.
            mid = str(rng.choice(high_risk_merchants))
            amount = round(float(np.clip(rng.uniform(110.0, 820.0), 20.0, 1200.0)), 2)
            base_p = 0.008
        is_fraud = 1 if rng.random() < base_p else 0
        pattern = "isolated_organic_fraud" if is_fraud else "organic_legitimate"
        channel = str(rng.choice(["web", "mobile", "pos"], p=[0.45, 0.40, 0.15]))
        raw_events.append((ts, aid, did, mid, amount, channel, f"10.0.{spec_idx % 200}", is_fraud, pattern))

    # Sort chronologically and assign deterministic privacy-safe transaction IDs & delayed labels
    raw_events.sort(key=lambda e: (e[0], e[1], e[2], e[3], e[4]))

    transactions: List[TransactionRecord] = []
    labels: List[FraudLabelRecord] = []
    seen_ts_offset: Dict[datetime, int] = {}

    for idx, (ts, aid, did, mid, amount, channel, ip_prefix, is_fraud, pattern) in enumerate(raw_events):
        # Ensure strictly distinct timestamps when collisions occur by adding deterministic microseconds
        bump = seen_ts_offset.get(ts, 0)
        seen_ts_offset[ts] = bump + 1
        final_ts = ts + timedelta(microseconds=bump)

        tx_id = pid(f"synthetic_tx_{idx:06d}", "tx")
        transactions.append(
            TransactionRecord(
                tx_id=tx_id,
                account_id=aid,
                device_id=did,
                merchant_id=mid,
                tx_timestamp=final_ts,
                amount_usd=amount,
                currency="USD",
                channel=channel,
                ip_prefix=ip_prefix,
            )
        )

        # Delayed label maturity: fraud chargebacks arrive in [min, max] days; clean labels mature at max days
        if is_fraud == 1:
            delay_days = int(rng.integers(sc.label_maturity_days_min, sc.label_maturity_days_max + 1))
            source = "chargeback" if delay_days >= 5 else "analyst_review"
        else:
            delay_days = sc.label_maturity_days_max
            source = "matured_clean"

        labels.append(
            FraudLabelRecord(
                tx_id=tx_id,
                is_fraud=is_fraud,
                tx_timestamp=final_ts,
                label_available_at=final_ts + timedelta(days=delay_days),
                label_source=source,  # type: ignore[arg-type]
                fraud_pattern=pattern,
            )
        )

    return DatasetBundle(
        schema_version=SCHEMA_VERSION,
        dataset_id=f"synthetic-graph-fraud-seed-{config.seed}",
        seed=config.seed,
        accounts=accounts,
        devices=devices,
        merchants=merchants,
        transactions=transactions,
        labels=labels,
    )


def build_hand_calculated_reference_bundle(
    salt: str = "synthetic-demo-salt-v1",
    digest_length: int = 12,
) -> Tuple[DatasetBundle, Dict[str, object]]:
    """Build a small deterministic fixture with exact hand-calculated expected graph metrics.

    Scenario timeline (UTC):
    - 3 ring accounts (A1, A2, A3) share device D_RING and hit high-risk merchant M_GIFT.
    - 2 household accounts (H1, H2) share device D_HOME at routine merchant M_GROC.
    - 1 cold-start unseen account (U1) on unseen device D_NEW at M_GIFT.

    Transactions:
    - T1 (Day 1, 10:00): A1 uses D_RING at M_GIFT ($500) -> fraud=1, matures Day 4, 10:00
    - T2 (Day 2, 10:00): A2 uses D_RING at M_GIFT ($450) -> fraud=1, matures Day 10, 10:00 (DELAYED!)
    - T3 (Day 3, 10:00): H1 uses D_HOME at M_GROC ($40)  -> fraud=0, matures Day 10, 10:00
    - T4 (Day 4, 12:00): H2 uses D_HOME at M_GROC ($55)  -> fraud=0, matures Day 11, 12:00
    - T5 (Day 5, 10:00): A3 uses D_RING at M_GIFT ($600) -> target transaction to verify!
        At T5 (Day 5, 10:00):
        - Strictly prior tx on D_RING: T1 (A1) and T2 (A2) => 2 distinct OTHER accounts on D_RING!
        - Prior matured fraud on D_RING: ONLY T1 (matured Day 4 10:00 <= Day 5 10:00).
          T2 matures Day 10 > Day 5, so T2 MUST NOT be counted as matured fraud! => matured_fraud_count == 1.
        - 2-hop ring co-accounts for A3: {A1, A2} => count == 2.
        - Shared merchant M_GIFT with co-accounts: both A1 and A2 used M_GIFT => count == 2.
        - High-risk MCC ratio on D_RING: 2/2 = 1.0.
        - A3 is cold-start unseen (0 prior tx), D_RING is seen (2 prior tx).
    - T6 (Day 6, 10:00): A1 uses D_RING at M_GIFT ($700) -> FUTURE relative to T5! Must NEVER affect T5!
    - T7 (Day 5, 11:00): H1 uses D_HOME at M_GROC ($45) -> legitimate shared device target:
        At T7:
        - Distinct OTHER accounts on D_HOME: {H2} => 1
        - High-risk MCC ratio on D_HOME: 0/2 = 0.0
        - Matured fraud on D_HOME: 0
    - T8 (Day 5, 12:00): U1 uses D_NEW at M_GIFT ($100) -> pure cold-start unseen account & unseen device.
    """
    t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def pid(raw: str, prefix: str) -> str:
        return pseudonymize_id(raw, prefix=prefix, salt=salt, digest_length=digest_length)  # type: ignore[arg-type]

    ids = {
        "A1": pid("ref_acc_1", "acc"),
        "A2": pid("ref_acc_2", "acc"),
        "A3": pid("ref_acc_3", "acc"),
        "H1": pid("ref_acc_h1", "acc"),
        "H2": pid("ref_acc_h2", "acc"),
        "U1": pid("ref_acc_u1", "acc"),
        "D_RING": pid("ref_dev_ring", "dev"),
        "D_HOME": pid("ref_dev_home", "dev"),
        "D_NEW": pid("ref_dev_new", "dev"),
        "M_GIFT": pid("ref_mer_gift", "mer"),
        "M_GROC": pid("ref_mer_groc", "mer"),
        "T1": pid("ref_tx_1", "tx"),
        "T2": pid("ref_tx_2", "tx"),
        "T3": pid("ref_tx_3", "tx"),
        "T4": pid("ref_tx_4", "tx"),
        "T5": pid("ref_tx_5", "tx"),
        "T6": pid("ref_tx_6", "tx"),
        "T7": pid("ref_tx_7", "tx"),
        "T8": pid("ref_tx_8", "tx"),
    }

    accounts = [
        AccountRecord(account_id=ids["A1"], created_at=t0 - timedelta(days=5), account_age_days_at_start=5.0, kyc_tier="basic", home_region="US-EAST", is_synthetic_ring_member=True),
        AccountRecord(account_id=ids["A2"], created_at=t0 - timedelta(days=4), account_age_days_at_start=4.0, kyc_tier="basic", home_region="US-EAST", is_synthetic_ring_member=True),
        AccountRecord(account_id=ids["A3"], created_at=t0 - timedelta(days=2), account_age_days_at_start=2.0, kyc_tier="basic", home_region="US-EAST", is_synthetic_ring_member=True),
        AccountRecord(account_id=ids["H1"], created_at=t0 - timedelta(days=300), account_age_days_at_start=300.0, kyc_tier="enhanced", home_region="US-WEST", is_shared_household_member=True),
        AccountRecord(account_id=ids["H2"], created_at=t0 - timedelta(days=280), account_age_days_at_start=280.0, kyc_tier="standard", home_region="US-WEST", is_shared_household_member=True),
        AccountRecord(account_id=ids["U1"], created_at=t0 + timedelta(days=4), account_age_days_at_start=0.0, kyc_tier="basic", home_region="US-MID"),
    ]
    devices = [
        DeviceRecord(device_id=ids["D_RING"], first_seen_at=t0, device_type="emulator", is_shared_legitimate=False, device_risk_hint=0.9),
        DeviceRecord(device_id=ids["D_HOME"], first_seen_at=t0 - timedelta(days=250), device_type="desktop_web", is_shared_legitimate=True, device_risk_hint=0.1),
        DeviceRecord(device_id=ids["D_NEW"], first_seen_at=t0 + timedelta(days=5), device_type="mobile_ios", is_shared_legitimate=False, device_risk_hint=0.1),
    ]
    merchants = [
        MerchantRecord(merchant_id=ids["M_GIFT"], onboarded_at=t0 - timedelta(days=60), category="gift_cards", country="US", base_chargeback_rate=0.06),
        MerchantRecord(merchant_id=ids["M_GROC"], onboarded_at=t0 - timedelta(days=400), category="groceries", country="US", base_chargeback_rate=0.005),
    ]

    txs = [
        TransactionRecord(tx_id=ids["T1"], account_id=ids["A1"], device_id=ids["D_RING"], merchant_id=ids["M_GIFT"], tx_timestamp=t0 + timedelta(days=1, hours=10), amount_usd=500.0, channel="mobile"),
        TransactionRecord(tx_id=ids["T2"], account_id=ids["A2"], device_id=ids["D_RING"], merchant_id=ids["M_GIFT"], tx_timestamp=t0 + timedelta(days=2, hours=10), amount_usd=450.0, channel="mobile"),
        TransactionRecord(tx_id=ids["T3"], account_id=ids["H1"], device_id=ids["D_HOME"], merchant_id=ids["M_GROC"], tx_timestamp=t0 + timedelta(days=3, hours=10), amount_usd=40.0, channel="web"),
        TransactionRecord(tx_id=ids["T4"], account_id=ids["H2"], device_id=ids["D_HOME"], merchant_id=ids["M_GROC"], tx_timestamp=t0 + timedelta(days=4, hours=12), amount_usd=55.0, channel="web"),
        TransactionRecord(tx_id=ids["T5"], account_id=ids["A3"], device_id=ids["D_RING"], merchant_id=ids["M_GIFT"], tx_timestamp=t0 + timedelta(days=5, hours=10), amount_usd=600.0, channel="mobile"),
        TransactionRecord(tx_id=ids["T7"], account_id=ids["H1"], device_id=ids["D_HOME"], merchant_id=ids["M_GROC"], tx_timestamp=t0 + timedelta(days=5, hours=11), amount_usd=45.0, channel="web"),
        TransactionRecord(tx_id=ids["T8"], account_id=ids["U1"], device_id=ids["D_NEW"], merchant_id=ids["M_GIFT"], tx_timestamp=t0 + timedelta(days=5, hours=12), amount_usd=100.0, channel="mobile"),
        TransactionRecord(tx_id=ids["T6"], account_id=ids["A1"], device_id=ids["D_RING"], merchant_id=ids["M_GIFT"], tx_timestamp=t0 + timedelta(days=6, hours=10), amount_usd=700.0, channel="mobile"),
    ]
    labels = [
        FraudLabelRecord(tx_id=ids["T1"], is_fraud=1, tx_timestamp=txs[0].tx_timestamp, label_available_at=t0 + timedelta(days=4, hours=10), label_source="chargeback", fraud_pattern="coordinated_fraud_ring"),
        FraudLabelRecord(tx_id=ids["T2"], is_fraud=1, tx_timestamp=txs[1].tx_timestamp, label_available_at=t0 + timedelta(days=10, hours=10), label_source="chargeback", fraud_pattern="coordinated_fraud_ring"),
        FraudLabelRecord(tx_id=ids["T3"], is_fraud=0, tx_timestamp=txs[2].tx_timestamp, label_available_at=t0 + timedelta(days=10, hours=10), label_source="matured_clean", fraud_pattern="legitimate_shared_device"),
        FraudLabelRecord(tx_id=ids["T4"], is_fraud=0, tx_timestamp=txs[3].tx_timestamp, label_available_at=t0 + timedelta(days=11, hours=12), label_source="matured_clean", fraud_pattern="legitimate_shared_device"),
        FraudLabelRecord(tx_id=ids["T5"], is_fraud=1, tx_timestamp=txs[4].tx_timestamp, label_available_at=t0 + timedelta(days=12, hours=10), label_source="chargeback", fraud_pattern="coordinated_fraud_ring"),
        FraudLabelRecord(tx_id=ids["T7"], is_fraud=0, tx_timestamp=txs[5].tx_timestamp, label_available_at=t0 + timedelta(days=12, hours=11), label_source="matured_clean", fraud_pattern="legitimate_shared_device"),
        FraudLabelRecord(tx_id=ids["T8"], is_fraud=0, tx_timestamp=txs[6].tx_timestamp, label_available_at=t0 + timedelta(days=12, hours=12), label_source="matured_clean", fraud_pattern="organic_legitimate"),
        FraudLabelRecord(tx_id=ids["T6"], is_fraud=1, tx_timestamp=txs[7].tx_timestamp, label_available_at=t0 + timedelta(days=13, hours=10), label_source="chargeback", fraud_pattern="coordinated_fraud_ring"),
    ]

    bundle = DatasetBundle(
        schema_version=SCHEMA_VERSION,
        dataset_id="reference-handcalc-v1",
        seed=0,
        accounts=accounts,
        devices=devices,
        merchants=merchants,
        transactions=txs,
        labels=labels,
    )

    expected = {
        "ids": ids,
        "T5_expected": {
            "device_prior_distinct_accounts_30d": 2.0,
            "device_matured_fraud_count": 1.0,  # Only T1 matured; T2 matures on Day 10!
            "two_hop_ring_account_count": 2.0,
            "two_hop_ring_matured_fraud_count": 1.0,
            "shared_merchant_with_co_accounts_30d": 2.0,
            "device_high_risk_mcc_ratio_30d": 1.0,
            "account_unseen_cold_start": 1.0,
            "device_unseen_cold_start": 0.0,
        },
        "T7_expected": {
            "device_prior_distinct_accounts_30d": 1.0,
            "device_matured_fraud_count": 0.0,
            "device_high_risk_mcc_ratio_30d": 0.0,
            "account_unseen_cold_start": 0.0,
            "device_unseen_cold_start": 0.0,
        },
        "T8_expected": {
            "device_prior_distinct_accounts_30d": 0.0,
            "device_matured_fraud_count": 0.0,
            "account_unseen_cold_start": 1.0,
            "device_unseen_cold_start": 1.0,
        },
    }
    return bundle, expected
