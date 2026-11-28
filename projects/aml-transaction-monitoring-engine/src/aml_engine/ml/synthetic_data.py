"""Deterministic seeded synthetic AML dataset generator.

Creates a multi-month synthetic account graph containing:
- Legitimate retail transfers, high-volume merchants, payroll runs, and treasury sweeps
- Borderline benign activity (e.g., occasional large legitimate bursts / real-estate deposits)
  that can trigger naive threshold rules (realistic false positives for ML prioritization)
- Five labeled suspicious typologies:
  STRUCTURING_SMURFING, RAPID_LAYERING_CHAIN, ROUND_TRIP_CYCLE,
  MULE_FAN_IN_FAN_OUT, and DORMANT_ACCOUNT_SPIKE.

100% synthetic data; zero real customer records or transactions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import random
from typing import Any, Dict, List


def generate_synthetic_dataset(
    seed: int = 42,
    num_days: int = 90,
    num_retail_accounts: int = 60,
    num_suspicious_episodes: int = 45,
) -> Dict[str, Any]:
    """Generate a reproducible synthetic dataset spanning `num_days` in UTC."""
    rng = random.Random(seed)
    start_dt = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)

    entities: List[Dict[str, Any]] = []
    accounts: List[Dict[str, Any]] = []
    transactions: List[Dict[str, Any]] = []
    tx_counter = 1

    def next_tx_id() -> str:
        nonlocal tx_counter
        tid = f"TX-SYN-{tx_counter:06d}"
        tx_counter += 1
        return tid

    def iso(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Create legitimate merchants, payroll, treasury, and retail population
    merchant_accs: List[str] = []
    for i in range(4):
        eid = f"ENT-SYN-MERCH-{i:02d}"
        aid = f"ACC-SYN-MERCH-{i:02d}"
        entities.append({"entity_id": eid, "name": f"Synthetic Merchant {i}", "risk_tier": "LOW", "jurisdiction": "US"})
        accounts.append({
            "account_id": aid,
            "entity_id": eid,
            "account_type": "MERCHANT",
            "opened_at": "2023-01-01T00:00:00Z",
            "baseline_monthly_usd": 200000.0,
        })
        merchant_accs.append(aid)

    payroll_eid = "ENT-SYN-PAYROLL-00"
    payroll_aid = "ACC-SYN-PAYROLL-00"
    treasury_aid = "ACC-SYN-TREASURY-00"
    entities.append({"entity_id": payroll_eid, "name": "Synthetic Payroll Corp", "risk_tier": "LOW", "jurisdiction": "US"})
    accounts.append({
        "account_id": payroll_aid,
        "entity_id": payroll_eid,
        "account_type": "PAYROLL_DISBURSEMENT",
        "opened_at": "2022-06-01T00:00:00Z",
        "baseline_monthly_usd": 350000.0,
    })
    accounts.append({
        "account_id": treasury_aid,
        "entity_id": payroll_eid,
        "account_type": "TREASURY",
        "opened_at": "2022-06-01T00:00:00Z",
        "baseline_monthly_usd": 600000.0,
    })

    retail_accs: List[str] = []
    for i in range(num_retail_accounts):
        eid = f"ENT-SYN-RET-{i:03d}"
        aid = f"ACC-SYN-RET-{i:03d}"
        tier = "LOW" if rng.random() < 0.85 else "MEDIUM"
        entities.append({"entity_id": eid, "name": f"Synthetic Retail {i}", "risk_tier": tier, "jurisdiction": "US"})
        accounts.append({
            "account_id": aid,
            "entity_id": eid,
            "account_type": "RETAIL",
            "opened_at": "2024-01-15T00:00:00Z",
            "baseline_monthly_usd": round(rng.uniform(4000.0, 14000.0), 2),
        })
        retail_accs.append(aid)

    # Create higher-risk corporate/shell accounts for suspicious typologies
    susp_accs: List[str] = []
    for i in range(30):
        eid = f"ENT-SYN-CORP-{i:03d}"
        aid = f"ACC-SYN-CORP-{i:03d}"
        tier = "HIGH" if i % 2 == 0 else "MEDIUM"
        juris = rng.choice(["PA", "CY", "VG", "HK", "SG", "US"])
        entities.append({"entity_id": eid, "name": f"Synthetic Holdings {i}", "risk_tier": tier, "jurisdiction": juris})
        accounts.append({
            "account_id": aid,
            "entity_id": eid,
            "account_type": "CORPORATE" if i % 3 != 0 else "RETAIL",
            "opened_at": "2025-09-01T00:00:00Z",
            "baseline_monthly_usd": round(rng.uniform(1500.0, 4500.0), 2),
        })
        susp_accs.append(aid)

    # 2. Generate background benign traffic across all days
    for day in range(num_days):
        day_dt = start_dt + timedelta(days=day)

        # Normal retail -> merchant card settlements & retail <-> retail ACH
        for _ in range(18):
            src = rng.choice(retail_accs)
            if rng.random() < 0.55:
                tgt = rng.choice(merchant_accs)
                chan = "CARD_SETTLEMENT"
                amt = round(rng.uniform(80.0, 2200.0), 2)
            else:
                tgt = rng.choice(retail_accs)
                if tgt == src:
                    continue
                chan = "ACH"
                amt = round(rng.uniform(120.0, 3200.0), 2)
            ts = day_dt + timedelta(minutes=rng.randint(0, 720))
            transactions.append({
                "tx_id": next_tx_id(),
                "source_account_id": src,
                "target_account_id": tgt,
                "amount": amt,
                "currency": rng.choice(["USD", "USD", "USD", "EUR", "GBP"]),
                "timestamp": iso(ts),
                "channel": chan,
                "scenario_label": "BENIGN",
            })

        # Merchant end-of-day sweep to treasury
        for m_acc in merchant_accs:
            sweep_ts = day_dt + timedelta(hours=11, minutes=rng.randint(0, 45))
            transactions.append({
                "tx_id": next_tx_id(),
                "source_account_id": m_acc,
                "target_account_id": treasury_aid,
                "amount": round(rng.uniform(8000.0, 28000.0), 2),
                "currency": "USD",
                "timestamp": iso(sweep_ts),
                "channel": "INTERNAL_TRANSFER",
                "scenario_label": "BENIGN",
            })

        # Bi-weekly payroll disbursement
        if day % 14 == 4:
            for emp in retail_accs[:12]:
                ts = day_dt + timedelta(hours=5, minutes=rng.randint(0, 10))
                transactions.append({
                    "tx_id": next_tx_id(),
                    "source_account_id": payroll_aid,
                    "target_account_id": emp,
                    "amount": round(rng.uniform(7600.0, 9400.0), 2),
                    "currency": "USD",
                    "timestamp": iso(ts),
                    "channel": "PAYROLL",
                    "scenario_label": "BENIGN",
                })

        # Borderline benign false-positive decoys (e.g., legitimate low-risk retail multi-deposit or home downpayment)
        if day % 5 == 2:
            benign_target = retail_accs[day % 20]
            base_t = day_dt + timedelta(hours=1)
            for k in range(3):
                src = retail_accs[(day + k + 25) % len(retail_accs)]
                if src == benign_target:
                    continue
                transactions.append({
                    "tx_id": next_tx_id(),
                    "source_account_id": src,
                    "target_account_id": benign_target,
                    "amount": round(rng.uniform(7600.0, 8400.0), 2),
                    "currency": "USD",
                    "timestamp": iso(base_t + timedelta(hours=k * 3)),
                    "channel": "ACH",
                    "scenario_label": "BENIGN",
                })

        # Occasional dormant legitimate retail spike (low-risk account receiving inheritance/bonus)
        if day >= 38 and day % 11 == 0:
            dorm_benign = retail_accs[40 + (day % 15)]
            transactions.append({
                "tx_id": next_tx_id(),
                "source_account_id": retail_accs[0],
                "target_account_id": dorm_benign,
                "amount": 350.0,
                "currency": "USD",
                "timestamp": iso(start_dt + timedelta(days=day - 35)),
                "channel": "ACH",
                "scenario_label": "BENIGN",
            })
            transactions.append({
                "tx_id": next_tx_id(),
                "source_account_id": treasury_aid,
                "target_account_id": dorm_benign,
                "amount": round(rng.uniform(21000.0, 26000.0), 2),
                "currency": "USD",
                "timestamp": iso(day_dt + timedelta(hours=4)),
                "channel": "ACH",
                "scenario_label": "BENIGN",
            })

    # 3. Inject labeled suspicious episodes evenly across the timeline for temporal splits
    typologies = [
        "STRUCTURING_SMURFING",
        "RAPID_LAYERING_CHAIN",
        "ROUND_TRIP_CYCLE",
        "MULE_FAN_IN_FAN_OUT",
        "DORMANT_ACCOUNT_SPIKE",
    ]

    for ep in range(num_suspicious_episodes):
        day_offset = int((ep + 1) * (num_days - 8) / (num_suspicious_episodes + 1)) + 3
        ep_dt = start_dt + timedelta(days=day_offset, hours=rng.randint(1, 6))
        typ = typologies[ep % len(typologies)]

        if typ == "STRUCTURING_SMURFING":
            target = susp_accs[ep % len(susp_accs)]
            senders = rng.sample(retail_accs, 4)
            for k, src in enumerate(senders):
                transactions.append({
                    "tx_id": next_tx_id(),
                    "source_account_id": src,
                    "target_account_id": target,
                    "amount": round(rng.uniform(8600.0, 9850.0), 2),
                    "currency": "USD",
                    "timestamp": iso(ep_dt + timedelta(hours=k * 2.5)),
                    "channel": "CASH_DEPOSIT",
                    "scenario_label": "STRUCTURING_SMURFING",
                })

        elif typ == "RAPID_LAYERING_CHAIN":
            chain_nodes = rng.sample(susp_accs, 4)
            amt = round(rng.uniform(25000.0, 65000.0), 2)
            cur_dt = ep_dt
            for h in range(3):
                transactions.append({
                    "tx_id": next_tx_id(),
                    "source_account_id": chain_nodes[h],
                    "target_account_id": chain_nodes[h + 1],
                    "amount": round(amt, 2),
                    "currency": "USD",
                    "timestamp": iso(cur_dt),
                    "channel": "WIRE",
                    "scenario_label": "RAPID_LAYERING_CHAIN",
                })
                amt *= rng.uniform(0.95, 0.99)
                cur_dt += timedelta(minutes=rng.randint(45, 120))

        elif typ == "ROUND_TRIP_CYCLE":
            cyc_nodes = rng.sample(susp_accs, 3)
            amt = round(rng.uniform(22000.0, 50000.0), 2)
            cur_dt = ep_dt
            edges = [
                (cyc_nodes[0], cyc_nodes[1]),
                (cyc_nodes[1], cyc_nodes[2]),
                (cyc_nodes[2], cyc_nodes[0]),
            ]
            for u, v in edges:
                transactions.append({
                    "tx_id": next_tx_id(),
                    "source_account_id": u,
                    "target_account_id": v,
                    "amount": round(amt, 2),
                    "currency": "USD",
                    "timestamp": iso(cur_dt),
                    "channel": "WIRE",
                    "scenario_label": "ROUND_TRIP_CYCLE",
                })
                amt *= rng.uniform(0.95, 0.985)
                cur_dt += timedelta(hours=rng.randint(2, 6))

        elif typ == "MULE_FAN_IN_FAN_OUT":
            hub = susp_accs[(ep * 3) % len(susp_accs)]
            exit_acc = susp_accs[(ep * 3 + 7) % len(susp_accs)]
            feeders = rng.sample(retail_accs, 5)
            total_in = 0.0
            for k, src in enumerate(feeders):
                amt = round(rng.uniform(4200.0, 6800.0), 2)
                total_in += amt
                transactions.append({
                    "tx_id": next_tx_id(),
                    "source_account_id": src,
                    "target_account_id": hub,
                    "amount": amt,
                    "currency": "USD",
                    "timestamp": iso(ep_dt + timedelta(hours=k * 1.2)),
                    "channel": "ACH",
                    "scenario_label": "MULE_FAN_IN_FAN_OUT",
                })
            transactions.append({
                "tx_id": next_tx_id(),
                "source_account_id": hub,
                "target_account_id": exit_acc,
                "amount": round(total_in * rng.uniform(0.92, 0.98), 2),
                "currency": "USD",
                "timestamp": iso(ep_dt + timedelta(hours=8)),
                "channel": "WIRE",
                "scenario_label": "MULE_FAN_IN_FAN_OUT",
            })

        elif typ == "DORMANT_ACCOUNT_SPIKE":
            # Ensure a clean 35+ day gap prior to burst for a dedicated dormant synthetic account
            d_eid = f"ENT-SYN-DORM-{ep:03d}"
            d_aid = f"ACC-SYN-DORM-{ep:03d}"
            entities.append({"entity_id": d_eid, "name": f"Synthetic Dormant {ep}", "risk_tier": "HIGH", "jurisdiction": "PA"})
            accounts.append({
                "account_id": d_aid,
                "entity_id": d_eid,
                "account_type": "RETAIL",
                "opened_at": "2024-01-01T00:00:00Z",
                "baseline_monthly_usd": 1200.0,
            })
            transactions.append({
                "tx_id": next_tx_id(),
                "source_account_id": retail_accs[ep % len(retail_accs)],
                "target_account_id": d_aid,
                "amount": 180.0,
                "currency": "USD",
                "timestamp": iso(ep_dt - timedelta(days=40)),
                "channel": "ACH",
                "scenario_label": "BENIGN",
            })
            transactions.append({
                "tx_id": next_tx_id(),
                "source_account_id": susp_accs[ep % len(susp_accs)],
                "target_account_id": d_aid,
                "amount": round(rng.uniform(28000.0, 55000.0), 2),
                "currency": "USD",
                "timestamp": iso(ep_dt),
                "channel": "WIRE",
                "scenario_label": "DORMANT_ACCOUNT_SPIKE",
            })

    transactions.sort(key=lambda t: (t["timestamp"], t["tx_id"]))
    return {
        "dataset_id": f"synthetic_aml_seed{seed}_d{num_days}",
        "schema_version": "1.0.0",
        "seed": seed,
        "entities": entities,
        "accounts": accounts,
        "transactions": transactions,
    }
