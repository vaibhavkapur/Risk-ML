#!/usr/bin/env python3
"""Generate deterministic valid, invalid, boundary, and hand-calculated fixtures in data/fixtures/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_fraud_engine.synthetic import build_hand_calculated_reference_bundle  # noqa: E402


def main() -> None:
    fixtures_dir = PROJECT_ROOT / "data" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Valid hand-calculated reference fixture + expected feature oracle
    bundle, expected = build_hand_calculated_reference_bundle()
    (fixtures_dir / "valid_reference_bundle.json").write_text(
        bundle.model_dump_json(indent=2), encoding="utf-8"
    )
    (fixtures_dir / "reference_expected_oracle.json").write_text(
        json.dumps(expected, indent=2), encoding="utf-8"
    )

    # 2. Invalid fixture: PII in ID, naive timestamp, negative amount, label maturing before tx
    invalid_payload = {
        "schema_version": "1.0.0",
        "dataset_id": "invalid-fixture-v1",
        "seed": 1,
        "accounts": [
            {
                "account_id": "user@example.com",
                "created_at": "2025-01-01T00:00:00",
                "account_age_days_at_start": -5.0,
                "kyc_tier": "basic",
                "home_region": "US",
            }
        ],
        "devices": [],
        "merchants": [],
        "transactions": [
            {
                "tx_id": "tx_000000000001",
                "account_id": "acc_000000000001",
                "device_id": "dev_000000000001",
                "merchant_id": "mer_000000000001",
                "tx_timestamp": "2025-01-02T10:00:00Z",
                "amount_usd": -25.0,
                "currency": "BTC",
                "channel": "fax",
            }
        ],
        "labels": [
            {
                "tx_id": "tx_000000000001",
                "is_fraud": 1,
                "tx_timestamp": "2025-01-02T10:00:00Z",
                "label_available_at": "2025-01-01T10:00:00Z",
                "label_source": "chargeback",
            }
        ],
    }
    (fixtures_dir / "invalid_contract_cases.json").write_text(
        json.dumps(invalid_payload, indent=2), encoding="utf-8"
    )

    # 3. Boundary fixture: unseen cold-start transaction and immature delayed-label transaction
    boundary_payload = {
        "description": "Boundary test transactions for unseen cold-start nodes and immature delayed labels",
        "unseen_cold_start_transaction": {
            "tx_id": "tx_9999999999aa",
            "account_id": "acc_9999999999aa",
            "device_id": "dev_9999999999aa",
            "merchant_id": "mer_9999999999aa",
            "tx_timestamp": "2025-04-01T12:00:00Z",
            "amount_usd": 0.01,
            "currency": "USD",
            "channel": "api",
            "ip_prefix": "203.0.113",
        },
        "max_amount_boundary_transaction": {
            "tx_id": "tx_9999999999bb",
            "account_id": expected["ids"]["A3"],
            "device_id": expected["ids"]["D_RING"],
            "merchant_id": expected["ids"]["M_GIFT"],
            "tx_timestamp": "2025-01-06T00:00:00Z",
            "amount_usd": 999999.99,
            "currency": "USD",
            "channel": "mobile",
            "ip_prefix": "198.51.100",
        },
    }
    (fixtures_dir / "boundary_cases.json").write_text(
        json.dumps(boundary_payload, indent=2), encoding="utf-8"
    )
    print(f"Fixtures written to {fixtures_dir}")


if __name__ == "__main__":
    main()
