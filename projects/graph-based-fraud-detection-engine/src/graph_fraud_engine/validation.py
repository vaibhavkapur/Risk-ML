"""Cross-entity and temporal dataset validation."""

from __future__ import annotations

from typing import List, Set

from graph_fraud_engine import SCHEMA_VERSION
from graph_fraud_engine.schemas import DatasetBundle, SchemaValidationError


def validate_dataset_bundle(bundle: DatasetBundle) -> None:
    """Validate cross-table referential integrity, uniqueness, and temporal consistency.

    Raises SchemaValidationError with an exhaustive list of violations if any rule fails.
    """
    errors: List[str] = []

    if bundle.schema_version != SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema_version '{bundle.schema_version}'; expected '{SCHEMA_VERSION}'"
        )

    account_ids: Set[str] = set()
    for acc in bundle.accounts:
        if acc.account_id in account_ids:
            errors.append(f"Duplicate account_id: {acc.account_id}")
        account_ids.add(acc.account_id)

    device_ids: Set[str] = set()
    for dev in bundle.devices:
        if dev.device_id in device_ids:
            errors.append(f"Duplicate device_id: {dev.device_id}")
        device_ids.add(dev.device_id)

    merchant_ids: Set[str] = set()
    for mer in bundle.merchants:
        if mer.merchant_id in merchant_ids:
            errors.append(f"Duplicate merchant_id: {mer.merchant_id}")
        merchant_ids.add(mer.merchant_id)

    tx_by_id = {}
    for tx in bundle.transactions:
        if tx.tx_id in tx_by_id:
            errors.append(f"Duplicate tx_id: {tx.tx_id}")
        tx_by_id[tx.tx_id] = tx

        if tx.account_id not in account_ids:
            errors.append(f"Transaction {tx.tx_id} references unknown account_id {tx.account_id}")
        if tx.device_id not in device_ids:
            errors.append(f"Transaction {tx.tx_id} references unknown device_id {tx.device_id}")
        if tx.merchant_id not in merchant_ids:
            errors.append(f"Transaction {tx.tx_id} references unknown merchant_id {tx.merchant_id}")

    seen_label_txs: Set[str] = set()
    for lbl in bundle.labels:
        if lbl.tx_id in seen_label_txs:
            errors.append(f"Duplicate label for tx_id: {lbl.tx_id}")
        seen_label_txs.add(lbl.tx_id)

        tx = tx_by_id.get(lbl.tx_id)
        if tx is None:
            errors.append(f"Label references unknown tx_id: {lbl.tx_id}")
        else:
            if lbl.tx_timestamp != tx.tx_timestamp:
                errors.append(
                    f"Label tx_timestamp mismatch for {lbl.tx_id}: "
                    f"{lbl.tx_timestamp.isoformat()} != {tx.tx_timestamp.isoformat()}"
                )
            if lbl.label_available_at < tx.tx_timestamp:
                errors.append(
                    f"Label for {lbl.tx_id} matures before transaction timestamp"
                )

    if errors:
        raise SchemaValidationError(
            f"Dataset validation failed with {len(errors)} error(s): {errors[0]}",
            errors=errors,
        )
