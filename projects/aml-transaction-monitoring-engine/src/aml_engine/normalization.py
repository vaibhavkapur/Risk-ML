"""Transaction normalization, FX conversion, entity resolution, and deduplication.

Pure domain module isolated from I/O. Handles missing entity mappings explicitly
with diagnostic statuses rather than silently dropping or fabricating links.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aml_engine.config import EngineConfig
from aml_engine.models import (
    Account,
    Entity,
    EntityMappingStatus,
    NormalizedTransaction,
    RawTransaction,
)
from aml_engine.validation import (
    AMLEngineError,
    format_utc_iso,
    parse_utc_timestamp,
    validate_transaction_dict,
)


@dataclass
class NormalizationBatchResult:
    normalized_transactions: List[NormalizedTransaction]
    duplicate_tx_ids: List[str]
    missing_entity_tx_ids: List[str]
    rejected_events: List[Dict[str, str]]
    warnings: List[str]


def compute_canonical_event_hash(tx: RawTransaction) -> str:
    payload = {
        "tx_id": tx.tx_id,
        "source_account_id": tx.source_account_id,
        "target_account_id": tx.target_account_id,
        "amount": round(tx.amount, 6),
        "currency": tx.currency,
        "timestamp": tx.timestamp,
        "channel": tx.channel,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_entity_mapping(
    source_account_id: str,
    target_account_id: str,
    accounts_by_id: Dict[str, Account],
    entities_by_id: Dict[str, Entity],
) -> Tuple[
    Optional[str],
    Optional[str],
    str,
    str,
    str,
    str,
    EntityMappingStatus,
    Optional[str],
]:
    src_acc = accounts_by_id.get(source_account_id)
    tgt_acc = accounts_by_id.get(target_account_id)

    src_ent = entities_by_id.get(src_acc.entity_id) if src_acc else None
    tgt_ent = entities_by_id.get(tgt_acc.entity_id) if tgt_acc else None

    issues: List[str] = []
    status = EntityMappingStatus.MAPPED

    if src_acc is None:
        issues.append(f"missing_source_account:{source_account_id}")
        status = EntityMappingStatus.MISSING_SOURCE_ACCOUNT
    elif src_ent is None:
        issues.append(f"missing_source_entity:{src_acc.entity_id}")
        status = EntityMappingStatus.MISSING_SOURCE_ENTITY

    if tgt_acc is None:
        issues.append(f"missing_target_account:{target_account_id}")
        status = (
            EntityMappingStatus.PARTIAL_OR_UNMAPPED
            if status != EntityMappingStatus.MAPPED
            else EntityMappingStatus.MISSING_TARGET_ACCOUNT
        )
    elif tgt_ent is None:
        issues.append(f"missing_target_entity:{tgt_acc.entity_id}")
        status = (
            EntityMappingStatus.PARTIAL_OR_UNMAPPED
            if status != EntityMappingStatus.MAPPED
            else EntityMappingStatus.MISSING_TARGET_ENTITY
        )

    warning = "; ".join(issues) if issues else None
    return (
        src_ent.entity_id if src_ent else None,
        tgt_ent.entity_id if tgt_ent else None,
        src_acc.account_type if src_acc else "UNKNOWN",
        tgt_acc.account_type if tgt_acc else "UNKNOWN",
        src_ent.risk_tier if src_ent else "UNKNOWN",
        tgt_ent.risk_tier if tgt_ent else "UNKNOWN",
        status,
        warning,
    )


def normalize_transactions(
    raw_transactions: Sequence[Dict[str, Any]],
    accounts_by_id: Dict[str, Account],
    entities_by_id: Dict[str, Entity],
    config: EngineConfig,
    strict: bool = False,
) -> NormalizationBatchResult:
    """Validate, deduplicate, convert currency to USD, and enrich with entity mapping.

    Output transactions are sorted deterministically by (timestamp_utc, tx_id).
    When strict=True, raises immediately on the first invalid/unsupported event;
    when strict=False, records rejected events explicitly in `rejected_events`.
    """
    normalized: List[NormalizedTransaction] = []
    seen_tx_ids: Dict[str, str] = {}
    duplicate_tx_ids: List[str] = []
    missing_entity_tx_ids: List[str] = []
    rejected_events: List[Dict[str, str]] = []
    warnings: List[str] = []

    fx_rates = config.fx_rates_to_usd
    supported_currencies = config.supported_currencies
    supported_channels = config.supported_channels

    for idx, raw_item in enumerate(raw_transactions):
        try:
            tx = validate_transaction_dict(
                raw_item,
                supported_currencies=supported_currencies,
                supported_channels=supported_channels,
            )
        except AMLEngineError as exc:
            if strict:
                raise
            tx_ref = str(raw_item.get("tx_id", f"index:{idx}")) if isinstance(raw_item, dict) else f"index:{idx}"
            rejected_events.append(
                {
                    "tx_id": tx_ref,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue

        event_hash = compute_canonical_event_hash(tx)
        if tx.tx_id in seen_tx_ids:
            duplicate_tx_ids.append(tx.tx_id)
            if seen_tx_ids[tx.tx_id] != event_hash:
                warnings.append(
                    f"Conflicting duplicate tx_id={tx.tx_id!r} with differing payload hash; keeping first occurrence."
                )
            continue
        seen_tx_ids[tx.tx_id] = event_hash

        dt_utc = parse_utc_timestamp(tx.timestamp)
        fx_rate = fx_rates[tx.currency]
        amount_usd = round(tx.amount * fx_rate, 6)

        (
            src_ent_id,
            tgt_ent_id,
            src_acc_type,
            tgt_acc_type,
            src_risk,
            tgt_risk,
            mapping_status,
            mapping_warning,
        ) = _resolve_entity_mapping(
            tx.source_account_id,
            tx.target_account_id,
            accounts_by_id,
            entities_by_id,
        )

        if mapping_status != EntityMappingStatus.MAPPED:
            missing_entity_tx_ids.append(tx.tx_id)
            warnings.append(f"tx_id={tx.tx_id}: {mapping_warning}")

        normalized.append(
            NormalizedTransaction(
                tx_id=tx.tx_id,
                canonical_event_hash=event_hash,
                source_account_id=tx.source_account_id,
                target_account_id=tx.target_account_id,
                source_entity_id=src_ent_id,
                target_entity_id=tgt_ent_id,
                source_account_type=src_acc_type,
                target_account_type=tgt_acc_type,
                source_risk_tier=src_risk,
                target_risk_tier=tgt_risk,
                original_amount=tx.amount,
                original_currency=tx.currency,
                fx_rate_to_usd=fx_rate,
                amount_usd=amount_usd,
                timestamp_utc=dt_utc,
                timestamp_iso=format_utc_iso(dt_utc),
                channel=tx.channel,
                entity_mapping_status=mapping_status,
                scenario_label=tx.scenario_label,
            )
        )

    normalized.sort(key=lambda t: (t.timestamp_utc, t.tx_id))
    return NormalizationBatchResult(
        normalized_transactions=normalized,
        duplicate_tx_ids=duplicate_tx_ids,
        missing_entity_tx_ids=missing_entity_tx_ids,
        rejected_events=rejected_events,
        warnings=warnings,
    )
