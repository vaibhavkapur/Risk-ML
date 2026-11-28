"""Observation snapshots, label maturity windows, and temporal holdout guards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import polars as pl

from chargeback_engine.config import MaturityConfig, SplitConfig
from chargeback_engine.contracts import (
    ContractValidationError,
    DisputeRecord,
    DuplicateDisputeError,
    LabelStatus,
    MerchantRecord,
    ReasonCategory,
    TemporalLeakageError,
    TransactionRecord,
    normalize_reason_code,
)


@dataclass(frozen=True)
class MaturityDiagnostics:
    as_of_ts: str
    maturity_mode: str
    total_transactions: int
    excluded_future_transactions: int
    total_disputes_input: int
    disputes_after_as_of_masked: int
    duplicate_disputes_deduplicated: int
    missing_reason_codes: int
    unrecognized_reason_codes: int
    mature_positive_count: int
    mature_negative_count: int
    immature_positive_count: int
    immature_negative_count: int
    trainable_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "as_of_ts": self.as_of_ts,
            "maturity_mode": self.maturity_mode,
            "total_transactions": self.total_transactions,
            "excluded_future_transactions": self.excluded_future_transactions,
            "total_disputes_input": self.total_disputes_input,
            "disputes_after_as_of_masked": self.disputes_after_as_of_masked,
            "duplicate_disputes_deduplicated": self.duplicate_disputes_deduplicated,
            "missing_reason_codes": self.missing_reason_codes,
            "unrecognized_reason_codes": self.unrecognized_reason_codes,
            "mature_positive_count": self.mature_positive_count,
            "mature_negative_count": self.mature_negative_count,
            "immature_positive_count": self.immature_positive_count,
            "immature_negative_count": self.immature_negative_count,
            "trainable_count": self.trainable_count,
        }


def compute_maturity_date(
    tx: TransactionRecord, default_maturity_days: int
) -> datetime:
    """Return explicit per-transaction maturity_date or settlement_ts + default_maturity_days."""
    if tx.explicit_maturity_date is not None:
        return tx.explicit_maturity_date
    return tx.settlement_ts + timedelta(days=default_maturity_days)


def resolve_observable_disputes(
    disputes: Sequence[DisputeRecord],
    as_of_ts: datetime,
    duplicate_policy: str = "raise",
) -> Tuple[Dict[str, DisputeRecord], int, int]:
    """Filter disputes to those observable at as_of_ts and enforce duplicate policy.

    Note: Duplicate detection runs on all input disputes referencing the same
    transaction_id so contract violations are never silently hidden by cutoff timing.
    """
    grouped_all: Dict[str, List[DisputeRecord]] = {}
    for d in disputes:
        grouped_all.setdefault(d.transaction_id, []).append(d)

    duplicate_tx_ids = [
        tx_id for tx_id, recs in grouped_all.items() if len(recs) > 1
    ]
    if duplicate_tx_ids and duplicate_policy == "raise":
        raise DuplicateDisputeError(duplicate_tx_ids)

    deduplicated_count = 0
    resolved: Dict[str, DisputeRecord] = {}
    masked_future_count = 0

    for tx_id, recs in grouped_all.items():
        # Sort deterministically by (dispute_ts, dispute_id)
        ordered = sorted(recs, key=lambda r: (r.dispute_ts, r.dispute_id))
        if len(ordered) > 1:
            deduplicated_count += len(ordered) - 1
        chosen = ordered[0]
        if chosen.dispute_ts > as_of_ts:
            masked_future_count += 1
            continue
        resolved[tx_id] = chosen

    return resolved, masked_future_count, deduplicated_count


def assign_maturity_labels(
    merchants: Sequence[MerchantRecord],
    transactions: Sequence[TransactionRecord],
    disputes: Sequence[DisputeRecord],
    as_of_ts: datetime,
    maturity_cfg: MaturityConfig,
    missing_reason_code_label: str = "UNKNOWN",
) -> Tuple[pl.DataFrame, MaturityDiagnostics]:
    """Build point-in-time observation table with strict label maturity semantics.

    Invariants:
      1. Transactions with transaction_ts > as_of_ts are excluded (cannot exist at snapshot).
      2. Disputes with dispute_ts > as_of_ts are masked (unobserved future events).
      3. Immature negatives (as_of_ts < maturity_date and no observed dispute) ALWAYS have
         label=None and is_trainable=False.
      4. In 'cohort_mature' mode (default), immature positives also have is_trainable=False
         so training/calibration prevalence is not distorted by right-censoring, while still
         preserving label=1 and status='immature_positive' for delayed-outcome monitoring.
    """
    merchant_map = {m.merchant_id: m for m in merchants}
    observable_disputes, masked_future_disputes, dup_deduped = resolve_observable_disputes(
        disputes=disputes,
        as_of_ts=as_of_ts,
        duplicate_policy=maturity_cfg.duplicate_dispute_policy,
    )

    rows: List[Dict[str, object]] = []
    excluded_future_tx = 0
    missing_reason_count = 0
    unrecognized_reason_count = 0
    counts = {
        LabelStatus.MATURE_POSITIVE: 0,
        LabelStatus.MATURE_NEGATIVE: 0,
        LabelStatus.IMMATURE_POSITIVE: 0,
        LabelStatus.IMMATURE_NEGATIVE: 0,
    }

    for tx in sorted(transactions, key=lambda t: (t.transaction_ts, t.transaction_id)):
        if tx.transaction_ts > as_of_ts:
            excluded_future_tx += 1
            continue

        maturity_date = compute_maturity_date(tx, maturity_cfg.default_maturity_days)
        is_mature = as_of_ts >= maturity_date
        days_since_settlement = (as_of_ts - tx.settlement_ts).total_seconds() / 86400.0
        dispute = observable_disputes.get(tx.transaction_id)

        has_observed_dispute = dispute is not None
        dispute_ts: Optional[datetime] = None
        dispute_delay_days: Optional[float] = None
        dispute_amount_usd: Optional[float] = None
        reason_code: Optional[str] = None
        reason_category: str = ReasonCategory.UNKNOWN.value
        is_missing_reason = False
        is_unrecognized_reason = False

        if dispute is not None:
            dispute_ts = dispute.dispute_ts
            dispute_delay_days = (
                dispute.dispute_ts - tx.settlement_ts
            ).total_seconds() / 86400.0
            dispute_amount_usd = dispute.dispute_amount_usd
            (
                reason_code,
                cat_enum,
                is_missing_reason,
                is_unrecognized_reason,
            ) = normalize_reason_code(
                dispute.reason_code, missing_label=missing_reason_code_label
            )
            reason_category = cat_enum.value
            if is_missing_reason:
                missing_reason_count += 1
            if is_unrecognized_reason:
                unrecognized_reason_count += 1

        if is_mature and has_observed_dispute:
            status = LabelStatus.MATURE_POSITIVE
            label: Optional[int] = 1
            is_trainable = True
        elif is_mature and not has_observed_dispute:
            status = LabelStatus.MATURE_NEGATIVE
            label = 0
            is_trainable = True
        elif not is_mature and has_observed_dispute:
            status = LabelStatus.IMMATURE_POSITIVE
            label = 1
            is_trainable = (
                maturity_cfg.maturity_mode == "exclude_immature_negatives_only"
            )
        else:
            status = LabelStatus.IMMATURE_NEGATIVE
            label = None
            is_trainable = False

        counts[status] += 1
        merchant = merchant_map.get(tx.merchant_id)
        rows.append(
            {
                "transaction_id": tx.transaction_id,
                "merchant_id": tx.merchant_id,
                "mcc_group": merchant.mcc_group if merchant else "unknown",
                "merchant_country": merchant.country if merchant else "XX",
                "merchant_onboarding_ts": (
                    merchant.onboarding_ts if merchant else tx.transaction_ts
                ),
                "avg_fulfillment_days": (
                    merchant.avg_fulfillment_days if merchant else 3.0
                ),
                "transaction_ts": tx.transaction_ts,
                "settlement_ts": tx.settlement_ts,
                "maturity_date": maturity_date,
                "days_since_settlement": round(days_since_settlement, 6),
                "amount": tx.amount,
                "currency": tx.currency,
                "amount_usd": tx.amount_usd,
                "channel": tx.channel.value,
                "is_cross_border": tx.is_cross_border,
                "has_3ds": tx.has_3ds,
                "has_avs_match": tx.has_avs_match,
                "customer_tenure_days": tx.customer_tenure_days,
                "is_mature": is_mature,
                "has_observed_dispute": has_observed_dispute,
                "dispute_ts": dispute_ts,
                "dispute_delay_days": (
                    round(dispute_delay_days, 6)
                    if dispute_delay_days is not None
                    else None
                ),
                "dispute_amount_usd": dispute_amount_usd,
                "reason_code": reason_code,
                "reason_category": reason_category,
                "is_missing_reason_code": is_missing_reason,
                "is_unrecognized_reason_code": is_unrecognized_reason,
                "label_status": status.value,
                "label": label,
                "is_trainable": is_trainable,
            }
        )

    schema = {
        "transaction_id": pl.Utf8,
        "merchant_id": pl.Utf8,
        "mcc_group": pl.Utf8,
        "merchant_country": pl.Utf8,
        "merchant_onboarding_ts": pl.Datetime("us", "UTC"),
        "avg_fulfillment_days": pl.Float64,
        "transaction_ts": pl.Datetime("us", "UTC"),
        "settlement_ts": pl.Datetime("us", "UTC"),
        "maturity_date": pl.Datetime("us", "UTC"),
        "days_since_settlement": pl.Float64,
        "amount": pl.Float64,
        "currency": pl.Utf8,
        "amount_usd": pl.Float64,
        "channel": pl.Utf8,
        "is_cross_border": pl.Boolean,
        "has_3ds": pl.Boolean,
        "has_avs_match": pl.Boolean,
        "customer_tenure_days": pl.Int64,
        "is_mature": pl.Boolean,
        "has_observed_dispute": pl.Boolean,
        "dispute_ts": pl.Datetime("us", "UTC"),
        "dispute_delay_days": pl.Float64,
        "dispute_amount_usd": pl.Float64,
        "reason_code": pl.Utf8,
        "reason_category": pl.Utf8,
        "is_missing_reason_code": pl.Boolean,
        "is_unrecognized_reason_code": pl.Boolean,
        "label_status": pl.Utf8,
        "label": pl.Int8,
        "is_trainable": pl.Boolean,
    }
    df = pl.DataFrame(rows, schema=schema)

    trainable_count = sum(1 for r in rows if r["is_trainable"])
    diag = MaturityDiagnostics(
        as_of_ts=as_of_ts.isoformat(),
        maturity_mode=maturity_cfg.maturity_mode,
        total_transactions=len(rows),
        excluded_future_transactions=excluded_future_tx,
        total_disputes_input=len(disputes),
        disputes_after_as_of_masked=masked_future_disputes,
        duplicate_disputes_deduplicated=dup_deduped,
        missing_reason_codes=missing_reason_count,
        unrecognized_reason_codes=unrecognized_reason_count,
        mature_positive_count=counts[LabelStatus.MATURE_POSITIVE],
        mature_negative_count=counts[LabelStatus.MATURE_NEGATIVE],
        immature_positive_count=counts[LabelStatus.IMMATURE_POSITIVE],
        immature_negative_count=counts[LabelStatus.IMMATURE_NEGATIVE],
        trainable_count=trainable_count,
    )
    return df, diag


def assign_temporal_splits(
    labeled_df: pl.DataFrame, split_cfg: SplitConfig
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split trainable (mature) records chronologically into train / calibration / test.

    Strictly enforces non-overlapping transaction timestamps across splits and
    verifies no immature negatives enter any split.
    """
    if labeled_df.is_empty():
        raise ContractValidationError("Cannot split an empty labeled dataframe")

    immature_neg_trainable = labeled_df.filter(
        pl.col("is_trainable") & (pl.col("label_status") == LabelStatus.IMMATURE_NEGATIVE.value)
    )
    if immature_neg_trainable.height > 0:
        raise TemporalLeakageError(
            "Invariant violation: immature negative records marked trainable"
        )

    trainable = labeled_df.filter(pl.col("is_trainable")).sort(
        ["transaction_ts", "transaction_id"]
    )
    n = trainable.height
    if n < 6:
        raise ContractValidationError(
            f"Insufficient trainable mature records for temporal split: {n} (need >= 6)"
        )

    train_end_idx = max(2, int(n * split_cfg.train_fraction))
    cal_end_idx = max(
        train_end_idx + 2,
        int(n * (split_cfg.train_fraction + split_cfg.calibration_fraction)),
    )
    if cal_end_idx >= n - 1:
        raise ContractValidationError(
            f"Temporal split leaves too few test records (n={n}, cal_end={cal_end_idx})"
        )

    train_df = trainable.slice(0, train_end_idx)
    cal_df = trainable.slice(train_end_idx, cal_end_idx - train_end_idx)
    test_df = trainable.slice(cal_end_idx, n - cal_end_idx)

    if split_cfg.embargo_days > 0:
        embargo = timedelta(days=split_cfg.embargo_days)
        train_max_ts = train_df["transaction_ts"].max()
        cal_df = cal_df.filter(pl.col("transaction_ts") > train_max_ts + embargo)
        if cal_df.is_empty():
            raise TemporalLeakageError(
                "Embargo eliminated all calibration records; reduce embargo_days or expand span"
            )
        cal_max_ts = cal_df["transaction_ts"].max()
        test_df = test_df.filter(pl.col("transaction_ts") > cal_max_ts + embargo)
        if test_df.is_empty():
            raise TemporalLeakageError(
                "Embargo eliminated all test records; reduce embargo_days or expand span"
            )

    validate_temporal_holdout(train_df, cal_df, test_df)
    return train_df, cal_df, test_df


def validate_temporal_holdout(
    train_df: pl.DataFrame,
    cal_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> None:
    """Assert strict chronological ordering and disjointness across temporal splits."""
    for name, frame in [("train", train_df), ("calibration", cal_df), ("test", test_df)]:
        if frame.is_empty():
            raise TemporalLeakageError(f"Temporal split '{name}' is empty")
        if frame.filter(pl.col("label").is_null()).height > 0:
            raise TemporalLeakageError(
                f"Split '{name}' contains null/immature labels"
            )

    train_max = train_df["transaction_ts"].max()
    cal_min = cal_df["transaction_ts"].min()
    cal_max = cal_df["transaction_ts"].max()
    test_min = test_df["transaction_ts"].min()

    if train_max > cal_min:
        raise TemporalLeakageError(
            f"Temporal leakage: train max ({train_max}) > calibration min ({cal_min})"
        )
    if cal_max > test_min:
        raise TemporalLeakageError(
            f"Temporal leakage: calibration max ({cal_max}) > test min ({test_min})"
        )

    train_ids = set(train_df["transaction_id"].to_list())
    cal_ids = set(cal_df["transaction_id"].to_list())
    test_ids = set(test_df["transaction_id"].to_list())
    if (train_ids & cal_ids) or (train_ids & test_ids) or (cal_ids & test_ids):
        raise TemporalLeakageError("Duplicate transaction_id across temporal splits")
