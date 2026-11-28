"""Review prioritization policy, non-punitive guardrails, and delayed-outcome monitoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from chargeback_engine.config import EconomicsConfig, PolicyConfig
from chargeback_engine.contracts import ConfigError, DisputeRecord
from chargeback_engine.models import compute_expected_loss_vectorized


@dataclass(frozen=True)
class ReviewDecision:
    transaction_id: str
    merchant_id: str
    action: str  # "manual_review" | "approve" (NEVER automated merchant punishment)
    rank: int
    calibrated_prob: float
    amount_usd: float
    expected_loss_usd: float
    expected_net_savings_usd: float
    merchant_advisory_flag: bool


def prioritize_reviews(
    scored_df: pl.DataFrame,
    probs: np.ndarray,
    economics: EconomicsConfig,
    policy: PolicyConfig,
    *,
    rank_by: str = "expected_net_savings",
) -> Tuple[List[ReviewDecision], Dict[str, Any]]:
    """Prioritize transactions for manual review under a fixed review capacity budget.

    Guardrails:
      - Disputes are not equivalent to fraud; actions are strictly limited to
        'manual_review' or 'approve'. Automated merchant suspension/termination is
        prohibited.
      - If a merchant's flagged share exceeds `policy.merchant_alert_rate_cap` on a
        batch with >= 5 transactions for that merchant, a non-punitive
        'analyst_merchant_health_check' advisory is recorded rather than automated holds.
    """
    if policy.allow_automated_merchant_punishment:
        raise ConfigError(
            "Automated merchant punishment is forbidden by policy invariants."
        )

    n = scored_df.height
    if n == 0:
        return [], {
            "total_scored": 0,
            "budget_capacity": 0,
            "reviewed_count": 0,
            "approved_count": 0,
            "total_expected_loss_usd": 0.0,
            "reviewed_expected_loss_usd": 0.0,
            "total_expected_net_savings_usd": 0.0,
            "merchant_advisories": [],
        }

    tx_ids = scored_df["transaction_id"].to_list()
    merchant_ids = scored_df["merchant_id"].to_list()
    amounts = scored_df["amount_usd"].to_numpy().astype(np.float64)
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    _, exp_loss, _, exp_net_savings = compute_expected_loss_vectorized(
        probs, amounts, economics
    )

    budget_capacity = max(1, int(math.floor(n * policy.review_budget_fraction)))

    # Deterministic sort key: primary score desc, prob desc, tx_id asc
    if rank_by == "expected_net_savings":
        indices = sorted(
            range(n),
            key=lambda i: (-float(exp_net_savings[i]), -float(probs[i]), tx_ids[i]),
        )
    elif rank_by == "probability":
        indices = sorted(
            range(n),
            key=lambda i: (-float(probs[i]), -float(exp_net_savings[i]), tx_ids[i]),
        )
    else:
        raise ConfigError(f"Unsupported rank_by strategy: {rank_by!r}")

    selected_set: set[int] = set()
    for idx in indices:
        if len(selected_set) >= budget_capacity:
            break
        if rank_by == "expected_net_savings":
            if float(exp_net_savings[idx]) < policy.min_expected_net_savings_usd:
                continue
        selected_set.add(idx)

    # Non-punitive merchant concentration monitoring
    merchant_total: Dict[str, int] = {}
    merchant_flagged: Dict[str, int] = {}
    for i, mid in enumerate(merchant_ids):
        merchant_total[mid] = merchant_total.get(mid, 0) + 1
        if i in selected_set:
            merchant_flagged[mid] = merchant_flagged.get(mid, 0) + 1

    advisory_merchants: set[str] = set()
    merchant_advisories: List[Dict[str, Any]] = []
    for mid in sorted(merchant_total):
        tot = merchant_total[mid]
        flg = merchant_flagged.get(mid, 0)
        rate = flg / tot
        if tot >= 5 and rate > policy.merchant_alert_rate_cap:
            advisory_merchants.add(mid)
            merchant_advisories.append(
                {
                    "merchant_id": mid,
                    "batch_tx_count": tot,
                    "flagged_count": flg,
                    "flagged_rate": round(rate, 4),
                    "recommendation": "analyst_merchant_health_check",
                    "automated_punitive_action": False,
                }
            )

    decisions: List[ReviewDecision] = []
    for rank_num, idx in enumerate(indices, start=1):
        action = "manual_review" if idx in selected_set else "approve"
        decisions.append(
            ReviewDecision(
                transaction_id=tx_ids[idx],
                merchant_id=merchant_ids[idx],
                action=action,
                rank=rank_num,
                calibrated_prob=round(float(probs[idx]), 6),
                amount_usd=round(float(amounts[idx]), 2),
                expected_loss_usd=round(float(exp_loss[idx]), 2),
                expected_net_savings_usd=round(float(exp_net_savings[idx]), 2),
                merchant_advisory_flag=merchant_ids[idx] in advisory_merchants,
            )
        )

    reviewed_indices = sorted(selected_set)
    summary: Dict[str, Any] = {
        "rank_by": rank_by,
        "total_scored": n,
        "budget_capacity": budget_capacity,
        "reviewed_count": len(reviewed_indices),
        "approved_count": n - len(reviewed_indices),
        "total_expected_loss_usd": round(float(exp_loss.sum()), 2),
        "reviewed_expected_loss_usd": round(
            float(exp_loss[reviewed_indices].sum()) if reviewed_indices else 0.0, 2
        ),
        "total_expected_net_savings_usd": round(
            float(exp_net_savings[reviewed_indices].sum())
            if reviewed_indices
            else 0.0,
            2,
        ),
        "merchant_advisories": merchant_advisories,
    }

    # If realized labels are available (offline simulation), report realized economics too.
    if "label" in scored_df.columns and scored_df["label"].null_count() == 0:
        y = scored_df["label"].to_numpy().astype(np.int64)
        lgd = amounts * (1.0 - economics.expected_recovery_rate) + economics.chargeback_fee_usd
        realized_loss = y * lgd
        total_realized = float(realized_loss.sum())
        rev_y = y[reviewed_indices] if reviewed_indices else np.array([], dtype=np.int64)
        rev_realized = float(realized_loss[reviewed_indices].sum()) if reviewed_indices else 0.0
        rev_n = len(reviewed_indices)
        realized_net_savings = (
            rev_realized * economics.review_prevention_effectiveness
            - rev_n * economics.review_cost_usd
        )
        summary.update(
            {
                "realized_disputes_total": int(y.sum()),
                "realized_disputes_in_review_queue": int(rev_y.sum()),
                "precision_at_review_budget": round(
                    float(rev_y.sum() / rev_n) if rev_n > 0 else 0.0, 6
                ),
                "recall_at_review_budget": round(
                    float(rev_y.sum() / y.sum()) if y.sum() > 0 else 0.0, 6
                ),
                "total_realized_loss_usd": round(total_realized, 2),
                "captured_realized_loss_usd": round(rev_realized, 2),
                "loss_capture_rate": round(
                    float(rev_realized / total_realized) if total_realized > 0 else 0.0,
                    6,
                ),
                "realized_net_savings_usd": round(float(realized_net_savings), 2),
            }
        )

    return decisions, summary


def fit_empirical_delay_cdf(
    train_mature_df: pl.DataFrame, max_days: int = 120
) -> np.ndarray:
    """Fit empirical cumulative dispute arrival curve F(d) for d in 0..max_days."""
    delays = (
        train_mature_df.filter(
            (pl.col("label") == 1) & pl.col("dispute_delay_days").is_not_null()
        )["dispute_delay_days"]
        .to_numpy()
        .astype(np.float64)
    )
    cdf = np.zeros(max_days + 1, dtype=np.float64)
    if len(delays) == 0:
        # Linear fallback if training set had no delay observations
        for d in range(max_days + 1):
            cdf[d] = min(1.0, d / max(1.0, float(max_days)))
        return cdf

    for d in range(max_days + 1):
        cdf[d] = float(np.mean(delays <= float(d)))
    return cdf


def _lookup_delay_cdf(cdf: np.ndarray, days_elapsed: float) -> float:
    if days_elapsed <= 0.0:
        return 0.0
    idx = min(len(cdf) - 1, int(math.floor(days_elapsed)))
    return float(cdf[idx])


def simulate_delayed_outcome_monitoring(
    monitored_df: pl.DataFrame,
    predicted_probs: np.ndarray,
    reviewed_tx_ids: Sequence[str],
    all_disputes: Sequence[DisputeRecord],
    delay_cdf: np.ndarray,
    start_as_of_ts: datetime,
    checkpoint_offsets_days: Sequence[int] = (0, 15, 30, 60, 90, 120),
) -> List[Dict[str, Any]]:
    """Simulate delayed-outcome monitoring as immature cohorts mature over time.

    At each checkpoint (start_as_of_ts + offset_days):
      - Only disputes with dispute_ts <= checkpoint_ts are visible.
      - Expected disputes by checkpoint_ts adjusts predicted_probs by each transaction's
        elapsed age through the empirical delay CDF F(age).
      - Tracks cohort maturation rate, O/E ratio, reviewed-queue observed precision,
        and drift alert status.
    """
    if monitored_df.is_empty():
        return []

    reviewed_set = set(reviewed_tx_ids)
    dispute_by_tx: Dict[str, DisputeRecord] = {}
    for d in sorted(all_disputes, key=lambda r: (r.dispute_ts, r.dispute_id)):
        dispute_by_tx.setdefault(d.transaction_id, d)

    rows = monitored_df.select(
        ["transaction_id", "settlement_ts", "maturity_date", "amount_usd"]
    ).to_dicts()
    probs = np.clip(np.asarray(predicted_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    snapshots: List[Dict[str, Any]] = []
    for offset in checkpoint_offsets_days:
        check_ts = start_as_of_ts + timedelta(days=int(offset))
        mature_cnt = 0
        observed_total = 0
        expected_total = 0.0
        observed_reviewed = 0
        expected_reviewed = 0.0
        cdf_sum = 0.0

        for i, row in enumerate(rows):
            tx_id = row["transaction_id"]
            settle_ts = row["settlement_ts"]
            mat_ts = row["maturity_date"]
            age_days = max(0.0, (check_ts - settle_ts).total_seconds() / 86400.0)
            frac_matured = _lookup_delay_cdf(delay_cdf, age_days)
            cdf_sum += frac_matured

            if check_ts >= mat_ts:
                mature_cnt += 1

            exp_i = float(probs[i]) * frac_matured
            expected_total += exp_i

            d_rec = dispute_by_tx.get(tx_id)
            obs_i = 1 if (d_rec is not None and d_rec.dispute_ts <= check_ts) else 0
            observed_total += obs_i

            if tx_id in reviewed_set:
                expected_reviewed += exp_i
                observed_reviewed += obs_i

        n = len(rows)
        oe_ratio = (observed_total / expected_total) if expected_total > 1e-6 else 1.0
        reviewed_n = len(reviewed_set)
        observed_rev_precision = (
            observed_reviewed / reviewed_n if reviewed_n > 0 else 0.0
        )

        # Conservative statistical alert band (avoid noisy alarms when expected count is small)
        if expected_total < 5.0:
            alert_status = "INSUFFICIENT_MATURATION"
        elif 0.70 <= oe_ratio <= 1.35:
            alert_status = "OK"
        elif 0.50 <= oe_ratio <= 1.65:
            alert_status = "WATCH"
        else:
            alert_status = "DRIFT_ALERT"

        snapshots.append(
            {
                "checkpoint_offset_days": int(offset),
                "checkpoint_ts": check_ts.isoformat(),
                "cohort_size": n,
                "mature_fraction": round(mature_cnt / n, 4),
                "mean_expected_arrival_completion": round(cdf_sum / n, 4),
                "expected_disputes_by_checkpoint": round(expected_total, 2),
                "observed_disputes_by_checkpoint": observed_total,
                "observed_to_expected_ratio": round(oe_ratio, 4),
                "reviewed_queue_size": reviewed_n,
                "reviewed_expected_disputes": round(expected_reviewed, 2),
                "reviewed_observed_disputes": observed_reviewed,
                "reviewed_observed_precision": round(observed_rev_precision, 6),
                "monitoring_status": alert_status,
            }
        )

    return snapshots
