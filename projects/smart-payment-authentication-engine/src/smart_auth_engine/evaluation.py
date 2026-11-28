"""Chronological policy evaluation respecting label maturity and reference baseline comparisons."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from smart_auth_engine.config import EngineConfig
from smart_auth_engine.contracts import (
    AuthAction,
    FinalAuthOutcome,
    PaymentEvent,
    RiskLabel,
)
from smart_auth_engine.inference import AuthenticationEngine


class ChronologicalOrderError(ValueError):
    """Raised when events supplied to chronological evaluation are not in non-decreasing timestamp order."""


class LabelAlignmentError(ValueError):
    """Raised when risk labels are missing or temporally inconsistent with events."""


@dataclass
class ChronologicalEvaluationReport:
    evaluation_as_of: datetime
    total_events: int
    matured_events: int
    immature_events: int
    action_counts: Dict[str, int]
    final_outcome_counts: Dict[str, int]
    mandatory_challenge_bypass_count: int
    override_bypass_count: int
    fallback_count: int
    matured_fraud_loss_rate: Optional[float]
    matured_genuine_acceptance_rate: Optional[float]
    challenge_rate: float
    decline_rate: float
    abstain_rate: float
    matured_gross_volume_minor: int
    matured_fraud_loss_minor: float
    total_friction_cost_minor: float
    daily_breakdown: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_as_of": self.evaluation_as_of.isoformat(),
            "total_events": self.total_events,
            "matured_events": self.matured_events,
            "immature_events": self.immature_events,
            "action_counts": self.action_counts,
            "final_outcome_counts": self.final_outcome_counts,
            "mandatory_challenge_bypass_count": self.mandatory_challenge_bypass_count,
            "override_bypass_count": self.override_bypass_count,
            "fallback_count": self.fallback_count,
            "matured_fraud_loss_rate": (
                None if self.matured_fraud_loss_rate is None else round(self.matured_fraud_loss_rate, 6)
            ),
            "matured_genuine_acceptance_rate": (
                None
                if self.matured_genuine_acceptance_rate is None
                else round(self.matured_genuine_acceptance_rate, 6)
            ),
            "challenge_rate": round(self.challenge_rate, 6),
            "decline_rate": round(self.decline_rate, 6),
            "abstain_rate": round(self.abstain_rate, 6),
            "matured_gross_volume_minor": self.matured_gross_volume_minor,
            "matured_fraud_loss_minor": round(self.matured_fraud_loss_minor, 2),
            "total_friction_cost_minor": round(self.total_friction_cost_minor, 2),
            "daily_breakdown": self.daily_breakdown,
        }


def evaluate_chronologically(
    events: Sequence[PaymentEvent],
    labels: Sequence[RiskLabel],
    engine: AuthenticationEngine,
    cfg: EngineConfig,
    evaluation_as_of: Optional[datetime] = None,
    record_store: bool = False,
) -> ChronologicalEvaluationReport:
    """Replay events strictly in chronological order and evaluate matured outcomes as of `evaluation_as_of`.

    - Rejects out-of-order event streams (ChronologicalOrderError).
    - Rejects missing or causally invalid labels (LabelAlignmentError).
    - Separates matured labels (label_available_at <= evaluation_as_of) from immature labels so
      future chargebacks never leak into reported loss metrics.
    - Explicitly audits mandatory-challenge and override non-bypass invariants on every event.
    """
    if not events:
        raise ValueError("Cannot run chronological evaluation on an empty event stream.")

    # Verify strict chronological non-decreasing order
    prev_ts = events[0].timestamp
    for idx in range(1, len(events)):
        if events[idx].timestamp < prev_ts:
            raise ChronologicalOrderError(
                f"Event {events[idx].payment_id} at {events[idx].timestamp.isoformat()} "
                f"precedes prior event {events[idx - 1].payment_id} at {prev_ts.isoformat()}."
            )
        prev_ts = events[idx].timestamp

    labels_by_id = {lbl.payment_id: lbl for lbl in labels}
    for ev in events:
        if ev.payment_id not in labels_by_id:
            raise LabelAlignmentError(f"Missing RiskLabel for payment_id={ev.payment_id!r}.")
        lbl = labels_by_id[ev.payment_id]
        if lbl.event_timestamp != ev.timestamp:
            raise LabelAlignmentError(
                f"Timestamp mismatch between PaymentEvent and RiskLabel for {ev.payment_id!r}."
            )

    if evaluation_as_of is None:
        # Default to max label maturity timestamp across the window (post-maturity retrospective audit)
        eval_as_of = max(labels_by_id[ev.payment_id].label_available_at for ev in events)
    else:
        if evaluation_as_of.tzinfo is None:
            raise ValueError("evaluation_as_of must be timezone-aware (UTC).")
        eval_as_of = evaluation_as_of.astimezone(timezone.utc)

    action_counts = {a.value: 0 for a in AuthAction}
    final_counts = {o.value: 0 for o in FinalAuthOutcome}
    mandatory_bypass = 0
    override_bypass = 0
    fallback_count = 0

    matured_events = 0
    immature_events = 0
    matured_volume = 0
    matured_fraud_loss = 0.0
    matured_genuine_total = 0
    matured_genuine_authorized = 0
    total_friction = 0.0

    daily_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "matured": 0,
            "immature": 0,
            "challenges": 0,
            "declines": 0,
            "matured_volume_minor": 0,
            "matured_fraud_loss_minor": 0.0,
        }
    )

    for ev in events:
        lbl = labels_by_id[ev.payment_id]
        is_mature = lbl.is_mature_at(eval_as_of)

        result = engine.evaluate_and_simulate_step_up(
            event=ev,
            actual_is_fraud=lbl.is_fraud if is_mature else None,
            record_store=record_store,
        )
        dec = result.decision
        action_counts[dec.action.value] += 1
        final_counts[result.final_outcome.value] += 1
        total_friction += result.friction_cost_minor

        if dec.fallback_used:
            fallback_count += 1
        if dec.mandatory_rule_triggered and dec.action in (AuthAction.APPROVE, AuthAction.ABSTAIN):
            mandatory_bypass += 1
        if dec.override_triggered and dec.action != AuthAction.DECLINE:
            override_bypass += 1

        day_key = ev.timestamp.strftime("%Y-%m-%d")
        d = daily_stats[day_key]
        d["total"] += 1
        if dec.action == AuthAction.CHALLENGE:
            d["challenges"] += 1
        if dec.action == AuthAction.DECLINE:
            d["declines"] += 1

        if is_mature:
            matured_events += 1
            matured_volume += ev.amount_minor
            matured_fraud_loss += result.expected_fraud_loss_minor
            d["matured"] += 1
            d["matured_volume_minor"] += ev.amount_minor
            d["matured_fraud_loss_minor"] += result.expected_fraud_loss_minor
            if not lbl.is_fraud:
                matured_genuine_total += 1
                if result.final_outcome == FinalAuthOutcome.AUTHORIZED:
                    matured_genuine_authorized += 1
        else:
            immature_events += 1
            d["immature"] += 1

    total = len(events)
    matured_loss_rate = (matured_fraud_loss / matured_volume) if matured_volume > 0 else None
    matured_accept_rate = (
        (matured_genuine_authorized / matured_genuine_total) if matured_genuine_total > 0 else None
    )

    daily_breakdown = []
    for day_key in sorted(daily_stats.keys()):
        row = daily_stats[day_key]
        daily_breakdown.append(
            {
                "date": day_key,
                "total_events": row["total"],
                "matured_events": row["matured"],
                "immature_events": row["immature"],
                "challenge_rate": round(row["challenges"] / max(row["total"], 1), 6),
                "decline_rate": round(row["declines"] / max(row["total"], 1), 6),
                "matured_fraud_loss_rate": (
                    round(row["matured_fraud_loss_minor"] / row["matured_volume_minor"], 6)
                    if row["matured_volume_minor"] > 0
                    else None
                ),
            }
        )

    return ChronologicalEvaluationReport(
        evaluation_as_of=eval_as_of,
        total_events=total,
        matured_events=matured_events,
        immature_events=immature_events,
        action_counts=action_counts,
        final_outcome_counts=final_counts,
        mandatory_challenge_bypass_count=mandatory_bypass,
        override_bypass_count=override_bypass,
        fallback_count=fallback_count,
        matured_fraud_loss_rate=matured_loss_rate,
        matured_genuine_acceptance_rate=matured_accept_rate,
        challenge_rate=action_counts[AuthAction.CHALLENGE.value] / float(total),
        decline_rate=action_counts[AuthAction.DECLINE.value] / float(total),
        abstain_rate=action_counts[AuthAction.ABSTAIN.value] / float(total),
        matured_gross_volume_minor=matured_volume,
        matured_fraud_loss_minor=matured_fraud_loss,
        total_friction_cost_minor=total_friction,
        daily_breakdown=daily_breakdown,
    )
