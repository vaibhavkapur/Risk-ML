"""Deterministic step-up authentication flow simulator."""

from __future__ import annotations

import hashlib
from datetime import timedelta
from typing import Optional

import numpy as np

from smart_auth_engine.config import EngineConfig
from smart_auth_engine.contracts import (
    AuthAction,
    ChallengeMethod,
    ChallengeOutcome,
    ChallengeStatus,
    FinalAuthOutcome,
    PaymentEvent,
    PolicyDecision,
    StepUpExecutionResult,
)


def _deterministic_rng(seed: int, decision_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{decision_id}".encode("utf-8")).digest()
    sub_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return np.random.default_rng(sub_seed)


def simulate_step_up(
    event: PaymentEvent,
    decision: PolicyDecision,
    cfg: EngineConfig,
    actual_is_fraud: Optional[bool] = None,
    seed: Optional[int] = None,
) -> StepUpExecutionResult:
    """Simulate end-to-end step-up authentication execution for a PolicyDecision.

    When `actual_is_fraud` is not supplied (e.g. live/online simulation endpoint),
    the simulator draws latent fraud state from `decision.calibrated_fraud_prob`
    (or a conservative 0.05 fallback rate if unavailable) using a deterministic per-decision RNG.
    """
    sim_seed = cfg.seed if seed is None else seed
    rng = _deterministic_rng(sim_seed, decision.decision_id)
    sim_cfg = cfg.step_up_simulation

    if actual_is_fraud is None:
        p = decision.calibrated_fraud_prob if decision.calibrated_fraud_prob is not None else 0.05
        is_fraud = bool(rng.random() < p)
    else:
        is_fraud = bool(actual_is_fraud)

    if decision.action == AuthAction.APPROVE:
        exp_loss = float(event.amount_minor) if is_fraud else 0.0
        return StepUpExecutionResult(
            decision=decision,
            challenge_outcome=None,
            final_outcome=FinalAuthOutcome.AUTHORIZED,
            expected_fraud_loss_minor=exp_loss,
            friction_cost_minor=0.0,
            audit_timestamp=decision.timestamp,
        )

    if decision.action == AuthAction.DECLINE:
        return StepUpExecutionResult(
            decision=decision,
            challenge_outcome=None,
            final_outcome=FinalAuthOutcome.DECLINED,
            expected_fraud_loss_minor=0.0,
            friction_cost_minor=0.0,
            audit_timestamp=decision.timestamp,
        )

    if decision.action == AuthAction.ABSTAIN:
        return StepUpExecutionResult(
            decision=decision,
            challenge_outcome=None,
            final_outcome=FinalAuthOutcome.HELD_FOR_REVIEW,
            expected_fraud_loss_minor=0.0,
            friction_cost_minor=0.0,
            audit_timestamp=decision.timestamp,
        )

    # Action == CHALLENGE -> simulate step-up challenge outcome
    if is_fraud:
        probs = [
            sim_cfg.fraud_pass_prob,
            sim_cfg.fraud_fail_prob,
            sim_cfg.fraud_abandon_prob,
            sim_cfg.fraud_timeout_prob,
        ]
    else:
        probs = [
            sim_cfg.genuine_pass_prob,
            sim_cfg.genuine_fail_prob,
            sim_cfg.genuine_abandon_prob,
            sim_cfg.genuine_timeout_prob,
        ]

    statuses = [
        ChallengeStatus.PASSED,
        ChallengeStatus.FAILED,
        ChallengeStatus.ABANDONED,
        ChallengeStatus.TIMED_OUT,
    ]
    chosen_idx = int(rng.choice(len(statuses), p=probs))
    status = statuses[chosen_idx]

    method = (
        decision.challenge_method
        if decision.challenge_method != ChallengeMethod.NONE
        else ChallengeMethod.OTP_SMS
    )
    base_latency = {
        ChallengeMethod.BIOMETRIC_APP: 4200,
        ChallengeMethod.OTP_SMS: 14500,
        ChallengeMethod.HARDWARE_TOKEN: 19000,
    }.get(method, 12000)
    if status == ChallengeStatus.TIMED_OUT:
        latency_ms = 60_000
    else:
        latency_ms = int(max(500, rng.normal(loc=base_latency, scale=base_latency * 0.2)))

    completed_at = decision.timestamp + timedelta(milliseconds=latency_ms)
    outcome = ChallengeOutcome(
        payment_id=event.payment_id,
        decision_id=decision.decision_id,
        challenge_method=method,
        status=status,
        latency_ms=latency_ms,
        completed_at=completed_at,
    )

    if status == ChallengeStatus.PASSED:
        final_outcome = FinalAuthOutcome.AUTHORIZED
        exp_loss = float(event.amount_minor) if is_fraud else 0.0
    elif status == ChallengeStatus.ABANDONED:
        final_outcome = FinalAuthOutcome.ABANDONED
        exp_loss = 0.0
    else:
        # FAILED or TIMED_OUT safely decline
        final_outcome = FinalAuthOutcome.DECLINED
        exp_loss = 0.0

    return StepUpExecutionResult(
        decision=decision,
        challenge_outcome=outcome,
        final_outcome=final_outcome,
        expected_fraud_loss_minor=exp_loss,
        friction_cost_minor=float(cfg.policy_optimization.friction_cost_minor),
        audit_timestamp=completed_at,
    )
