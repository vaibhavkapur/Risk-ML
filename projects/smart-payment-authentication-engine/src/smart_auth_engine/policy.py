"""Pure domain authentication policy, mandatory rules, overrides, deterministic fallback, and constrained optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from smart_auth_engine.config import EngineConfig
from smart_auth_engine.contracts import (
    AuthAction,
    ChallengeMethod,
    FeatureContribution,
    PaymentEvent,
    PolicyDecision,
    RiskLabel,
    build_stable_decision_id,
)
from smart_auth_engine.features import ExtractedFeatures


@dataclass(frozen=True)
class PolicyThresholds:
    """Risk probability thresholds separating approve, challenge, and decline bands."""

    challenge_threshold: float
    decline_threshold: float

    def __post_init__(self) -> None:
        if not (0.0 < self.challenge_threshold < self.decline_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < challenge_threshold < decline_threshold <= 1, "
                f"got ({self.challenge_threshold}, {self.decline_threshold})."
            )


def evaluate_mandatory_challenge_rules(event: PaymentEvent, cfg: EngineConfig) -> List[str]:
    """Evaluate mandatory step-up rules that NO optimization policy may bypass."""
    rules = cfg.mandatory_challenge_rules
    reasons: List[str] = []
    if event.amount_minor >= rules.high_amount_minor:
        reasons.append("MANDATORY_HIGH_AMOUNT")
    if event.merchant_category in rules.high_risk_categories:
        reasons.append("MANDATORY_HIGH_RISK_CATEGORY")
    if rules.require_challenge_on_emulator and event.device.is_emulator is True:
        reasons.append("MANDATORY_EMULATOR_DETECTED")
    if (
        event.is_cross_border
        and event.amount_minor >= rules.require_challenge_on_cross_border_high_amount_minor
    ):
        reasons.append("MANDATORY_CROSS_BORDER_HIGH_AMOUNT")
    if (
        event.device.ip_velocity_1h is not None
        and event.device.ip_velocity_1h >= rules.max_ip_velocity_1h_without_challenge
    ):
        reasons.append("MANDATORY_IP_VELOCITY_SPIKE")
    return reasons


def evaluate_high_risk_overrides(event: PaymentEvent, cfg: EngineConfig) -> List[str]:
    """Evaluate high-risk hard overrides that force DECLINE regardless of model score."""
    ov = cfg.high_risk_overrides
    reasons: List[str] = []
    if (
        event.prior_chargebacks_90d is not None
        and event.prior_chargebacks_90d >= ov.decline_prior_chargebacks_90d_gte
    ):
        reasons.append("OVERRIDE_PRIOR_CHARGEBACKS")
    if (
        event.device.ip_velocity_1h is not None
        and event.device.ip_velocity_1h >= ov.decline_ip_velocity_1h_gte
    ):
        reasons.append("OVERRIDE_EXTREME_IP_VELOCITY")
    if (
        ov.decline_emulator_and_tor
        and event.device.is_emulator is True
        and event.device.is_tor_or_vpn is True
    ):
        reasons.append("OVERRIDE_EMULATOR_AND_TOR")
    if (
        event.prior_chargebacks_90d is not None
        and event.prior_chargebacks_90d >= 1
        and event.amount_minor >= ov.decline_amount_minor_with_chargeback
        and (event.device.is_tor_or_vpn is True or event.device.browser_fingerprint_mismatch is True)
    ):
        reasons.append("OVERRIDE_CHARGEBACK_HIGH_AMOUNT_ANOMALY")
    return reasons


def select_challenge_method(event: PaymentEvent) -> ChallengeMethod:
    """Deterministically select simulated step-up challenge method."""
    if event.amount_minor >= 200_000:
        return ChallengeMethod.HARDWARE_TOKEN
    if event.device.biometric_enrolled is True and event.device.is_emulator is not True:
        return ChallengeMethod.BIOMETRIC_APP
    return ChallengeMethod.OTP_SMS


def evaluate_deterministic_fallback(
    event: PaymentEvent,
    missing_fields: Sequence[str],
    missing_ratio: float,
    fallback_reasons: Sequence[str],
    mandatory_reasons: Sequence[str],
    cfg: EngineConfig,
) -> Tuple[AuthAction, List[str]]:
    """Deterministic fallback action when features are missing or the risk model is unavailable.

    Precedence:
    1. Mandatory challenge rules (cannot be bypassed by fallback).
    2. High-amount outage decline cap.
    3. Excessive missing telemetry -> ABSTAIN (for manual/downstream policy review).
    4. Medium-amount outage/missingness -> CHALLENGE.
    5. Low-amount fallback -> APPROVE (if model outage) or CHALLENGE (if risk signals suspicious).
    """
    fb = cfg.fallback
    reasons = list(fallback_reasons)

    if event.amount_minor >= fb.outage_decline_amount_minor:
        reasons.append("FALLBACK_DECLINE_AMOUNT_CAP")
        return AuthAction.DECLINE, reasons

    if mandatory_reasons:
        reasons.extend(mandatory_reasons)
        reasons.append("FALLBACK_MANDATORY_CHALLENGE_PRESERVED")
        return AuthAction.CHALLENGE, reasons

    if missing_ratio >= fb.abstain_missing_feature_ratio:
        reasons.append("FALLBACK_EXCESSIVE_MISSING_FEATURES_ABSTAIN")
        return AuthAction.ABSTAIN, reasons

    if event.amount_minor >= fb.outage_challenge_amount_minor:
        reasons.append("FALLBACK_CHALLENGE_AMOUNT_BAND")
        return AuthAction.CHALLENGE, reasons

    if event.device.is_tor_or_vpn is True or event.device.browser_fingerprint_mismatch is True:
        reasons.append("FALLBACK_SUSPICIOUS_DEVICE_CHALLENGE")
        return AuthAction.CHALLENGE, reasons

    reasons.append("FALLBACK_LOW_AMOUNT_APPROVE")
    return AuthAction.APPROVE, reasons


def decide_authentication_action(
    event: PaymentEvent,
    extracted: ExtractedFeatures,
    thresholds: PolicyThresholds,
    cfg: EngineConfig,
    model_version: str,
    calibrated_prob: Optional[float] = None,
    raw_prob: Optional[float] = None,
    explanations: Optional[List[FeatureContribution]] = None,
    model_outage: bool = False,
    outage_reason: Optional[str] = None,
) -> PolicyDecision:
    """Pure domain policy decision function with strict precedence invariants.

    Invariants:
    - High-risk overrides always yield DECLINE.
    - Mandatory challenge rules can NEVER be bypassed to APPROVE or ABSTAIN (only escalated to DECLINE).
    - Missing critical features, excessive missingness, or model outage always invoke deterministic fallback.
    """
    mandatory_reasons = evaluate_mandatory_challenge_rules(event, cfg)
    override_reasons = evaluate_high_risk_overrides(event, cfg)
    mandatory_triggered = len(mandatory_reasons) > 0
    override_triggered = len(override_reasons) > 0

    decision_id = build_stable_decision_id(
        payment_id=event.payment_id,
        timestamp=event.timestamp,
        model_version=model_version,
        feature_version=extracted.feature_version,
        policy_version=cfg.policy_version,
    )

    # 1. High-risk overrides always dominate (cannot be bypassed)
    if override_triggered:
        reasons = list(override_reasons)
        if mandatory_triggered:
            reasons.extend(mandatory_reasons)
        return PolicyDecision(
            decision_id=decision_id,
            payment_id=event.payment_id,
            timestamp=event.timestamp,
            action=AuthAction.DECLINE,
            calibrated_fraud_prob=None if model_outage else calibrated_prob,
            raw_fraud_prob=None if model_outage else raw_prob,
            challenge_method=ChallengeMethod.NONE,
            mandatory_rule_triggered=mandatory_triggered,
            override_triggered=True,
            fallback_used=False,
            reason_codes=reasons,
            explanations=explanations or [],
            missing_features=list(extracted.missing_fields),
            missing_feature_ratio=round(extracted.missing_ratio, 6),
            model_version=model_version,
            feature_version=extracted.feature_version,
            policy_version=cfg.policy_version,
        )

    # 2. Detect missing-feature or model-outage fallback conditions
    missing_critical = [
        f for f in cfg.fallback.critical_features if f in extracted.missing_fields
    ]
    fallback_reasons: List[str] = []
    if model_outage or calibrated_prob is None:
        fallback_reasons.append(outage_reason or "FALLBACK_MODEL_OUTAGE")
    if missing_critical:
        fallback_reasons.append(f"FALLBACK_MISSING_CRITICAL_FEATURES:{','.join(sorted(missing_critical))}")
    if extracted.missing_ratio > cfg.fallback.max_missing_feature_ratio_for_model:
        fallback_reasons.append("FALLBACK_HIGH_MISSING_FEATURE_RATIO")

    if fallback_reasons:
        fb_action, fb_reason_codes = evaluate_deterministic_fallback(
            event=event,
            missing_fields=extracted.missing_fields,
            missing_ratio=extracted.missing_ratio,
            fallback_reasons=fallback_reasons,
            mandatory_reasons=mandatory_reasons,
            cfg=cfg,
        )
        method = (
            select_challenge_method(event)
            if fb_action == AuthAction.CHALLENGE
            else ChallengeMethod.NONE
        )
        return PolicyDecision(
            decision_id=decision_id,
            payment_id=event.payment_id,
            timestamp=event.timestamp,
            action=fb_action,
            calibrated_fraud_prob=None,
            raw_fraud_prob=None,
            challenge_method=method,
            mandatory_rule_triggered=mandatory_triggered,
            override_triggered=False,
            fallback_used=True,
            reason_codes=fb_reason_codes,
            explanations=[],
            missing_features=list(extracted.missing_fields),
            missing_feature_ratio=round(extracted.missing_ratio, 6),
            model_version=model_version,
            feature_version=extracted.feature_version,
            policy_version=cfg.policy_version,
        )

    # 3. Calibrated probability threshold decision + non-bypassable mandatory challenge guard
    assert calibrated_prob is not None
    reasons: List[str] = []
    if calibrated_prob >= thresholds.decline_threshold:
        action = AuthAction.DECLINE
        reasons.append("POLICY_MODEL_DECLINE_THRESHOLD")
        if mandatory_triggered:
            reasons.extend(mandatory_reasons)
    elif mandatory_triggered:
        action = AuthAction.CHALLENGE
        reasons.extend(mandatory_reasons)
        if calibrated_prob >= thresholds.challenge_threshold:
            reasons.append("POLICY_MODEL_CHALLENGE_THRESHOLD")
    elif calibrated_prob >= thresholds.challenge_threshold:
        action = AuthAction.CHALLENGE
        reasons.append("POLICY_MODEL_CHALLENGE_THRESHOLD")
    else:
        action = AuthAction.APPROVE
        reasons.append("POLICY_MODEL_FRICTIONLESS_APPROVE")

    method = (
        select_challenge_method(event)
        if action == AuthAction.CHALLENGE
        else ChallengeMethod.NONE
    )
    return PolicyDecision(
        decision_id=decision_id,
        payment_id=event.payment_id,
        timestamp=event.timestamp,
        action=action,
        calibrated_fraud_prob=round(float(calibrated_prob), 6),
        raw_fraud_prob=None if raw_prob is None else round(float(raw_prob), 6),
        challenge_method=method,
        mandatory_rule_triggered=mandatory_triggered,
        override_triggered=False,
        fallback_used=False,
        reason_codes=reasons,
        explanations=explanations or [],
        missing_features=list(extracted.missing_fields),
        missing_feature_ratio=round(extracted.missing_ratio, 6),
        model_version=model_version,
        feature_version=extracted.feature_version,
        policy_version=cfg.policy_version,
    )


@dataclass
class PolicyEvaluationPoint:
    challenge_threshold: float
    decline_threshold: float
    fraud_loss_rate: float
    challenge_rate: float
    acceptance_rate: float
    gross_approval_rate: float
    decline_rate: float
    abstain_rate: float
    expected_total_cost_minor: float
    expected_fraud_loss_minor: float
    expected_friction_cost_minor: float
    expected_false_decline_cost_minor: float
    mandatory_challenge_count: int
    override_decline_count: int
    feasible: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_threshold": round(self.challenge_threshold, 6),
            "decline_threshold": round(self.decline_threshold, 6),
            "fraud_loss_rate": round(self.fraud_loss_rate, 6),
            "challenge_rate": round(self.challenge_rate, 6),
            "acceptance_rate": round(self.acceptance_rate, 6),
            "gross_approval_rate": round(self.gross_approval_rate, 6),
            "decline_rate": round(self.decline_rate, 6),
            "abstain_rate": round(self.abstain_rate, 6),
            "expected_total_cost_minor": round(self.expected_total_cost_minor, 2),
            "expected_fraud_loss_minor": round(self.expected_fraud_loss_minor, 2),
            "expected_friction_cost_minor": round(self.expected_friction_cost_minor, 2),
            "expected_false_decline_cost_minor": round(self.expected_false_decline_cost_minor, 2),
            "mandatory_challenge_count": self.mandatory_challenge_count,
            "override_decline_count": self.override_decline_count,
            "feasible": self.feasible,
        }


@dataclass
class PolicyOptimizationResult:
    selected_thresholds: PolicyThresholds
    selected_point: PolicyEvaluationPoint
    constraints_satisfied: bool
    pareto_frontier: List[PolicyEvaluationPoint]
    evaluated_candidates: int


def evaluate_policy_candidate(
    events: Sequence[PaymentEvent],
    labels: Sequence[RiskLabel],
    calibrated_probs: np.ndarray,
    thresholds: PolicyThresholds,
    cfg: EngineConfig,
    precomputed_mandatory: Optional[Sequence[bool]] = None,
    precomputed_override: Optional[Sequence[bool]] = None,
) -> PolicyEvaluationPoint:
    """Evaluate expected fraud loss, challenge rate, and genuine acceptance for a threshold pair."""
    n = len(events)
    if n == 0:
        raise ValueError("Cannot evaluate policy on an empty dataset.")

    opt = cfg.policy_optimization
    amounts = np.array([ev.amount_minor for ev in events], dtype=np.float64)
    is_fraud = np.array([1.0 if lbl.is_fraud else 0.0 for lbl in labels], dtype=np.float64)
    total_volume = float(amounts.sum())
    n_genuine = float((1.0 - is_fraud).sum())

    if precomputed_mandatory is None:
        mandatory = np.array([len(evaluate_mandatory_challenge_rules(ev, cfg)) > 0 for ev in events], dtype=bool)
    else:
        mandatory = np.asarray(precomputed_mandatory, dtype=bool)

    if precomputed_override is None:
        override = np.array([len(evaluate_high_risk_overrides(ev, cfg)) > 0 for ev in events], dtype=bool)
    else:
        override = np.asarray(precomputed_override, dtype=bool)

    probs = np.asarray(calibrated_probs, dtype=np.float64)

    # Action arrays strictly obeying precedence: override -> decline_threshold -> mandatory -> challenge_threshold -> approve
    is_decline = override | (probs >= thresholds.decline_threshold)
    is_challenge = (~is_decline) & (mandatory | (probs >= thresholds.challenge_threshold))
    is_approve = (~is_decline) & (~is_challenge)

    # Expected fraud loss (minor units)
    fraud_escape_on_challenge = 1.0 - opt.challenge_fraud_catch_rate
    exp_fraud_loss = float(
        (amounts * is_fraud * is_approve).sum()
        + (amounts * is_fraud * is_challenge * fraud_escape_on_challenge).sum()
    )

    # Expected friction cost on challenges
    exp_friction_cost = float(is_challenge.sum() * opt.friction_cost_minor)

    # Expected opportunity cost on false declines + genuine challenge abandonments
    opp_rate = opt.decline_opportunity_cost_bps / 10_000.0
    genuine_abandon = 1.0 - opt.challenge_genuine_pass_rate
    exp_false_decline_cost = float(
        (amounts * (1.0 - is_fraud) * is_decline * opp_rate).sum()
        + (amounts * (1.0 - is_fraud) * is_challenge * genuine_abandon * opp_rate).sum()
    )

    total_cost = exp_fraud_loss + exp_friction_cost + exp_false_decline_cost
    fraud_loss_rate = exp_fraud_loss / max(total_volume, 1.0)
    challenge_rate = float(is_challenge.mean())
    decline_rate = float(is_decline.mean())

    accepted_genuine = float(
        ((1.0 - is_fraud) * is_approve).sum()
        + ((1.0 - is_fraud) * is_challenge * opt.challenge_genuine_pass_rate).sum()
    )
    acceptance_rate = accepted_genuine / max(n_genuine, 1.0)

    accepted_gross = float(
        is_approve.sum()
        + ((1.0 - is_fraud) * is_challenge * opt.challenge_genuine_pass_rate).sum()
        + (is_fraud * is_challenge * fraud_escape_on_challenge).sum()
    )
    gross_approval_rate = accepted_gross / float(n)

    feasible = (
        challenge_rate <= opt.max_challenge_rate + 1e-12
        and fraud_loss_rate <= opt.max_fraud_loss_rate + 1e-12
    )

    return PolicyEvaluationPoint(
        challenge_threshold=thresholds.challenge_threshold,
        decline_threshold=thresholds.decline_threshold,
        fraud_loss_rate=fraud_loss_rate,
        challenge_rate=challenge_rate,
        acceptance_rate=acceptance_rate,
        gross_approval_rate=gross_approval_rate,
        decline_rate=decline_rate,
        abstain_rate=0.0,
        expected_total_cost_minor=total_cost,
        expected_fraud_loss_minor=exp_fraud_loss,
        expected_friction_cost_minor=exp_friction_cost,
        expected_false_decline_cost_minor=exp_false_decline_cost,
        mandatory_challenge_count=int((mandatory & ~override).sum()),
        override_decline_count=int(override.sum()),
        feasible=feasible,
    )


def _extract_pareto_frontier(points: Sequence[PolicyEvaluationPoint]) -> List[PolicyEvaluationPoint]:
    """Extract non-dominated points over (minimize fraud_loss_rate, minimize challenge_rate, maximize acceptance_rate)."""
    sorted_pts = sorted(points, key=lambda p: (p.fraud_loss_rate, p.challenge_rate, -p.acceptance_rate))
    frontier: List[PolicyEvaluationPoint] = []
    for cand in sorted_pts:
        dominated = False
        for existing in frontier:
            if (
                existing.fraud_loss_rate <= cand.fraud_loss_rate + 1e-12
                and existing.challenge_rate <= cand.challenge_rate + 1e-12
                and existing.acceptance_rate >= cand.acceptance_rate - 1e-12
                and (
                    existing.fraud_loss_rate < cand.fraud_loss_rate - 1e-12
                    or existing.challenge_rate < cand.challenge_rate - 1e-12
                    or existing.acceptance_rate > cand.acceptance_rate + 1e-12
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(cand)
    return frontier


def optimize_policy_thresholds(
    events: Sequence[PaymentEvent],
    labels: Sequence[RiskLabel],
    calibrated_probs: np.ndarray,
    cfg: EngineConfig,
) -> PolicyOptimizationResult:
    """Optimize (challenge_threshold, decline_threshold) under fraud-loss and challenge-rate constraints."""
    opt = cfg.policy_optimization
    steps = opt.threshold_grid_steps
    challenge_grid = np.linspace(0.02, 0.55, steps)
    decline_grid = np.linspace(0.15, 0.95, steps)

    mandatory = [len(evaluate_mandatory_challenge_rules(ev, cfg)) > 0 for ev in events]
    override = [len(evaluate_high_risk_overrides(ev, cfg)) > 0 for ev in events]

    candidates: List[PolicyEvaluationPoint] = []
    for t_c in challenge_grid:
        for t_d in decline_grid:
            if t_d <= t_c + 0.03:
                continue
            pt = evaluate_policy_candidate(
                events=events,
                labels=labels,
                calibrated_probs=calibrated_probs,
                thresholds=PolicyThresholds(float(t_c), float(t_d)),
                cfg=cfg,
                precomputed_mandatory=mandatory,
                precomputed_override=override,
            )
            candidates.append(pt)

    feasible_pts = [p for p in candidates if p.feasible]
    if feasible_pts:
        best = min(
            feasible_pts,
            key=lambda p: (p.expected_total_cost_minor, p.fraud_loss_rate, p.challenge_rate),
        )
        constraints_satisfied = True
    else:
        # Closest-to-feasible fallback with explicit constraints_satisfied=False (no silent success)
        def _violation(p: PolicyEvaluationPoint) -> float:
            v_c = max(0.0, p.challenge_rate - opt.max_challenge_rate) / max(opt.max_challenge_rate, 1e-6)
            v_f = max(0.0, p.fraud_loss_rate - opt.max_fraud_loss_rate) / max(opt.max_fraud_loss_rate, 1e-6)
            return v_c + v_f

        best = min(candidates, key=lambda p: (_violation(p), p.expected_total_cost_minor))
        constraints_satisfied = False

    frontier = _extract_pareto_frontier(candidates)
    return PolicyOptimizationResult(
        selected_thresholds=PolicyThresholds(best.challenge_threshold, best.decline_threshold),
        selected_point=best,
        constraints_satisfied=constraints_satisfied,
        pareto_frontier=frontier,
        evaluated_candidates=len(candidates),
    )
