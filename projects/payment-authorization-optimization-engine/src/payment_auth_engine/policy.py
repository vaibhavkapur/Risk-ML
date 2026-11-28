"""Baseline, unconstrained EV, and constrained risk-aware authorization policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from payment_auth_engine.config import EngineConfig
from payment_auth_engine.models import ActionPrediction
from payment_auth_engine.schemas import (
    EVALUABLE_ACTIONS,
    AuthorizationAttempt,
    ConstraintCheckResult,
    DecisionRecord,
    PolicyAction,
)


@dataclass(frozen=True)
class PolicyChoice:
    """Selected action plus audit metadata from a policy decision."""

    action: PolicyAction
    prediction: ActionPrediction
    reason_codes: List[str]
    policy_notes: str
    constraint_checks: List[ConstraintCheckResult]
    any_constraint_breach: bool


class AuthorizationPolicy:
    """Typed interface for authorization decision policies."""

    name: str = "abstract_policy"

    def choose(
        self,
        attempt: AuthorizationAttempt,
        predictions: Dict[PolicyAction, ActionPrediction],
        *,
        allowed_actions: Optional[Sequence[PolicyAction]] = None,
    ) -> PolicyChoice:
        raise NotImplementedError


def _feasible_actions(
    predictions: Dict[PolicyAction, ActionPrediction],
    allowed_actions: Optional[Sequence[PolicyAction]],
) -> List[PolicyAction]:
    pool = list(allowed_actions) if allowed_actions is not None else list(EVALUABLE_ACTIONS)
    return [a for a in pool if a in predictions and predictions[a].feasible]


def _all_non_decline_unavailable(predictions: Dict[PolicyAction, ActionPrediction]) -> bool:
    non_decline = [
        a for a in (PolicyAction.APPROVE_PRIMARY, PolicyAction.ROUTE_SECONDARY, PolicyAction.STEP_UP_3DS)
        if a in predictions
    ]
    return bool(non_decline) and all(not predictions[a].feasible for a in non_decline)


class BaselineThresholdPolicy(AuthorizationPolicy):
    """Static risk-score threshold baseline with no smart routing or 3DS step-up."""

    name = "baseline_threshold"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.threshold = config.policy.baseline_risk_threshold

    def choose(
        self,
        attempt: AuthorizationAttempt,
        predictions: Dict[PolicyAction, ActionPrediction],
        *,
        allowed_actions: Optional[Sequence[PolicyAction]] = None,
    ) -> PolicyChoice:
        feasible = _feasible_actions(predictions, allowed_actions)
        reason_codes: List[str] = []

        if _all_non_decline_unavailable(predictions):
            decline_pred = predictions[PolicyAction.DECLINE_POLICY]
            return PolicyChoice(
                action=PolicyAction.BLOCK_PROCESSOR_OUTAGE,
                prediction=decline_pred,
                reason_codes=["ALL_PROCESSORS_OUTAGE_FAIL_CLOSED"],
                policy_notes="All payment processors unavailable; failing closed.",
                constraint_checks=[
                    ConstraintCheckResult(
                        constraint_name="processor_availability",
                        limit_value=1.0,
                        observed_value=0.0,
                        passed=False,
                        message="Zero healthy or degraded processors available.",
                    )
                ],
                any_constraint_breach=False,
            )

        effective_risk = attempt.risk_score
        if effective_risk is None:
            effective_risk = 0.25 + self.config.policy.missing_feature_risk_penalty
            reason_codes.append("MISSING_RISK_SCORE_FALLBACK")

        if effective_risk > self.threshold:
            chosen = PolicyAction.DECLINE_POLICY
            reason_codes.append("BASELINE_RISK_THRESHOLD_EXCEEDED")
            notes = f"Risk score {effective_risk:.3f} exceeds baseline threshold {self.threshold:.3f}."
        elif PolicyAction.APPROVE_PRIMARY in feasible:
            chosen = PolicyAction.APPROVE_PRIMARY
            reason_codes.append("BASELINE_PRIMARY_ROUTE")
            notes = f"Risk score {effective_risk:.3f} within baseline threshold {self.threshold:.3f}."
        elif PolicyAction.ROUTE_SECONDARY in feasible:
            chosen = PolicyAction.ROUTE_SECONDARY
            reason_codes.append("PROCESSOR_OUTAGE_FAILOVER")
            notes = "Primary processor unavailable; failed over to secondary processor."
        elif PolicyAction.STEP_UP_3DS in feasible:
            chosen = PolicyAction.STEP_UP_3DS
            reason_codes.append("ELIGIBILITY_RESTRICTED_STEP_UP")
            notes = "Retry eligibility restricted feasible action to STEP_UP_3DS."
        else:
            chosen = PolicyAction.DECLINE_POLICY
            reason_codes.append("NO_FEASIBLE_APPROVE_ACTION")
            notes = "No feasible approval route; declining."

        pred = predictions[chosen]
        return PolicyChoice(
            action=chosen,
            prediction=pred,
            reason_codes=reason_codes,
            policy_notes=notes,
            constraint_checks=[],
            any_constraint_breach=False,
        )


class UnconstrainedEVPolicy(AuthorizationPolicy):
    """Maximizes expected net value without global fraud-loss or latency constraints."""

    name = "unconstrained_ev"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def choose(
        self,
        attempt: AuthorizationAttempt,
        predictions: Dict[PolicyAction, ActionPrediction],
        *,
        allowed_actions: Optional[Sequence[PolicyAction]] = None,
    ) -> PolicyChoice:
        if _all_non_decline_unavailable(predictions):
            decline_pred = predictions[PolicyAction.DECLINE_POLICY]
            return PolicyChoice(
                action=PolicyAction.BLOCK_PROCESSOR_OUTAGE,
                prediction=decline_pred,
                reason_codes=["ALL_PROCESSORS_OUTAGE_FAIL_CLOSED"],
                policy_notes="All payment processors unavailable; failing closed.",
                constraint_checks=[],
                any_constraint_breach=False,
            )

        feasible = _feasible_actions(predictions, allowed_actions)
        if not feasible:
            feasible = [PolicyAction.DECLINE_POLICY]

        best_action = max(feasible, key=lambda a: (predictions[a].expected_net_value_cents, -predictions[a].estimated_latency_ms))
        pred = predictions[best_action]
        reasons = [f"MAX_UNCONSTRAINED_EV:{best_action.value}"]
        if best_action == PolicyAction.ROUTE_SECONDARY and not predictions[PolicyAction.APPROVE_PRIMARY].feasible:
            reasons.append("PROCESSOR_OUTAGE_FAILOVER")
        if pred.missing_features:
            reasons.append("MISSING_FEATURES_IMPUTED")

        return PolicyChoice(
            action=best_action,
            prediction=pred,
            reason_codes=reasons,
            policy_notes=f"Selected highest unconstrained EV action ({pred.expected_net_value_cents:.2f} cents).",
            constraint_checks=[],
            any_constraint_breach=False,
        )


class ConstrainedRiskAwarePolicy(AuthorizationPolicy):
    """Lagrangian dual-constrained risk-aware policy enforcing loss and latency budgets.

    Guarantees:
    - Explicit handling of missing features via conservative uncertainty inflation.
    - Processor outage failover (or fail-closed when all processors are down).
    - Explicit constraint checks on every decision (zero hidden breaches).
    """

    name = "constrained_risk_aware"

    def __init__(
        self,
        config: EngineConfig,
        *,
        lambda_loss: float = 1.0,
        lambda_latency: float = 0.02,
    ) -> None:
        self.config = config
        self.lambda_loss = lambda_loss
        self.lambda_latency = lambda_latency
        self.calibration_audit: Dict[str, float] = {}

    def _effective_metrics(
        self, attempt: AuthorizationAttempt, pred: ActionPrediction
    ) -> Tuple[float, float, float]:
        """Return (conservative_expected_loss_cents, effective_loss_bps_if_approved, latency_excess_ms)."""
        if pred.action == PolicyAction.DECLINE_POLICY or pred.acceptance_prob <= 0.0:
            return 0.0, 0.0, 0.0

        n_missing = len(pred.missing_features)
        if n_missing > 0:
            mult = self.config.policy.missing_feature_uncertainty_multiplier ** min(n_missing, 3)
            # Conservative uplift on fraud probability when risk/velocity features are absent
            added_risk = self.config.policy.missing_feature_risk_penalty * (n_missing / 5.0)
            lgd_cents = (
                attempt.amount_cents * self.config.economics.loss_given_fraud_multiplier
                + self.config.economics.chargeback_fee_cents
            )
            cons_fraud = min(1.0, pred.fraud_prob_given_approved * mult + added_risk)
            cons_loss = pred.acceptance_prob * cons_fraud * lgd_cents
        else:
            cons_loss = pred.expected_loss_cents

        exp_approved_amt = max(1.0, pred.acceptance_prob * attempt.amount_cents)
        eff_loss_bps = 10_000.0 * (cons_loss / exp_approved_amt)
        lat_excess = max(0.0, pred.estimated_latency_ms - self.config.policy.max_p95_latency_ms)
        return cons_loss, eff_loss_bps, lat_excess

    def _lagrangian_score(self, attempt: AuthorizationAttempt, pred: ActionPrediction) -> float:
        if pred.action == PolicyAction.DECLINE_POLICY:
            return 0.0
        cons_loss, _, lat_excess = self._effective_metrics(attempt, pred)
        margin_rate = self.config.economics.merchant_margin_bps / 10_000.0
        exp_rev = pred.acceptance_prob * (attempt.amount_cents * margin_rate)
        action_cost = max(0.0, exp_rev - pred.expected_loss_cents - pred.expected_net_value_cents)
        # Dual penalty on expected loss and latency above budget
        return (
            exp_rev
            - action_cost
            - (1.0 + self.lambda_loss) * cons_loss
            - self.lambda_latency * (pred.estimated_latency_ms + 3.0 * lat_excess)
        )

    def fit_dual_multipliers(
        self,
        cal_attempts: Sequence[AuthorizationAttempt],
        cal_predictions: Sequence[Dict[PolicyAction, ActionPrediction]],
    ) -> Dict[str, float]:
        """Grid-search dual multipliers on calibration fold to maximize net acceptance within budgets."""
        max_loss_bps = self.config.policy.max_expected_loss_bps
        max_p95_lat = self.config.policy.max_p95_latency_ms

        best_feasible: Optional[Tuple[float, float, float, Dict[str, float]]] = None
        safest_fallback: Optional[Tuple[float, float, float, Dict[str, float]]] = None

        for lam_loss in self.config.policy.dual_lambda_loss_grid:
            for lam_lat in self.config.policy.dual_lambda_latency_grid:
                self.lambda_loss = float(lam_loss)
                self.lambda_latency = float(lam_lat)

                total_acc = 0.0
                total_loss_cents = 0.0
                total_approved_amt = 0.0
                total_net_val = 0.0
                latencies: List[float] = []

                for attempt, preds in zip(cal_attempts, cal_predictions):
                    choice = self.choose(attempt, preds)
                    p = choice.prediction
                    cons_loss, _, _ = self._effective_metrics(attempt, p)
                    total_acc += p.acceptance_prob
                    total_loss_cents += cons_loss
                    total_approved_amt += p.acceptance_prob * attempt.amount_cents
                    total_net_val += p.expected_net_value_cents
                    latencies.append(p.estimated_latency_ms)

                n = max(1, len(cal_attempts))
                net_acc_rate = total_acc / n
                loss_bps = 10_000.0 * (total_loss_cents / max(1.0, total_approved_amt))
                p95_lat = float(np.percentile(np.asarray(latencies, dtype=np.float64), 95)) if latencies else 0.0

                stats = {
                    "lambda_loss": float(lam_loss),
                    "lambda_latency": float(lam_lat),
                    "cal_net_acceptance_rate": round(net_acc_rate, 6),
                    "cal_expected_loss_bps": round(loss_bps, 4),
                    "cal_p95_latency_ms": round(p95_lat, 2),
                    "cal_expected_net_value_cents": round(total_net_val / n, 4),
                }

                # Objective: maximize net value (with tie-break on net acceptance) subject to constraints
                objective = (total_net_val / n) + 5.0 * net_acc_rate
                if loss_bps <= max_loss_bps and p95_lat <= max_p95_lat:
                    if best_feasible is None or objective > best_feasible[0]:
                        best_feasible = (objective, float(lam_loss), float(lam_lat), stats)

                # Track minimum violation fallback in case grid is overly tight
                violation = max(0.0, loss_bps - max_loss_bps) + max(0.0, (p95_lat - max_p95_lat) / 10.0)
                if safest_fallback is None or violation < safest_fallback[0]:
                    safest_fallback = (violation, float(lam_loss), float(lam_lat), stats)

        chosen = best_feasible if best_feasible is not None else safest_fallback
        assert chosen is not None
        self.lambda_loss = chosen[1]
        self.lambda_latency = chosen[2]
        self.calibration_audit = dict(chosen[3])
        self.calibration_audit["constraints_satisfied_on_cal"] = best_feasible is not None
        return self.calibration_audit

    def choose(
        self,
        attempt: AuthorizationAttempt,
        predictions: Dict[PolicyAction, ActionPrediction],
        *,
        allowed_actions: Optional[Sequence[PolicyAction]] = None,
    ) -> PolicyChoice:
        if _all_non_decline_unavailable(predictions):
            decline_pred = predictions[PolicyAction.DECLINE_POLICY]
            return PolicyChoice(
                action=PolicyAction.BLOCK_PROCESSOR_OUTAGE,
                prediction=decline_pred,
                reason_codes=["ALL_PROCESSORS_OUTAGE_FAIL_CLOSED"],
                policy_notes="All payment processors are in OUTAGE; failing closed safely.",
                constraint_checks=[
                    ConstraintCheckResult(
                        constraint_name="processor_availability",
                        limit_value=1.0,
                        observed_value=0.0,
                        passed=False,
                        message="All configured processors in OUTAGE state; blocked rather than silently failing.",
                    )
                ],
                any_constraint_breach=False,
            )

        feasible = _feasible_actions(predictions, allowed_actions)
        if not feasible:
            feasible = [PolicyAction.DECLINE_POLICY]

        # Enforce hard per-attempt latency ceiling unless only higher-latency step-up is explicitly requested
        max_lat = self.config.policy.max_p95_latency_ms
        latency_safe = [
            a for a in feasible if a == PolicyAction.DECLINE_POLICY or predictions[a].estimated_latency_ms <= max_lat * 1.15
        ]
        candidate_pool = latency_safe if latency_safe else feasible

        best_action = max(
            candidate_pool,
            key=lambda a: (self._lagrangian_score(attempt, predictions[a]), -predictions[a].estimated_latency_ms),
        )
        pred = predictions[best_action]
        cons_loss, eff_loss_bps, _ = self._effective_metrics(attempt, pred)

        reasons: List[str] = [f"CONSTRAINED_OPTIMAL:{best_action.value}"]
        if best_action == PolicyAction.ROUTE_SECONDARY and not predictions[PolicyAction.APPROVE_PRIMARY].feasible:
            reasons.append("PROCESSOR_OUTAGE_FAILOVER")
        if pred.missing_features:
            reasons.append("MISSING_FEATURES_CONSERVATIVE_PENALTY")

        # Hard single-attempt guard: never silently breach extreme loss ceiling (4x portfolio budget)
        hard_attempt_loss_cap_bps = self.config.policy.max_expected_loss_bps * 4.0
        if best_action != PolicyAction.DECLINE_POLICY and eff_loss_bps > hard_attempt_loss_cap_bps:
            # Try STEP_UP_3DS if feasible and within cap; otherwise DECLINE_POLICY
            step_up = predictions.get(PolicyAction.STEP_UP_3DS)
            if (
                step_up is not None
                and step_up.feasible
                and PolicyAction.STEP_UP_3DS in feasible
                and self._effective_metrics(attempt, step_up)[1] <= hard_attempt_loss_cap_bps
            ):
                best_action = PolicyAction.STEP_UP_3DS
                pred = step_up
                cons_loss, eff_loss_bps, _ = self._effective_metrics(attempt, pred)
                reasons.append("HARD_LOSS_CAP_STEP_UP_3DS")
            elif PolicyAction.DECLINE_POLICY in feasible:
                best_action = PolicyAction.DECLINE_POLICY
                pred = predictions[PolicyAction.DECLINE_POLICY]
                cons_loss, eff_loss_bps, _ = 0.0, 0.0, 0.0
                reasons.append("HARD_LOSS_CAP_DECLINE")

        checks = [
            ConstraintCheckResult(
                constraint_name="attempt_hard_loss_cap_bps",
                limit_value=round(hard_attempt_loss_cap_bps, 2),
                observed_value=round(eff_loss_bps, 2),
                passed=eff_loss_bps <= hard_attempt_loss_cap_bps + 1e-9,
                message=f"Conservative attempt loss {eff_loss_bps:.2f} bps vs hard cap {hard_attempt_loss_cap_bps:.2f} bps.",
            ),
            ConstraintCheckResult(
                constraint_name="attempt_latency_guard_ms",
                limit_value=round(max_lat * 1.15, 2),
                observed_value=round(pred.estimated_latency_ms, 2),
                passed=pred.estimated_latency_ms <= max_lat * 1.15 + 1e-9,
                message=f"Estimated latency {pred.estimated_latency_ms:.1f} ms vs guard {max_lat * 1.15:.1f} ms.",
            ),
        ]
        any_breach = any(not c.passed for c in checks)

        notes = (
            f"Constrained policy (lambda_loss={self.lambda_loss}, lambda_latency={self.lambda_latency}) "
            f"selected {best_action.value} with conservative loss {cons_loss:.2f} cents ({eff_loss_bps:.2f} bps)."
        )
        return PolicyChoice(
            action=best_action,
            prediction=pred,
            reason_codes=reasons,
            policy_notes=notes,
            constraint_checks=checks,
            any_constraint_breach=any_breach,
        )


def audit_batch_constraints(
    attempts: Sequence[AuthorizationAttempt],
    decisions: Sequence[DecisionRecord],
    *,
    max_expected_loss_bps: float,
    max_p95_latency_ms: float,
) -> Dict[str, object]:
    """Audit portfolio-level constraints so zero hidden constraint breaches can occur."""
    if len(attempts) != len(decisions):
        raise ValueError("attempts and decisions must have identical length")
    if not decisions:
        return {
            "n_decisions": 0,
            "net_acceptance_rate": 0.0,
            "expected_loss_bps": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "decision_level_breaches": 0,
            "portfolio_breaches": [],
            "passed": True,
        }

    total_acc = sum(d.predicted_acceptance_prob for d in decisions)
    total_loss_cents = sum(d.expected_loss_cents for d in decisions)
    total_approved_amt = sum(d.predicted_acceptance_prob * a.amount_cents for a, d in zip(attempts, decisions))
    latencies = np.asarray([d.estimated_latency_ms for d in decisions], dtype=np.float64)

    net_acc_rate = total_acc / len(decisions)
    expected_loss_bps = 10_000.0 * (total_loss_cents / max(1.0, total_approved_amt))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))

    decision_breaches = sum(1 for d in decisions if d.any_constraint_breach)
    portfolio_breaches: List[str] = []
    if expected_loss_bps > max_expected_loss_bps:
        portfolio_breaches.append(
            f"PORTFOLIO_EXPECTED_LOSS_BREACH:{expected_loss_bps:.2f}bps>{max_expected_loss_bps:.2f}bps"
        )
    if p95_lat > max_p95_latency_ms:
        portfolio_breaches.append(
            f"PORTFOLIO_P95_LATENCY_BREACH:{p95_lat:.2f}ms>{max_p95_latency_ms:.2f}ms"
        )

    return {
        "n_decisions": len(decisions),
        "net_acceptance_rate": round(net_acc_rate, 6),
        "expected_loss_bps": round(expected_loss_bps, 4),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "decision_level_breaches": decision_breaches,
        "portfolio_breaches": portfolio_breaches,
        "passed": decision_breaches == 0 and len(portfolio_breaches) == 0,
    }
