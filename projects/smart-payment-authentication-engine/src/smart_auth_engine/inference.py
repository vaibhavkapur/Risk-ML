"""Inference orchestration, structured logging, diagnostic counters, and replay support."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from smart_auth_engine.config import EngineConfig
from smart_auth_engine.contracts import (
    AuthAction,
    FeatureContribution,
    PaymentEvent,
    PolicyDecision,
    StepUpExecutionResult,
)
from smart_auth_engine.features import ExtractedFeatures, extract_features
from smart_auth_engine.policy import PolicyThresholds, decide_authentication_action
from smart_auth_engine.step_up import simulate_step_up
from smart_auth_engine.store import DecisionStore, InMemoryDecisionStore

logger = logging.getLogger("smart_auth_engine.inference")


class RiskScorerProtocol(Protocol):
    """Minimal scorer interface decoupled from concrete ML frameworks."""

    model_version: str
    feature_version: str

    def predict_proba_event(
        self,
        event: PaymentEvent,
    ) -> tuple[float, float, list[FeatureContribution]]:
        ...


@dataclass
class DiagnosticCounters:
    total_requests: int = 0
    action_counts: Dict[str, int] = field(
        default_factory=lambda: {a.value: 0 for a in AuthAction}
    )
    mandatory_rule_hits: int = 0
    override_hits: int = 0
    fallback_hits: int = 0
    model_outage_hits: int = 0
    missing_feature_fallbacks: int = 0
    step_up_simulations: int = 0
    replay_requests: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "action_counts": dict(self.action_counts),
            "mandatory_rule_hits": self.mandatory_rule_hits,
            "override_hits": self.override_hits,
            "fallback_hits": self.fallback_hits,
            "model_outage_hits": self.model_outage_hits,
            "missing_feature_fallbacks": self.missing_feature_fallbacks,
            "step_up_simulations": self.step_up_simulations,
            "replay_requests": self.replay_requests,
        }


class AuthenticationEngine:
    """Orchestrates feature extraction, model scoring, policy evaluation, step-up simulation, and replay."""

    def __init__(
        self,
        cfg: EngineConfig,
        thresholds: PolicyThresholds,
        model: Optional[Any] = None,
        store: Optional[DecisionStore] = None,
        model_available: bool = True,
    ) -> None:
        self.cfg = cfg
        self.thresholds = thresholds
        self.model = model
        self.store: DecisionStore = store if store is not None else InMemoryDecisionStore()
        self.model_available = model_available and (model is not None)
        self.counters = DiagnosticCounters()

    @property
    def active_model_version(self) -> str:
        if self.model is not None and hasattr(self.model, "model_version"):
            return str(self.model.model_version)
        return "unavailable_model"

    def set_model_outage(self, outage: bool) -> None:
        """Toggle simulated model outage for resilience testing and operational drills."""
        self.model_available = (not outage) and (self.model is not None)

    def _score_safely(
        self,
        event: PaymentEvent,
        extracted: ExtractedFeatures,
    ) -> tuple[Optional[float], Optional[float], list[FeatureContribution], bool, Optional[str]]:
        if not self.model_available or self.model is None:
            return None, None, [], True, "FALLBACK_MODEL_UNAVAILABLE"
        try:
            if hasattr(self.model, "predict_proba_features"):
                cal_p, raw_p, explanations = self.model.predict_proba_features(extracted)
            else:
                cal_p, raw_p, explanations = self.model.predict_proba_event(event)
            if not (0.0 <= float(cal_p) <= 1.0):
                raise ValueError(f"Out-of-range calibrated probability: {cal_p}")
            return float(cal_p), float(raw_p), explanations, False, None
        except Exception as exc:
            logger.warning(
                json.dumps(
                    {
                        "event": "model_scoring_failure",
                        "payment_id": event.payment_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    sort_keys=True,
                )
            )
            return None, None, [], True, f"FALLBACK_MODEL_ERROR:{type(exc).__name__}"

    def evaluate(
        self,
        event: PaymentEvent,
        record_store: bool = True,
        simulate_outage: bool = False,
    ) -> PolicyDecision:
        """Evaluate a payment event and return a PolicyDecision with audit logging."""
        extracted = extract_features(event)

        if simulate_outage:
            cal_p, raw_p, explanations, outage, outage_reason = (
                None,
                None,
                [],
                True,
                "FALLBACK_SIMULATED_OUTAGE",
            )
        else:
            cal_p, raw_p, explanations, outage, outage_reason = self._score_safely(event, extracted)

        decision = decide_authentication_action(
            event=event,
            extracted=extracted,
            thresholds=self.thresholds,
            cfg=self.cfg,
            model_version=self.active_model_version,
            calibrated_prob=cal_p,
            raw_prob=raw_p,
            explanations=explanations,
            model_outage=outage,
            outage_reason=outage_reason,
        )

        # Update diagnostic counters
        self.counters.total_requests += 1
        self.counters.action_counts[decision.action.value] += 1
        if decision.mandatory_rule_triggered:
            self.counters.mandatory_rule_hits += 1
        if decision.override_triggered:
            self.counters.override_hits += 1
        if decision.fallback_used:
            self.counters.fallback_hits += 1
            if outage:
                self.counters.model_outage_hits += 1
            if any(r.startswith("FALLBACK_MISSING_") or r.startswith("FALLBACK_HIGH_MISSING") for r in decision.reason_codes):
                self.counters.missing_feature_fallbacks += 1

        if record_store:
            self.store.save_record(event=event, decision=decision, step_up=None)

        logger.info(
            json.dumps(
                {
                    "event": "policy_decision",
                    "decision_id": decision.decision_id,
                    "payment_id": decision.payment_id,
                    "action": decision.action.value,
                    "calibrated_fraud_prob": decision.calibrated_fraud_prob,
                    "mandatory_rule_triggered": decision.mandatory_rule_triggered,
                    "override_triggered": decision.override_triggered,
                    "fallback_used": decision.fallback_used,
                    "reason_codes": decision.reason_codes,
                    "model_version": decision.model_version,
                    "policy_version": decision.policy_version,
                },
                sort_keys=True,
            )
        )
        return decision

    def evaluate_and_simulate_step_up(
        self,
        event: PaymentEvent,
        actual_is_fraud: Optional[bool] = None,
        record_store: bool = True,
        simulate_outage: bool = False,
    ) -> StepUpExecutionResult:
        """Evaluate policy and simulate the complete step-up authentication flow."""
        decision = self.evaluate(
            event=event,
            record_store=False,
            simulate_outage=simulate_outage,
        )
        result = simulate_step_up(
            event=event,
            decision=decision,
            cfg=self.cfg,
            actual_is_fraud=actual_is_fraud,
        )
        self.counters.step_up_simulations += 1
        if record_store:
            self.store.save_record(event=event, decision=decision, step_up=result)
        return result

    def replay_decision(self, decision_id: str) -> Dict[str, Any]:
        """Deterministically replay a previously recorded decision and verify parity."""
        record = self.store.get_record(decision_id)
        if record is None:
            raise KeyError(f"Decision ID not found for replay: {decision_id}")
        event, original_decision, _ = record
        replayed = self.evaluate(
            event=event,
            record_store=False,
            simulate_outage=original_decision.fallback_used and (original_decision.calibrated_fraud_prob is None) and any(
                "OUTAGE" in r or "UNAVAILABLE" in r for r in original_decision.reason_codes
            ),
        )
        self.counters.replay_requests += 1
        exact_match = (
            replayed.decision_id == original_decision.decision_id
            and replayed.action == original_decision.action
            and replayed.mandatory_rule_triggered == original_decision.mandatory_rule_triggered
            and replayed.override_triggered == original_decision.override_triggered
            and replayed.fallback_used == original_decision.fallback_used
            and replayed.reason_codes == original_decision.reason_codes
        )
        return {
            "decision_id": decision_id,
            "exact_match": exact_match,
            "original": original_decision.model_dump(mode="json"),
            "replayed": replayed.model_dump(mode="json"),
        }
