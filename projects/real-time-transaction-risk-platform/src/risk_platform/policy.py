"""Pure domain risk decision policy with conservative stale-cache, unknown-input, and timeout fallbacks."""

from __future__ import annotations

from typing import Dict, List, Optional

from risk_platform.config import ServingConfig
from risk_platform.schemas import (
    DEFAULT_FX_TO_USD,
    FeatureAttribution,
    FeatureVector,
    RiskDecision,
    TransactionEvent,
    compute_decision_id,
)

POLICY_VERSION = "policy.v1"
FALLBACK_MODEL_VERSION = "fallback.rules.v1"


class PolicyEngine:
    """Stateless risk decision engine separating policy thresholds & fallbacks from ML inference."""

    def __init__(
        self,
        serving_config: ServingConfig,
        policy_version: str = POLICY_VERSION,
        fx_to_usd: Optional[Dict[str, float]] = None,
    ) -> None:
        self.cfg = serving_config
        self.policy_version = policy_version
        self.fx_to_usd = dict(fx_to_usd or DEFAULT_FX_TO_USD)

    def _base_reason_codes(self, fv: FeatureVector) -> List[str]:
        codes: List[str] = []
        v = fv.values
        if v["tx_count_5m"] >= 3.0:
            codes.append("HIGH_VELOCITY_5M")
        if v["tx_count_1h"] >= 6.0:
            codes.append("HIGH_VELOCITY_1H")
        if v["country_changed_from_prev"] >= 1.0:
            codes.append("COUNTRY_CHANGED")
        if v["amount_ratio_to_24h_avg"] >= 4.0:
            codes.append("AMOUNT_SPIKE_VS_24H")
        if v["high_risk_context_score"] >= 0.70:
            codes.append("HIGH_RISK_CONTEXT")
        if fv.feature_status == "COLD_START":
            codes.append("ACCOUNT_COLD_START")
        return codes

    def evaluate_model_score(
        self,
        event: TransactionEvent,
        feature_vector: FeatureVector,
        calibrated_score: float,
        raw_score: float,
        model_version: str,
        attributions: List[FeatureAttribution],
        latency_ms: float,
        decided_at_ms: int,
    ) -> RiskDecision:
        """Apply risk thresholds plus conservative safeguards for stale features and unknown inputs."""
        amount_usd = event.amount_usd(self.fx_to_usd)
        reason_codes = self._base_reason_codes(feature_vector)
        fallback_used = False
        fallback_reason: Optional[str] = None

        if calibrated_score >= self.cfg.decline_threshold:
            action = "DECLINE"
            reason_codes.append("SCORE_ABOVE_DECLINE_THRESHOLD")
        elif calibrated_score >= self.cfg.review_threshold:
            action = "REVIEW"
            reason_codes.append("SCORE_ABOVE_REVIEW_THRESHOLD")
        else:
            action = "APPROVE"

        # Conservative safeguard 1: stale cache features
        if feature_vector.feature_status == "STALE":
            reason_codes.append("STALE_FEATURES")
            if action == "APPROVE" and (
                amount_usd >= self.cfg.stale_review_amount_usd
                or feature_vector.values["high_risk_context_score"] >= 0.50
            ):
                action = "REVIEW"
                fallback_used = True
                fallback_reason = "STALE_FEATURES"
                reason_codes.append("STALE_CACHE_CONSERVATIVE_ESCALATION")

        # Conservative safeguard 2: unknown inputs (unknown MCC / unknown country)
        if feature_vector.unknown_input_flags:
            reason_codes.extend(feature_vector.unknown_input_flags)
            if action == "APPROVE" and (
                amount_usd >= self.cfg.stale_review_amount_usd
                or calibrated_score >= self.cfg.review_threshold * 0.65
            ):
                action = "REVIEW"
                fallback_used = True
                fallback_reason = fallback_reason or "UNKNOWN_INPUT_CONSERVATIVE"
                reason_codes.append("UNKNOWN_INPUT_CONSERVATIVE_ESCALATION")

        decision_id = compute_decision_id(
            event_id=event.event_id,
            model_version=model_version,
            feature_version=feature_vector.feature_version,
            policy_version=self.policy_version,
            as_of_ts_ms=feature_vector.as_of_ts_ms,
        )

        return RiskDecision(
            decision_id=decision_id,
            event_id=event.event_id,
            account_id=event.account_id,
            action=action,
            risk_score=float(calibrated_score),
            raw_score=float(raw_score),
            model_version=model_version,
            feature_version=feature_vector.feature_version,
            policy_version=self.policy_version,
            feature_freshness_ms=feature_vector.feature_freshness_ms,
            feature_status=feature_vector.feature_status,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            reason_codes=reason_codes,
            top_attributions=attributions,
            latency_ms=latency_ms,
            decided_at_ms=decided_at_ms,
        )

    def evaluate_fallback(
        self,
        event: TransactionEvent,
        feature_vector: FeatureVector,
        fallback_reason: str,
        latency_ms: float,
        decided_at_ms: int,
    ) -> RiskDecision:
        """Deterministic conservative rule-based fallback invoked on model timeout or error."""
        amount_usd = event.amount_usd(self.fx_to_usd)
        v = feature_vector.values
        reason_codes = self._base_reason_codes(feature_vector)
        reason_codes.append(f"FALLBACK:{fallback_reason}")

        if feature_vector.feature_status == "STALE":
            reason_codes.append("STALE_FEATURES")
        if feature_vector.unknown_input_flags:
            reason_codes.extend(feature_vector.unknown_input_flags)

        # Conservative deterministic rule scoring
        if (
            amount_usd >= self.cfg.fallback_decline_amount_usd
            or v["tx_count_5m"] >= 4.0
            or (v["country_changed_from_prev"] >= 1.0 and v["high_risk_context_score"] >= 0.70)
        ):
            action = "DECLINE"
            fallback_score = 0.90
            reason_codes.append("FALLBACK_CONSERVATIVE_DECLINE")
        elif (
            amount_usd >= self.cfg.fallback_review_amount_usd
            or feature_vector.feature_status == "STALE"
            or bool(feature_vector.unknown_input_flags)
            or v["high_risk_context_score"] >= 0.55
            or v["tx_count_5m"] >= 2.0
        ):
            action = "REVIEW"
            fallback_score = 0.55
            reason_codes.append("FALLBACK_CONSERVATIVE_REVIEW")
        else:
            action = "APPROVE"
            fallback_score = 0.15
            reason_codes.append("FALLBACK_LOW_RISK_APPROVE")

        decision_id = compute_decision_id(
            event_id=event.event_id,
            model_version=FALLBACK_MODEL_VERSION,
            feature_version=feature_vector.feature_version,
            policy_version=self.policy_version,
            as_of_ts_ms=feature_vector.as_of_ts_ms,
        )

        return RiskDecision(
            decision_id=decision_id,
            event_id=event.event_id,
            account_id=event.account_id,
            action=action,
            risk_score=fallback_score,
            raw_score=None,
            model_version=FALLBACK_MODEL_VERSION,
            feature_version=feature_vector.feature_version,
            policy_version=self.policy_version,
            feature_freshness_ms=feature_vector.feature_freshness_ms,
            feature_status=feature_vector.feature_status,
            fallback_used=True,
            fallback_reason=fallback_reason,
            reason_codes=reason_codes,
            top_attributions=[],
            latency_ms=latency_ms,
            decided_at_ms=decided_at_ms,
        )
