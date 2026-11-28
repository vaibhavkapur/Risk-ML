"""Adaptive thresholds, step-up recommendations, analyst evidence, and fallback policy."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import EngineConfig
from .models import ATORiskModelBundle, rule_oracle_score
from .schemas import (
    AnalystEvidence,
    EvidenceItem,
    FallbackReason,
    FeatureVector,
    LoginSessionEvent,
    ModelScore,
    PolicyDecision,
    StepUpAction,
    make_decision_id,
)


_FEATURE_EXPLANATIONS: Dict[str, str] = {
    "is_new_device_for_user": "Device has not been seen for this user within the active lookback window.",
    "is_impossible_travel": "Geographic velocity from prior login exceeds physically plausible travel speed.",
    "geo_velocity_kmh": "Estimated travel speed (km/h) from previous known login coordinates.",
    "ip_reputation_risk": "Network reputation risk score for the observed synthetic IP address.",
    "device_trust_score": "Synthetic device attestation / trust score (lower indicates higher risk).",
    "failed_attempts_1h": "Failed authentication attempts for the user within the prior 1 hour.",
    "failed_attempts_24h": "Failed authentication attempts for the user within the prior 24 hours.",
    "sensitive_action_count": "Number of high-risk account changes attempted in the session.",
    "session_action_velocity_per_min": "Rate of sensitive session actions per minute.",
    "auth_factor_downgrade": "Session authenticated with a weaker factor than user's prior strongest factor.",
    "credential_stuffing_burst": "Rapid burst of failed login attempts exceeding threshold.",
    "hour_of_day_deviation": "Circular hour deviation from user's historical login time.",
    "is_shared_network_ip": "IP address matches shared corporate/campus NAT pattern (mitigates false blocks).",
    "profile_expired": "User's historical behavioral profile expired past the configured TTL.",
    "missing_feature_ratio": "Fraction of optional raw telemetry fields missing from the event.",
}


def compute_adaptive_thresholds(
    event: LoginSessionEvent,
    fv: FeatureVector,
    config: EngineConfig,
    drift_alert_active: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute adaptive step-up thresholds adjusted for account tier, shared network, and drift."""
    base = config.policy.base_thresholds
    adjustments: Dict[str, float] = {}

    delta = 0.0
    if event.account_tier == "high_value":
        adjustments["high_value_account"] = config.policy.high_value_account_delta
        delta += config.policy.high_value_account_delta

    if (
        fv.values["is_shared_network_ip"] >= 1.0
        and fv.values["credential_stuffing_burst"] == 0.0
        and fv.values["is_impossible_travel"] == 0.0
        and fv.values["sensitive_action_count"] == 0.0
    ):
        adjustments["shared_network_dampening"] = config.policy.shared_network_dampening_delta
        delta += config.policy.shared_network_dampening_delta

    if drift_alert_active:
        adjustments["drift_tightening"] = config.policy.drift_tightening_delta
        delta += config.policy.drift_tightening_delta

    def _clamp(val: float, lo: float, hi: float) -> float:
        return round(max(lo, min(hi, val)), 6)

    otp = _clamp(base.challenge_otp + delta, 0.10, 0.60)
    webauthn = _clamp(base.challenge_webauthn + delta, otp + 0.10, 0.85)
    block = _clamp(base.block_and_reset + delta, webauthn + 0.08, 0.96)

    thresholds = {
        "challenge_otp": otp,
        "challenge_webauthn": webauthn,
        "block_and_reset": block,
    }
    return thresholds, adjustments


def _collect_triggered_rules(
    fv: FeatureVector,
    config: EngineConfig,
    fallback_reason: FallbackReason,
) -> List[str]:
    v = fv.values
    rules: List[str] = []
    if v["failed_attempts_1h"] >= config.policy.repeated_attempt_hard_block:
        rules.append("HARD_BLOCK_REPEATED_ATTEMPTS_1H")
    elif v["failed_attempts_1h"] >= config.policy.repeated_attempt_hard_step_up:
        rules.append("HARD_STEP_UP_REPEATED_ATTEMPTS_1H")
    if v["is_impossible_travel"] >= 1.0:
        rules.append("IMPOSSIBLE_TRAVEL_DETECTED")
    if v["credential_stuffing_burst"] >= 1.0:
        rules.append("CREDENTIAL_STUFFING_BURST")
    if v["sensitive_action_count"] >= 2.0 and v["is_new_device_for_user"] >= 1.0:
        rules.append("NEW_DEVICE_SENSITIVE_ACTION_BURST")
    if v["auth_factor_downgrade"] >= 1.0 and v["is_new_device_for_user"] >= 1.0:
        rules.append("AUTH_FACTOR_DOWNGRADE_ON_NEW_DEVICE")
    if v["is_shared_network_ip"] >= 1.0:
        rules.append("SHARED_NETWORK_CONTEXT_DAMPENING")
    if fv.profile_expired:
        rules.append("USER_PROFILE_EXPIRED_TTL")
    if fv.missing_fields:
        rules.append(f"MISSING_TELEMETRY:{','.join(fv.missing_fields)}")
    if fallback_reason != FallbackReason.NONE:
        rules.append(f"FALLBACK_POLICY_ACTIVE:{fallback_reason.value}")
    return rules


def build_analyst_evidence(
    fv: FeatureVector,
    action: StepUpAction,
    calibrated_score: float,
    triggered_rules: List[str],
    model_bundle: Optional[ATORiskModelBundle],
    fallback_reason: FallbackReason,
) -> AnalystEvidence:
    """Produce ranked, human-readable analyst evidence and playbook guidance."""
    v = fv.values
    scored_items: List[Tuple[float, EvidenceItem]] = []

    if model_bundle is not None:
        weights = model_bundle.linear_feature_weights
        stats = model_bundle.reference_feature_stats
        for fname, val in v.items():
            if fname not in _FEATURE_EXPLANATIONS:
                continue
            fstat = stats.get(fname, {})
            mean = fstat.get("lr_mean", fstat.get("mean", 0.0))
            scale = fstat.get("lr_scale", fstat.get("std", 1.0))
            z = (val - mean) / max(scale, 0.10)
            w = weights.get(fname, 0.0)
            contrib = w * z
            if abs(contrib) < 0.05 and val == 0.0:
                continue
            direction = (
                "risk_increasing"
                if contrib > 0.02
                else ("risk_mitigating" if contrib < -0.02 else "context")
            )
            scored_items.append(
                (
                    abs(contrib),
                    EvidenceItem(
                        factor=fname,
                        direction=direction,
                        contribution=round(float(contrib), 4),
                        observed_value=round(float(val), 4),
                        explanation=_FEATURE_EXPLANATIONS[fname],
                    ),
                )
            )
    else:
        # Deterministic fallback explanations using transparent rule weights
        fallback_signals = [
            ("is_impossible_travel", 0.28 * v["is_impossible_travel"]),
            ("ip_reputation_risk", 0.24 * v["ip_reputation_risk"]),
            ("device_trust_score", 0.18 * (0.5 - v["device_trust_score"])),
            ("sensitive_action_count", 0.20 * min(v["sensitive_action_count"] / 3.0, 1.0)),
            ("failed_attempts_1h", 0.16 * min(v["failed_attempts_1h"] / 5.0, 1.0)),
            ("is_new_device_for_user", 0.12 * v["is_new_device_for_user"]),
            ("auth_factor_downgrade", 0.12 * v["auth_factor_downgrade"]),
            ("profile_expired", 0.05 * v["profile_expired"]),
            ("is_shared_network_ip", -0.06 * v["is_shared_network_ip"]),
        ]
        for fname, contrib in fallback_signals:
            if abs(contrib) < 0.01:
                continue
            direction = "risk_increasing" if contrib > 0 else "risk_mitigating"
            scored_items.append(
                (
                    abs(contrib),
                    EvidenceItem(
                        factor=fname,
                        direction=direction,
                        contribution=round(float(contrib), 4),
                        observed_value=round(float(v[fname]), 4),
                        explanation=_FEATURE_EXPLANATIONS[fname],
                    ),
                )
            )

    scored_items.sort(key=lambda pair: (-pair[0], pair[1].factor))
    top_factors = [item for _, item in scored_items[:5]]

    playbooks = {
        StepUpAction.ALLOW: "No analyst intervention required; continue passive session monitoring.",
        StepUpAction.CHALLENGE_OTP: "Require one-time passcode step-up; log outcome for device trust graduation.",
        StepUpAction.CHALLENGE_WEBAUTHN: "Require phishing-resistant WebAuthn/passkey challenge and queue for rapid analyst triage.",
        StepUpAction.BLOCK_AND_RESET: "Block session immediately, revoke active tokens, initiate password/MFA reset, and open high-priority ATO case.",
    }

    mode_str = (
        f"fallback rule policy ({fallback_reason.value})"
        if fallback_reason != FallbackReason.NONE
        else "calibrated ML + adaptive threshold policy"
    )
    summary = (
        f"Recommended {action.value} (risk={calibrated_score:.3f}) via {mode_str}; "
        f"{len(triggered_rules)} rule signal(s) triggered."
    )

    return AnalystEvidence(
        summary=summary,
        top_factors=top_factors,
        triggered_rules=triggered_rules,
        missing_fields=list(fv.missing_fields),
        profile_expired=fv.profile_expired,
        recommended_playbook=playbooks[action],
    )


def decide_step_up_action(
    event: LoginSessionEvent,
    fv: FeatureVector,
    model_score: Optional[ModelScore],
    config: EngineConfig,
    model_bundle: Optional[ATORiskModelBundle] = None,
    fallback_reason: FallbackReason = FallbackReason.NONE,
    drift_alert_active: bool = False,
) -> PolicyDecision:
    """Apply adaptive thresholds, guardrails, and fallback policy to produce a PolicyDecision."""
    # Automatically trigger fallback when missing feature ratio exceeds configured threshold
    active_fallback = fallback_reason
    if (
        active_fallback == FallbackReason.NONE
        and fv.values["missing_feature_ratio"] >= config.features.missing_feature_fallback_ratio
    ):
        active_fallback = FallbackReason.EXCESSIVE_MISSING_FEATURES
    if active_fallback == FallbackReason.NONE and model_score is None:
        active_fallback = FallbackReason.MODEL_UNAVAILABLE

    thresholds, adjustments = compute_adaptive_thresholds(
        event=event,
        fv=fv,
        config=config,
        drift_alert_active=drift_alert_active,
    )
    triggered_rules = _collect_triggered_rules(fv, config, active_fallback)

    if active_fallback != FallbackReason.NONE:
        # Deterministic conservative rule-based fallback policy
        fallback_score = rule_oracle_score(fv.values)
        # Missing telemetry adds conservative uncertainty cushion so degraded events don't fail open silently
        if active_fallback in (
            FallbackReason.EXCESSIVE_MISSING_FEATURES,
            FallbackReason.STORE_UNAVAILABLE,
        ):
            fallback_score = min(1.0, round(fallback_score + 0.15, 6))
        calibrated_score = fallback_score
        anomaly_score = model_score.anomaly_score if model_score is not None else fallback_score
        supervised_prob = model_score.supervised_probability if model_score is not None else fallback_score
        model_version = (
            f"fallback:{active_fallback.value}"
            if model_score is None
            else f"{model_score.model_version}+fallback:{active_fallback.value}"
        )
    else:
        assert model_score is not None
        calibrated_score = model_score.calibrated_risk_score
        anomaly_score = model_score.anomaly_score
        supervised_prob = model_score.supervised_probability
        model_version = model_score.model_version

    # Threshold-based step-up action
    if calibrated_score >= thresholds["block_and_reset"]:
        action = StepUpAction.BLOCK_AND_RESET
    elif calibrated_score >= thresholds["challenge_webauthn"]:
        action = StepUpAction.CHALLENGE_WEBAUTHN
    elif calibrated_score >= thresholds["challenge_otp"]:
        action = StepUpAction.CHALLENGE_OTP
    else:
        action = StepUpAction.ALLOW

    # Deterministic safety guardrails (repeated attempts, severe takeover signals, expired profile)
    failed_1h = fv.values["failed_attempts_1h"]
    if failed_1h >= config.policy.repeated_attempt_hard_block:
        action = StepUpAction.BLOCK_AND_RESET
    elif failed_1h >= config.policy.repeated_attempt_hard_step_up and action in (
        StepUpAction.ALLOW,
        StepUpAction.CHALLENGE_OTP,
    ):
        action = StepUpAction.CHALLENGE_WEBAUTHN

    # Compromised session high-confidence guardrail: impossible travel + sensitive changes
    if fv.values["is_impossible_travel"] >= 1.0 and fv.values["sensitive_action_count"] >= 2.0:
        action = StepUpAction.BLOCK_AND_RESET

    # Feature expiry guardrail: when historical profile expired and new device/sensitive action appears,
    # never allow silently without at least OTP step-up
    if fv.profile_expired and action == StepUpAction.ALLOW and (
        fv.values["is_new_device_for_user"] >= 1.0 or fv.values["sensitive_action_count"] >= 1.0
    ):
        action = StepUpAction.CHALLENGE_OTP

    # Store outage / excessive missingness guardrail: never fail completely open on sensitive actions
    if (
        active_fallback in (FallbackReason.STORE_UNAVAILABLE, FallbackReason.EXCESSIVE_MISSING_FEATURES)
        and action == StepUpAction.ALLOW
    ):
        action = StepUpAction.CHALLENGE_OTP

    evidence = build_analyst_evidence(
        fv=fv,
        action=action,
        calibrated_score=calibrated_score,
        triggered_rules=triggered_rules,
        model_bundle=None if active_fallback != FallbackReason.NONE else model_bundle,
        fallback_reason=active_fallback,
    )

    decision_id = make_decision_id(
        event_id=event.event_id,
        timestamp=event.timestamp,
        model_version=model_version,
        feature_version=fv.feature_version,
        policy_version=config.versions.policy_version,
    )

    return PolicyDecision(
        decision_id=decision_id,
        event_id=event.event_id,
        user_id=event.user_id,
        session_id=event.session_id,
        timestamp=event.timestamp,
        schema_version=event.schema_version,
        feature_version=fv.feature_version,
        model_version=model_version,
        policy_version=config.versions.policy_version,
        action=action,
        calibrated_risk_score=calibrated_score,
        anomaly_score=anomaly_score,
        supervised_probability=supervised_prob,
        thresholds_applied=thresholds,
        threshold_adjustments=adjustments,
        fallback_used=active_fallback != FallbackReason.NONE,
        fallback_reason=active_fallback,
        drift_alert_active=drift_alert_active,
        evidence=evidence,
        features_snapshot=dict(fv.values),
    )
