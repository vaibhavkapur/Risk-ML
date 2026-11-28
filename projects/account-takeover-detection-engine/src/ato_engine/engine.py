"""Core ATO Detection Engine orchestrator isolated from HTTP/CLI I/O."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

from .audit import DecisionAuditLogger, DiagnosticCounters
from .config import EngineConfig
from .drift import evaluate_drift
from .features import compute_features, extract_features_from_store
from .models import ATORiskModelBundle
from .policy import decide_step_up_action
from .schemas import (
    ContractValidationError,
    DriftReport,
    FallbackReason,
    FeatureVector,
    LoginSessionEvent,
    ModelScore,
    PolicyDecision,
    StoreUnavailableError,
    UnsupportedInputError,
    parse_and_validate_event,
)
from .store import InMemoryStateStore, RedisStateStore, StateStore


def build_state_store(config: EngineConfig) -> StateStore:
    """Construct configured StateStore adapter (in-memory default or optional Redis)."""
    if config.store.backend == "redis":
        assert config.store.redis_url is not None
        return RedisStateStore(
            redis_url=config.store.redis_url,
            key_prefix=config.store.redis_key_prefix,
        )
    return InMemoryStateStore()


class ATODetectionEngine:
    """Stateful, strictly causal Account Takeover detection engine.

    Enforces:
    - Input contract & supported-value validation
    - Strictly causal feature extraction (prior events only; current event recorded AFTER scoring)
    - Graceful fallback policy on model unavailability, model staleness, store outage,
      excessive missing features, or critical drift circuit breaker
    - Mandatory audit logging of every decision
    """

    def __init__(
        self,
        config: EngineConfig,
        model_bundle: Optional[ATORiskModelBundle] = None,
        store: Optional[StateStore] = None,
        audit_logger: Optional[DecisionAuditLogger] = None,
    ) -> None:
        self.config = config
        self.model_bundle = model_bundle
        self.store: StateStore = store if store is not None else build_state_store(config)
        self.audit_logger = (
            audit_logger
            if audit_logger is not None
            else DecisionAuditLogger(log_path=config.audit.log_path)
        )
        self.counters = DiagnosticCounters()
        self.drift_alert_active: bool = False
        self.critical_drift_breaker: bool = False
        self._recent_decisions: List[PolicyDecision] = []
        self._recent_features: List[FeatureVector] = []
        self._max_drift_buffer: int = 500

    def evaluate(
        self,
        payload: Dict[str, Any] | LoginSessionEvent,
        record_to_store: bool = True,
    ) -> PolicyDecision:
        """Validate, extract causal features, score, apply adaptive/fallback policy, and log."""
        t_start = time.perf_counter()
        try:
            event = parse_and_validate_event(payload)
        except UnsupportedInputError:
            self.counters.unsupported_input_errors += 1
            raise
        except ContractValidationError:
            self.counters.validation_errors += 1
            raise

        fallback_reason = FallbackReason.NONE

        # 1. Causal feature extraction (with store outage fallback)
        try:
            fv = extract_features_from_store(
                event=event,
                store=self.store,
                config=self.config.features,
                feature_version=self.config.versions.feature_version,
            )
        except StoreUnavailableError:
            fallback_reason = FallbackReason.STORE_UNAVAILABLE
            fv = compute_features(
                event=event,
                user_history=[],
                ip_history=[],
                has_expired_history=False,
                config=self.config.features,
                feature_version=self.config.versions.feature_version,
            )

        # 2. Model scoring (with unavailable / stale / drift-breaker fallback)
        model_score: Optional[ModelScore] = None
        if self.model_bundle is None:
            if fallback_reason == FallbackReason.NONE:
                fallback_reason = FallbackReason.MODEL_UNAVAILABLE
        elif self.model_bundle.is_stale(
            now=event.timestamp, max_age_days=self.config.policy.stale_model_max_age_days
        ):
            if fallback_reason == FallbackReason.NONE:
                fallback_reason = FallbackReason.MODEL_STALE
            model_score = self.model_bundle.predict_one(fv)
        elif self.critical_drift_breaker:
            if fallback_reason == FallbackReason.NONE:
                fallback_reason = FallbackReason.CRITICAL_DRIFT_CIRCUIT_BREAKER
            model_score = self.model_bundle.predict_one(fv)
        else:
            model_score = self.model_bundle.predict_one(fv)

        # 3. Adaptive step-up policy & analyst evidence
        decision = decide_step_up_action(
            event=event,
            fv=fv,
            model_score=model_score,
            config=self.config,
            model_bundle=self.model_bundle,
            fallback_reason=fallback_reason,
            drift_alert_active=self.drift_alert_active,
        )

        # 4. Record event AFTER causal feature extraction so event never leaks into itself
        if record_to_store:
            try:
                self.store.record_event(event)
            except StoreUnavailableError:
                pass

        # 5. Mandatory decision audit log + counters + drift buffer
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self.audit_logger.log_decision(decision, latency_ms=latency_ms)
        self.counters.record_decision(decision, latency_ms=latency_ms)

        self._recent_decisions.append(decision)
        self._recent_features.append(fv)
        if len(self._recent_decisions) > self._max_drift_buffer:
            self._recent_decisions = self._recent_decisions[-self._max_drift_buffer :]
            self._recent_features = self._recent_features[-self._max_drift_buffer :]

        return decision

    def evaluate_batch(
        self,
        events: Iterable[Dict[str, Any] | LoginSessionEvent],
        record_to_store: bool = True,
    ) -> List[PolicyDecision]:
        """Validate and evaluate a batch of events in chronological order."""
        parsed: List[LoginSessionEvent] = []
        for raw in events:
            try:
                parsed.append(parse_and_validate_event(raw))
            except UnsupportedInputError:
                self.counters.unsupported_input_errors += 1
                raise
            except ContractValidationError:
                self.counters.validation_errors += 1
                raise
        parsed.sort(key=lambda e: (e.timestamp, e.event_id))
        return [self.evaluate(ev, record_to_store=record_to_store) for ev in parsed]

    def check_drift(
        self,
        decisions: Optional[List[PolicyDecision]] = None,
        feature_vectors: Optional[List[FeatureVector]] = None,
        activate_circuit_breaker: bool = True,
    ) -> DriftReport:
        """Run drift analysis against the model's calibration baseline."""
        if self.model_bundle is None:
            return DriftReport(
                sample_size=0,
                score_psi=0.0,
                feature_psi={},
                missingness_rate=0.0,
                baseline_missingness_rate=0.0,
                expiry_rate=0.0,
                alerts=[],
                critical_drift_detected=False,
            )
        decs = decisions if decisions is not None else self._recent_decisions
        fvs = feature_vectors if feature_vectors is not None else self._recent_features
        report = evaluate_drift(decs, fvs, self.model_bundle, self.config.drift)
        self.counters.drift_checks_run += 1
        self.drift_alert_active = len(report.alerts) > 0
        if report.critical_drift_detected:
            self.counters.critical_drift_events += 1
            if activate_circuit_breaker:
                self.critical_drift_breaker = True
        return report

    def health_status(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.store.ping() else "degraded_fallback_active",
            "store_healthy": self.store.ping(),
            "model_loaded": self.model_bundle is not None,
            "model_version": self.model_bundle.model_version if self.model_bundle else None,
            "schema_version": self.config.versions.schema_version,
            "feature_version": self.config.versions.feature_version,
            "policy_version": self.config.versions.policy_version,
            "drift_alert_active": self.drift_alert_active,
            "critical_drift_breaker": self.critical_drift_breaker,
            "audit_decisions_logged": self.audit_logger.count(),
            "counters": self.counters.snapshot(),
        }
