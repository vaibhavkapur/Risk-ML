"""End-to-end real-time risk scoring service with bounded inference timeout and audit logging."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from risk_platform.audit import AuditLogger
from risk_platform.config import PlatformConfig
from risk_platform.metrics import MetricsRegistry
from risk_platform.model import RiskModelBackend
from risk_platform.policy import PolicyEngine
from risk_platform.schemas import (
    FeatureAttribution,
    FeatureVector,
    IngestionResult,
    RiskDecision,
    TransactionEvent,
)
from risk_platform.store import FeatureStore

logger = logging.getLogger("risk_platform.service")


class RiskScoringService:
    """Real-time scoring orchestrator wiring feature store, model inference, policy, metrics, and audit."""

    def __init__(
        self,
        config: PlatformConfig,
        store: FeatureStore,
        model: RiskModelBackend,
        policy: Optional[PolicyEngine] = None,
        metrics: Optional[MetricsRegistry] = None,
        audit_logger: Optional[AuditLogger] = None,
        enforce_async_timeout: bool = True,
    ) -> None:
        self.config = config
        self.store = store
        self.model = model
        self.policy = policy or PolicyEngine(
            serving_config=config.serving,
            policy_version=config.schema.policy_version,
            fx_to_usd=config.schema.fx_to_usd,
        )
        self.metrics = metrics or MetricsRegistry()
        self.audit_logger = audit_logger or AuditLogger(path=None)
        self._enforce_async_timeout = enforce_async_timeout
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="risk-model")

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def ingest_event(self, event: TransactionEvent) -> IngestionResult:
        """Ingest a validated transaction event into the streaming feature store."""
        result = self.store.ingest(event)
        self.metrics.inc(f"ingest_{result.status.lower()}")
        return result

    def _run_model_with_timeout(
        self, feature_vector: FeatureVector, timeout_ms: float
    ) -> Tuple[float, float, List[FeatureAttribution]]:
        if not self._enforce_async_timeout:
            t0 = time.perf_counter()
            res = self.model.predict(feature_vector)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms > timeout_ms:
                raise FuturesTimeoutError(
                    f"Inference exceeded budget: {elapsed_ms:.2f}ms > {timeout_ms:.2f}ms"
                )
            return res

        future = self._executor.submit(self.model.predict, feature_vector)
        try:
            return future.result(timeout=max(0.0005, timeout_ms / 1000.0))
        except FuturesTimeoutError:
            future.cancel()
            raise

    def score_transaction(
        self,
        event: TransactionEvent,
        as_of_ts_ms: Optional[int] = None,
        ingest_after_score: bool = False,
        record_audit: bool = True,
    ) -> Tuple[RiskDecision, FeatureVector]:
        """Score a transaction with feature freshness tracking, timeout fallback, and audit logging."""
        t_start = time.perf_counter()
        eval_ts_ms = int(as_of_ts_ms) if as_of_ts_ms is not None else event.event_ts_ms

        feature_vector = self.store.get_features(
            event=event,
            as_of_ts_ms=eval_ts_ms,
            max_staleness_ms=self.config.serving.max_feature_staleness_ms,
        )
        self.metrics.inc(f"feature_status_{feature_vector.feature_status.lower()}")

        try:
            cal_score, raw_score, attributions = self._run_model_with_timeout(
                feature_vector=feature_vector,
                timeout_ms=self.config.serving.inference_timeout_ms,
            )
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            decision = self.policy.evaluate_model_score(
                event=event,
                feature_vector=feature_vector,
                calibrated_score=cal_score,
                raw_score=raw_score,
                model_version=self.model.model_version,
                attributions=attributions,
                latency_ms=latency_ms,
                decided_at_ms=eval_ts_ms,
            )
        except FuturesTimeoutError:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            self.metrics.inc("model_timeout_fallbacks")
            logger.warning(
                "Model inference timed out for event_id=%s after %.2fms (budget=%.2fms)",
                event.event_id,
                latency_ms,
                self.config.serving.inference_timeout_ms,
            )
            decision = self.policy.evaluate_fallback(
                event=event,
                feature_vector=feature_vector,
                fallback_reason="MODEL_TIMEOUT",
                latency_ms=latency_ms,
                decided_at_ms=eval_ts_ms,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            self.metrics.inc("model_error_fallbacks")
            logger.error("Model inference error for event_id=%s: %s", event.event_id, exc)
            decision = self.policy.evaluate_fallback(
                event=event,
                feature_vector=feature_vector,
                fallback_reason="MODEL_ERROR",
                latency_ms=latency_ms,
                decided_at_ms=eval_ts_ms,
            )

        self.metrics.inc("decisions_total")
        self.metrics.inc(f"decision_action_{decision.action.lower()}")
        if decision.fallback_used:
            self.metrics.inc("decisions_fallback_total")
        self.metrics.observe_latency(decision.latency_ms)

        if record_audit:
            self.audit_logger.log_decision(event, feature_vector, decision)

        if ingest_after_score:
            self.ingest_event(event)

        return decision, feature_vector

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": self.config.name,
            "environment": self.config.environment,
            "model_version": self.model.model_version,
            "feature_version": self.model.feature_version,
            "policy_version": self.policy.policy_version,
            "store": self.store.stats(),
        }
