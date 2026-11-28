"""FastAPI service interface for the Payment Authorization Optimization Engine.

Offline synthetic research prototype only — not for live payment processing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from payment_auth_engine import FEATURE_SCHEMA_VERSION, SCHEMA_VERSION, __version__
from payment_auth_engine.engine import AuthorizationEngine
from payment_auth_engine.schemas import (
    ContractValidationError,
    DecisionRecord,
    ProcessorHealthStatus,
    RetryEligibilityResult,
    validate_attempt,
)
from payment_auth_engine.taxonomy import evaluate_retry_eligibility


class ProcessorHealthUpdateRequest(BaseModel):
    status: ProcessorHealthStatus
    latency_multiplier: float = Field(default=1.0, ge=1.0, le=20.0)
    approval_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = "api_update"


def create_app(engine: AuthorizationEngine) -> FastAPI:
    """Create a FastAPI app bound to a configured AuthorizationEngine instance."""
    app = FastAPI(
        title="Payment Authorization Optimization Engine",
        version=__version__,
        description=(
            "Offline synthetic research prototype for risk-aware payment authorization "
            "optimization, retry safety, and calibrated expected-loss policy enforcement."
        ),
    )

    @app.exception_handler(ContractValidationError)
    async def _handle_contract_error(_request: Any, exc: ContractValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error_code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "mode": "offline_synthetic_prototype",
            "version": __version__,
            "schema_version": SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "policy_name": engine.policy.name,
            "model_version": engine.model_bundle.model_version,
            "config_hash": engine.config.config_hash(),
        }

    @app.post("/v1/authorize", response_model=DecisionRecord)
    def authorize(payload: Dict[str, Any]) -> DecisionRecord:
        attempt = validate_attempt(payload)
        return engine.decide(attempt)

    @app.post("/v1/retry-eligibility", response_model=RetryEligibilityResult)
    def retry_eligibility(payload: Dict[str, Any]) -> RetryEligibilityResult:
        attempt = validate_attempt(payload)
        idem_seen = engine.idempotency_ledger.has_idempotency_key(attempt.idempotency_key)
        prior_approved = engine.idempotency_ledger.is_transaction_approved(attempt.transaction_id)
        return evaluate_retry_eligibility(
            attempt,
            idempotency_seen=idem_seen,
            prior_approved=prior_approved,
            max_retries=engine.config.retry.max_retries,
            allow_soft_decline_retry=engine.config.retry.allow_soft_decline_retry,
            allow_technical_failover=engine.config.retry.allow_technical_failover,
        )

    @app.get("/v1/diagnostics")
    def diagnostics() -> Dict[str, Any]:
        return {
            "policy_name": engine.policy.name,
            "model_version": engine.model_bundle.model_version,
            "diagnostics": engine.diagnostics.snapshot(),
            "processors": {
                pid: state.model_dump(mode="json")
                for pid, state in engine.health_registry.get_all().items()
            },
        }

    @app.post("/v1/processors/{processor_id}/health")
    def update_processor_health(processor_id: str, body: ProcessorHealthUpdateRequest) -> Dict[str, Any]:
        try:
            state = engine.health_registry.set_health(
                processor_id,
                body.status,
                latency_multiplier=body.latency_multiplier,
                approval_penalty=body.approval_penalty,
                reason=body.reason,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return state.model_dump(mode="json")

    return app
