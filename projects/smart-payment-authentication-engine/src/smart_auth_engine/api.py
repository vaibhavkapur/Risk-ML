"""FastAPI service exposing payment authentication, step-up simulation, replay, and diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from smart_auth_engine.contracts import (
    ErrorResponse,
    PaymentEvent,
    PolicyDecision,
    StepUpExecutionResult,
)
from smart_auth_engine.inference import AuthenticationEngine


class StepUpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: PaymentEvent
    actual_is_fraud: Optional[bool] = None
    simulate_outage: bool = False


class AuthenticateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: PaymentEvent
    simulate_outage: bool = False


def create_app(engine: AuthenticationEngine) -> FastAPI:
    """Create a FastAPI application bound to a configured AuthenticationEngine instance."""
    app = FastAPI(
        title="Smart Payment Authentication Engine",
        version="0.1.0",
        description=(
            "Educational / research prototype for calibrated fraud risk modeling, "
            "constrained authentication policy optimization, and deterministic step-up simulation."
        ),
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        err = ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Invalid payment authentication request payload.",
            details={"errors": exc.errors()},
        )
        return JSONResponse(status_code=422, content=err.model_dump(mode="json"))

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok" if engine.model_available else "degraded_fallback_active",
            "model_available": engine.model_available,
            "model_version": engine.active_model_version,
            "feature_version": engine.cfg.feature_version,
            "policy_version": engine.cfg.policy_version,
            "thresholds": {
                "challenge_threshold": engine.thresholds.challenge_threshold,
                "decline_threshold": engine.thresholds.decline_threshold,
            },
        }

    @app.post("/v1/authenticate", response_model=PolicyDecision)
    async def authenticate(req: AuthenticateRequest) -> PolicyDecision:
        return engine.evaluate(
            event=req.event,
            record_store=True,
            simulate_outage=req.simulate_outage,
        )

    @app.post("/v1/step-up/simulate", response_model=StepUpExecutionResult)
    async def step_up_simulate(req: StepUpRequest) -> StepUpExecutionResult:
        return engine.evaluate_and_simulate_step_up(
            event=req.event,
            actual_is_fraud=req.actual_is_fraud,
            record_store=True,
            simulate_outage=req.simulate_outage,
        )

    @app.get("/v1/decisions/{decision_id}")
    async def get_decision(decision_id: str) -> Dict[str, Any]:
        rec = engine.store.get_record(decision_id)
        if rec is None:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error_code="DECISION_NOT_FOUND",
                    message=f"No decision recorded with decision_id={decision_id!r}.",
                ).model_dump(mode="json"),
            )
        event, decision, step_up = rec
        return {
            "event": event.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "step_up": step_up.model_dump(mode="json") if step_up is not None else None,
        }

    @app.post("/v1/replay/{decision_id}")
    async def replay_decision(decision_id: str) -> Dict[str, Any]:
        try:
            return engine.replay_decision(decision_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error_code="DECISION_NOT_FOUND",
                    message=str(exc),
                ).model_dump(mode="json"),
            ) from exc

    @app.get("/v1/metrics")
    async def get_metrics() -> Dict[str, Any]:
        return {
            "counters": engine.counters.to_dict(),
            "model_available": engine.model_available,
            "model_version": engine.active_model_version,
            "policy_version": engine.cfg.policy_version,
        }

    return app
