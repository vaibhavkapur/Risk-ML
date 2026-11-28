"""FastAPI scoring and ingestion API exposing feature freshness, model versions, and audit replay."""

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from risk_platform.audit import replay_audit_decisions
from risk_platform.errors import (
    ContractValidationError,
    DuplicateConflictError,
    SensitiveDataViolationError,
    UnsupportedCurrencyError,
    UnsupportedSchemaVersionError,
)
from risk_platform.model import CalibratedLinearRiskModel
from risk_platform.schemas import TransactionEvent, parse_utc_timestamp_ms
from risk_platform.service import RiskScoringService


class ScoreRequest(BaseModel):
    event: Dict[str, Any] = Field(..., description="Canonical tx.v1 transaction payload")
    as_of_ts: Optional[str] = Field(
        default=None, description="Optional ISO-8601 UTC scoring evaluation timestamp"
    )
    ingest_after_score: bool = Field(
        default=False, description="Whether to ingest the event into the online store after scoring"
    )


def _parse_event_or_http_error(payload: Dict[str, Any], service: RiskScoringService) -> TransactionEvent:
    try:
        return TransactionEvent.from_dict(
            payload,
            supported_versions=service.config.schema.supported_versions,
            supported_currencies=service.config.schema.supported_currencies,
            max_amount_cents=service.config.schema.max_amount_cents,
        )
    except SensitiveDataViolationError as exc:
        raise HTTPException(
            status_code=422, detail={"error_code": "SENSITIVE_DATA_VIOLATION", "message": str(exc)}
        ) from exc
    except UnsupportedSchemaVersionError as exc:
        raise HTTPException(
            status_code=422, detail={"error_code": "UNSUPPORTED_SCHEMA_VERSION", "message": str(exc)}
        ) from exc
    except UnsupportedCurrencyError as exc:
        raise HTTPException(
            status_code=422, detail={"error_code": "UNSUPPORTED_CURRENCY", "message": str(exc)}
        ) from exc
    except ContractValidationError as exc:
        raise HTTPException(
            status_code=422, detail={"error_code": "CONTRACT_VALIDATION_ERROR", "message": str(exc)}
        ) from exc


def create_app(service: RiskScoringService) -> FastAPI:
    """Create a configured FastAPI application bound to a RiskScoringService instance."""
    app = FastAPI(
        title="Real-Time Transaction Risk Platform",
        version="0.1.0",
        description="Low-latency, versioned transaction risk scoring API (offline prototype).",
    )
    app.state.service = service

    @app.get("/health")
    def get_health() -> Dict[str, Any]:
        return service.health()

    @app.get("/metrics")
    def get_metrics() -> Dict[str, Any]:
        return {
            "service": service.metrics.snapshot(),
            "store": service.store.stats(),
        }

    @app.get("/v1/models/active")
    def get_active_model() -> Dict[str, Any]:
        model = service.model
        meta: Dict[str, Any] = {
            "model_version": model.model_version,
            "feature_version": model.feature_version,
            "policy_version": service.policy.policy_version,
        }
        if isinstance(model, CalibratedLinearRiskModel):
            meta["artifact_sha256"] = model.artifact_sha256
            meta["platt_calibrator"] = {"platt_a": model.platt_a, "platt_b": model.platt_b}
            meta["evaluation"] = model.training_metadata.get("evaluation", {})
            meta["splits"] = model.training_metadata.get("splits", {})
        return meta

    @app.post("/v1/events/ingest")
    def post_ingest_event(payload: Dict[str, Any]) -> Dict[str, Any]:
        event = _parse_event_or_http_error(payload, service)
        try:
            result = service.ingest_event(event)
        except DuplicateConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error_code": "DUPLICATE_CONFLICT", "message": str(exc)},
            ) from exc
        return result.to_dict()

    @app.post("/v1/score")
    def post_score(request: ScoreRequest, response: Response) -> Dict[str, Any]:
        event = _parse_event_or_http_error(request.event, service)
        as_of_ms: Optional[int] = None
        if request.as_of_ts is not None:
            try:
                as_of_ms = parse_utc_timestamp_ms(request.as_of_ts, "as_of_ts")
            except ContractValidationError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={"error_code": "INVALID_AS_OF_TS", "message": str(exc)},
                ) from exc

        decision, feature_vector = service.score_transaction(
            event=event,
            as_of_ts_ms=as_of_ms,
            ingest_after_score=request.ingest_after_score,
            record_audit=True,
        )
        response.headers["X-Decision-Id"] = decision.decision_id
        response.headers["X-Model-Version"] = decision.model_version
        response.headers["X-Feature-Version"] = decision.feature_version
        response.headers["X-Policy-Version"] = decision.policy_version

        return {
            "decision": decision.to_dict(),
            "features": feature_vector.to_dict(),
        }

    @app.post("/v1/audit/replay")
    def post_audit_replay() -> Dict[str, Any]:
        chain_check = service.audit_logger.verify_hash_chain()
        if not isinstance(service.model, CalibratedLinearRiskModel):
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "UNSUPPORTED_MODEL_BACKEND",
                    "message": "Audit replay requires a CalibratedLinearRiskModel backend",
                },
            )
        records = service.audit_logger.read_records()
        replay_check = replay_audit_decisions(records, service.model, service.policy)
        return {
            "hash_chain": chain_check,
            "replay": replay_check,
        }

    return app
