"""FastAPI HTTP adapter for the Account Takeover Detection Engine.

Domain logic stays isolated in `ato_engine.engine`; this module handles HTTP
serialization, status codes, and lifecycle wiring only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import __version__
from .audit import DecisionAuditLogger
from .config import EngineConfig, load_config
from .engine import ATODetectionEngine
from .models import ATORiskModelBundle
from .schemas import ContractValidationError, UnsupportedInputError


class BatchEvaluateRequest(BaseModel):
    events: List[Dict[str, Any]]
    record_to_store: bool = True


def create_app(
    engine: Optional[ATODetectionEngine] = None,
    config_path: Optional[str | Path] = None,
    model_path: Optional[str | Path] = None,
) -> FastAPI:
    """Create a FastAPI application wrapping an ATODetectionEngine instance."""
    if engine is None:
        cfg = load_config(config_path) if config_path is not None else EngineConfig()
        bundle: Optional[ATORiskModelBundle] = None
        if model_path is not None and Path(model_path).exists():
            bundle = ATORiskModelBundle.load(model_path)
        engine = ATODetectionEngine(
            config=cfg,
            model_bundle=bundle,
            audit_logger=DecisionAuditLogger(log_path=cfg.audit.log_path),
        )

    app = FastAPI(
        title="Account Takeover Detection Engine",
        version=__version__,
        description=(
            "Research prototype API for synthetic login/session ATO scoring, "
            "calibrated step-up recommendations, analyst evidence, and drift diagnostics."
        ),
    )
    app.state.engine = engine

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return app.state.engine.health_status()

    @app.get("/v1/metrics")
    def metrics() -> Dict[str, Any]:
        return {
            "counters": app.state.engine.counters.snapshot(),
            "audit_decisions_logged": app.state.engine.audit_logger.count(),
            "drift_alert_active": app.state.engine.drift_alert_active,
            "critical_drift_breaker": app.state.engine.critical_drift_breaker,
        }

    @app.post("/v1/evaluate")
    def evaluate_event(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            decision = app.state.engine.evaluate(payload)
        except UnsupportedInputError as exc:
            raise HTTPException(status_code=400, detail={"error": "unsupported_input", "message": str(exc)}) from exc
        except ContractValidationError as exc:
            raise HTTPException(status_code=422, detail={"error": "contract_validation_error", "message": str(exc)}) from exc
        return decision.model_dump(mode="json")

    @app.post("/v1/evaluate/batch")
    def evaluate_batch(req: BatchEvaluateRequest) -> Dict[str, Any]:
        try:
            decisions = app.state.engine.evaluate_batch(
                req.events, record_to_store=req.record_to_store
            )
        except UnsupportedInputError as exc:
            raise HTTPException(status_code=400, detail={"error": "unsupported_input", "message": str(exc)}) from exc
        except ContractValidationError as exc:
            raise HTTPException(status_code=422, detail={"error": "contract_validation_error", "message": str(exc)}) from exc
        return {
            "count": len(decisions),
            "decisions": [d.model_dump(mode="json") for d in decisions],
        }

    @app.post("/v1/drift/check")
    def run_drift_check() -> Dict[str, Any]:
        report = app.state.engine.check_drift()
        return report.model_dump(mode="json")

    return app
