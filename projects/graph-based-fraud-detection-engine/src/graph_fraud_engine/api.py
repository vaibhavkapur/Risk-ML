"""FastAPI HTTP service exposing risk scoring, graph evidence, and analyst review endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query

from graph_fraud_engine import __version__
from graph_fraud_engine.schemas import (
    AnalystDispositionRecord,
    AnalystDispositionRequest,
    ModelNotReadyError,
    ReviewCase,
    RiskScoreRequest,
    RiskScoreResponse,
    SchemaValidationError,
    UnsupportedOperationError,
)
from graph_fraud_engine.service import FraudDetectionService, ReviewCaseNotFoundError


def create_app(service: Optional[FraudDetectionService] = None) -> FastAPI:
    """Create a FastAPI application wired to a FraudDetectionService instance."""
    app = FastAPI(
        title="Graph-Based Fraud Detection Engine",
        version=__version__,
        description=(
            "Educational/research prototype serving calibrated temporal-graph fraud scores, "
            "graph-traced evidence paths, and analyst review dispositions."
        ),
    )
    app.state.service = service

    def _get_service() -> FraudDetectionService:
        svc: Optional[FraudDetectionService] = app.state.service
        if svc is None:
            raise HTTPException(status_code=503, detail="FraudDetectionService is not initialized")
        return svc

    @app.get("/health")
    def health() -> Dict[str, Any]:
        svc: Optional[FraudDetectionService] = app.state.service
        return {
            "status": "ok" if svc is not None else "degraded",
            "version": __version__,
            "models_loaded": sorted(svc.models.keys()) if svc is not None else [],
        }

    @app.get("/metrics")
    def metrics() -> Dict[str, Any]:
        return _get_service().get_diagnostics()

    @app.post("/v1/score", response_model=RiskScoreResponse)
    def score_transaction(request: RiskScoreRequest) -> RiskScoreResponse:
        svc = _get_service()
        try:
            return svc.score_transaction(request)
        except UnsupportedOperationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SchemaValidationError as exc:
            raise HTTPException(status_code=422, detail={"message": str(exc), "errors": exc.errors}) from exc
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/reviews", response_model=List[ReviewCase])
    def list_reviews(status: Optional[str] = Query(default=None)) -> List[ReviewCase]:
        svc = _get_service()
        try:
            return svc.list_review_cases(status=status)
        except UnsupportedOperationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/reviews/{case_id}", response_model=ReviewCase)
    def get_review(case_id: str) -> ReviewCase:
        svc = _get_service()
        try:
            return svc.get_review_case(case_id)
        except ReviewCaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/reviews/{case_id}/disposition", response_model=AnalystDispositionRecord)
    def submit_disposition(
        case_id: str, request: AnalystDispositionRequest
    ) -> AnalystDispositionRecord:
        svc = _get_service()
        try:
            return svc.submit_disposition(case_id, request)
        except ReviewCaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app
