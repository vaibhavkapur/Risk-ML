"""Optional FastAPI application adapter wrapping AMLMonitoringService.

Requires optional dependencies from `requirements-optional.txt` (`fastapi`, `uvicorn`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from aml_engine.api.service import AMLMonitoringService
from aml_engine.validation import (
    CaseWorkflowError,
    UnsupportedInputError,
    ValidationError,
)


def create_fastapi_app(service: AMLMonitoringService) -> Any:
    """Create a FastAPI ASGI application backed by `AMLMonitoringService`."""
    try:
        from fastapi import FastAPI, HTTPException  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Optional dependency 'fastapi' is not installed. "
            "Use `aml_engine.api.http_server.create_http_server` for zero-dependency HTTP serving, "
            "or install extras with: pip install -r requirements-optional.txt"
        ) from exc

    app = FastAPI(
        title="AML Transaction Monitoring Engine (Educational Prototype)",
        version="0.1.0",
        description="Advisory transaction-monitoring API with explainable alerts and audited case review.",
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return service.health()

    @app.post("/monitor")
    def monitor(payload: Dict[str, Any], strict_validation: bool = False) -> Dict[str, Any]:
        try:
            return service.monitor_batch(payload, strict_validation=strict_validation)
        except (ValidationError, UnsupportedInputError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/alerts")
    def list_alerts(priority: Optional[str] = None) -> Dict[str, Any]:
        return {"alerts": service.list_alerts(priority=priority)}

    @app.get("/alerts/{alert_id}")
    def get_alert(alert_id: str) -> Dict[str, Any]:
        res = service.get_alert_with_provenance(alert_id)
        if res is None:
            raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")
        return res

    @app.get("/cases")
    def list_cases(status: Optional[str] = None) -> Dict[str, Any]:
        return {"cases": service.list_cases(status=status)}

    @app.get("/cases/{case_id}")
    def get_case(case_id: str) -> Dict[str, Any]:
        res = service.get_case(case_id)
        if res is None:
            raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
        return res

    @app.post("/cases/{case_id}/review")
    def review_case(case_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.review_case(
                case_id=case_id,
                reviewer_id=str(body.get("reviewer_id", "")),
                new_status=str(body.get("new_status", "")),
                rationale=str(body.get("rationale", "")),
                new_priority=body.get("new_priority"),
                override_reason_code=body.get("override_reason_code"),
                timestamp_iso=body.get("timestamp_iso"),
            )
        except (ValidationError, CaseWorkflowError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/runs")
    def list_runs() -> Dict[str, Any]:
        return {"runs": service.list_runs()}

    return app
