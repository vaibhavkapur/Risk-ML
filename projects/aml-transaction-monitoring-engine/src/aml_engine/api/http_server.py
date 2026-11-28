"""Zero-dependency HTTP/JSON server and in-process router wrapping AMLMonitoringService."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from aml_engine.api.service import AMLMonitoringService
from aml_engine.validation import (
    CaseWorkflowError,
    UnsupportedInputError,
    ValidationError,
)


def dispatch_http_request(
    service: AMLMonitoringService,
    method: str,
    raw_path: str,
    body_bytes: bytes = b"",
) -> Tuple[int, Dict[str, Any]]:
    """Pure HTTP request dispatcher shared by HTTPServer and in-process tests."""
    parsed = urlparse(raw_path)
    path = parsed.path.rstrip("/") or "/"
    qs = parse_qs(parsed.query)
    verb = method.upper()

    if verb == "GET":
        if path == "/health":
            return 200, service.health()
        if path == "/alerts":
            priority = qs.get("priority", [None])[0]
            return 200, {"alerts": service.list_alerts(priority=priority)}
        if path.startswith("/alerts/"):
            alert_id = path.split("/", 2)[2]
            res = service.get_alert_with_provenance(alert_id)
            if res is None:
                return 404, {"error": f"Alert not found: {alert_id}"}
            return 200, res
        if path == "/cases":
            status = qs.get("status", [None])[0]
            return 200, {"cases": service.list_cases(status=status)}
        if path.startswith("/cases/"):
            case_id = path.split("/", 2)[2]
            res = service.get_case(case_id)
            if res is None:
                return 404, {"error": f"Case not found: {case_id}"}
            return 200, res
        if path == "/runs":
            return 200, {"runs": service.list_runs()}
        return 404, {"error": f"Unknown route: {path}"}

    if verb == "POST":
        try:
            raw_text = body_bytes.decode("utf-8") if body_bytes else "{}"
            try:
                body = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"Malformed JSON request body: {exc}") from exc
            if not isinstance(body, dict):
                raise ValidationError("Request body must be a JSON object.")

            if path == "/monitor":
                strict = bool(body.pop("strict_validation", False))
                return 200, service.monitor_batch(body, strict_validation=strict)
            if path.startswith("/cases/") and path.endswith("/review"):
                parts = path.split("/")
                if len(parts) == 4:
                    case_id = parts[2]
                    result = service.review_case(
                        case_id=case_id,
                        reviewer_id=str(body.get("reviewer_id", "")),
                        new_status=str(body.get("new_status", "")),
                        rationale=str(body.get("rationale", "")),
                        new_priority=body.get("new_priority"),
                        override_reason_code=body.get("override_reason_code"),
                        timestamp_iso=body.get("timestamp_iso"),
                    )
                    return 200, result
            return 404, {"error": f"Unknown route: {path}"}
        except (ValidationError, UnsupportedInputError, CaseWorkflowError) as exc:
            return 400, {"error_type": type(exc).__name__, "error": str(exc)}

    return 405, {"error": f"Method not allowed: {verb}"}


def create_http_server(
    service: AMLMonitoringService,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> HTTPServer:
    """Create a stdlib HTTPServer bound to `service`."""

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            status, payload = dispatch_http_request(service, "GET", self.path)
            self._send_json(status, payload)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            status, payload = dispatch_http_request(service, "POST", self.path, raw)
            self._send_json(status, payload)

    return HTTPServer((host, port), Handler)
