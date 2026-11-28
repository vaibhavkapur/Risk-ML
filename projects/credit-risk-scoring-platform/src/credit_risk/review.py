"""Manual review queue persistence and lightweight local web/CLI review interface.

Strictly decision-support: every queued application requires explicit human
reviewer inspection and disposition with recorded rationale.
"""

from __future__ import annotations

from datetime import datetime, timezone
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import parse_qs

from credit_risk.contracts import ApplicantRecord, ScoringDecisionOutput
from credit_risk.errors import UnsupportedInputError

ALLOWED_HUMAN_DISPOSITIONS = (
    "HUMAN_REVIEW_CLEARED",
    "HUMAN_REQUEST_DOCUMENTATION",
    "HUMAN_ESCALATED_COMMITTEE",
    "HUMAN_DECLINE_RECOMMENDATION",
)


class ManualReviewStore:
    """JSON-backed manual review queue and human disposition audit log."""

    def __init__(self, store_path: Union[str, Path]) -> None:
        self.store_path = Path(store_path)

    def _load_state(self) -> Dict[str, Any]:
        if not self.store_path.exists():
            return {
                "schema_version": "1.0.0",
                "automated_decision_allowed": False,
                "notice": (
                    "Educational manual review queue. All cases require human underwriting "
                    "review; no automated lending decisions are made."
                ),
                "cases": {},
            }
        return json.loads(self.store_path.read_text(encoding="utf-8"))

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def enqueue(
        self,
        applicant: ApplicantRecord,
        decision: ScoringDecisionOutput,
    ) -> Dict[str, Any]:
        state = self._load_state()
        case_record = {
            "decision_id": decision.decision_id,
            "application_id": applicant.application_id,
            "review_status": "PENDING_MANUAL_REVIEW",
            "automated_decision_allowed": False,
            "manual_review_required": True,
            "applicant_snapshot": applicant.to_dict(),
            "scoring_output": decision.to_dict(),
            "human_reviews": [],
        }
        state["cases"][decision.decision_id] = case_record
        self._save_state(state)
        return case_record

    def list_cases(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        state = self._load_state()
        cases = list(state["cases"].values())
        if status_filter:
            cases = [c for c in cases if c["review_status"] == status_filter]
        # Sort highest PD first for reviewer triage.
        cases.sort(key=lambda c: (-float(c["scoring_output"]["calibrated_pd"]), c["decision_id"]))
        return cases

    def get_case(self, decision_id: str) -> Optional[Dict[str, Any]]:
        state = self._load_state()
        return state["cases"].get(decision_id)

    def record_human_review(
        self,
        decision_id: str,
        reviewer_id: str,
        disposition: str,
        rationale: str,
        reviewed_at_utc: Optional[str] = None,
    ) -> Dict[str, Any]:
        if disposition not in ALLOWED_HUMAN_DISPOSITIONS:
            raise UnsupportedInputError(
                f"Unsupported human disposition '{disposition}'. Allowed: {ALLOWED_HUMAN_DISPOSITIONS}",
                details={"disposition": disposition},
            )
        if not reviewer_id.strip():
            raise ValueError("reviewer_id must be a non-empty string.")
        if not rationale.strip():
            raise ValueError("rationale must be a non-empty explanation of the human review.")

        state = self._load_state()
        if decision_id not in state["cases"]:
            raise KeyError(f"Decision ID '{decision_id}' not found in manual review store.")

        ts = reviewed_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        review_entry = {
            "reviewer_id": reviewer_id.strip(),
            "disposition": disposition,
            "rationale": rationale.strip(),
            "reviewed_at_utc": ts,
            "automated_decision": False,
        }
        case = state["cases"][decision_id]
        case["human_reviews"].append(review_entry)
        case["review_status"] = disposition
        self._save_state(state)
        return case


def render_review_queue_html(cases: List[Dict[str, Any]]) -> str:
    """Render a self-contained HTML manual-review workbench for human underwriters."""
    rows_html: List[str] = []
    for c in cases:
        s = c["scoring_output"]
        a = c["applicant_snapshot"]
        exps = "".join(
            f"<li><code>{html.escape(e['reason_code'])}</code>: {html.escape(e['summary'])}</li>"
            for e in s.get("explanations", [])
        )
        warns = "".join(
            f"<div class='warn'>{html.escape(w)}</div>" for w in s.get("warnings", [])
        )
        reviews_html = "".join(
            f"<li><strong>{html.escape(r['disposition'])}</strong> by {html.escape(r['reviewer_id'])} "
            f"({html.escape(r['reviewed_at_utc'])}): {html.escape(r['rationale'])}</li>"
            for r in c.get("human_reviews", [])
        )
        options_html = "".join(
            f"<option value='{d}'>{d}</option>" for d in ALLOWED_HUMAN_DISPOSITIONS
        )
        rows_html.append(
            f"""
            <div class="card">
              <div class="card-header">
                <div>
                  <strong>{html.escape(c['application_id'])}</strong>
                  <span class="badge">{html.escape(s['risk_band'])}</span>
                  <span class="status">{html.escape(c['review_status'])}</span>
                </div>
                <div>Decision ID: <code>{html.escape(c['decision_id'])}</code></div>
              </div>
              <div class="metrics">
                <div>Calibrated PD: <strong>{s['calibrated_pd']:.2%}</strong> (raw: {s['raw_score']:.2%})</div>
                <div>Priority: <strong>{html.escape(s['queue_priority'])}</strong> (SLA: {s['review_sla_hours']}h)</div>
                <div>Action: <code>{html.escape(s['recommended_action'])}</code></div>
                <div>Income: {a['annual_income_usd']} | Loan: {a['loan_amount_usd']} | Score: {a['credit_score']} | DTI: {a['debt_to_income_ratio']}</div>
              </div>
              {warns}
              <div class="section-title">Top Decision Explanations (Log-Odds)</div>
              <ul>{exps}</ul>
              <div class="section-title">Human Review Audit Trail</div>
              <ul>{reviews_html or '<li>No human review recorded yet.</li>'}</ul>
              <form method="POST" action="/review">
                <input type="hidden" name="decision_id" value="{html.escape(c['decision_id'])}" />
                <input type="text" name="reviewer_id" placeholder="Reviewer ID (e.g. analyst-01)" required />
                <select name="disposition">{options_html}</select>
                <input type="text" name="rationale" placeholder="Human underwriting rationale..." required style="min-width:260px" />
                <button type="submit">Record Human Review</button>
              </form>
            </div>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Credit Risk Manual Review Workbench (Educational Prototype)</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f6f8fa; color: #1f2328; }}
    .banner {{ background: #fff8c5; border: 1px solid #d4a72c; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; }}
    .card {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .badge {{ background: #ddf4ff; color: #0969da; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-left: 8px; }}
    .status {{ background: #eaeef2; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-left: 6px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 6px; font-size: 14px; margin-bottom: 8px; }}
    .warn {{ background: #ffebe9; color: #cf222e; padding: 6px 10px; border-radius: 4px; margin: 6px 0; font-size: 13px; }}
    .section-title {{ font-weight: 600; margin-top: 10px; font-size: 13px; color: #57606a; }}
    form {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }}
    input, select, button {{ padding: 6px 10px; border-radius: 4px; border: 1px solid #d0d7de; }}
    button {{ background: #1f883d; color: #fff; border: none; cursor: pointer; }}
  </style>
</head>
<body>
  <h1>Credit Risk Scoring Platform — Manual Review Workbench</h1>
  <div class="banner">
    <strong>Decision-Support Only (Educational Prototype):</strong>
    Automated lending decisions are disabled. All probabilities of default are synthetic prototype estimates
    intended to prioritize and inform human underwriting review.
  </div>
  {''.join(rows_html) if rows_html else '<p>No cases currently in the manual review queue.</p>'}
</body>
</html>
"""


def create_review_http_server(
    store: ManualReviewStore,
    host: str = "127.0.0.1",
    port: int = 8088,
) -> HTTPServer:
    """Create a local-only HTTP server serving the manual review HTML workbench & JSON API."""

    class ReviewHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/cases":
                payload = json.dumps(
                    {"automated_decision_allowed": False, "cases": store.list_cases()},
                    indent=2,
                ).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", payload)
            else:
                page = render_review_queue_html(store.list_cases()).encode("utf-8")
                self._send(200, "text/html; charset=utf-8", page)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/review":
                self._send(404, "text/plain; charset=utf-8", b"Not Found")
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")
            form = parse_qs(raw_body)
            decision_id = form.get("decision_id", [""])[0]
            reviewer_id = form.get("reviewer_id", [""])[0]
            disposition = form.get("disposition", [""])[0]
            rationale = form.get("rationale", [""])[0]
            try:
                store.record_human_review(
                    decision_id=decision_id,
                    reviewer_id=reviewer_id,
                    disposition=disposition,
                    rationale=rationale,
                )
            except Exception as exc:  # noqa: BLE001
                self._send(400, "text/plain; charset=utf-8", f"Review error: {exc}".encode("utf-8"))
                return
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return HTTPServer((host, port), ReviewHandler)
