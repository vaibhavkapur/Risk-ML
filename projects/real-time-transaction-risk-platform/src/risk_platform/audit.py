"""Append-only replayable JSONL audit log with hash chaining and deterministic replay verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Union

from risk_platform.model import CalibratedLinearRiskModel
from risk_platform.policy import PolicyEngine
from risk_platform.schemas import FeatureVector, RiskDecision, TransactionEvent, parse_utc_timestamp_ms
from risk_platform.store import InMemoryFeatureStore


class AuditLogger:
    """Append-only JSONL audit log recording versioned inputs, features, and decisions for replay."""

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._prev_hash = "0" * 64
        self._seq = 0
        self._memory_records: List[Dict[str, Any]] = []

        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self._recover_chain_tail()

    def _recover_chain_tail(self) -> None:
        assert self.path is not None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._seq = int(rec.get("seq", self._seq))
                self._prev_hash = str(rec.get("record_hash", self._prev_hash))

    def append(self, record_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._seq += 1
            body = {
                "audit_version": "audit.v1",
                "seq": self._seq,
                "record_type": record_type,
                "prev_hash": self._prev_hash,
                "payload": payload,
            }
            canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            rec_hash = hashlib.sha256(canonical).hexdigest()
            record = dict(body)
            record["record_hash"] = rec_hash
            self._prev_hash = rec_hash

            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
            else:
                self._memory_records.append(record)
            return record

    def log_decision(
        self,
        event: TransactionEvent,
        feature_vector: FeatureVector,
        decision: RiskDecision,
    ) -> Dict[str, Any]:
        return self.append(
            "SCORING_DECISION",
            {
                "event": event.to_dict(),
                "features": feature_vector.to_dict(),
                "decision": decision.to_dict(),
            },
        )

    def read_records(self) -> List[Dict[str, Any]]:
        with self._lock:
            if self.path is None:
                return list(self._memory_records)
            if not self.path.exists():
                return []
            records: List[Dict[str, Any]] = []
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records

    def verify_hash_chain(self) -> Dict[str, Any]:
        """Verify tamper-evidence of the audit log hash chain."""
        records = self.read_records()
        expected_prev = "0" * 64
        for idx, rec in enumerate(records, start=1):
            if rec.get("prev_hash") != expected_prev:
                return {
                    "valid": False,
                    "records_checked": idx,
                    "error": f"Chain broken at seq {rec.get('seq')}",
                }
            body = {k: v for k, v in rec.items() if k != "record_hash"}
            canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            computed = hashlib.sha256(canonical).hexdigest()
            if computed != rec.get("record_hash"):
                return {
                    "valid": False,
                    "records_checked": idx,
                    "error": f"Hash mismatch at seq {rec.get('seq')}",
                }
            expected_prev = computed
        return {"valid": True, "records_checked": len(records), "error": None}


def replay_audit_decisions(
    records: List[Dict[str, Any]],
    model: CalibratedLinearRiskModel,
    policy: PolicyEngine,
) -> Dict[str, Any]:
    """Deterministically replay SCORING_DECISION audit records against model + policy.

    Verifies that every non-fallback decision recorded with the matching model_version
    reproduces the exact decision_id, action, and calibrated risk_score.
    """
    decision_records = [r for r in records if r.get("record_type") == "SCORING_DECISION"]
    replayed = 0
    matched = 0
    skipped_fallback = 0
    discrepancies: List[Dict[str, Any]] = []

    for rec in decision_records:
        payload = rec["payload"]
        orig_dec = payload["decision"]
        if orig_dec.get("fallback_used") and orig_dec.get("model_version") != model.model_version:
            skipped_fallback += 1
            continue

        ev = TransactionEvent.from_dict(payload["event"])
        fv_raw = payload["features"]
        as_of_ms = parse_utc_timestamp_ms(fv_raw["as_of_ts"], "as_of_ts")
        fv = FeatureVector(
            event_id=fv_raw["event_id"],
            account_id=fv_raw["account_id"],
            feature_version=fv_raw["feature_version"],
            as_of_ts_ms=as_of_ms,
            values={k: float(v) for k, v in fv_raw["values"].items()},
            feature_freshness_ms=fv_raw.get("feature_freshness_ms"),
            feature_status=fv_raw.get("feature_status", "FRESH"),
            unknown_input_flags=list(fv_raw.get("unknown_input_flags", [])),
        )

        cal_score, raw_score, attributions = model.predict(fv)
        replayed_dec = policy.evaluate_model_score(
            event=ev,
            feature_vector=fv,
            calibrated_score=cal_score,
            raw_score=raw_score,
            model_version=model.model_version,
            attributions=attributions,
            latency_ms=0.0,
            decided_at_ms=as_of_ms,
        )
        replayed += 1

        score_diff = abs(round(replayed_dec.risk_score, 6) - float(orig_dec["risk_score"]))
        if (
            replayed_dec.decision_id == orig_dec["decision_id"]
            and replayed_dec.action == orig_dec["action"]
            and score_diff <= 1e-6
        ):
            matched += 1
        else:
            discrepancies.append(
                {
                    "event_id": ev.event_id,
                    "original_action": orig_dec["action"],
                    "replayed_action": replayed_dec.action,
                    "original_score": orig_dec["risk_score"],
                    "replayed_score": round(replayed_dec.risk_score, 6),
                    "original_decision_id": orig_dec["decision_id"],
                    "replayed_decision_id": replayed_dec.decision_id,
                }
            )

    return {
        "total_decision_records": len(decision_records),
        "replayed": replayed,
        "matched": matched,
        "skipped_fallback": skipped_fallback,
        "exact_match": len(discrepancies) == 0,
        "discrepancies": discrepancies[:20],
    }
