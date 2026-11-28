"""Structured diagnostics, processor health registry, and idempotency ledger."""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Set

import numpy as np

from payment_auth_engine.config import EngineConfig
from payment_auth_engine.schemas import (
    DecisionRecord,
    PolicyAction,
    ProcessorHealthState,
    ProcessorHealthStatus,
)

logger = logging.getLogger("payment_auth_engine")


class ProcessorHealthRegistry:
    """In-memory registry tracking processor health for resilience and outage failover."""

    def __init__(self, config: EngineConfig) -> None:
        self._states: Dict[str, ProcessorHealthState] = {
            pid: ProcessorHealthState(processor_id=pid, status=ProcessorHealthStatus.HEALTHY)
            for pid in config.processors
        }

    def set_health(
        self,
        processor_id: str,
        status: ProcessorHealthStatus,
        *,
        latency_multiplier: float = 1.0,
        approval_penalty: float = 0.0,
        reason: str = "manual_update",
    ) -> ProcessorHealthState:
        if processor_id not in self._states:
            raise KeyError(f"UNKNOWN_PROCESSOR: {processor_id!r} not in {sorted(self._states)}")
        state = ProcessorHealthState(
            processor_id=processor_id,
            status=status,
            latency_multiplier=latency_multiplier,
            approval_penalty=approval_penalty,
            reason=reason,
        )
        self._states[processor_id] = state
        return state

    def get_all(self) -> Dict[str, ProcessorHealthState]:
        return dict(self._states)

    def reset_all_healthy(self) -> None:
        for pid in list(self._states):
            self._states[pid] = ProcessorHealthState(processor_id=pid, status=ProcessorHealthStatus.HEALTHY)


class IdempotencyLedger:
    """Tracks processed idempotency keys and approved transaction IDs to prevent double-charging."""

    def __init__(self) -> None:
        self._by_idempotency_key: Dict[str, DecisionRecord] = {}
        self._approved_transactions: Set[str] = set()

    def has_idempotency_key(self, key: str) -> bool:
        return key in self._by_idempotency_key

    def get_by_idempotency_key(self, key: str) -> Optional[DecisionRecord]:
        return self._by_idempotency_key.get(key)

    def is_transaction_approved(self, transaction_id: str) -> bool:
        return transaction_id in self._approved_transactions

    def mark_transaction_approved(self, transaction_id: str) -> None:
        self._approved_transactions.add(transaction_id)

    def record(self, decision: DecisionRecord) -> None:
        if decision.idempotency_key not in self._by_idempotency_key:
            self._by_idempotency_key[decision.idempotency_key] = decision

    def clear(self) -> None:
        self._by_idempotency_key.clear()
        self._approved_transactions.clear()


class DiagnosticsCollector:
    """Structured diagnostic counters and JSON event log for policy audit and observability."""

    def __init__(self, *, max_events: int = 500, emit_logs: bool = False) -> None:
        self.max_events = max_events
        self.emit_logs = emit_logs
        self._counters: Counter[str] = Counter()
        self._action_counts: Counter[str] = Counter()
        self._latencies_ms: List[float] = []
        self._events: List[Dict[str, Any]] = []

    def record_decision(self, decision: DecisionRecord) -> None:
        self._counters["total_requests"] += 1
        self._action_counts[decision.action.value] += 1
        self._latencies_ms.append(decision.estimated_latency_ms)

        if decision.action == PolicyAction.BLOCK_HARD_DECLINE:
            self._counters["hard_decline_blocks"] += 1
        elif decision.action == PolicyAction.HOLD_RECONCILE:
            self._counters["uncertain_outcome_holds"] += 1
        elif decision.action == PolicyAction.DEDUPLICATE_RETURN_PRIOR:
            self._counters["duplicate_attempt_blocks"] += 1
        elif decision.action == PolicyAction.BLOCK_PROCESSOR_OUTAGE:
            self._counters["outage_fail_closed_blocks"] += 1

        if "PROCESSOR_OUTAGE_FAILOVER" in decision.explanation.reason_codes:
            self._counters["outage_failovers"] += 1
        if decision.explanation.missing_features:
            self._counters["missing_feature_requests"] += 1
        if decision.any_constraint_breach:
            self._counters["constraint_breaches"] += 1

        event = {
            "decision_id": decision.decision_id,
            "attempt_id": decision.attempt_id,
            "transaction_id": decision.transaction_id,
            "policy": decision.policy_name,
            "action": decision.action.value,
            "processor": decision.processor_selected,
            "reason_codes": decision.explanation.reason_codes,
            "missing_features": decision.explanation.missing_features,
            "any_constraint_breach": decision.any_constraint_breach,
        }
        if len(self._events) < self.max_events:
            self._events.append(event)
        if self.emit_logs:
            logger.info(json.dumps(event, sort_keys=True))

    def snapshot(self) -> Dict[str, Any]:
        lats = np.asarray(self._latencies_ms, dtype=np.float64) if self._latencies_ms else np.empty(0)
        return {
            "counters": dict(sorted(self._counters.items())),
            "action_counts": dict(sorted(self._action_counts.items())),
            "latency_summary_ms": {
                "p50": round(float(np.percentile(lats, 50)), 2) if len(lats) else 0.0,
                "p95": round(float(np.percentile(lats, 95)), 2) if len(lats) else 0.0,
                "max": round(float(np.max(lats)), 2) if len(lats) else 0.0,
            },
            "recent_events_count": len(self._events),
        }

    def reset(self) -> None:
        self._counters.clear()
        self._action_counts.clear()
        self._latencies_ms.clear()
        self._events.clear()
