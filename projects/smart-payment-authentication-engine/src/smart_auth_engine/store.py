"""Pluggable decision and replay storage backends (in-memory, durable JSONL, optional Redis)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from smart_auth_engine.contracts import PaymentEvent, PolicyDecision, StepUpExecutionResult


class DecisionStore(ABC):
    """Abstract storage interface for audit decisions and replay inputs."""

    @abstractmethod
    def save_record(
        self,
        event: PaymentEvent,
        decision: PolicyDecision,
        step_up: Optional[StepUpExecutionResult] = None,
    ) -> None:
        """Persist a payment event, policy decision, and optional step-up result."""

    @abstractmethod
    def get_record(
        self,
        decision_id: str,
    ) -> Optional[Tuple[PaymentEvent, PolicyDecision, Optional[StepUpExecutionResult]]]:
        """Fetch a persisted record by decision_id."""

    @abstractmethod
    def list_decision_ids(self) -> List[str]:
        """List stored decision IDs in insertion order."""


class InMemoryDecisionStore(DecisionStore):
    """Default zero-dependency in-memory store."""

    def __init__(self) -> None:
        self._records: Dict[str, Tuple[PaymentEvent, PolicyDecision, Optional[StepUpExecutionResult]]] = {}

    def save_record(
        self,
        event: PaymentEvent,
        decision: PolicyDecision,
        step_up: Optional[StepUpExecutionResult] = None,
    ) -> None:
        self._records[decision.decision_id] = (event, decision, step_up)

    def get_record(
        self,
        decision_id: str,
    ) -> Optional[Tuple[PaymentEvent, PolicyDecision, Optional[StepUpExecutionResult]]]:
        return self._records.get(decision_id)

    def list_decision_ids(self) -> List[str]:
        return list(self._records.keys())


class JsonlDecisionStore(DecisionStore):
    """Append-only durable JSONL decision store supporting process restart and replay."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Tuple[PaymentEvent, PolicyDecision, Optional[StepUpExecutionResult]]] = {}
        self._reload_from_disk()

    def _reload_from_disk(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ev = PaymentEvent.model_validate(row["event"])
                dec = PolicyDecision.model_validate(row["decision"])
                su = (
                    StepUpExecutionResult.model_validate(row["step_up"])
                    if row.get("step_up") is not None
                    else None
                )
                self._cache[dec.decision_id] = (ev, dec, su)

    def save_record(
        self,
        event: PaymentEvent,
        decision: PolicyDecision,
        step_up: Optional[StepUpExecutionResult] = None,
    ) -> None:
        self._cache[decision.decision_id] = (event, decision, step_up)
        row = {
            "event": event.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "step_up": step_up.model_dump(mode="json") if step_up is not None else None,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def get_record(
        self,
        decision_id: str,
    ) -> Optional[Tuple[PaymentEvent, PolicyDecision, Optional[StepUpExecutionResult]]]:
        return self._cache.get(decision_id)

    def list_decision_ids(self) -> List[str]:
        return list(self._cache.keys())


class OptionalRedisDecisionStore(DecisionStore):
    """Optional Redis adapter; requires `redis` package or an injected Redis-compatible client."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "smart_auth:decision:",
        client: Optional[Any] = None,
    ) -> None:
        self.key_prefix = key_prefix
        self.index_key = f"{key_prefix}index"
        if client is not None:
            self._client = client
        else:
            try:
                import redis  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Redis backend requested, but optional dependency `redis` is not installed. "
                    "Install with `pip install .[redis]` or switch storage.backend to `memory` or `jsonl`."
                ) from exc
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def save_record(
        self,
        event: PaymentEvent,
        decision: PolicyDecision,
        step_up: Optional[StepUpExecutionResult] = None,
    ) -> None:
        row = {
            "event": event.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "step_up": step_up.model_dump(mode="json") if step_up is not None else None,
        }
        key = f"{self.key_prefix}{decision.decision_id}"
        self._client.set(key, json.dumps(row, sort_keys=True))
        self._client.rpush(self.index_key, decision.decision_id)

    def get_record(
        self,
        decision_id: str,
    ) -> Optional[Tuple[PaymentEvent, PolicyDecision, Optional[StepUpExecutionResult]]]:
        raw = self._client.get(f"{self.key_prefix}{decision_id}")
        if raw is None:
            return None
        row = json.loads(raw)
        ev = PaymentEvent.model_validate(row["event"])
        dec = PolicyDecision.model_validate(row["decision"])
        su = (
            StepUpExecutionResult.model_validate(row["step_up"])
            if row.get("step_up") is not None
            else None
        )
        return ev, dec, su

    def list_decision_ids(self) -> List[str]:
        items = self._client.lrange(self.index_key, 0, -1)
        return list(dict.fromkeys(items))
