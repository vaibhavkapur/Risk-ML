"""Streaming feature store with idempotent ingestion, watermarks, freshness, and checkpoint recovery."""

from __future__ import annotations

from bisect import insort
from collections import defaultdict
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from risk_platform.errors import DuplicateConflictError, FeatureStoreError
from risk_platform.features import compute_point_in_time_features
from risk_platform.schemas import (
    DEFAULT_FX_TO_USD,
    FeatureVector,
    IngestionResult,
    TransactionEvent,
)


class FeatureStore(Protocol):
    """Pluggable feature store boundary for online serving and ingestion."""

    def ingest(self, event: TransactionEvent) -> IngestionResult: ...

    def get_features(
        self,
        event: TransactionEvent,
        as_of_ts_ms: Optional[int] = None,
        max_staleness_ms: Optional[int] = None,
    ) -> FeatureVector: ...

    def stats(self) -> Dict[str, Any]: ...


class InMemoryFeatureStore:
    """Thread-safe sliding-window streaming feature store with checkpoint/restore support.

    Guarantees:
    - Idempotent ingestion keyed by event_id with SHA-256 payload conflict detection.
    - Watermark-based late-event classification (ACCEPTED, LATE_ACCEPTED, LATE_DROPPED).
    - Deterministic sorted insertion by (event_ts_ms, event_id) for exact offline/online parity.
    - Feature freshness tracking and stale-cache detection.
    - Atomic JSON checkpoint persistence for restart recovery.
    """

    def __init__(
        self,
        allowed_lateness_ms: int = 300_000,
        retention_window_ms: int = 90_000_000,
        max_events_per_account: int = 2000,
        max_staleness_ms: int = 600_000,
        fx_to_usd: Optional[Dict[str, float]] = None,
    ) -> None:
        self.allowed_lateness_ms = int(allowed_lateness_ms)
        self.retention_window_ms = int(retention_window_ms)
        self.max_events_per_account = int(max_events_per_account)
        self.max_staleness_ms = int(max_staleness_ms)
        self.fx_to_usd = dict(fx_to_usd or DEFAULT_FX_TO_USD)

        self._lock = threading.RLock()
        self._seen_events: Dict[str, str] = {}  # event_id -> payload_hash
        # Per account: sorted list of (event_ts_ms, event_id, TransactionEvent)
        self._account_events: Dict[str, List[Tuple[int, str, TransactionEvent]]] = defaultdict(list)
        self._account_last_updated_ms: Dict[str, int] = {}
        self._watermark_ms: int = 0
        self._forced_stale_accounts: Dict[str, int] = {}  # Optional test/ops override for cache age

        self._counters: Dict[str, int] = {
            "accepted": 0,
            "duplicate_ignored": 0,
            "duplicate_conflict": 0,
            "late_accepted": 0,
            "late_dropped": 0,
        }

    def ingest(self, event: TransactionEvent) -> IngestionResult:
        """Idempotently ingest a transaction event into the sliding-window store."""
        payload_hash = event.payload_hash()
        with self._lock:
            existing_hash = self._seen_events.get(event.event_id)
            if existing_hash is not None:
                if existing_hash == payload_hash:
                    self._counters["duplicate_ignored"] += 1
                    return IngestionResult(
                        event_id=event.event_id,
                        status="DUPLICATE_IGNORED",
                        watermark_ms=self._watermark_ms,
                        account_event_count=len(self._account_events.get(event.account_id, [])),
                        payload_hash=payload_hash,
                        reason="Identical event_id and payload hash already ingested",
                    )
                self._counters["duplicate_conflict"] += 1
                raise DuplicateConflictError(
                    f"Conflicting duplicate payload for event_id '{event.event_id}': "
                    f"existing={existing_hash[:12]}, incoming={payload_hash[:12]}"
                )

            is_late = self._watermark_ms > 0 and event.event_ts_ms < self._watermark_ms
            if is_late and (self._watermark_ms - event.event_ts_ms) > self.allowed_lateness_ms:
                self._seen_events[event.event_id] = payload_hash
                self._counters["late_dropped"] += 1
                return IngestionResult(
                    event_id=event.event_id,
                    status="LATE_DROPPED",
                    watermark_ms=self._watermark_ms,
                    account_event_count=len(self._account_events.get(event.account_id, [])),
                    payload_hash=payload_hash,
                    reason=(
                        f"Event older than allowed_lateness_ms ({self.allowed_lateness_ms}ms) "
                        f"behind watermark ({self._watermark_ms - event.event_ts_ms}ms late)"
                    ),
                )

            self._seen_events[event.event_id] = payload_hash
            if event.event_ts_ms > self._watermark_ms:
                self._watermark_ms = event.event_ts_ms

            bucket = self._account_events[event.account_id]
            insort(bucket, (event.event_ts_ms, event.event_id, event))

            # Prune expired events outside retention_window_ms relative to latest account/stream watermark
            min_keep_ts = self._watermark_ms - self.retention_window_ms
            if bucket and bucket[0][0] < min_keep_ts:
                bucket[:] = [item for item in bucket if item[0] >= min_keep_ts]
            if len(bucket) > self.max_events_per_account:
                del bucket[: len(bucket) - self.max_events_per_account]

            effective_update_ms = (
                event.ingest_ts_ms if event.ingest_ts_ms is not None else event.event_ts_ms
            )
            prev_upd = self._account_last_updated_ms.get(event.account_id, 0)
            self._account_last_updated_ms[event.account_id] = max(prev_upd, effective_update_ms)

            status = "LATE_ACCEPTED" if is_late else "ACCEPTED"
            self._counters["late_accepted" if is_late else "accepted"] += 1
            return IngestionResult(
                event_id=event.event_id,
                status=status,
                watermark_ms=self._watermark_ms,
                account_event_count=len(bucket),
                payload_hash=payload_hash,
                reason="Accepted within allowed lateness window" if is_late else None,
            )

    def mark_account_cache_timestamp(self, account_id: str, last_updated_ms: int) -> None:
        """Override account cache freshness timestamp (useful for stale-cache simulation/testing)."""
        with self._lock:
            self._forced_stale_accounts[account_id] = int(last_updated_ms)

    def get_features(
        self,
        event: TransactionEvent,
        as_of_ts_ms: Optional[int] = None,
        max_staleness_ms: Optional[int] = None,
    ) -> FeatureVector:
        """Compute point-in-time features from the online store with freshness classification."""
        eval_ts = int(as_of_ts_ms) if as_of_ts_ms is not None else event.event_ts_ms
        staleness_limit = (
            int(max_staleness_ms) if max_staleness_ms is not None else self.max_staleness_ms
        )

        with self._lock:
            bucket = self._account_events.get(event.account_id, [])
            prior_events = [item[2] for item in bucket]

            if event.account_id in self._forced_stale_accounts:
                last_updated = self._forced_stale_accounts[event.account_id]
                freshness_ms: Optional[int] = max(0, eval_ts - last_updated)
                feature_status = "STALE" if freshness_ms > staleness_limit else "FRESH"
            elif not prior_events:
                freshness_ms = None
                feature_status = "COLD_START"
            else:
                # Freshness compares scoring evaluation time against the latest known stream watermark
                # or account update timestamp.
                ref_ts = max(
                    self._watermark_ms,
                    self._account_last_updated_ms.get(event.account_id, 0),
                )
                freshness_ms = max(0, eval_ts - ref_ts)
                feature_status = "STALE" if freshness_ms > staleness_limit else "FRESH"

        return compute_point_in_time_features(
            event=event,
            prior_account_events=prior_events,
            fx_to_usd=self.fx_to_usd,
            feature_freshness_ms=freshness_ms,
            feature_status=feature_status,
        )

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "watermark_ms": self._watermark_ms,
                "tracked_accounts": len(self._account_events),
                "seen_event_ids": len(self._seen_events),
                "retained_events": sum(len(v) for v in self._account_events.values()),
                "counters": dict(self._counters),
            }

    def save_checkpoint(self, path: Union[str, Path]) -> str:
        """Atomically persist store state to JSON for deterministic restart recovery."""
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "checkpoint_version": "ckpt.v1",
                "allowed_lateness_ms": self.allowed_lateness_ms,
                "retention_window_ms": self.retention_window_ms,
                "max_events_per_account": self.max_events_per_account,
                "max_staleness_ms": self.max_staleness_ms,
                "fx_to_usd": self.fx_to_usd,
                "watermark_ms": self._watermark_ms,
                "seen_events": dict(self._seen_events),
                "account_last_updated_ms": dict(self._account_last_updated_ms),
                "counters": dict(self._counters),
                "events": [
                    item[2].to_dict()
                    for acct_id in sorted(self._account_events.keys())
                    for item in self._account_events[acct_id]
                ],
            }

        tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(tmp_path, ckpt_path)
        return str(ckpt_path)

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "InMemoryFeatureStore":
        """Restore feature store state from a previously saved checkpoint."""
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FeatureStoreError(f"Checkpoint file not found: {ckpt_path}")

        try:
            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise FeatureStoreError(f"Failed to read checkpoint '{ckpt_path}': {exc}") from exc

        if payload.get("checkpoint_version") != "ckpt.v1":
            raise FeatureStoreError(
                f"Unsupported checkpoint_version: {payload.get('checkpoint_version')}"
            )

        store = cls(
            allowed_lateness_ms=int(payload["allowed_lateness_ms"]),
            retention_window_ms=int(payload["retention_window_ms"]),
            max_events_per_account=int(payload["max_events_per_account"]),
            max_staleness_ms=int(payload["max_staleness_ms"]),
            fx_to_usd=payload.get("fx_to_usd"),
        )
        store._watermark_ms = int(payload["watermark_ms"])
        store._seen_events = {str(k): str(v) for k, v in payload["seen_events"].items()}
        store._account_last_updated_ms = {
            str(k): int(v) for k, v in payload["account_last_updated_ms"].items()
        }
        store._counters = {str(k): int(v) for k, v in payload["counters"].items()}

        for ev_dict in payload.get("events", []):
            ev = TransactionEvent.from_dict(ev_dict)
            insort(store._account_events[ev.account_id], (ev.event_ts_ms, ev.event_id, ev))

        return store


class RedisFeatureStoreAdapter:
    """Optional Redis-backed feature store adapter boundary.

    Not activated by default in offline prototype mode; explicitly raises FeatureStoreError
    if the optional `redis` dependency or Redis server is unavailable.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", key_prefix: str = "rtrp:") -> None:
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise FeatureStoreError(
                "Optional 'redis' package is not installed. Install with `pip install .[redis]` "
                "or use InMemoryFeatureStore in offline prototype mode."
            ) from exc

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        try:
            self._client.ping()
        except Exception as exc:
            raise FeatureStoreError(
                f"Cannot reach Redis at '{redis_url}': {exc}. Use InMemoryFeatureStore for offline runs."
            ) from exc

    def ingest(self, event: TransactionEvent) -> IngestionResult:
        raise FeatureStoreError("RedisFeatureStoreAdapter is an optional integration boundary; run Redis server to enable.")

    def get_features(
        self,
        event: TransactionEvent,
        as_of_ts_ms: Optional[int] = None,
        max_staleness_ms: Optional[int] = None,
    ) -> FeatureVector:
        raise FeatureStoreError("RedisFeatureStoreAdapter is an optional integration boundary; run Redis server to enable.")

    def stats(self) -> Dict[str, Any]:
        return {"adapter": "redis", "redis_url": self.redis_url}
