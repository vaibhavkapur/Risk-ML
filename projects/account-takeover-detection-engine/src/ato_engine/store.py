"""Causal event state store abstraction with in-memory default and optional Redis adapter."""

from __future__ import annotations

from bisect import insort
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from typing import Dict, List, Optional, Protocol, Tuple

from .schemas import LoginSessionEvent, StoreUnavailableError


class StateStore(Protocol):
    """Abstract interface for causal historical event lookups."""

    def record_event(self, event: LoginSessionEvent) -> None:
        """Persist an event for future causal lookups."""

    def get_user_history(
        self,
        user_id: str,
        before_ts: datetime,
        window_seconds: int,
    ) -> List[LoginSessionEvent]:
        """Return events for user_id strictly before before_ts within window_seconds."""

    def has_expired_user_history(
        self,
        user_id: str,
        before_ts: datetime,
        ttl_seconds: int,
    ) -> bool:
        """Return True if user had events strictly before TTL cutoff and none active within TTL."""

    def get_ip_history(
        self,
        ip_address: str,
        before_ts: datetime,
        window_seconds: int,
    ) -> List[LoginSessionEvent]:
        """Return events for ip_address strictly before before_ts within window_seconds."""

    def ping(self) -> bool:
        """Return True if store backend is healthy."""


class InMemoryStateStore:
    """Deterministic in-memory causal store suitable for offline evaluation and local serving."""

    def __init__(self) -> None:
        self._user_events: Dict[str, List[Tuple[float, str, LoginSessionEvent]]] = defaultdict(list)
        self._ip_events: Dict[str, List[Tuple[float, str, LoginSessionEvent]]] = defaultdict(list)
        self._seen_event_ids: set[str] = set()
        self._available: bool = True

    def set_available(self, available: bool) -> None:
        """Toggle availability to simulate store outages in fault-injection tests."""
        self._available = available

    def ping(self) -> bool:
        return self._available

    def _check_available(self) -> None:
        if not self._available:
            raise StoreUnavailableError("InMemoryStateStore simulated outage")

    def record_event(self, event: LoginSessionEvent) -> None:
        self._check_available()
        if event.event_id in self._seen_event_ids:
            return
        self._seen_event_ids.add(event.event_id)
        ts_epoch = event.timestamp.astimezone(timezone.utc).timestamp()
        entry = (ts_epoch, event.event_id, event)
        insort(self._user_events[event.user_id], entry)
        if event.ip_address is not None:
            insort(self._ip_events[event.ip_address], entry)

    def get_user_history(
        self,
        user_id: str,
        before_ts: datetime,
        window_seconds: int,
    ) -> List[LoginSessionEvent]:
        self._check_available()
        upper = before_ts.astimezone(timezone.utc).timestamp()
        lower = (before_ts.astimezone(timezone.utc) - timedelta(seconds=window_seconds)).timestamp()
        return [
            ev
            for ts_epoch, _, ev in self._user_events.get(user_id, ())
            if lower <= ts_epoch < upper
        ]

    def has_expired_user_history(
        self,
        user_id: str,
        before_ts: datetime,
        ttl_seconds: int,
    ) -> bool:
        self._check_available()
        upper = before_ts.astimezone(timezone.utc).timestamp()
        cutoff = (before_ts.astimezone(timezone.utc) - timedelta(seconds=ttl_seconds)).timestamp()
        events = self._user_events.get(user_id, ())
        has_older = any(ts_epoch < cutoff for ts_epoch, _, _ in events)
        has_active = any(cutoff <= ts_epoch < upper for ts_epoch, _, _ in events)
        return has_older and not has_active

    def get_ip_history(
        self,
        ip_address: str,
        before_ts: datetime,
        window_seconds: int,
    ) -> List[LoginSessionEvent]:
        self._check_available()
        upper = before_ts.astimezone(timezone.utc).timestamp()
        lower = (before_ts.astimezone(timezone.utc) - timedelta(seconds=window_seconds)).timestamp()
        return [
            ev
            for ts_epoch, _, ev in self._ip_events.get(ip_address, ())
            if lower <= ts_epoch < upper
        ]


class RedisStateStore:
    """Optional Redis-backed causal state store using sorted sets (ZSET) scored by UTC epoch.

    Requires the optional `redis` package and a running Redis server.
    Never called automatically unless explicitly configured via `store.backend: redis`.
    """

    def __init__(self, redis_url: str, key_prefix: str = "ato_engine:") -> None:
        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise StoreUnavailableError(
                "Optional dependency 'redis' is not installed. Install with `pip install .[redis]`."
            ) from exc
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._prefix = key_prefix

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    def _user_key(self, user_id: str) -> str:
        return f"{self._prefix}user:{user_id}"

    def _ip_key(self, ip_address: str) -> str:
        return f"{self._prefix}ip:{ip_address}"

    def record_event(self, event: LoginSessionEvent) -> None:
        try:
            ts_epoch = event.timestamp.astimezone(timezone.utc).timestamp()
            payload = event.model_dump_json()
            pipe = self._client.pipeline()
            pipe.zadd(self._user_key(event.user_id), {payload: ts_epoch})
            if event.ip_address is not None:
                pipe.zadd(self._ip_key(event.ip_address), {payload: ts_epoch})
            pipe.execute()
        except Exception as exc:
            raise StoreUnavailableError(f"Redis record_event failed: {exc}") from exc

    def get_user_history(
        self,
        user_id: str,
        before_ts: datetime,
        window_seconds: int,
    ) -> List[LoginSessionEvent]:
        try:
            upper = before_ts.astimezone(timezone.utc).timestamp()
            lower = (before_ts.astimezone(timezone.utc) - timedelta(seconds=window_seconds)).timestamp()
            raw_items = self._client.zrangebyscore(
                self._user_key(user_id), min=lower, max=f"({upper}"
            )
            return [LoginSessionEvent.model_validate_json(item) for item in raw_items]
        except Exception as exc:
            raise StoreUnavailableError(f"Redis get_user_history failed: {exc}") from exc

    def has_expired_user_history(
        self,
        user_id: str,
        before_ts: datetime,
        ttl_seconds: int,
    ) -> bool:
        try:
            upper = before_ts.astimezone(timezone.utc).timestamp()
            cutoff = (before_ts.astimezone(timezone.utc) - timedelta(seconds=ttl_seconds)).timestamp()
            older_count = self._client.zcount(self._user_key(user_id), min="-inf", max=f"({cutoff}")
            active_count = self._client.zcount(self._user_key(user_id), min=cutoff, max=f"({upper}")
            return int(older_count) > 0 and int(active_count) == 0
        except Exception as exc:
            raise StoreUnavailableError(f"Redis has_expired_user_history failed: {exc}") from exc

    def get_ip_history(
        self,
        ip_address: str,
        before_ts: datetime,
        window_seconds: int,
    ) -> List[LoginSessionEvent]:
        try:
            upper = before_ts.astimezone(timezone.utc).timestamp()
            lower = (before_ts.astimezone(timezone.utc) - timedelta(seconds=window_seconds)).timestamp()
            raw_items = self._client.zrangebyscore(
                self._ip_key(ip_address), min=lower, max=f"({upper}"
            )
            return [LoginSessionEvent.model_validate_json(item) for item in raw_items]
        except Exception as exc:
            raise StoreUnavailableError(f"Redis get_ip_history failed: {exc}") from exc
