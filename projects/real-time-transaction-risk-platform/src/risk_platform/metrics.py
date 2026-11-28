"""Thread-safe diagnostic counters and latency percentile tracking."""

from __future__ import annotations

from collections import defaultdict
import threading
from typing import Any, Dict, List
import numpy as np


class MetricsRegistry:
    """In-process structured metrics collector for counters and latency distributions."""

    def __init__(self, max_latency_samples: int = 50_000) -> None:
        self._lock = threading.RLock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._latencies_ms: List[float] = []
        self._max_samples = int(max_latency_samples)

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += int(value)

    def observe_latency(self, latency_ms: float) -> None:
        with self._lock:
            self._latencies_ms.append(float(latency_ms))
            if len(self._latencies_ms) > self._max_samples:
                del self._latencies_ms[: len(self._latencies_ms) - self._max_samples]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters = dict(sorted(self._counters.items()))
            samples = list(self._latencies_ms)

        if samples:
            arr = np.array(samples, dtype=float)
            latency_stats = {
                "count": len(samples),
                "min_ms": round(float(np.min(arr)), 4),
                "mean_ms": round(float(np.mean(arr)), 4),
                "p50_ms": round(float(np.percentile(arr, 50)), 4),
                "p95_ms": round(float(np.percentile(arr, 95)), 4),
                "p99_ms": round(float(np.percentile(arr, 99)), 4),
                "max_ms": round(float(np.max(arr)), 4),
            }
        else:
            latency_stats = {
                "count": 0,
                "min_ms": 0.0,
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "max_ms": 0.0,
            }

        return {
            "counters": counters,
            "scoring_latency": latency_stats,
        }
