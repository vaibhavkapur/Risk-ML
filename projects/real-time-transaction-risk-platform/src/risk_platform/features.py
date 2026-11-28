"""Pure domain time-window feature definitions and offline/online parity engine.

Single source of truth for feat.v1 feature computation used identically by both
offline point-in-time batch extraction and online sliding-window serving.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from risk_platform.schemas import (
    DEFAULT_FX_TO_USD,
    FeatureVector,
    KNOWN_MERCHANT_CATEGORIES,
    TransactionEvent,
)

FEATURE_VERSION = "feat.v1"

FEATURE_NAMES: Tuple[str, ...] = (
    "amount_usd_log1p",
    "tx_count_5m",
    "tx_count_1h",
    "tx_count_24h",
    "tx_amount_sum_1h_usd",
    "tx_amount_sum_24h_usd",
    "amount_ratio_to_24h_avg",
    "distinct_merchants_24h",
    "distinct_devices_24h",
    "country_changed_from_prev",
    "seconds_since_prev_tx_log1p",
    "high_risk_context_score",
)

WINDOW_5M_MS = 5 * 60 * 1000
WINDOW_1H_MS = 60 * 60 * 1000
WINDOW_24H_MS = 24 * 60 * 60 * 1000

_CHANNEL_RISK: Dict[str, float] = {
    "card_present": 0.0,
    "recurring": 0.0,
    "ecommerce": 0.25,
    "in_app": 0.25,
}

_MCC_RISK: Dict[str, float] = {
    "grocery": 0.0,
    "utilities": 0.0,
    "restaurant": 0.0,
    "travel": 0.15,
    "electronics": 0.20,
    "gaming": 0.25,
    "digital_goods": 0.30,
    "unknown": 0.35,
    "crypto_exchange": 0.45,
}

KNOWN_COUNTRIES = frozenset({"US", "GB", "DE", "FR", "CA", "SG", "AU", "NL", "BR", "NG"})


def _sort_key(event: TransactionEvent) -> Tuple[int, str]:
    """Canonical deterministic temporal ordering key."""
    return (event.event_ts_ms, event.event_id)


def detect_unknown_input_flags(event: TransactionEvent) -> List[str]:
    """Identify unknown or conservative-handling categorical attributes."""
    flags: List[str] = []
    if event.merchant_category == "unknown" or event.merchant_category not in KNOWN_MERCHANT_CATEGORIES:
        flags.append(f"UNKNOWN_MCC:{event.merchant_category}")
    if event.country_code not in KNOWN_COUNTRIES:
        flags.append(f"UNKNOWN_COUNTRY:{event.country_code}")
    return flags


def compute_point_in_time_features(
    event: TransactionEvent,
    prior_account_events: Sequence[TransactionEvent],
    fx_to_usd: Optional[Dict[str, float]] = None,
    feature_freshness_ms: Optional[int] = None,
    feature_status: str = "FRESH",
) -> FeatureVector:
    """Compute exact feat.v1 point-in-time feature vector from strictly prior account events.

    Window semantics:
      Half-open interval (t - W, t] over events h belonging to event.account_id
      such that (h.event_ts_ms, h.event_id) < (event.event_ts_ms, event.event_id)
      and h.event_id != event.event_id.
    """
    rates = fx_to_usd or DEFAULT_FX_TO_USD
    t = event.event_ts_ms
    curr_key = _sort_key(event)
    amount_usd = event.amount_usd(rates)

    cutoff_5m = t - WINDOW_5M_MS
    cutoff_1h = t - WINDOW_1H_MS
    cutoff_24h = t - WINDOW_24H_MS

    tx_count_5m = 0
    tx_count_1h = 0
    tx_count_24h = 0
    tx_amount_sum_1h_usd = 0.0
    tx_amount_sum_24h_usd = 0.0
    merchants_24h = set()
    devices_24h = set()

    most_recent_prior: Optional[TransactionEvent] = None
    most_recent_key: Optional[Tuple[int, str]] = None

    for h in prior_account_events:
        if h.account_id != event.account_id:
            continue
        if h.event_id == event.event_id:
            continue
        h_key = _sort_key(h)
        if h_key >= curr_key:
            continue
        if h.event_ts_ms <= cutoff_24h:
            continue

        h_amt = h.amount_usd(rates)
        tx_count_24h += 1
        tx_amount_sum_24h_usd += h_amt
        merchants_24h.add(h.merchant_id)
        devices_24h.add(h.device_id)

        if h.event_ts_ms > cutoff_1h:
            tx_count_1h += 1
            tx_amount_sum_1h_usd += h_amt

        if h.event_ts_ms > cutoff_5m:
            tx_count_5m += 1

        if most_recent_key is None or h_key > most_recent_key:
            most_recent_key = h_key
            most_recent_prior = h

    if tx_count_24h > 0:
        avg_24h = tx_amount_sum_24h_usd / float(tx_count_24h)
        amount_ratio_to_24h_avg = amount_usd / (avg_24h + 1.0)
    else:
        amount_ratio_to_24h_avg = 1.0

    if most_recent_prior is not None:
        country_changed = 1.0 if most_recent_prior.country_code != event.country_code else 0.0
        delta_sec = max(0.0, min(86400.0, (t - most_recent_prior.event_ts_ms) / 1000.0))
        seconds_since_prev_tx_log1p = math.log1p(delta_sec)
    else:
        country_changed = 0.0
        seconds_since_prev_tx_log1p = math.log1p(86400.0)

    channel_risk = _CHANNEL_RISK.get(event.channel, 0.25)
    mcc_risk = _MCC_RISK.get(event.merchant_category, 0.35)
    high_risk_context = min(1.5, channel_risk + mcc_risk + 0.5 * event.ip_risk_score)

    values: Dict[str, float] = {
        "amount_usd_log1p": math.log1p(amount_usd),
        "tx_count_5m": float(tx_count_5m),
        "tx_count_1h": float(tx_count_1h),
        "tx_count_24h": float(tx_count_24h),
        "tx_amount_sum_1h_usd": float(tx_amount_sum_1h_usd),
        "tx_amount_sum_24h_usd": float(tx_amount_sum_24h_usd),
        "amount_ratio_to_24h_avg": float(amount_ratio_to_24h_avg),
        "distinct_merchants_24h": float(len(merchants_24h)),
        "distinct_devices_24h": float(len(devices_24h)),
        "country_changed_from_prev": float(country_changed),
        "seconds_since_prev_tx_log1p": float(seconds_since_prev_tx_log1p),
        "high_risk_context_score": float(high_risk_context),
    }

    return FeatureVector(
        event_id=event.event_id,
        account_id=event.account_id,
        feature_version=FEATURE_VERSION,
        as_of_ts_ms=t,
        values=values,
        feature_freshness_ms=feature_freshness_ms,
        feature_status=feature_status,
        unknown_input_flags=detect_unknown_input_flags(event),
    )


class OfflineFeatureExtractor:
    """Deterministic point-in-time batch feature builder for training and parity evaluation."""

    def __init__(self, fx_to_usd: Optional[Dict[str, float]] = None) -> None:
        self._fx_to_usd = dict(fx_to_usd or DEFAULT_FX_TO_USD)

    def extract_batch(self, events: Iterable[TransactionEvent]) -> Dict[str, FeatureVector]:
        """Deduplicate by event_id, sort canonically, and compute point-in-time feature vectors."""
        deduped: Dict[str, TransactionEvent] = {}
        for ev in events:
            if ev.event_id not in deduped:
                deduped[ev.event_id] = ev

        ordered = sorted(deduped.values(), key=_sort_key)
        history_by_account: Dict[str, List[TransactionEvent]] = defaultdict(list)
        results: Dict[str, FeatureVector] = {}

        for ev in ordered:
            hist = history_by_account[ev.account_id]
            # Prune events strictly older than 24h to keep batch extraction fast and exact
            cutoff_24h = ev.event_ts_ms - WINDOW_24H_MS
            if hist and hist[0].event_ts_ms <= cutoff_24h:
                hist = [h for h in hist if h.event_ts_ms > cutoff_24h]
                history_by_account[ev.account_id] = hist

            fv = compute_point_in_time_features(
                event=ev,
                prior_account_events=hist,
                fx_to_usd=self._fx_to_usd,
                feature_freshness_ms=0 if hist else None,
                feature_status="FRESH" if hist else "COLD_START",
            )
            results[ev.event_id] = fv
            hist.append(ev)

        return results


@dataclass(frozen=True)
class ParityMismatch:
    event_id: str
    feature: str
    offline_value: float
    online_value: float
    abs_diff: float


@dataclass(frozen=True)
class ParityReport:
    passed: bool
    events_compared: int
    features_per_event: int
    max_abs_diff: float
    tolerance: float
    mismatches: List[ParityMismatch]

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "events_compared": self.events_compared,
            "features_per_event": self.features_per_event,
            "max_abs_diff": self.max_abs_diff,
            "tolerance": self.tolerance,
            "mismatch_count": len(self.mismatches),
            "mismatches": [
                {
                    "event_id": m.event_id,
                    "feature": m.feature,
                    "offline_value": m.offline_value,
                    "online_value": m.online_value,
                    "abs_diff": m.abs_diff,
                }
                for m in self.mismatches[:25]
            ],
        }


def check_feature_parity(
    offline_vectors: Dict[str, FeatureVector],
    online_vectors: Dict[str, FeatureVector],
    atol: float = 1e-9,
) -> ParityReport:
    """Compare offline and online feature vectors across matching event_ids within tolerance."""
    common_ids = sorted(set(offline_vectors.keys()) & set(online_vectors.keys()))
    mismatches: List[ParityMismatch] = []
    max_abs_diff = 0.0

    for eid in common_ids:
        off_fv = offline_vectors[eid]
        on_fv = online_vectors[eid]
        for fname in FEATURE_NAMES:
            off_val = float(off_fv.values[fname])
            on_val = float(on_fv.values[fname])
            diff = abs(off_val - on_val)
            if diff > max_abs_diff:
                max_abs_diff = diff
            if diff > atol:
                mismatches.append(
                    ParityMismatch(
                        event_id=eid,
                        feature=fname,
                        offline_value=off_val,
                        online_value=on_val,
                        abs_diff=diff,
                    )
                )

    missing_in_online = set(offline_vectors.keys()) - set(online_vectors.keys())
    missing_in_offline = set(online_vectors.keys()) - set(offline_vectors.keys())
    passed = (len(mismatches) == 0) and (len(missing_in_online) == 0) and (len(missing_in_offline) == 0)

    return ParityReport(
        passed=passed,
        events_compared=len(common_ids),
        features_per_event=len(FEATURE_NAMES),
        max_abs_diff=max_abs_diff,
        tolerance=atol,
        mismatches=mismatches,
    )
