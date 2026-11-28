"""Strictly causal session and rolling user/device/network feature engineering.

Invariant: NO future or same-timestamp events may enter any feature calculation.
All rolling statistics use strictly prior events (history_ts < event_ts) within
their configured TTL windows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Dict, Iterable, List, Sequence, Tuple

from . import FEATURE_VERSION
from .config import FeaturesConfig
from .schemas import (
    AUTH_METHOD_STRENGTH,
    ContractValidationError,
    FeatureVector,
    LoginSessionEvent,
)
from .store import StateStore


FEATURE_NAMES: Tuple[str, ...] = (
    "is_new_device_for_user",
    "device_age_hours_for_user",
    "user_prior_event_count_30d",
    "failed_attempts_1h",
    "failed_attempts_24h",
    "distinct_ips_24h",
    "distinct_users_on_ip_24h",
    "is_shared_network_ip",
    "geo_velocity_kmh",
    "is_impossible_travel",
    "ip_reputation_risk",
    "ip_reputation_missing",
    "device_trust_score",
    "device_trust_missing",
    "hour_of_day_deviation",
    "session_action_velocity_per_min",
    "sensitive_action_count",
    "auth_factor_downgrade",
    "time_since_last_login_hours",
    "profile_expired",
    "missing_feature_ratio",
    "credential_stuffing_burst",
)

EARTH_RADIUS_KM = 6371.0
MAX_CAPPED_VELOCITY_KMH = 15000.0
DEFAULT_IMPUTED_IP_REPUTATION = 0.35
DEFAULT_IMPUTED_DEVICE_TRUST = 0.50


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two WGS84 coordinates."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def circular_hour_distance(hour_a: float, hour_b: float) -> float:
    """Minimum distance in hours on a 24-hour clock (in [0, 12])."""
    diff = abs(hour_a - hour_b) % 24.0
    return min(diff, 24.0 - diff)


def assert_no_future_events(
    target_ts: datetime,
    history: Sequence[LoginSessionEvent],
) -> None:
    """Enforce the causal invariant: every history event must be strictly earlier than target_ts."""
    target_utc = target_ts.astimezone(timezone.utc)
    for ev in history:
        if ev.timestamp.astimezone(timezone.utc) >= target_utc:
            raise ContractValidationError(
                f"Causal violation: historical event {ev.event_id} at {ev.timestamp.isoformat()} "
                f"is not strictly before target timestamp {target_utc.isoformat()}"
            )


def compute_features(
    event: LoginSessionEvent,
    user_history: Sequence[LoginSessionEvent],
    ip_history: Sequence[LoginSessionEvent],
    has_expired_history: bool,
    config: FeaturesConfig,
    feature_version: str = FEATURE_VERSION,
) -> FeatureVector:
    """Pure, deterministic feature computation separated from I/O.

    All events in `user_history` and `ip_history` MUST precede `event.timestamp`.
    """
    target_ts = event.timestamp.astimezone(timezone.utc)
    assert_no_future_events(target_ts, user_history)
    assert_no_future_events(target_ts, ip_history)

    short_cutoff = target_ts - timedelta(seconds=config.short_window_seconds)
    medium_cutoff = target_ts - timedelta(seconds=config.medium_window_seconds)
    ttl_cutoff = target_ts - timedelta(seconds=config.feature_ttl_seconds)

    # Filter strictly to active non-expired TTL window
    active_user_history = [
        ev for ev in user_history if ev.timestamp.astimezone(timezone.utc) >= ttl_cutoff
    ]
    active_ip_history = [
        ev for ev in ip_history if ev.timestamp.astimezone(timezone.utc) >= medium_cutoff
    ]

    # Missing raw telemetry tracking
    missing_fields: List[str] = []
    if event.device_id is None:
        missing_fields.append("device_id")
    if event.device_trust_score is None:
        missing_fields.append("device_trust_score")
    if event.ip_address is None:
        missing_fields.append("ip_address")
    if event.ip_reputation_risk is None:
        missing_fields.append("ip_reputation_risk")
    if event.geo_lat is None or event.geo_lon is None:
        missing_fields.append("geo_coordinates")
    missing_feature_ratio = round(len(missing_fields) / 5.0, 6)

    # 1 & 2. Device novelty & age for user
    if event.device_id is None:
        is_new_device = 1.0
        device_age_hours = 0.0
    else:
        same_device_events = [
            ev for ev in active_user_history if ev.device_id == event.device_id
        ]
        if same_device_events:
            first_seen = min(ev.timestamp.astimezone(timezone.utc) for ev in same_device_events)
            is_new_device = 0.0
            device_age_hours = (target_ts - first_seen).total_seconds() / 3600.0
        else:
            is_new_device = 1.0
            device_age_hours = 0.0

    # 3. User prior count in active window
    user_prior_count = float(len(active_user_history))

    # 4 & 5. Failed login attempts (1h, 24h)
    failed_outcomes = {"failure", "challenge_failed"}
    failed_1h = float(
        sum(
            1
            for ev in active_user_history
            if ev.timestamp.astimezone(timezone.utc) >= short_cutoff
            and ev.auth_outcome in failed_outcomes
        )
    )
    failed_24h = float(
        sum(
            1
            for ev in active_user_history
            if ev.timestamp.astimezone(timezone.utc) >= medium_cutoff
            and ev.auth_outcome in failed_outcomes
        )
    )

    # 6. Distinct IPs for user in 24h (prior + current if present)
    user_ips_24h = {
        ev.ip_address
        for ev in active_user_history
        if ev.timestamp.astimezone(timezone.utc) >= medium_cutoff and ev.ip_address is not None
    }
    if event.ip_address is not None:
        user_ips_24h.add(event.ip_address)
    distinct_ips_24h = float(len(user_ips_24h))

    # 7 & 8. Shared network / IP fan-out in 24h
    ip_users_24h = {ev.user_id for ev in active_ip_history}
    if event.ip_address is not None:
        ip_users_24h.add(event.user_id)
    distinct_users_on_ip_24h = float(len(ip_users_24h))

    ip_failed_1h = sum(
        1
        for ev in active_ip_history
        if ev.timestamp.astimezone(timezone.utc) >= short_cutoff
        and ev.auth_outcome in failed_outcomes
    )
    ip_success_24h = sum(
        1 for ev in active_ip_history if ev.auth_outcome in {"success", "challenge_passed"}
    )
    # Shared network = explicitly known shared NAT OR multiple users succeeding peacefully on IP
    is_shared_network = (
        1.0
        if (
            event.is_known_shared_network
            or (
                distinct_users_on_ip_24h >= float(config.shared_network_user_threshold)
                and ip_success_24h >= config.shared_network_user_threshold
                and ip_failed_1h < config.burst_failed_attempts_1h
            )
        )
        else 0.0
    )

    # 9 & 10. Geo velocity (impossible travel)
    geo_velocity_kmh = 0.0
    if event.geo_lat is not None and event.geo_lon is not None:
        prior_geo_events = [
            ev
            for ev in active_user_history
            if ev.geo_lat is not None and ev.geo_lon is not None
        ]
        if prior_geo_events:
            last_geo = max(prior_geo_events, key=lambda e: e.timestamp)
            dist_km = haversine_km(
                last_geo.geo_lat, last_geo.geo_lon, event.geo_lat, event.geo_lon  # type: ignore[arg-type]
            )
            delta_hours = max(
                (target_ts - last_geo.timestamp.astimezone(timezone.utc)).total_seconds() / 3600.0,
                1.0 / 60.0,
            )
            geo_velocity_kmh = min(dist_km / delta_hours, MAX_CAPPED_VELOCITY_KMH)
    is_impossible_travel = 1.0 if geo_velocity_kmh > config.max_plausible_velocity_kmh else 0.0

    # 11-14. IP reputation & device trust with explicit missingness indicators
    ip_reputation_missing = 1.0 if event.ip_reputation_risk is None else 0.0
    ip_reputation_risk = (
        DEFAULT_IMPUTED_IP_REPUTATION
        if event.ip_reputation_risk is None
        else float(event.ip_reputation_risk)
    )
    device_trust_missing = 1.0 if event.device_trust_score is None else 0.0
    device_trust_score = (
        DEFAULT_IMPUTED_DEVICE_TRUST
        if event.device_trust_score is None
        else float(event.device_trust_score)
    )

    # 15. Circular hour-of-day deviation from user's historical mean login hour
    current_hour = target_ts.hour + target_ts.minute / 60.0
    if active_user_history:
        angles = [
            2.0
            * math.pi
            * (
                (
                    ev.timestamp.astimezone(timezone.utc).hour
                    + ev.timestamp.astimezone(timezone.utc).minute / 60.0
                )
                / 24.0
            )
            for ev in active_user_history
        ]
        mean_sin = sum(math.sin(a) for a in angles) / len(angles)
        mean_cos = sum(math.cos(a) for a in angles) / len(angles)
        if abs(mean_sin) < 1e-9 and abs(mean_cos) < 1e-9:
            hour_deviation = 0.0
        else:
            mean_angle = math.atan2(mean_sin, mean_cos) % (2.0 * math.pi)
            mean_hour = (mean_angle / (2.0 * math.pi)) * 24.0
            hour_deviation = circular_hour_distance(current_hour, mean_hour)
    else:
        hour_deviation = 0.0

    # 16 & 17. Session sensitive actions & velocity
    sensitive_action_count = float(len(event.sensitive_actions))
    duration_minutes = max(event.session_duration_seconds / 60.0, 0.5)
    session_action_velocity = sensitive_action_count / duration_minutes

    # 18. Auth factor downgrade (relative to user's strongest prior successful auth)
    prior_strengths = [
        AUTH_METHOD_STRENGTH.get(ev.auth_method, 1)
        for ev in active_user_history
        if ev.auth_outcome in {"success", "challenge_passed"}
    ]
    current_strength = AUTH_METHOD_STRENGTH.get(event.auth_method, 1)
    auth_factor_downgrade = (
        1.0 if (prior_strengths and current_strength < max(prior_strengths)) else 0.0
    )

    # 19. Time since last non-expired login
    if active_user_history:
        last_ts = max(ev.timestamp.astimezone(timezone.utc) for ev in active_user_history)
        time_since_last_hours = (target_ts - last_ts).total_seconds() / 3600.0
    else:
        time_since_last_hours = -1.0

    # 20. Feature expiry indicator
    profile_expired_flag = 1.0 if has_expired_history else 0.0

    # 22. Credential stuffing / repeated attempts burst
    burst_threshold = float(config.burst_failed_attempts_1h)
    credential_stuffing_burst = (
        1.0
        if (
            failed_1h >= burst_threshold
            or (ip_failed_1h >= burst_threshold and is_shared_network == 0.0)
        )
        else 0.0
    )

    values: Dict[str, float] = {
        "is_new_device_for_user": is_new_device,
        "device_age_hours_for_user": round(device_age_hours, 6),
        "user_prior_event_count_30d": user_prior_count,
        "failed_attempts_1h": failed_1h,
        "failed_attempts_24h": failed_24h,
        "distinct_ips_24h": distinct_ips_24h,
        "distinct_users_on_ip_24h": distinct_users_on_ip_24h,
        "is_shared_network_ip": is_shared_network,
        "geo_velocity_kmh": round(geo_velocity_kmh, 6),
        "is_impossible_travel": is_impossible_travel,
        "ip_reputation_risk": round(ip_reputation_risk, 6),
        "ip_reputation_missing": ip_reputation_missing,
        "device_trust_score": round(device_trust_score, 6),
        "device_trust_missing": device_trust_missing,
        "hour_of_day_deviation": round(hour_deviation, 6),
        "session_action_velocity_per_min": round(session_action_velocity, 6),
        "sensitive_action_count": sensitive_action_count,
        "auth_factor_downgrade": auth_factor_downgrade,
        "time_since_last_login_hours": round(time_since_last_hours, 6),
        "profile_expired": profile_expired_flag,
        "missing_feature_ratio": missing_feature_ratio,
        "credential_stuffing_burst": credential_stuffing_burst,
    }

    return FeatureVector(
        event_id=event.event_id,
        user_id=event.user_id,
        timestamp=target_ts,
        feature_version=feature_version,
        values=values,
        missing_fields=missing_fields,
        profile_expired=has_expired_history,
        history_events_considered=len(active_user_history) + len(active_ip_history),
    )


def extract_features_from_store(
    event: LoginSessionEvent,
    store: StateStore,
    config: FeaturesConfig,
    feature_version: str = FEATURE_VERSION,
) -> FeatureVector:
    """Query causal history from a StateStore and compute FeatureVector without mutating store."""
    user_history = store.get_user_history(
        user_id=event.user_id,
        before_ts=event.timestamp,
        window_seconds=config.feature_ttl_seconds,
    )
    has_expired = store.has_expired_user_history(
        user_id=event.user_id,
        before_ts=event.timestamp,
        ttl_seconds=config.feature_ttl_seconds,
    )
    ip_history: List[LoginSessionEvent] = []
    if event.ip_address is not None:
        ip_history = store.get_ip_history(
            ip_address=event.ip_address,
            before_ts=event.timestamp,
            window_seconds=config.medium_window_seconds,
        )
    return compute_features(
        event=event,
        user_history=user_history,
        ip_history=ip_history,
        has_expired_history=has_expired,
        config=config,
        feature_version=feature_version,
    )


def extract_stream_features(
    events: Iterable[LoginSessionEvent],
    store: StateStore,
    config: FeaturesConfig,
    feature_version: str = FEATURE_VERSION,
    record_events: bool = True,
) -> List[FeatureVector]:
    """Extract causal features sequentially for a chronologically ordered stream of events."""
    sorted_events = sorted(events, key=lambda e: (e.timestamp, e.event_id))
    vectors: List[FeatureVector] = []
    for ev in sorted_events:
        fv = extract_features_from_store(
            event=ev,
            store=store,
            config=config,
            feature_version=feature_version,
        )
        vectors.append(fv)
        if record_events:
            store.record_event(ev)
    return vectors


def vector_to_array_row(fv: FeatureVector) -> List[float]:
    """Convert a FeatureVector to a fixed-order numeric list matching FEATURE_NAMES."""
    return [float(fv.values[name]) for name in FEATURE_NAMES]
