"""Deterministic synthetic event stream generator for ATO detection research.

Generates strictly synthetic login, device, network, and session events using
RFC 5737 documentation IP blocks (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24).
No real user records, credentials, or device fingerprints are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import random
from typing import Dict, List, Tuple

from .schemas import LoginSessionEvent


SYNTHETIC_CITIES: Tuple[Tuple[str, float, float], ...] = (
    ("US", 40.7128, -74.0060),  # New York
    ("US", 37.7749, -122.4194),  # San Francisco
    ("US", 41.8781, -87.6298),  # Chicago
    ("GB", 51.5074, -0.1278),  # London
    ("DE", 52.5200, 13.4050),  # Berlin
    ("SG", 1.3521, 103.8198),  # Singapore
    ("JP", 35.6762, 139.6503),  # Tokyo
    ("AU", -33.8688, 151.2093),  # Sydney
)

SHARED_NETWORK_IPS: Tuple[str, ...] = (
    "198.51.100.200",
    "198.51.100.201",
    "198.51.100.202",
)

ATTACKER_IPS: Tuple[str, ...] = (
    "203.0.113.10",
    "203.0.113.11",
    "203.0.113.12",
    "203.0.113.13",
    "203.0.113.14",
    "203.0.113.15",
)


@dataclass(frozen=True)
class SyntheticUserProfile:
    user_id: str
    home_country: str
    home_lat: float
    home_lon: float
    primary_device_id: str
    home_ip: str
    typical_hour: int
    preferred_auth: str
    account_tier: str


def _build_user_profiles(rng: random.Random, num_users: int) -> List[SyntheticUserProfile]:
    profiles: List[SyntheticUserProfile] = []
    for idx in range(num_users):
        country, lat, lon = rng.choice(SYNTHETIC_CITIES[:5])
        home_ip = f"192.0.2.{(idx % 220) + 10}"
        profiles.append(
            SyntheticUserProfile(
                user_id=f"usr_{idx:04d}",
                home_country=country,
                home_lat=lat + rng.uniform(-0.15, 0.15),
                home_lon=lon + rng.uniform(-0.15, 0.15),
                primary_device_id=f"dev_{idx:04d}_primary",
                home_ip=home_ip,
                typical_hour=rng.randint(8, 20),
                preferred_auth=rng.choices(["webauthn", "otp", "sso", "password"], weights=[40, 30, 15, 15])[0],
                account_tier="high_value" if idx % 5 == 0 else "standard",
            )
        )
    return profiles


def generate_synthetic_dataset(
    seed: int = 42,
    num_users: int = 90,
    num_days: int = 45,
    base_events_per_day: int = 55,
    ato_rate: float = 0.08,
    missing_telemetry_rate: float = 0.05,
    label_delay_hours_range: Tuple[float, float] = (12.0, 72.0),
    start_time: datetime | None = None,
) -> List[LoginSessionEvent]:
    """Generate a chronological synthetic stream covering normal, new-device, shared-network,
    missing-data, delayed-label, and compromised-session patterns.
    """
    rng = random.Random(seed)
    profiles = _build_user_profiles(rng, num_users=num_users)
    t0 = start_time or datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    events: List[LoginSessionEvent] = []
    seq = 0

    for day in range(num_days):
        for _ in range(base_events_per_day):
            profile = rng.choice(profiles)
            is_ato = rng.random() < ato_rate

            # Pick timestamp centered around typical hour for benign, shifted for ATO
            if is_ato:
                hour_offset = rng.choice([-9, -7, -6, 6, 8, 10])
                hour = (profile.typical_hour + hour_offset) % 24
            else:
                hour = (profile.typical_hour + int(round(rng.gauss(0.0, 2.0)))) % 24
            minute = rng.randint(0, 59)
            second = rng.randint(0, 59)
            ts = t0 + timedelta(days=day, hours=hour, minutes=minute, seconds=second)

            delay_hours = rng.uniform(*label_delay_hours_range)
            label_avail = ts + timedelta(hours=delay_hours)

            if is_ato:
                # Optionally prepend a short burst of failed login attempts (credential stuffing)
                burst_count = rng.choices([0, 2, 4, 6], weights=[25, 25, 30, 20])[0]
                attacker_ip = rng.choice(ATTACKER_IPS)
                attacker_dev = f"dev_atk_{rng.randint(100, 999)}"
                # Choose a distant geo for impossible travel / foreign login
                distant_cities = [c for c in SYNTHETIC_CITIES if c[0] != profile.home_country]
                atk_country, atk_lat, atk_lon = rng.choice(distant_cities)
                atk_lat += rng.uniform(-0.2, 0.2)
                atk_lon += rng.uniform(-0.2, 0.2)

                for b in range(burst_count):
                    seq += 1
                    fail_ts = ts - timedelta(minutes=(burst_count - b) * 3)
                    if fail_ts < t0:
                        fail_ts = ts - timedelta(seconds=(burst_count - b) * 20)
                    events.append(
                        LoginSessionEvent(
                            event_id=f"syn_{seq:06d}",
                            timestamp=fail_ts,
                            user_id=profile.user_id,
                            session_id=f"ses_atk_f_{seq:06d}",
                            event_type="login_attempt",
                            auth_outcome="failure",
                            auth_method="password",
                            account_tier=profile.account_tier,
                            device_id=attacker_dev,
                            device_trust_score=round(rng.uniform(0.05, 0.35), 3),
                            ip_address=attacker_ip,
                            ip_reputation_risk=round(rng.uniform(0.65, 0.98), 3),
                            is_known_shared_network=False,
                            geo_lat=round(atk_lat, 4),
                            geo_lon=round(atk_lon, 4),
                            country_code=atk_country,
                            sensitive_actions=[],
                            session_duration_seconds=round(rng.uniform(5.0, 25.0), 1),
                            label=1,
                            label_available_at=fail_ts + timedelta(hours=delay_hours),
                        )
                    )

                # Compromised session event
                seq += 1
                sens_pool = ["change_mfa", "change_email", "add_payee", "wire_transfer_setup", "export_data"]
                n_sens = rng.choices([1, 2, 3], weights=[30, 45, 25])[0]
                sensitive_actions = rng.sample(sens_pool, k=n_sens)
                # Stealthy ATOs (~25%): residential proxy in home country, moderate trust, single sensitive action
                stealth = rng.random() < 0.25
                events.append(
                    LoginSessionEvent(
                        event_id=f"syn_{seq:06d}",
                        timestamp=ts,
                        user_id=profile.user_id,
                        session_id=f"ses_atk_{seq:06d}",
                        event_type="session_activity" if n_sens >= 2 else "login_attempt",
                        auth_outcome="success",
                        auth_method="password" if (not stealth or rng.random() < 0.5) else profile.preferred_auth,
                        account_tier=profile.account_tier,
                        device_id=attacker_dev,
                        device_trust_score=round(
                            rng.uniform(0.45, 0.78) if stealth else rng.uniform(0.05, 0.42), 3
                        ),
                        ip_address=attacker_ip,
                        ip_reputation_risk=round(
                            rng.uniform(0.22, 0.58) if stealth else rng.uniform(0.60, 0.98), 3
                        ),
                        is_known_shared_network=False,
                        geo_lat=round(profile.home_lat + rng.uniform(-0.35, 0.35), 4)
                        if stealth
                        else round(atk_lat, 4),
                        geo_lon=round(profile.home_lon + rng.uniform(-0.35, 0.35), 4)
                        if stealth
                        else round(atk_lon, 4),
                        country_code=profile.home_country if stealth else atk_country,
                        sensitive_actions=sensitive_actions if not stealth else (sensitive_actions[:1] if rng.random() < 0.7 else []),
                        session_duration_seconds=round(
                            rng.uniform(60.0, 240.0) if stealth else rng.uniform(20.0, 90.0), 1
                        ),
                        label=1,
                        label_available_at=label_avail,
                    )
                )
            else:
                # Benign event: normal, benign new device (~12%), benign shared network (~14%), or benign travel/VPN (~6%)
                seq += 1
                sub_roll = rng.random()
                is_new_benign_dev = sub_roll < 0.12
                is_shared_net = 0.12 <= sub_roll < 0.26
                is_benign_travel_vpn = 0.26 <= sub_roll < 0.32

                device_id: str | None = (
                    f"dev_{profile.user_id}_new_{day // 7}"
                    if (is_new_benign_dev or (is_benign_travel_vpn and rng.random() < 0.4))
                    else profile.primary_device_id
                )
                if is_shared_net:
                    ip_addr: str | None = rng.choice(SHARED_NETWORK_IPS)
                elif is_benign_travel_vpn:
                    ip_addr = f"198.51.100.{rng.randint(20, 120)}"
                else:
                    ip_addr = profile.home_ip

                if is_benign_travel_vpn:
                    device_trust: float | None = round(rng.uniform(0.50, 0.82), 3)
                    ip_risk: float | None = round(rng.uniform(0.25, 0.62), 3)
                    t_country, t_lat, t_lon = rng.choice(SYNTHETIC_CITIES)
                    lat: float | None = round(t_lat + rng.uniform(-0.2, 0.2), 4)
                    lon: float | None = round(t_lon + rng.uniform(-0.2, 0.2), 4)
                    country: str | None = t_country
                else:
                    device_trust = round(
                        rng.uniform(0.65, 0.92) if is_new_benign_dev else rng.uniform(0.75, 0.99), 3
                    )
                    ip_risk = round(
                        rng.uniform(0.08, 0.30) if is_shared_net else rng.uniform(0.01, 0.22), 3
                    )
                    lat = round(profile.home_lat + rng.uniform(-0.2, 0.2), 4)
                    lon = round(profile.home_lon + rng.uniform(-0.2, 0.2), 4)
                    country = profile.home_country

                # Occasional benign typo / failed attempt (~5%) and occasional password fallback (~6%)
                outcome = "failure" if rng.random() < 0.05 else "success"
                auth_m = "password" if (is_benign_travel_vpn and rng.random() < 0.25) else profile.preferred_auth
                # Occasional benign sensitive action (~3%)
                sens = ["change_email"] if (outcome == "success" and rng.random() < 0.03) else []

                # Inject missing telemetry occasionally
                if rng.random() < missing_telemetry_rate:
                    miss_choice = rng.choice(["geo", "ip_rep", "dev_trust"])
                    if miss_choice == "geo":
                        lat, lon, country = None, None, None
                    elif miss_choice == "ip_rep":
                        ip_risk = None
                    else:
                        device_trust = None

                events.append(
                    LoginSessionEvent(
                        event_id=f"syn_{seq:06d}",
                        timestamp=ts,
                        user_id=profile.user_id,
                        session_id=f"ses_ben_{seq:06d}",
                        event_type="login_attempt",
                        auth_outcome=outcome,
                        auth_method=auth_m,
                        account_tier=profile.account_tier,
                        device_id=device_id,
                        device_trust_score=device_trust,
                        ip_address=ip_addr,
                        ip_reputation_risk=ip_risk,
                        is_known_shared_network=is_shared_net,
                        geo_lat=lat,
                        geo_lon=lon,
                        country_code=country,
                        sensitive_actions=sens,
                        session_duration_seconds=round(rng.uniform(100.0, 900.0), 1),
                        label=0,
                        label_available_at=label_avail,
                    )
                )

    events.sort(key=lambda e: (e.timestamp, e.event_id))
    return events
