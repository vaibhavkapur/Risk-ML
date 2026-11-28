# Data Contracts — Account Takeover Detection Engine

**Schema Version:** `1.0.0`  
**Feature Version:** `1.0.0`  
**Policy Version:** `1.0.0`  
**Scope:** Strictly synthetic identifiers and RFC 5737 documentation IP addresses. No real credentials, raw passwords, customer PII, or production device fingerprints are permitted (`extra="forbid"` rejects unexpected fields).

---

## 1. Input Contract: `LoginSessionEvent`

| Field | Type | Required | Units / Allowed Values / Semantics |
|---|---|---|---|
| `event_id` | `string` | Yes | Unique synthetic event identifier (`^[A-Za-z0-9._:-]{1,128}$`) |
| `schema_version` | `string` | No (default `"1.0.0"`) | Must be `"1.0.0"`; other values raise `UnsupportedInputError` (HTTP 400) |
| `timestamp` | `datetime` | Yes | **Timezone-aware ISO-8601 UTC** timestamp; naive timestamps rejected (`ContractValidationError` / HTTP 422) |
| `user_id` | `string` | Yes | Synthetic user identifier (`^[A-Za-z0-9._:-]{1,128}$`) |
| `session_id` | `string` | Yes | Synthetic session identifier |
| `event_type` | `string` | No (default `"login_attempt"`) | One of `login_attempt`, `session_activity`, `mfa_challenge`, `password_reset` |
| `auth_outcome` | `string` | No (default `"success"`) | One of `success`, `failure`, `challenge_passed`, `challenge_failed` |
| `auth_method` | `string` | No (default `"password"`) | One of `password`, `otp`, `sso`, `webauthn` |
| `account_tier` | `string` | No (default `"standard"`) | One of `standard`, `high_value` |
| `device_id` | `string \| null` | No | Synthetic device token; `null` counted as missing telemetry |
| `device_trust_score` | `float \| null` | No | Synthetic attestation score in `[0.0, 1.0]` |
| `ip_address` | `string \| null` | No | Valid IPv4/IPv6 syntax (synthetic RFC 5737 in fixtures) |
| `ip_reputation_risk` | `float \| null` | No | Network risk score in `[0.0, 1.0]` |
| `is_known_shared_network` | `bool` | No (default `false`) | Optional known corporate/campus NAT indicator |
| `geo_lat` | `float \| null` | No | Latitude in `[-90.0, 90.0]`; must pair with `geo_lon` |
| `geo_lon` | `float \| null` | No | Longitude in `[-180.0, 180.0]`; must pair with `geo_lat` |
| `country_code` | `string \| null` | No | ISO-2 uppercase code (`^[A-Z]{2}$`) |
| `sensitive_actions` | `list[string]` | No (default `[]`) | Subset of `change_mfa`, `change_email`, `change_password`, `add_payee`, `wire_transfer_setup`, `export_data`, `disable_alerts`, `api_key_create` |
| `session_duration_seconds` | `float` | No (default `0.0`) | Non-negative duration in seconds (`>= 0.0`) |
| `label` | `int \| null` | No | Offline training/evaluation only: `0` (benign) or `1` (ATO) |
| `label_available_at` | `datetime \| null` | No | Timezone-aware UTC timestamp when delayed label matured; must be `>= timestamp` and requires non-null `label` |

---

## 2. Feature Contract: `FeatureVector` (`feature_version = "1.0.0"`)

All 22 features are deterministic floats computed strictly from events with `history_ts < event.timestamp` within their configured rolling TTL windows:

1. `is_new_device_for_user` (`0` or `1`)
2. `device_age_hours_for_user` (hours, `>= 0.0`)
3. `user_prior_event_count_30d` (count in active TTL window)
4. `failed_attempts_1h` (count in prior 3600s)
5. `failed_attempts_24h` (count in prior 86400s)
6. `distinct_ips_24h` (count of distinct non-null IPs for user in prior 24h + current)
7. `distinct_users_on_ip_24h` (count of distinct users on current IP in prior 24h + current)
8. `is_shared_network_ip` (`0` or `1`; known flag or >= 4 distinct peacefully authenticating users on IP without burst failures)
9. `geo_velocity_kmh` (km/h via Haversine distance, capped at `15000.0`, minimum 1-minute denominator)
10. `is_impossible_travel` (`0` or `1`; `geo_velocity_kmh > 900.0`)
11. `ip_reputation_risk` (`[0.0, 1.0]`, imputed to `0.35` when missing)
12. `ip_reputation_missing` (`0` or `1`)
13. `device_trust_score` (`[0.0, 1.0]`, imputed to `0.50` when missing)
14. `device_trust_missing` (`0` or `1`)
15. `hour_of_day_deviation` (circular hour distance in `[0.0, 12.0]`)
16. `session_action_velocity_per_min` (`sensitive_action_count / max(duration_min, 0.5)`)
17. `sensitive_action_count` (count of sensitive actions in current event)
18. `auth_factor_downgrade` (`0` or `1` vs user's strongest prior successful factor)
19. `time_since_last_login_hours` (hours since last active non-expired event, or `-1.0`)
20. `profile_expired` (`0` or `1`; `1` when user has events older than TTL and zero active events)
21. `missing_feature_ratio` (fraction of 5 optional raw telemetry fields missing, in `[0.0, 1.0]`)
22. `credential_stuffing_burst` (`0` or `1`; `failed_attempts_1h >= 5` or non-shared IP failed burst `>= 5`)

---

## 3. Output Contract: `PolicyDecision`

| Field | Type | Description |
|---|---|---|
| `decision_id` | `string` | Deterministic ID `dec_<16-hex>` from SHA-256 of `(event_id, timestamp, model_version, feature_version, policy_version)` |
| `event_id`, `user_id`, `session_id` | `string` | Echoed canonical identifiers |
| `timestamp` | `datetime` | UTC ISO-8601 event timestamp |
| `schema_version`, `feature_version`, `model_version`, `policy_version` | `string` | Full lineage versions |
| `action` | `StepUpAction` | `ALLOW`, `CHALLENGE_OTP`, `CHALLENGE_WEBAUTHN`, or `BLOCK_AND_RESET` |
| `calibrated_risk_score` | `float` | Final risk probability in `[0.0, 1.0]` |
| `anomaly_score` | `float` | Normalized IsolationForest anomaly score in `[0.0, 1.0]` |
| `supervised_probability` | `float` | Blended supervised probability in `[0.0, 1.0]` |
| `thresholds_applied` | `dict[str, float]` | Effective adaptive thresholds (`challenge_otp`, `challenge_webauthn`, `block_and_reset`) |
| `threshold_adjustments` | `dict[str, float]` | Named deltas applied (`high_value_account`, `shared_network_dampening`, `drift_tightening`) |
| `fallback_used` | `bool` | `true` when deterministic fallback policy handled the decision |
| `fallback_reason` | `FallbackReason` | `none`, `model_unavailable`, `model_stale`, `store_unavailable`, `excessive_missing_features`, `critical_drift_circuit_breaker` |
| `drift_alert_active` | `bool` | Whether a drift alert is currently active on the engine |
| `evidence` | `AnalystEvidence` | Summary, top 5 ranked `EvidenceItem` attributions, `triggered_rules`, `missing_fields`, `profile_expired`, `recommended_playbook` |
| `features_snapshot` | `dict[str, float]` | Exact causal feature values used for the decision |

---

## 4. Error Taxonomy & Fixtures

| Error Class | HTTP Status | Trigger Examples |
|---|---|---|
| `ContractValidationError` | `422` | Naive timestamp, latitude/longitude out of bounds, partial geo pair, invalid IP syntax, `label_available_at < timestamp`, negative duration, unexpected extra fields, causal timestamp violation |
| `UnsupportedInputError` | `400` | Unknown `schema_version`, unknown `event_type`, unknown `auth_method`, unknown `sensitive_actions` |
| `ConfigurationError` | Startup fail | Missing config file, malformed YAML/JSON, non-monotonic thresholds, invalid fractions |
| `ModelArtifactError` | Startup fail | Missing model/manifest file, SHA-256 checksum mismatch, incompatible `feature_names` |

**Fixtures:**
- Valid reference oracle fixture: `data/fixtures/reference_oracle.json`
- Invalid, unsupported, and boundary fixtures: `data/fixtures/invalid_and_boundary_events.json`
