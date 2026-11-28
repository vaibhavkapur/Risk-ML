# Data and Scoring Contracts (`v1.0.0`)

All schemas are enforced programmatically in [`src/credit_risk/contracts.py`](../src/credit_risk/contracts.py) and tested against deterministic fixtures in [`data/fixtures/`](../data/fixtures/).

## 1. Point-in-Time Applicant Input Contract (`ApplicantRecord`)

Represents the origination-time applicant snapshot available **at or before** `application_timestamp`.

| Field | Type | Unit / Domain | Nullable | Bounds / Allowed Values |
|---|---|---|---|---|
| `application_id` | `string` | Unique loan application ID | No | Non-empty string |
| `applicant_id` | `string` | Synthetic applicant ID | No | Non-empty string |
| `schema_version` | `string` | Semver | No | `"1.0.0"` |
| `application_timestamp` | `string` | ISO-8601 UTC timestamp | No | Valid UTC datetime |
| `feature_as_of_timestamp` | `string` | ISO-8601 UTC timestamp | No | Must satisfy `feature_as_of_timestamp <= application_timestamp` |
| `annual_income_usd` | `float` | USD / year | Yes | `[0.0, 5_000_000.0]` |
| `loan_amount_usd` | `float` | USD | No | `(0.0, 500_000.0]` |
| `loan_term_months` | `int` | Months | No | `{12, 24, 36, 48, 60, 72}` |
| `debt_to_income_ratio` | `float` | Ratio (monthly debt / monthly income) | Yes | `[0.0, 2.5]` |
| `credit_score` | `int` | Bureau score points | Yes | `[300, 850]` |
| `credit_history_months` | `int` | Months | Yes | `[0, 720]` |
| `revolving_utilization_ratio` | `float` | Ratio (balance / credit limit) | Yes | `[0.0, 1.5]` |
| `delinquencies_24m` | `int` | Event count (prior 24 months) | Yes | `[0, 24]` |
| `inquiries_6m` | `int` | Event count (prior 6 months) | Yes | `[0, 30]` |
| `employment_length_years` | `float` | Years | Yes | `[0.0, 60.0]` |
| `home_ownership` | `string` | Categorical enum | No | `RENT`, `MORTGAGE`, `OWN`, `OTHER` |
| `loan_purpose` | `string` | Categorical enum | No | `DEBT_CONSOLIDATION`, `HOME_IMPROVEMENT`, `EDUCATION`, `SMALL_BUSINESS`, `MAJOR_PURCHASE`, `OTHER` |
| `age_band` | `string` | Monitoring-only subgroup | No | `18_29`, `30_44`, `45_59`, `60_PLUS` (**excluded from model features**) |
| `region_tier` | `string` | Monitoring-only subgroup | No | `METRO`, `SUBURBAN`, `RURAL` (**excluded from model features**) |

### Leakage Firewall Rules

1. **Forbidden post-origination fields:** Presence of any key in `FORBIDDEN_POST_ORIGINATION_FIELDS` (`days_past_due_30d`, `days_past_due_90d`, `post_origination_payment_count`, `collections_status`, `chargeoff_amount`, `default_event_date`, `recovery_amount`, `first_payment_default`, `months_since_last_payment`, `post_funding_balance`, `default_flag`, `label_status`) raises `PointInTimeLeakageError` (`POINT_IN_TIME_LEAKAGE_ERROR`).
2. **Timestamp causality:** `feature_as_of_timestamp > application_timestamp` raises `PointInTimeLeakageError`.
3. **Strict schema closure:** Any unrecognized field raises `ContractValidationError` (`CONTRACT_VALIDATION_ERROR`) to prevent silent feature drift.

## 2. Point-in-Time Label Contract (`LabelRecord`)

| Field | Type | Unit / Domain | Nullable | Invariants |
|---|---|---|---|---|
| `application_id` | `string` | Foreign key to `ApplicantRecord` | No | Non-empty string |
| `application_timestamp` | `string` | ISO-8601 UTC timestamp | No | Matches origination time |
| `observation_window_days` | `int` | Days | No | `[1, 3650]` (default `365`) |
| `label_maturity_date` | `string` | ISO-8601 UTC timestamp | No | Must satisfy `>= application_timestamp + observation_window_days` |
| `as_of_cutoff_date` | `string` | ISO-8601 UTC timestamp | No | Snapshot extraction date |
| `default_flag` | `int` | Binary (`0`=non-default, `1`=default) | Yes | Required non-null when `label_status == "MATURED"` |
| `label_status` | `string` | Enum | No | `MATURED` or `CENSORED_IMMATURE` |

### Label Maturity Invariants

- If `as_of_cutoff_date < label_maturity_date`, marking `label_status == "MATURED"` raises `PointInTimeLeakageError`.
- When `require_matured=True`, any `CENSORED_IMMATURE` record raises `PointInTimeLeakageError` during validation and is automatically excluded from `train`/`calibration`/`test` splits during `temporal_split_dataset()`.

## 3. Scoring Output Contract (`ScoringDecisionOutput`)

Every scoring response is strictly decision-support for human underwriting review:

| Field | Type | Semantics |
|---|---|---|
| `decision_id` | `string` | Deterministic SHA-256 prefix (`DEC-<16 HEX>`) derived from canonical input + version metadata |
| `application_id` | `string` | Echoed applicant identifier |
| `schema_version` | `string` | Input/output contract version (`"1.0.0"`) |
| `feature_version` | `string` | Feature pipeline version (`"1.0.0"`) |
| `model_id` | `string` | Trained estimator identifier (e.g. `gbdt_baseline_v1`) |
| `calibration_method` | `string` | `isotonic` or `sigmoid` |
| `policy_version` | `string` | Manual review policy version (`"1.0.0"`) |
| `raw_score` | `float` | Uncalibrated estimator probability in `(0, 1)` |
| `calibrated_pd` | `float` | Monotone calibrated probability of default in `(0, 1)` |
| `risk_band` | `string` | `LOW_RISK_REVIEW`, `MODERATE_RISK_REVIEW`, `HIGH_RISK_REVIEW`, or `VERY_HIGH_RISK_ESCALATION` |
| `queue_priority` | `string` | `standard`, `elevated`, `high`, or `urgent` |
| `review_sla_hours` | `int` | Target human review SLA in hours (`48`, `24`, `12`, `4`) |
| `recommended_action` | `string` | Human underwriting queue routing action |
| `automated_decision_allowed` | `bool` | **Always `false`** |
| `manual_review_required` | `bool` | **Always `true`** |
| `explanations` | `list[FeatureContribution]` | Top-K local log-odds attributions with standardized reason codes (`RC01`–`RC15`) |
| `warnings` | `list[string]` | Missing-feature imputation notices and operational caveats |
| `scored_at_utc` | `string` | ISO-8601 UTC scoring timestamp |

## 4. Error Taxonomy

| Exception Class | `error_code` | Trigger |
|---|---|---|
| `ConfigurationError` | `CONFIGURATION_ERROR` | Missing YAML file, malformed YAML, invalid split ordering, or non-monotone policy thresholds |
| `ContractValidationError` | `CONTRACT_VALIDATION_ERROR` | Out-of-range numeric value, invalid enum, NaN/Inf, unknown column, or missing required field |
| `PointInTimeLeakageError` | `POINT_IN_TIME_LEAKAGE_ERROR` | Post-origination field present, `feature_as_of_timestamp > application_timestamp`, or premature matured label |
| `InsufficientSubgroupSampleError` | `INSUFFICIENT_SUBGROUP_SAMPLE_ERROR` | Subgroup size `< min_subgroup_sample_size` in strict evaluation mode |
| `UnsupportedInputError` | `UNSUPPORTED_INPUT_ERROR` | Unsupported model backend/calibrator, missing optional `xgboost`, automated lending enabled, or invalid review disposition |
| `CalibrationMonotonicityError` | `CALIBRATION_MONOTONICITY_ERROR` | Fitted calibrator violates monotonicity beyond configured tolerance |

## 5. Reference Fixtures (`data/fixtures/`)

- `valid_applicants.json`: Valid applicant, multi-missing-field applicant, minimum boundary (`credit_score=300`, `loan_amount_usd=1.0`), and maximum boundary (`credit_score=850`, `loan_amount_usd=500000.0`).
- `invalid_applicants.json`: Out-of-range credit score, zero loan amount, unsupported loan term, and unknown extra field.
- `leakage_applicants.json`: `days_past_due_30d`, future `feature_as_of_timestamp`, and `collections_status`/`chargeoff_amount`.
- `label_fixtures.json`: Matured labels, valid censored immature label, and invalid premature matured label.
- `reference_oracle.json`: Hand-calculated logistic PD, Brier score, AUROC, PR-AUC, PSI, and risk-band routing expectations.
