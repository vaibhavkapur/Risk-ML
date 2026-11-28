"""Point-in-time applicant feature, label, and scoring output contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from credit_risk.errors import ContractValidationError, PointInTimeLeakageError

SCHEMA_VERSION = "1.0.0"

ALLOWED_LOAN_TERMS: FrozenSet[int] = frozenset({12, 24, 36, 48, 60, 72})
ALLOWED_HOME_OWNERSHIP: FrozenSet[str] = frozenset({"RENT", "MORTGAGE", "OWN", "OTHER"})
ALLOWED_LOAN_PURPOSES: FrozenSet[str] = frozenset(
    {
        "DEBT_CONSOLIDATION",
        "HOME_IMPROVEMENT",
        "EDUCATION",
        "SMALL_BUSINESS",
        "MAJOR_PURCHASE",
        "OTHER",
    }
)
ALLOWED_AGE_BANDS: FrozenSet[str] = frozenset({"18_29", "30_44", "45_59", "60_PLUS"})
ALLOWED_REGION_TIERS: FrozenSet[str] = frozenset({"METRO", "SUBURBAN", "RURAL"})
ALLOWED_LABEL_STATUSES: FrozenSet[str] = frozenset({"MATURED", "CENSORED_IMMATURE"})

# Post-origination columns that MUST NEVER appear in point-in-time applicant features.
FORBIDDEN_POST_ORIGINATION_FIELDS: FrozenSet[str] = frozenset(
    {
        "days_past_due_30d",
        "days_past_due_90d",
        "post_origination_payment_count",
        "collections_status",
        "chargeoff_amount",
        "default_event_date",
        "recovery_amount",
        "first_payment_default",
        "months_since_last_payment",
        "post_funding_balance",
        "default_flag",
        "label_status",
    }
)

# Monitoring-only subgroup fields (strictly excluded from model feature matrices).
PROTECTED_SUBGROUP_FIELDS: Tuple[str, ...] = ("age_band", "region_tier")

ALLOWED_APPLICANT_KEYS: FrozenSet[str] = frozenset(
    {
        "application_id",
        "applicant_id",
        "schema_version",
        "application_timestamp",
        "feature_as_of_timestamp",
        "annual_income_usd",
        "loan_amount_usd",
        "loan_term_months",
        "debt_to_income_ratio",
        "credit_score",
        "credit_history_months",
        "revolving_utilization_ratio",
        "delinquencies_24m",
        "inquiries_6m",
        "employment_length_years",
        "home_ownership",
        "loan_purpose",
        "age_band",
        "region_tier",
    }
)


def parse_iso_utc(value: Any, field_name: str) -> datetime:
    """Parse an ISO-8601 date or UTC timestamp strictly into a timezone-aware datetime."""
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(
            f"Field '{field_name}' must be a non-empty ISO-8601 string.",
            details={"field": field_name, "value": value},
        )
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ContractValidationError(
            f"Field '{field_name}' is not a valid ISO-8601 timestamp: '{value}'.",
            details={"field": field_name, "value": value},
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _validate_optional_float(
    value: Any,
    field_name: str,
    min_val: float,
    max_val: float,
    allow_none: bool = True,
    strict_min: bool = False,
) -> Optional[float]:
    if value is None:
        if allow_none:
            return None
        raise ContractValidationError(
            f"Required numeric field '{field_name}' cannot be null.",
            details={"field": field_name},
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(
            f"Field '{field_name}' must be a number, got {type(value).__name__}.",
            details={"field": field_name, "value": value},
        )
    fval = float(value)
    if math.isnan(fval) or math.isinf(fval):
        raise ContractValidationError(
            f"Field '{field_name}' must be finite (no NaN/Inf).",
            details={"field": field_name, "value": str(value)},
        )
    if (strict_min and fval <= min_val) or (not strict_min and fval < min_val) or fval > max_val:
        op = ">" if strict_min else ">="
        raise ContractValidationError(
            f"Field '{field_name}'={fval} out of bounds ({op} {min_val} and <= {max_val}).",
            details={"field": field_name, "value": fval, "min": min_val, "max": max_val},
        )
    return fval


def _validate_optional_int(
    value: Any,
    field_name: str,
    min_val: int,
    max_val: int,
    allow_none: bool = True,
) -> Optional[int]:
    if value is None:
        if allow_none:
            return None
        raise ContractValidationError(
            f"Required integer field '{field_name}' cannot be null.",
            details={"field": field_name},
        )
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        else:
            raise ContractValidationError(
                f"Field '{field_name}' must be an integer, got {type(value).__name__}.",
                details={"field": field_name, "value": value},
            )
    if value < min_val or value > max_val:
        raise ContractValidationError(
            f"Field '{field_name}'={value} out of bounds ([{min_val}, {max_val}]).",
            details={"field": field_name, "value": value, "min": min_val, "max": max_val},
        )
    return value


def _validate_enum(value: Any, field_name: str, allowed: FrozenSet[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ContractValidationError(
            f"Field '{field_name}' has unsupported value '{value}'. Allowed: {sorted(allowed)}",
            details={"field": field_name, "value": value, "allowed": sorted(allowed)},
        )
    return value


@dataclass(frozen=True)
class ApplicantRecord:
    """Point-in-time applicant origination snapshot (contract v1.0.0)."""

    application_id: str
    applicant_id: str
    schema_version: str
    application_timestamp: str
    feature_as_of_timestamp: str
    annual_income_usd: Optional[float]
    loan_amount_usd: float
    loan_term_months: int
    debt_to_income_ratio: Optional[float]
    credit_score: Optional[int]
    credit_history_months: Optional[int]
    revolving_utilization_ratio: Optional[float]
    delinquencies_24m: Optional[int]
    inquiries_6m: Optional[int]
    employment_length_years: Optional[float]
    home_ownership: str
    loan_purpose: str
    age_band: str
    region_tier: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def application_dt(self) -> datetime:
        return parse_iso_utc(self.application_timestamp, "application_timestamp")


def validate_applicant_record(raw: Dict[str, Any]) -> ApplicantRecord:
    """Validate raw applicant payload against point-in-time schema and leakage rules."""
    if not isinstance(raw, dict):
        raise ContractValidationError(
            "Applicant payload must be a JSON object / dict.",
            details={"type": type(raw).__name__},
        )

    # 1. Leakage firewall: reject any post-origination or outcome fields immediately.
    leaked_fields = sorted(set(raw.keys()).intersection(FORBIDDEN_POST_ORIGINATION_FIELDS))
    if leaked_fields:
        raise PointInTimeLeakageError(
            f"Post-origination or outcome fields detected in applicant features: {leaked_fields}",
            details={
                "application_id": raw.get("application_id"),
                "leaked_fields": leaked_fields,
            },
        )

    # 2. Reject unknown unexpected fields to prevent silent schema drift.
    unknown_fields = sorted(set(raw.keys()) - ALLOWED_APPLICANT_KEYS)
    if unknown_fields:
        raise ContractValidationError(
            f"Unknown fields not defined in contract v{SCHEMA_VERSION}: {unknown_fields}",
            details={"unknown_fields": unknown_fields},
        )

    # 3. Validate identifiers & schema version.
    application_id = raw.get("application_id")
    applicant_id = raw.get("applicant_id")
    if not isinstance(application_id, str) or not application_id.strip():
        raise ContractValidationError("Field 'application_id' must be a non-empty string.")
    if not isinstance(applicant_id, str) or not applicant_id.strip():
        raise ContractValidationError("Field 'applicant_id' must be a non-empty string.")

    schema_version = raw.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        raise ContractValidationError(
            f"Unsupported schema_version '{schema_version}'; expected '{SCHEMA_VERSION}'.",
            details={"schema_version": schema_version, "expected": SCHEMA_VERSION},
        )

    # 4. Validate point-in-time timestamp invariant: feature_as_of_timestamp <= application_timestamp.
    app_dt = parse_iso_utc(raw.get("application_timestamp"), "application_timestamp")
    as_of_dt = parse_iso_utc(raw.get("feature_as_of_timestamp"), "feature_as_of_timestamp")
    if as_of_dt > app_dt:
        raise PointInTimeLeakageError(
            "Point-in-time violation: feature_as_of_timestamp is after application_timestamp.",
            details={
                "application_id": application_id,
                "application_timestamp": raw.get("application_timestamp"),
                "feature_as_of_timestamp": raw.get("feature_as_of_timestamp"),
            },
        )

    # 5. Validate numeric and categorical bounds.
    annual_income_usd = _validate_optional_float(
        raw.get("annual_income_usd"), "annual_income_usd", 0.0, 5_000_000.0, allow_none=True
    )
    loan_amount_usd = _validate_optional_float(
        raw.get("loan_amount_usd"),
        "loan_amount_usd",
        0.0,
        500_000.0,
        allow_none=False,
        strict_min=True,
    )
    assert loan_amount_usd is not None

    loan_term_months = _validate_optional_int(
        raw.get("loan_term_months"), "loan_term_months", 12, 72, allow_none=False
    )
    assert loan_term_months is not None
    if loan_term_months not in ALLOWED_LOAN_TERMS:
        raise ContractValidationError(
            f"Unsupported loan_term_months={loan_term_months}. Allowed: {sorted(ALLOWED_LOAN_TERMS)}",
            details={"field": "loan_term_months", "value": loan_term_months},
        )

    debt_to_income_ratio = _validate_optional_float(
        raw.get("debt_to_income_ratio"), "debt_to_income_ratio", 0.0, 2.5, allow_none=True
    )
    credit_score = _validate_optional_int(
        raw.get("credit_score"), "credit_score", 300, 850, allow_none=True
    )
    credit_history_months = _validate_optional_int(
        raw.get("credit_history_months"), "credit_history_months", 0, 720, allow_none=True
    )
    revolving_utilization_ratio = _validate_optional_float(
        raw.get("revolving_utilization_ratio"),
        "revolving_utilization_ratio",
        0.0,
        1.5,
        allow_none=True,
    )
    delinquencies_24m = _validate_optional_int(
        raw.get("delinquencies_24m"), "delinquencies_24m", 0, 24, allow_none=True
    )
    inquiries_6m = _validate_optional_int(
        raw.get("inquiries_6m"), "inquiries_6m", 0, 30, allow_none=True
    )
    employment_length_years = _validate_optional_float(
        raw.get("employment_length_years"), "employment_length_years", 0.0, 60.0, allow_none=True
    )

    home_ownership = _validate_enum(raw.get("home_ownership"), "home_ownership", ALLOWED_HOME_OWNERSHIP)
    loan_purpose = _validate_enum(raw.get("loan_purpose"), "loan_purpose", ALLOWED_LOAN_PURPOSES)
    age_band = _validate_enum(raw.get("age_band"), "age_band", ALLOWED_AGE_BANDS)
    region_tier = _validate_enum(raw.get("region_tier"), "region_tier", ALLOWED_REGION_TIERS)

    return ApplicantRecord(
        application_id=application_id.strip(),
        applicant_id=applicant_id.strip(),
        schema_version=schema_version,
        application_timestamp=app_dt.isoformat().replace("+00:00", "Z"),
        feature_as_of_timestamp=as_of_dt.isoformat().replace("+00:00", "Z"),
        annual_income_usd=annual_income_usd,
        loan_amount_usd=loan_amount_usd,
        loan_term_months=loan_term_months,
        debt_to_income_ratio=debt_to_income_ratio,
        credit_score=credit_score,
        credit_history_months=credit_history_months,
        revolving_utilization_ratio=revolving_utilization_ratio,
        delinquencies_24m=delinquencies_24m,
        inquiries_6m=inquiries_6m,
        employment_length_years=employment_length_years,
        home_ownership=home_ownership,
        loan_purpose=loan_purpose,
        age_band=age_band,
        region_tier=region_tier,
    )


@dataclass(frozen=True)
class LabelRecord:
    """Point-in-time default outcome contract with explicit label maturity."""

    application_id: str
    application_timestamp: str
    observation_window_days: int
    label_maturity_date: str
    as_of_cutoff_date: str
    default_flag: Optional[int]
    label_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_label_record(raw: Dict[str, Any], require_matured: bool = False) -> LabelRecord:
    """Validate label contract and enforce observation-window maturity."""
    if not isinstance(raw, dict):
        raise ContractValidationError("Label payload must be a dict.")

    application_id = raw.get("application_id")
    if not isinstance(application_id, str) or not application_id.strip():
        raise ContractValidationError("Label field 'application_id' must be a non-empty string.")

    app_dt = parse_iso_utc(raw.get("application_timestamp"), "application_timestamp")
    window_days = _validate_optional_int(
        raw.get("observation_window_days"), "observation_window_days", 1, 3650, allow_none=False
    )
    assert window_days is not None

    expected_maturity = app_dt + timedelta(days=window_days)
    maturity_dt = parse_iso_utc(raw.get("label_maturity_date"), "label_maturity_date")
    cutoff_dt = parse_iso_utc(raw.get("as_of_cutoff_date"), "as_of_cutoff_date")

    if maturity_dt < expected_maturity - timedelta(seconds=1):
        raise PointInTimeLeakageError(
            "Label maturity date is earlier than application_timestamp + observation_window_days.",
            details={
                "application_id": application_id,
                "label_maturity_date": raw.get("label_maturity_date"),
                "expected_min_maturity": expected_maturity.isoformat(),
            },
        )

    label_status = _validate_enum(raw.get("label_status"), "label_status", ALLOWED_LABEL_STATUSES)
    default_flag = raw.get("default_flag")
    if default_flag is not None:
        if isinstance(default_flag, bool) or default_flag not in (0, 1):
            raise ContractValidationError(
                f"Label 'default_flag' must be 0, 1, or null; got {default_flag!r}.",
                details={"application_id": application_id, "default_flag": default_flag},
            )

    is_matured_by_time = cutoff_dt >= maturity_dt
    if label_status == "MATURED" and not is_matured_by_time:
        raise PointInTimeLeakageError(
            "Immature label falsely marked as MATURED before observation window elapsed.",
            details={
                "application_id": application_id,
                "label_maturity_date": maturity_dt.isoformat(),
                "as_of_cutoff_date": cutoff_dt.isoformat(),
            },
        )
    if label_status == "MATURED" and default_flag is None:
        raise ContractValidationError(
            "MATURED label must provide a non-null default_flag (0 or 1).",
            details={"application_id": application_id},
        )
    if label_status == "CENSORED_IMMATURE" and require_matured:
        raise PointInTimeLeakageError(
            "Immature/right-censored label cannot be included when require_matured=True.",
            details={"application_id": application_id},
        )

    return LabelRecord(
        application_id=application_id.strip(),
        application_timestamp=app_dt.isoformat().replace("+00:00", "Z"),
        observation_window_days=window_days,
        label_maturity_date=maturity_dt.isoformat().replace("+00:00", "Z"),
        as_of_cutoff_date=cutoff_dt.isoformat().replace("+00:00", "Z"),
        default_flag=default_flag,
        label_status=label_status,
    )


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: Optional[float]
    contribution_log_odds: float
    direction: str  # "INCREASES_RISK" or "DECREASES_RISK"
    reason_code: str
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoringDecisionOutput:
    """Versioned decision-support output contract (manual review only)."""

    decision_id: str
    application_id: str
    schema_version: str
    feature_version: str
    model_id: str
    calibration_method: str
    policy_version: str
    raw_score: float
    calibrated_pd: float
    risk_band: str
    queue_priority: str
    review_sla_hours: int
    recommended_action: str
    automated_decision_allowed: bool
    manual_review_required: bool
    explanations: List[FeatureContribution] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    scored_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
