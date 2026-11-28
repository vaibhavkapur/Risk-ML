"""Versioned point-in-time feature extraction pipeline.

Strictly separates feature transformation from I/O, enforces leakage controls,
excludes protected subgroup attributes from model inputs, fits imputation and
scaling parameters exclusively on the training split, and emits explicit
missingness indicators.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from credit_risk.contracts import (
    ALLOWED_HOME_OWNERSHIP,
    ALLOWED_LOAN_PURPOSES,
    ApplicantRecord,
    PROTECTED_SUBGROUP_FIELDS,
    validate_applicant_record,
)
from credit_risk.errors import ContractValidationError

FEATURE_VERSION = "1.0.0"

RAW_NUMERIC_FIELDS: Tuple[str, ...] = (
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
)

NULLABLE_NUMERIC_FIELDS: Tuple[str, ...] = (
    "annual_income_usd",
    "debt_to_income_ratio",
    "credit_score",
    "credit_history_months",
    "revolving_utilization_ratio",
    "delinquencies_24m",
    "inquiries_6m",
    "employment_length_years",
)

DERIVED_NUMERIC_FIELDS: Tuple[str, ...] = (
    "loan_to_income_ratio",
    "estimated_payment_to_income",
    "log_annual_income",
    "log_loan_amount",
)

HOME_OWNERSHIP_CATEGORIES: Tuple[str, ...] = tuple(sorted(ALLOWED_HOME_OWNERSHIP))
LOAN_PURPOSE_CATEGORIES: Tuple[str, ...] = tuple(sorted(ALLOWED_LOAN_PURPOSES))


class FeaturePipeline:
    """Deterministic point-in-time feature transformer with explicit missingness handling."""

    def __init__(self, feature_version: str = FEATURE_VERSION) -> None:
        self.feature_version = feature_version
        self.medians_: Dict[str, float] = {}
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = self._build_feature_names()
        self.is_fitted_: bool = False

        # Sanity invariant: protected subgroup columns never appear in model features.
        for protected in PROTECTED_SUBGROUP_FIELDS:
            assert protected not in self.feature_names_, f"Protected attribute '{protected}' leaked!"

    @staticmethod
    def _build_feature_names() -> List[str]:
        names: List[str] = []
        names.extend(RAW_NUMERIC_FIELDS)
        names.extend(DERIVED_NUMERIC_FIELDS)
        names.extend(f"missing_{col}" for col in NULLABLE_NUMERIC_FIELDS)
        names.extend(f"home_ownership_{cat}" for cat in HOME_OWNERSHIP_CATEGORIES)
        names.extend(f"loan_purpose_{cat}" for cat in LOAN_PURPOSE_CATEGORIES)
        return names

    def _coerce_records(
        self, records: Sequence[ApplicantRecord | Dict[str, Any]]
    ) -> List[ApplicantRecord]:
        validated: List[ApplicantRecord] = []
        for rec in records:
            if isinstance(rec, ApplicantRecord):
                validated.append(rec)
            else:
                validated.append(validate_applicant_record(rec))
        return validated

    def fit(self, records: Sequence[ApplicantRecord | Dict[str, Any]]) -> "FeaturePipeline":
        """Fit training-split medians and standardization statistics."""
        validated = self._coerce_records(records)
        if not validated:
            raise ContractValidationError("Cannot fit FeaturePipeline on an empty record sequence.")

        medians: Dict[str, float] = {}
        for col in RAW_NUMERIC_FIELDS:
            vals = [getattr(r, col) for r in validated if getattr(r, col) is not None]
            if not vals:
                medians[col] = 0.0
            else:
                medians[col] = float(np.median(np.asarray(vals, dtype=np.float64)))
        self.medians_ = medians

        raw_matrix = self._extract_unscaled_matrix(validated)
        self.means_ = np.mean(raw_matrix, axis=0)
        stds = np.std(raw_matrix, axis=0)
        # Prevent divide-by-zero for constant indicators.
        self.stds_ = np.where(stds < 1e-8, 1.0, stds)
        self.is_fitted_ = True
        return self

    def _extract_unscaled_matrix(self, validated: Sequence[ApplicantRecord]) -> np.ndarray:
        rows: List[List[float]] = []
        for rec in validated:
            raw_vals: Dict[str, float] = {}
            missing_flags: List[float] = []

            for col in RAW_NUMERIC_FIELDS:
                val = getattr(rec, col)
                is_missing = val is None
                if col in NULLABLE_NUMERIC_FIELDS:
                    missing_flags.append(1.0 if is_missing else 0.0)
                raw_vals[col] = self.medians_[col] if is_missing else float(val)

            income_safe = max(raw_vals["annual_income_usd"], 1_000.0)
            loan_amt = raw_vals["loan_amount_usd"]
            term_m = max(raw_vals["loan_term_months"], 1.0)

            loan_to_income = loan_amt / income_safe
            # Approximate monthly payment ratio (simple amortized proxy at 10% APR / 12)
            monthly_rate = 0.10 / 12.0
            factor = (monthly_rate * (1.0 + monthly_rate) ** term_m) / (
                ((1.0 + monthly_rate) ** term_m) - 1.0
            )
            est_monthly_pmt = loan_amt * factor
            est_pti = est_monthly_pmt / (income_safe / 12.0)

            derived = [
                loan_to_income,
                est_pti,
                math.log1p(max(raw_vals["annual_income_usd"], 0.0)),
                math.log1p(max(loan_amt, 0.0)),
            ]

            home_ohe = [1.0 if rec.home_ownership == cat else 0.0 for cat in HOME_OWNERSHIP_CATEGORIES]
            purpose_ohe = [1.0 if rec.loan_purpose == cat else 0.0 for cat in LOAN_PURPOSE_CATEGORIES]

            row = [raw_vals[c] for c in RAW_NUMERIC_FIELDS] + derived + missing_flags + home_ohe + purpose_ohe
            rows.append(row)

        return np.asarray(rows, dtype=np.float64)

    def transform(
        self,
        records: Sequence[ApplicantRecord | Dict[str, Any]],
        standardized: bool = True,
    ) -> np.ndarray:
        """Transform point-in-time applicant records into a deterministic numeric matrix."""
        if not self.is_fitted_ or self.means_ is None or self.stds_ is None:
            raise RuntimeError("FeaturePipeline must be fitted before transform().")
        validated = self._coerce_records(records)
        if not validated:
            return np.empty((0, len(self.feature_names_)), dtype=np.float64)
        matrix = self._extract_unscaled_matrix(validated)
        if standardized:
            return (matrix - self.means_) / self.stds_
        return matrix

    def fit_transform(
        self,
        records: Sequence[ApplicantRecord | Dict[str, Any]],
        standardized: bool = True,
    ) -> np.ndarray:
        self.fit(records)
        return self.transform(records, standardized=standardized)

    def missing_fields_for_record(self, rec: ApplicantRecord) -> List[str]:
        return [col for col in NULLABLE_NUMERIC_FIELDS if getattr(rec, col) is None]

    def to_state_dict(self) -> Dict[str, Any]:
        if not self.is_fitted_ or self.means_ is None or self.stds_ is None:
            raise RuntimeError("Cannot serialize unfitted FeaturePipeline.")
        return {
            "feature_version": self.feature_version,
            "feature_names": list(self.feature_names_),
            "medians": dict(self.medians_),
            "means": self.means_.tolist(),
            "stds": self.stds_.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "FeaturePipeline":
        pipe = cls(feature_version=str(state["feature_version"]))
        pipe.feature_names_ = list(state["feature_names"])
        pipe.medians_ = {str(k): float(v) for k, v in state["medians"].items()}
        pipe.means_ = np.asarray(state["means"], dtype=np.float64)
        pipe.stds_ = np.asarray(state["stds"], dtype=np.float64)
        pipe.is_fitted_ = True
        return pipe
