"""Versioned feature engineering, missing-feature handling, and temporal splits."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from payment_auth_engine import FEATURE_SCHEMA_VERSION
from payment_auth_engine.schemas import AuthorizationAttempt, HistoricalOutcome, PolicyAction

HIGH_RISK_MCCS = frozenset({"5967", "6051", "7995", "4829", "5816"})
CARD_NETWORKS = ("visa", "mastercard", "amex", "discover")
PROCESSORS = ("proc_a", "proc_b", "proc_c")
OPTIONAL_FEATURE_FIELDS = (
    "risk_score",
    "velocity_1h",
    "velocity_24h",
    "account_age_days",
    "bin_approval_rate_30d",
)


@dataclass
class ImputationStats:
    """Training-derived imputation values (fit only on training split to avoid leakage)."""

    risk_score: float = 0.25
    velocity_1h: float = 1.0
    velocity_24h: float = 3.0
    account_age_days: float = 180.0
    bin_approval_rate_30d: float = 0.88


@dataclass
class ExtractedFeatures:
    """Feature matrix plus metadata for explanations and diagnostics."""

    matrix: np.ndarray
    feature_names: List[str]
    missing_fields_per_row: List[List[str]]
    schema_version: str = FEATURE_SCHEMA_VERSION


@dataclass
class FeatureExtractor:
    """Deterministic versioned feature extractor with explicit missingness indicators."""

    schema_version: str = FEATURE_SCHEMA_VERSION
    imputation: ImputationStats = field(default_factory=ImputationStats)
    is_fitted: bool = False

    @staticmethod
    def base_feature_names() -> List[str]:
        names = [
            "log_amount",
            "is_cross_border",
            "is_ecommerce",
            "is_recurring",
            "has_3ds",
            "has_network_token",
            "attempt_number",
            "high_risk_mcc",
            "risk_score_imputed",
            "velocity_1h_log",
            "velocity_24h_log",
            "account_age_log",
            "bin_approval_rate_imputed",
            "risk_score_missing",
            "velocity_1h_missing",
            "velocity_24h_missing",
            "account_age_missing",
            "bin_approval_rate_missing",
            "missing_feature_ratio",
        ]
        names.extend(f"network_{n}" for n in CARD_NETWORKS)
        names.extend(f"processor_{p}" for p in PROCESSORS)
        return names

    @staticmethod
    def action_feature_names() -> List[str]:
        return [
            "action_approve_primary",
            "action_route_secondary",
            "action_step_up_3ds",
            "sec_x_cross_border",
            "sec_x_low_bin",
            "stepup_x_risk",
        ]

    def full_feature_names(self) -> List[str]:
        return self.base_feature_names() + self.action_feature_names()

    def fit(self, attempts: Sequence[AuthorizationAttempt]) -> "FeatureExtractor":
        """Compute deterministic median imputation statistics from training attempts only."""
        if not attempts:
            raise ValueError("Cannot fit FeatureExtractor on empty attempt sequence")

        def _median_or_default(values: List[float], default: float) -> float:
            if not values:
                return default
            return float(np.median(np.asarray(values, dtype=np.float64)))

        self.imputation = ImputationStats(
            risk_score=_median_or_default(
                [a.risk_score for a in attempts if a.risk_score is not None], 0.25
            ),
            velocity_1h=_median_or_default(
                [float(a.velocity_1h) for a in attempts if a.velocity_1h is not None], 1.0
            ),
            velocity_24h=_median_or_default(
                [float(a.velocity_24h) for a in attempts if a.velocity_24h is not None], 3.0
            ),
            account_age_days=_median_or_default(
                [float(a.account_age_days) for a in attempts if a.account_age_days is not None], 180.0
            ),
            bin_approval_rate_30d=_median_or_default(
                [a.bin_approval_rate_30d for a in attempts if a.bin_approval_rate_30d is not None], 0.88
            ),
        )
        self.is_fitted = True
        return self

    def transform(
        self,
        attempts: Sequence[AuthorizationAttempt],
        actions: Optional[Sequence[PolicyAction]] = None,
    ) -> ExtractedFeatures:
        """Transform authorization attempts (and optional actions) into a numeric feature matrix."""
        if actions is not None and len(actions) != len(attempts):
            raise ValueError(f"actions length ({len(actions)}) must match attempts length ({len(attempts)})")

        base_names = self.base_feature_names()
        names = base_names + (self.action_feature_names() if actions is not None else [])
        rows: List[List[float]] = []
        missing_per_row: List[List[str]] = []

        for idx, attempt in enumerate(attempts):
            missing: List[str] = []

            if attempt.risk_score is None:
                risk_val = self.imputation.risk_score
                risk_miss = 1.0
                missing.append("risk_score")
            else:
                risk_val = attempt.risk_score
                risk_miss = 0.0

            if attempt.velocity_1h is None:
                v1_val = self.imputation.velocity_1h
                v1_miss = 1.0
                missing.append("velocity_1h")
            else:
                v1_val = float(attempt.velocity_1h)
                v1_miss = 0.0

            if attempt.velocity_24h is None:
                v24_val = self.imputation.velocity_24h
                v24_miss = 1.0
                missing.append("velocity_24h")
            else:
                v24_val = float(attempt.velocity_24h)
                v24_miss = 0.0

            if attempt.account_age_days is None:
                age_val = self.imputation.account_age_days
                age_miss = 1.0
                missing.append("account_age_days")
            else:
                age_val = float(attempt.account_age_days)
                age_miss = 0.0

            if attempt.bin_approval_rate_30d is None:
                bin_val = self.imputation.bin_approval_rate_30d
                bin_miss = 1.0
                missing.append("bin_approval_rate_30d")
            else:
                bin_val = attempt.bin_approval_rate_30d
                bin_miss = 0.0

            missing_ratio = len(missing) / float(len(OPTIONAL_FEATURE_FIELDS))
            row = [
                math.log1p(attempt.amount_cents / 100.0),
                1.0 if attempt.issuer_country != attempt.merchant_country else 0.0,
                1.0 if attempt.entry_mode == "ecommerce" else 0.0,
                1.0 if attempt.is_recurring else 0.0,
                1.0 if attempt.has_3ds else 0.0,
                1.0 if attempt.has_network_token else 0.0,
                float(attempt.attempt_number),
                1.0 if attempt.merchant_category_code in HIGH_RISK_MCCS else 0.0,
                risk_val,
                math.log1p(v1_val),
                math.log1p(v24_val),
                math.log1p(age_val),
                bin_val,
                risk_miss,
                v1_miss,
                v24_miss,
                age_miss,
                bin_miss,
                missing_ratio,
            ]
            row.extend(1.0 if attempt.card_network == n else 0.0 for n in CARD_NETWORKS)
            row.extend(1.0 if attempt.processor_id == p else 0.0 for p in PROCESSORS)

            if actions is not None:
                act = actions[idx]
                is_sec = 1.0 if act == PolicyAction.ROUTE_SECONDARY else 0.0
                is_step = 1.0 if act == PolicyAction.STEP_UP_3DS else 0.0
                is_xb = 1.0 if attempt.issuer_country != attempt.merchant_country else 0.0
                is_low_bin = 1.0 if bin_val < 0.85 else 0.0
                row.extend(
                    [
                        1.0 if act == PolicyAction.APPROVE_PRIMARY else 0.0,
                        is_sec,
                        is_step,
                        is_sec * is_xb,
                        is_sec * is_low_bin,
                        is_step * risk_val,
                    ]
                )

            rows.append(row)
            missing_per_row.append(missing)

        matrix = np.asarray(rows, dtype=np.float64) if rows else np.empty((0, len(names)), dtype=np.float64)
        return ExtractedFeatures(
            matrix=matrix,
            feature_names=names,
            missing_fields_per_row=missing_per_row,
            schema_version=self.schema_version,
        )


def temporal_train_cal_test_split(
    timestamps: Sequence[datetime],
    train_fraction: float,
    calibration_fraction: float,
    test_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strictly chronological split indices with non-overlapping temporal boundaries."""
    n = len(timestamps)
    if n < 10:
        raise ValueError(f"Need at least 10 samples for temporal split, got {n}")
    if abs(train_fraction + calibration_fraction + test_fraction - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    order = np.argsort(np.asarray([ts.timestamp() for ts in timestamps], dtype=np.float64), kind="mergesort")
    n_train = max(1, int(round(n * train_fraction)))
    n_cal = max(1, int(round(n * calibration_fraction)))
    if n_train + n_cal >= n:
        n_cal = max(1, n - n_train - 1)

    train_idx = order[:n_train]
    cal_idx = order[n_train : n_train + n_cal]
    test_idx = order[n_train + n_cal :]

    # Verify strict temporal ordering invariant
    assert timestamps[int(train_idx[-1])] <= timestamps[int(cal_idx[0])], "Temporal leakage between train and cal"
    assert timestamps[int(cal_idx[-1])] <= timestamps[int(test_idx[0])], "Temporal leakage between cal and test"

    return train_idx, cal_idx, test_idx


def filter_matured_label_indices(
    outcomes: Sequence[HistoricalOutcome],
    candidate_indices: Sequence[int],
    as_of_utc: datetime,
    min_maturity_days: int,
) -> np.ndarray:
    """Return subset of candidate indices whose fraud/loss labels have matured by as_of_utc."""
    cutoff = as_of_utc - timedelta(days=min_maturity_days)
    matured: List[int] = []
    for idx in candidate_indices:
        out = outcomes[int(idx)]
        if (
            out.authorized
            and out.fraud_confirmed is not None
            and out.label_maturity_days >= min_maturity_days
            and out.attempt_timestamp_utc <= cutoff
        ):
            matured.append(int(idx))
    return np.asarray(matured, dtype=np.int64)
