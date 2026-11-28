"""Deterministic synthetic loan dataset generator and fixture I/O utilities.

Strictly generates synthetic, non-personal educational loan data with explicit
point-in-time semantics, missingness, optional temporal distribution shift, and
label maturity windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from credit_risk.config import DataConfig
from credit_risk.contracts import (
    ApplicantRecord,
    LabelRecord,
    SCHEMA_VERSION,
    parse_iso_utc,
    validate_applicant_record,
    validate_label_record,
)


@dataclass(frozen=True)
class SyntheticDataset:
    applicants: List[ApplicantRecord]
    labels: List[LabelRecord]
    true_pd: List[float]
    metadata: Dict[str, Any]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def generate_synthetic_dataset(
    cfg: DataConfig,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate deterministic point-in-time synthetic loan applications and matured labels."""
    rng = np.random.default_rng(seed)

    start_dt = parse_iso_utc(cfg.start_date, "data.start_date")
    end_dt = parse_iso_utc(cfg.end_date, "data.end_date")
    shift_dt = parse_iso_utc(cfg.shift_start_date, "data.shift_start_date")
    cutoff_dt = parse_iso_utc(cfg.as_of_cutoff_date, "data.as_of_cutoff_date")

    total_seconds = max(1, int((end_dt - start_dt).total_seconds()))
    n = cfg.n_applicants

    # Generate sorted origination timestamps so temporal splits are clean and realistic.
    offsets = np.sort(rng.integers(0, total_seconds, size=n))
    app_dts = [start_dt + timedelta(seconds=int(s)) for s in offsets]

    home_choices = np.array(["RENT", "MORTGAGE", "OWN", "OTHER"])
    home_probs = np.array([0.42, 0.40, 0.14, 0.04])

    purpose_choices = np.array(
        [
            "DEBT_CONSOLIDATION",
            "HOME_IMPROVEMENT",
            "EDUCATION",
            "SMALL_BUSINESS",
            "MAJOR_PURCHASE",
            "OTHER",
        ]
    )
    purpose_probs = np.array([0.46, 0.16, 0.08, 0.12, 0.10, 0.08])

    term_choices = np.array([12, 24, 36, 48, 60, 72])
    term_probs = np.array([0.05, 0.10, 0.40, 0.15, 0.25, 0.05])

    age_choices = np.array(["18_29", "30_44", "45_59", "60_PLUS"])
    age_probs = np.array([0.22, 0.38, 0.27, 0.13])

    region_choices = np.array(["METRO", "SUBURBAN", "RURAL"])
    region_probs = np.array([0.45, 0.37, 0.18])

    applicants: List[ApplicantRecord] = []
    labels: List[LabelRecord] = []
    true_pds: List[float] = []

    matured_count = 0
    censored_count = 0
    default_count = 0

    for i in range(n):
        app_dt = app_dts[i]
        # Bureau snapshot taken 0 to 48 hours before application submission.
        lag_hours = int(rng.integers(0, 49))
        as_of_dt = app_dt - timedelta(hours=lag_hours)

        is_shifted = app_dt >= shift_dt
        shift = cfg.shift_magnitude if is_shifted else 0.0

        age_band = str(rng.choice(age_choices, p=age_probs))
        region_tier = str(rng.choice(region_choices, p=region_probs))
        home_ownership = str(rng.choice(home_choices, p=home_probs))
        loan_purpose = str(rng.choice(purpose_choices, p=purpose_probs))
        loan_term = int(rng.choice(term_choices, p=term_probs))

        # Base economic features with mild temporal distribution shift after shift_start_date.
        credit_score_raw = int(
            np.clip(np.round(rng.normal(685.0 - 22.0 * shift, 68.0)), 350, 845)
        )
        annual_income_raw = float(
            np.clip(np.round(np.exp(rng.normal(11.12 - 0.06 * shift, 0.48)), 2), 18_000.0, 450_000.0)
        )
        loan_amount = float(
            np.clip(
                np.round(annual_income_raw * rng.uniform(0.08, 0.42), 2),
                1_500.0,
                120_000.0,
            )
        )
        dti_raw = float(
            np.clip(np.round(rng.normal(0.27 + 0.09 * shift, 0.11), 4), 0.02, 1.20)
        )
        util_raw = float(
            np.clip(np.round(rng.beta(2.2 + 0.8 * shift, 3.0), 4), 0.0, 1.15)
        )
        history_months_raw = int(
            np.clip(np.round(rng.normal(135.0, 62.0)), 6, 480)
        )
        delinq_rate = 0.28 + 0.15 * shift + max(0.0, (660 - credit_score_raw) / 140.0) * 0.45
        delinq_raw = int(np.clip(rng.poisson(delinq_rate), 0, 12))
        inq_raw = int(np.clip(rng.poisson(1.1 + 0.4 * shift), 0, 14))
        emp_years_raw = float(np.clip(np.round(rng.exponential(5.5), 1), 0.0, 40.0))

        # True structural probability of default (depends ONLY on financial risk factors,
        # never on protected subgroup columns directly, plus mild nonlinear interactions).
        lti = loan_amount / max(annual_income_raw, 10_000.0)
        purpose_risk = {
            "SMALL_BUSINESS": 0.42,
            "DEBT_CONSOLIDATION": 0.15,
            "OTHER": 0.18,
            "MAJOR_PURCHASE": -0.05,
            "HOME_IMPROVEMENT": -0.15,
            "EDUCATION": -0.10,
        }[loan_purpose]
        home_risk = {
            "RENT": 0.16,
            "OTHER": 0.22,
            "MORTGAGE": -0.14,
            "OWN": -0.24,
        }[home_ownership]

        # Nonlinear risk drivers: compound distress when high DTI meets high utilization
        # and severe credit-score cliff below 620.
        compound_distress = 1.65 * max(0.0, dti_raw - 0.32) * max(0.0, util_raw - 0.55)
        subprime_cliff = 0.85 * max(0.0, (620.0 - credit_score_raw) / 100.0) ** 1.5

        logit = (
            -2.35
            - 0.0115 * (credit_score_raw - 680.0)
            + 2.35 * (dti_raw - 0.28)
            + 1.45 * (util_raw - 0.45)
            + 0.42 * delinq_raw
            + 0.14 * inq_raw
            + 1.35 * (lti - 0.22)
            + 0.012 * (loan_term - 36.0)
            - 0.0022 * (history_months_raw - 120.0)
            - 0.032 * (emp_years_raw - 5.0)
            + purpose_risk
            + home_risk
            + compound_distress
            + subprime_cliff
            + (0.12 * shift)  # mild macro stress during shifted window
        )
        pd_true = float(_sigmoid(np.array([logit]))[0])
        true_pds.append(pd_true)

        # Inject controlled missingness into nullable bureau / income fields.
        m_rate = cfg.missing_rate
        credit_score = None if rng.random() < m_rate * 0.7 else credit_score_raw
        annual_income = None if rng.random() < m_rate * 0.6 else annual_income_raw
        dti = None if rng.random() < m_rate else dti_raw
        util = None if rng.random() < m_rate * 1.1 else util_raw
        history_months = None if rng.random() < m_rate else history_months_raw
        delinq = None if rng.random() < m_rate * 0.8 else delinq_raw
        inq = None if rng.random() < m_rate * 0.8 else inq_raw
        emp_years = None if rng.random() < m_rate * 1.2 else emp_years_raw

        app_id = f"APP-{app_dt.strftime('%Y%m')}-{i + 1:05d}"
        applicant_id = f"USR-{100000 + i:06d}"

        record_dict = {
            "application_id": app_id,
            "applicant_id": applicant_id,
            "schema_version": SCHEMA_VERSION,
            "application_timestamp": app_dt.isoformat().replace("+00:00", "Z"),
            "feature_as_of_timestamp": as_of_dt.isoformat().replace("+00:00", "Z"),
            "annual_income_usd": annual_income,
            "loan_amount_usd": loan_amount,
            "loan_term_months": loan_term,
            "debt_to_income_ratio": dti,
            "credit_score": credit_score,
            "credit_history_months": history_months,
            "revolving_utilization_ratio": util,
            "delinquencies_24m": delinq,
            "inquiries_6m": inq,
            "employment_length_years": emp_years,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "age_band": age_band,
            "region_tier": region_tier,
        }
        applicants.append(validate_applicant_record(record_dict))

        maturity_dt = app_dt + timedelta(days=cfg.observation_window_days)
        if cutoff_dt >= maturity_dt:
            default_flag = int(rng.random() < pd_true)
            label_status = "MATURED"
            matured_count += 1
            default_count += default_flag
        else:
            default_flag = None
            label_status = "CENSORED_IMMATURE"
            censored_count += 1

        label_dict = {
            "application_id": app_id,
            "application_timestamp": app_dt.isoformat().replace("+00:00", "Z"),
            "observation_window_days": cfg.observation_window_days,
            "label_maturity_date": maturity_dt.isoformat().replace("+00:00", "Z"),
            "as_of_cutoff_date": cutoff_dt.isoformat().replace("+00:00", "Z"),
            "default_flag": default_flag,
            "label_status": label_status,
        }
        labels.append(validate_label_record(label_dict, require_matured=False))

    metadata = {
        "source": "synthetic_generator_v1",
        "seed": seed,
        "n_applicants": n,
        "matured_labels": matured_count,
        "censored_immature_labels": censored_count,
        "observed_default_rate_matured": (
            round(default_count / matured_count, 6) if matured_count > 0 else None
        ),
        "mean_true_pd": round(float(np.mean(true_pds)), 6),
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "shift_start_date": cfg.shift_start_date,
        "shift_magnitude": cfg.shift_magnitude,
        "observation_window_days": cfg.observation_window_days,
        "as_of_cutoff_date": cfg.as_of_cutoff_date,
        "contains_real_customer_data": False,
    }
    return SyntheticDataset(
        applicants=applicants,
        labels=labels,
        true_pd=true_pds,
        metadata=metadata,
    )


def load_json_fixture(path: Union[str, Path]) -> Any:
    """Load a JSON fixture file deterministically."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json_file(path: Union[str, Path], payload: Any) -> Path:
    """Persist structured JSON output with deterministic formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


def temporal_split_dataset(
    applicants: List[ApplicantRecord],
    labels: List[LabelRecord],
    train_end_date: str,
    calibration_end_date: str,
    test_end_date: str,
    require_label_maturity: bool = True,
) -> Dict[str, Tuple[List[ApplicantRecord], List[LabelRecord]]]:
    """Partition dataset strictly by application_timestamp into train, calibration, and test.

    Enforces label maturity and verifies 1-to-1 application_id alignment.
    """
    label_by_id = {lbl.application_id: lbl for lbl in labels}
    train_end = parse_iso_utc(train_end_date, "train_end_date")
    cal_end = parse_iso_utc(calibration_end_date, "calibration_end_date")
    test_end = parse_iso_utc(test_end_date, "test_end_date")

    # End-of-day inclusive semantics for date-only boundaries (YYYY-MM-DD).
    if len(train_end_date.strip()) == 10:
        train_end = train_end + timedelta(days=1) - timedelta(microseconds=1)
    if len(calibration_end_date.strip()) == 10:
        cal_end = cal_end + timedelta(days=1) - timedelta(microseconds=1)
    if len(test_end_date.strip()) == 10:
        test_end = test_end + timedelta(days=1) - timedelta(microseconds=1)

    splits: Dict[str, Tuple[List[ApplicantRecord], List[LabelRecord]]] = {
        "train": ([], []),
        "calibration": ([], []),
        "test": ([], []),
        "censored_excluded": ([], []),
    }

    for rec in applicants:
        lbl = label_by_id.get(rec.application_id)
        if lbl is None:
            continue
        if lbl.label_status != "MATURED":
            if require_label_maturity:
                splits["censored_excluded"][0].append(rec)
                splits["censored_excluded"][1].append(lbl)
                continue
        app_dt = rec.application_dt
        if app_dt <= train_end:
            splits["train"][0].append(rec)
            splits["train"][1].append(lbl)
        elif app_dt <= cal_end:
            splits["calibration"][0].append(rec)
            splits["calibration"][1].append(lbl)
        elif app_dt <= test_end:
            splits["test"][0].append(rec)
            splits["test"][1].append(lbl)

    return splits
