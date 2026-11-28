"""Contract validation, point-in-time leakage prevention, and label maturity tests."""

from pathlib import Path
import pytest

from credit_risk.contracts import (
    PROTECTED_SUBGROUP_FIELDS,
    validate_applicant_record,
    validate_label_record,
)
from credit_risk.data import load_json_fixture, temporal_split_dataset
from credit_risk.errors import ContractValidationError, PointInTimeLeakageError
from credit_risk.features import FeaturePipeline

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def test_valid_and_boundary_applicant_fixtures_pass():
    records = load_json_fixture(FIXTURES_DIR / "valid_applicants.json")
    assert len(records) == 4
    validated = [validate_applicant_record(r) for r in records]
    assert validated[0].application_id == "FIX-VALID-001"
    assert validated[1].credit_score is None
    assert validated[2].credit_score == 300
    assert validated[3].credit_score == 850


def test_invalid_applicant_fixtures_raise_contract_validation_error():
    cases = load_json_fixture(FIXTURES_DIR / "invalid_applicants.json")
    for case in cases:
        with pytest.raises(ContractValidationError) as exc_info:
            validate_applicant_record(case["payload"])
        assert exc_info.value.error_code == case["expected_error"]


def test_post_origination_and_future_timestamp_leakage_fixtures_blocked():
    cases = load_json_fixture(FIXTURES_DIR / "leakage_applicants.json")
    assert len(cases) >= 3
    for case in cases:
        with pytest.raises(PointInTimeLeakageError) as exc_info:
            validate_applicant_record(case["payload"])
        assert exc_info.value.error_code == case["expected_error"]


def test_label_maturity_contract_and_immature_exclusion():
    lbl_fixtures = load_json_fixture(FIXTURES_DIR / "label_fixtures.json")

    matured = [validate_label_record(r, require_matured=True) for r in lbl_fixtures["valid_matured"]]
    assert [m.default_flag for m in matured] == [0, 1]

    # Censored immature labels are valid when require_matured=False, blocked when True.
    for raw in lbl_fixtures["valid_censored_immature"]:
        rec = validate_label_record(raw, require_matured=False)
        assert rec.label_status == "CENSORED_IMMATURE"
        assert rec.default_flag is None
        with pytest.raises(PointInTimeLeakageError):
            validate_label_record(raw, require_matured=True)

    # Immature label falsely marked MATURED must always raise PointInTimeLeakageError.
    for raw in lbl_fixtures["invalid_premature_matured_leakage"]:
        with pytest.raises(PointInTimeLeakageError):
            validate_label_record(raw, require_matured=False)


def test_temporal_split_excludes_censored_immature_records():
    valid_apps = [validate_applicant_record(r) for r in load_json_fixture(FIXTURES_DIR / "valid_applicants.json")]
    lbl_fixtures = load_json_fixture(FIXTURES_DIR / "label_fixtures.json")
    matured = [validate_label_record(r) for r in lbl_fixtures["valid_matured"]]

    # Add an immature label matching the 3rd applicant.
    immature_lbl = validate_label_record(
        {
            "application_id": valid_apps[2].application_id,
            "application_timestamp": valid_apps[2].application_timestamp,
            "observation_window_days": 365,
            "label_maturity_date": "2024-06-01T00:00:00Z",
            "as_of_cutoff_date": "2024-01-01T00:00:00Z",
            "default_flag": None,
            "label_status": "CENSORED_IMMATURE",
        }
    )

    splits = temporal_split_dataset(
        applicants=valid_apps[:3],
        labels=matured + [immature_lbl],
        train_end_date="2023-06-30",
        calibration_end_date="2023-12-31",
        test_end_date="2024-06-30",
        require_label_maturity=True,
    )
    assert len(splits["train"][0]) == 2
    assert len(splits["censored_excluded"][0]) == 1
    assert splits["censored_excluded"][0][0].application_id == valid_apps[2].application_id


def test_protected_subgroup_columns_excluded_from_feature_pipeline():
    pipe = FeaturePipeline()
    for protected in PROTECTED_SUBGROUP_FIELDS:
        assert protected not in pipe.feature_names_
        assert not any(name.startswith(f"{protected}_") for name in pipe.feature_names_)
