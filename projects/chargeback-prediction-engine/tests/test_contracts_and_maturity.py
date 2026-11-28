"""Domain correctness tests: reference oracle, immature labels, and missing reason codes."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from chargeback_engine.config import MaturityConfig
from chargeback_engine.contracts import (
    ContractValidationError,
    LabelStatus,
    ReasonCategory,
    UnsupportedInputError,
    normalize_reason_code,
    parse_utc_timestamp,
    validate_dataset_contracts,
)
from chargeback_engine.maturity import assign_maturity_labels
from chargeback_engine.pipeline import verify_reference_fixture


def test_reference_fixture_matches_hand_calculated_oracle(fixtures_dir):
    summary = verify_reference_fixture(fixtures_dir)
    assert summary["status"] == "passed"
    assert summary["checked_transactions"] == 6
    assert summary["diagnostics"]["immature_negative_count"] == 2
    assert summary["diagnostics"]["disputes_after_as_of_masked"] == 1
    assert summary["diagnostics"]["missing_reason_codes"] == 1


def test_immature_negatives_are_always_excluded(reference_raw_data):
    merchants_raw, tx_raw, disputes_raw, expected = reference_raw_data
    merchants, txs, disputes = validate_dataset_contracts(
        merchants_raw, tx_raw, disputes_raw
    )
    as_of = parse_utc_timestamp(expected["as_of_ts"], "as_of_ts")

    # Both maturity modes MUST strictly exclude immature negatives
    for mode in ("cohort_mature", "exclude_immature_negatives_only"):
        df, diag = assign_maturity_labels(
            merchants,
            txs,
            disputes,
            as_of_ts=as_of,
            maturity_cfg=MaturityConfig(default_maturity_days=90, maturity_mode=mode),
        )
        immature_neg = df.filter(
            df["label_status"] == LabelStatus.IMMATURE_NEGATIVE.value
        )
        assert immature_neg.height == 2
        assert immature_neg["label"].null_count() == 2
        assert immature_neg["is_trainable"].sum() == 0

        if mode == "cohort_mature":
            assert diag.trainable_count == 3  # 2 mature pos + 1 mature neg
        else:
            assert diag.trainable_count == 4  # + 1 immature pos, still 0 immature neg


def test_missing_and_unrecognized_reason_codes_handled_deterministically(
    reference_raw_data,
):
    code, cat, is_miss, is_unrec = normalize_reason_code(None)
    assert (code, cat, is_miss, is_unrec) == ("UNKNOWN", ReasonCategory.UNKNOWN, True, False)

    code2, cat2, is_miss2, is_unrec2 = normalize_reason_code("   ")
    assert (code2, cat2, is_miss2, is_unrec2) == ("UNKNOWN", ReasonCategory.UNKNOWN, True, False)

    code3, cat3, is_miss3, is_unrec3 = normalize_reason_code("99.9")
    assert (code3, cat3, is_miss3, is_unrec3) == ("99.9", ReasonCategory.UNKNOWN, False, True)

    # End-to-end check on missing + unrecognized codes in disputes
    merchants_raw, tx_raw, disputes_raw, expected = reference_raw_data
    modified_disputes = [dict(d) for d in disputes_raw]
    modified_disputes[0]["reason_code"] = "99.9"  # unrecognized
    merchants, txs, disputes = validate_dataset_contracts(
        merchants_raw, tx_raw, modified_disputes
    )
    df, diag = assign_maturity_labels(
        merchants,
        txs,
        disputes,
        as_of_ts=parse_utc_timestamp(expected["as_of_ts"], "as_of_ts"),
        maturity_cfg=MaturityConfig(),
    )
    assert diag.missing_reason_codes == 1
    assert diag.unrecognized_reason_codes == 1
    by_id = {r["transaction_id"]: r for r in df.to_dicts()}
    assert by_id["TX_001"]["is_unrecognized_reason_code"] is True
    assert by_id["TX_003"]["reason_code"] == "UNKNOWN"
    assert by_id["TX_003"]["is_missing_reason_code"] is True


def test_future_transactions_excluded_at_earlier_snapshot(reference_raw_data):
    merchants_raw, tx_raw, disputes_raw, _ = reference_raw_data
    merchants, txs, disputes = validate_dataset_contracts(
        merchants_raw, tx_raw, disputes_raw
    )
    early_snapshot = datetime(2025, 4, 20, tzinfo=timezone.utc)
    df, diag = assign_maturity_labels(
        merchants, txs, disputes, as_of_ts=early_snapshot, maturity_cfg=MaturityConfig()
    )
    # Only TX_001, TX_002, TX_003 existed on or before 2025-04-20
    assert diag.total_transactions == 3
    assert diag.excluded_future_transactions == 3
    # DSP_002 happened 2025-05-05 so it must be masked at 2025-04-20
    by_id = {r["transaction_id"]: r for r in df.to_dicts()}
    assert by_id["TX_003"]["has_observed_dispute"] is False
    assert by_id["TX_003"]["label_status"] == LabelStatus.IMMATURE_NEGATIVE.value


def test_invalid_contract_fixtures_rejected(reference_raw_data, invalid_cases):
    merchants_raw, tx_raw, disputes_raw, _ = reference_raw_data

    with pytest.raises(ContractValidationError, match="amount"):
        validate_dataset_contracts(
            merchants_raw, [invalid_cases["negative_amount_tx"]], []
        )

    with pytest.raises(UnsupportedInputError, match="currency"):
        validate_dataset_contracts(
            merchants_raw, [invalid_cases["unsupported_currency_tx"]], []
        )

    with pytest.raises(ContractValidationError, match="timezone"):
        validate_dataset_contracts(
            merchants_raw, [invalid_cases["naive_timestamp_tx"]], []
        )

    with pytest.raises(ContractValidationError, match="settlement_ts"):
        validate_dataset_contracts(
            merchants_raw, [invalid_cases["settlement_before_tx"]], []
        )

    with pytest.raises(ContractValidationError, match="precedes transaction_ts"):
        validate_dataset_contracts(
            merchants_raw, tx_raw, [invalid_cases["dispute_before_tx"]]
        )
