"""Boundary, failure, and operational resilience tests."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from chargeback_engine.cli import main as cli_main
from chargeback_engine.config import MaturityConfig, load_config
from chargeback_engine.contracts import (
    ConfigError,
    DuplicateDisputeError,
    TemporalLeakageError,
    parse_utc_timestamp,
    validate_dataset_contracts,
)
from chargeback_engine.maturity import assign_maturity_labels, validate_temporal_holdout


def test_duplicate_disputes_raise_under_strict_policy(reference_raw_data, invalid_cases):
    merchants_raw, tx_raw, disputes_raw, expected = reference_raw_data
    disputes_with_dup = disputes_raw + [invalid_cases["duplicate_dispute"]]
    m, t, d = validate_dataset_contracts(merchants_raw, tx_raw, disputes_with_dup)
    as_of = parse_utc_timestamp(expected["as_of_ts"], "as_of")

    with pytest.raises(DuplicateDisputeError) as exc_info:
        assign_maturity_labels(
            m,
            t,
            d,
            as_of_ts=as_of,
            maturity_cfg=MaturityConfig(duplicate_dispute_policy="raise"),
        )
    assert "TX_001" in exc_info.value.duplicate_transaction_ids


def test_duplicate_disputes_deduplicated_deterministically_when_configured(
    reference_raw_data, invalid_cases
):
    merchants_raw, tx_raw, disputes_raw, expected = reference_raw_data
    disputes_with_dup = disputes_raw + [invalid_cases["duplicate_dispute"]]
    m, t, d = validate_dataset_contracts(merchants_raw, tx_raw, disputes_with_dup)
    as_of = parse_utc_timestamp(expected["as_of_ts"], "as_of")

    df, diag = assign_maturity_labels(
        m,
        t,
        d,
        as_of_ts=as_of,
        maturity_cfg=MaturityConfig(duplicate_dispute_policy="deduplicate_earliest"),
    )
    assert diag.duplicate_disputes_deduplicated == 1
    row = df.filter(pl.col("transaction_id") == "TX_001").to_dicts()[0]
    # Earliest dispute DSP_001 (2025-02-10) kept over DSP_001_DUP (2025-02-15)
    assert row["dispute_ts"] == datetime(2025, 2, 10, tzinfo=timezone.utc)


def test_temporal_holdout_rejects_overlapping_splits_and_null_labels():
    t1 = datetime(2025, 1, 10, tzinfo=timezone.utc)
    t2 = datetime(2025, 2, 10, tzinfo=timezone.utc)
    t3 = datetime(2025, 3, 10, tzinfo=timezone.utc)

    # Overlap: train max (t3) > cal min (t2)
    train_bad = pl.DataFrame({"transaction_id": ["A"], "transaction_ts": [t3], "label": [1]})
    cal_bad = pl.DataFrame({"transaction_id": ["B"], "transaction_ts": [t2], "label": [0]})
    test_ok = pl.DataFrame({"transaction_id": ["C"], "transaction_ts": [t3], "label": [0]})
    with pytest.raises(TemporalLeakageError, match="train max"):
        validate_temporal_holdout(train_bad, cal_bad, test_ok)

    # Null/immature label injected into test split
    train_ok = pl.DataFrame({"transaction_id": ["A"], "transaction_ts": [t1], "label": [1]})
    cal_ok = pl.DataFrame({"transaction_id": ["B"], "transaction_ts": [t2], "label": [0]})
    test_null = pl.DataFrame(
        {"transaction_id": ["C"], "transaction_ts": [t3], "label": [None]},
        schema={"transaction_id": pl.Utf8, "transaction_ts": pl.Datetime("us", "UTC"), "label": pl.Int8},
    )
    with pytest.raises(TemporalLeakageError, match="null/immature"):
        validate_temporal_holdout(train_ok, cal_ok, test_null)


def test_missing_and_malformed_config_operational_errors(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("split:\n  train_fraction: 0.9\n  calibration_fraction: 0.5\n  test_fraction: 0.2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(bad_yaml)

    # CLI must return non-zero exit code (no silent success)
    exit_code = cli_main(["demo", "--config", str(tmp_path / "does_not_exist.yaml")])
    assert exit_code == 2
