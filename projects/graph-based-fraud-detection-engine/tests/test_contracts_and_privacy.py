"""Tests for data contracts, privacy-safe IDs, config validation, and operational error handling."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from graph_fraud_engine.config import ConfigurationError, load_config
from graph_fraud_engine.privacy import (
    PrivacyViolationError,
    is_privacy_safe_id,
    pseudonymize_id,
)
from graph_fraud_engine.schemas import (
    DatasetBundle,
    FraudLabelRecord,
    SchemaValidationError,
    TransactionRecord,
)
from graph_fraud_engine.synthetic import build_hand_calculated_reference_bundle
from graph_fraud_engine.validation import validate_dataset_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_privacy_safe_pseudonymization_and_pii_rejection() -> None:
    aid1 = pseudonymize_id("synthetic_acc_1", "acc", salt="test-salt")
    aid2 = pseudonymize_id("synthetic_acc_1", "acc", salt="test-salt")
    assert aid1 == aid2
    assert is_privacy_safe_id(aid1, "acc")
    assert not is_privacy_safe_id(aid1, "dev")

    for pii_value in ("alice@example.com", "123-45-6789", "4111-1111-1111-1111"):
        with pytest.raises(PrivacyViolationError):
            pseudonymize_id(pii_value, "acc", salt="test-salt")


def test_missing_or_invalid_configuration_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "missing.yaml")

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("seed: 42\npolicy:\n  review_threshold: 0.9\n  block_threshold: 0.2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid configuration"):
        load_config(bad_yaml)

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="non-empty"):
        load_config(empty_yaml)


def test_valid_reference_bundle_passes_validation() -> None:
    bundle, _ = build_hand_calculated_reference_bundle()
    validate_dataset_bundle(bundle)


def test_invalid_contract_rejections() -> None:
    now = datetime(2025, 1, 10, 12, 0, tzinfo=timezone.utc)

    # Naive timestamp rejected
    with pytest.raises(ValidationError):
        TransactionRecord(
            tx_id="tx_abcdef123456",
            account_id="acc_abcdef123456",
            device_id="dev_abcdef123456",
            merchant_id="mer_abcdef123456",
            tx_timestamp=datetime(2025, 1, 10, 12, 0),
            amount_usd=100.0,
        )

    # Unsupported currency rejected
    with pytest.raises(ValidationError):
        TransactionRecord(
            tx_id="tx_abcdef123456",
            account_id="acc_abcdef123456",
            device_id="dev_abcdef123456",
            merchant_id="mer_abcdef123456",
            tx_timestamp=now,
            amount_usd=100.0,
            currency="BTC",
        )

    # Label maturing before transaction rejected
    with pytest.raises(ValidationError):
        FraudLabelRecord(
            tx_id="tx_abcdef123456",
            is_fraud=1,
            tx_timestamp=now,
            label_available_at=now - timedelta(hours=1),
        )


def test_cross_entity_referential_integrity_validation() -> None:
    bundle, _ = build_hand_calculated_reference_bundle()
    bad_tx = TransactionRecord(
        tx_id="tx_deadbeef1234",
        account_id="acc_000000000000",  # Not in bundle.accounts
        device_id=bundle.devices[0].device_id,
        merchant_id=bundle.merchants[0].merchant_id,
        tx_timestamp=datetime(2025, 1, 10, 12, 0, tzinfo=timezone.utc),
        amount_usd=50.0,
    )
    broken = bundle.model_copy(update={"transactions": bundle.transactions + [bad_tx]})
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_dataset_bundle(broken)
    assert any("unknown account_id" in err for err in exc_info.value.errors)


def test_disk_fixtures_if_present() -> None:
    fixtures_dir = PROJECT_ROOT / "data" / "fixtures"
    valid_path = fixtures_dir / "valid_reference_bundle.json"
    invalid_path = fixtures_dir / "invalid_contract_cases.json"
    if valid_path.exists():
        loaded = DatasetBundle.model_validate_json(valid_path.read_text(encoding="utf-8"))
        validate_dataset_bundle(loaded)
    if invalid_path.exists():
        raw = json.loads(invalid_path.read_text(encoding="utf-8"))
        with pytest.raises(ValidationError):
            DatasetBundle.model_validate(raw)
