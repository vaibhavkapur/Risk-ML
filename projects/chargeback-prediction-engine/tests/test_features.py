"""Causal feature engineering tests: anti-leakage and merchant cold-start shrinkage."""

from __future__ import annotations

import pytest

from chargeback_engine.config import FeatureConfig, MaturityConfig
from chargeback_engine.contracts import parse_utc_timestamp, validate_dataset_contracts
from chargeback_engine.features import build_causal_features
from chargeback_engine.maturity import assign_maturity_labels


def test_zero_future_dispute_leakage_in_causal_features(reference_raw_data):
    """Mutating a future dispute must not alter ANY feature of an earlier transaction."""
    merchants_raw, tx_raw, disputes_raw, expected = reference_raw_data
    as_of = parse_utc_timestamp("2025-12-01T00:00:00Z", "as_of")
    fcfg = FeatureConfig()

    m1, t1, d1 = validate_dataset_contracts(merchants_raw, tx_raw, disputes_raw)
    lab1, _ = assign_maturity_labels(m1, t1, d1, as_of_ts=as_of, maturity_cfg=MaturityConfig())
    feat1, meta = build_causal_features(lab1, fcfg, prior_dispute_rate=0.02)

    # Inject a new dispute on TX_002 reported on 2025-04-20 (AFTER TX_003 on 2025-04-15!)
    extra_dispute = {
        "dispute_id": "DSP_FUTURE_TEST",
        "transaction_id": "TX_002",
        "dispute_ts": "2025-04-20T00:00:00Z",
        "reason_code": "13.1",
        "dispute_amount": 200.0,
        "currency": "USD",
    }
    m2, t2, d2 = validate_dataset_contracts(
        merchants_raw, tx_raw, disputes_raw + [extra_dispute]
    )
    lab2, _ = assign_maturity_labels(m2, t2, d2, as_of_ts=as_of, maturity_cfg=MaturityConfig())
    feat2, _ = build_causal_features(lab2, fcfg, prior_dispute_rate=0.02)

    by_id_1 = {r["transaction_id"]: r for r in feat1.to_dicts()}
    by_id_2 = {r["transaction_id"]: r for r in feat2.to_dicts()}

    # TX_001, TX_002, and TX_003 ALL occurred before 2025-04-20 -> features MUST be identical!
    for tx_id in ("TX_001", "TX_002", "TX_003"):
        for col in meta.feature_columns:
            assert by_id_1[tx_id][col] == pytest.approx(by_id_2[tx_id][col]), (
                f"Leakage detected in {tx_id}.{col}"
            )

    # TX_004 occurred on 2025-05-10 (AFTER 2025-04-20) -> MUST reflect the newly reported dispute!
    assert by_id_2["TX_004"]["m_disputes_reported_30d"] == by_id_1["TX_004"]["m_disputes_reported_30d"] + 1
    assert by_id_2["TX_004"]["m_prior_mature_dispute_count"] == 2.0


def test_cold_start_merchant_shrinks_smoothly_to_prior(reference_raw_data):
    merchants_raw, tx_raw, disputes_raw, expected = reference_raw_data
    as_of = parse_utc_timestamp(expected["as_of_ts"], "as_of")
    m, t, d = validate_dataset_contracts(merchants_raw, tx_raw, disputes_raw)
    lab, _ = assign_maturity_labels(m, t, d, as_of_ts=as_of, maturity_cfg=MaturityConfig())
    feat, _ = build_causal_features(lab, FeatureConfig(), prior_dispute_rate=0.035)

    by_id = {r["transaction_id"]: r for r in feat.to_dicts()}
    # M_NEW has 0 mature prior txns -> smoothed rate must equal prior 0.035 exactly
    assert by_id["TX_005"]["m_is_cold_start"] == 1.0
    assert by_id["TX_005"]["m_prior_mature_tx_count"] == 0.0
    assert by_id["TX_005"]["m_smoothed_dispute_rate"] == pytest.approx(0.035)
    assert by_id["TX_006"]["m_is_cold_start"] == 1.0
    assert by_id["TX_006"]["m_smoothed_dispute_rate"] == pytest.approx(0.035)
