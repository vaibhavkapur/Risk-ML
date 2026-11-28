"""Review prioritization, non-punitive guardrails, and delayed-outcome monitoring tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from chargeback_engine.config import EconomicsConfig, PolicyConfig
from chargeback_engine.contracts import ConfigError
from chargeback_engine.pipeline import run_pipeline
from chargeback_engine.policy import prioritize_reviews
from chargeback_engine.synthetic import generate_synthetic_dataset


def test_review_prioritization_respects_budget_and_non_punitive_guardrails():
    df = pl.DataFrame(
        {
            "transaction_id": [f"TX_{i:02d}" for i in range(20)],
            "merchant_id": ["M_HIGH"] * 10 + ["M_LOW"] * 10,
            "amount_usd": [400.0] * 10 + [50.0] * 10,
            "label": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0] + [0] * 10,
        }
    )
    probs = np.array([0.35, 0.30, 0.25, 0.22, 0.20] + [0.01] * 15)
    econ = EconomicsConfig()
    policy = PolicyConfig(
        review_budget_fraction=0.20,  # budget = 4 txns
        min_expected_net_savings_usd=0.0,
        allow_automated_merchant_punishment=False,
        merchant_alert_rate_cap=0.25,
    )

    decisions, summary = prioritize_reviews(df, probs, econ, policy)
    assert summary["budget_capacity"] == 4
    assert summary["reviewed_count"] == 4
    assert all(d.action in {"manual_review", "approve"} for d in decisions)
    # 4/10 = 40% of M_HIGH flagged -> advisory issued, NEVER automated punishment
    assert len(summary["merchant_advisories"]) == 1
    adv = summary["merchant_advisories"][0]
    assert adv["merchant_id"] == "M_HIGH"
    assert adv["recommendation"] == "analyst_merchant_health_check"
    assert adv["automated_punitive_action"] is False


def test_punitive_merchant_policy_rejected_at_config_validation():
    bad_policy = PolicyConfig(allow_automated_merchant_punishment=True)
    with pytest.raises(ConfigError, match="prohibited"):
        bad_policy.validate()


def test_delayed_outcome_monitoring_matures_monotonically(default_config):
    merchants, txs, disputes = generate_synthetic_dataset(
        default_config.synthetic, seed=default_config.seed
    )
    result = run_pipeline(default_config, merchants, txs, disputes)
    snaps = result.delayed_outcome_monitoring
    assert len(snaps) == 6

    # Mature fraction and observed disputes must be non-decreasing across future checkpoints
    mat_fracs = [s["mature_fraction"] for s in snaps]
    obs_counts = [s["observed_disputes_by_checkpoint"] for s in snaps]
    exp_counts = [s["expected_disputes_by_checkpoint"] for s in snaps]

    assert mat_fracs == sorted(mat_fracs)
    assert obs_counts == sorted(obs_counts)
    assert exp_counts == sorted(exp_counts)
    assert mat_fracs[-1] >= 0.95  # By +120d after as_of_ts, immature cohorts have matured
