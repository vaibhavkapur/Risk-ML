"""Reproducibility, replay determinism, and operational failure-mode tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from payment_auth_engine.config import ConfigurationError, load_config
from payment_auth_engine.engine import AuthorizationEngine, run_full_experiment
from payment_auth_engine.models import AuthorizationModelBundle
from payment_auth_engine.policy import ConstrainedRiskAwarePolicy
from payment_auth_engine.synthetic import generate_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_exact_reproducibility_with_fixed_seed_and_config() -> None:
    config = load_config(ROOT / "config" / "default.yaml")
    run_a = run_full_experiment(config)
    run_b = run_full_experiment(config)

    assert run_a["model_summary"] == run_b["model_summary"]
    assert run_a["policy_comparison"] == run_b["policy_comparison"]
    assert run_a["policy_audit"] == run_b["policy_audit"]
    assert run_a["constrained_sample_decisions"] == run_b["constrained_sample_decisions"]


def test_replay_produces_identical_decision_ids_and_outputs() -> None:
    config = load_config(ROOT / "config" / "default.yaml")
    dataset = generate_synthetic_dataset(config)
    bundle, _, cal_idx, test_idx = AuthorizationModelBundle.train(
        dataset.attempts, dataset.outcomes, config, dataset.as_of_utc
    )
    cal_attempts = [dataset.attempts[int(i)] for i in cal_idx]
    cal_preds = bundle.predict_batch_all_actions(cal_attempts)
    test_attempts = [dataset.attempts[int(i)] for i in test_idx[:100]]

    policy = ConstrainedRiskAwarePolicy(config)
    policy.fit_dual_multipliers(cal_attempts, cal_preds)
    engine = AuthorizationEngine(config=config, model_bundle=bundle, policy=policy)

    first = engine.decide_batch(test_attempts)
    replayed = engine.replay(test_attempts)

    assert len(first) == len(replayed) == 100
    for d1, d2 in zip(first, replayed):
        assert d1.decision_id == d2.decision_id
        assert d1.action == d2.action
        assert d1.predicted_acceptance_prob == d2.predicted_acceptance_prob
        assert d1.expected_loss_cents == d2.expected_loss_cents


def test_operational_missing_and_invalid_configuration_fail_explicitly(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as exc_missing:
        load_config(tmp_path / "missing.yaml")
    assert exc_missing.value.code == "CONFIG_NOT_FOUND"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("seed: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigurationError) as exc_yaml:
        load_config(bad_yaml)
    assert exc_yaml.value.code == "CONFIG_PARSE_ERROR"

    bad_fractions = tmp_path / "bad_frac.yaml"
    bad_fractions.write_text(
        (ROOT / "config" / "default.yaml").read_text(encoding="utf-8").replace("test_fraction: 0.25", "test_fraction: 0.90"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError) as exc_val:
        load_config(bad_fractions)
    assert exc_val.value.code == "CONFIG_VALIDATION_ERROR"
