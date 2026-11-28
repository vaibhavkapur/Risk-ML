"""M3 Integration tests: Constrained policy optimization, tradeoffs, mandatory challenge guard, and high-risk overrides."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from smart_auth_engine.config import load_config
from smart_auth_engine.contracts import AuthAction, DeviceSignals, PaymentEvent
from smart_auth_engine.data_generator import generate_synthetic_dataset
from smart_auth_engine.features import extract_feature_matrix, extract_features
from smart_auth_engine.inference import AuthenticationEngine
from smart_auth_engine.policy import (
    PolicyThresholds,
    decide_authentication_action,
    optimize_policy_thresholds,
)
from smart_auth_engine.training import ReferenceOracleScorer, train_calibrated_risk_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_constrained_policy_optimization_and_pareto_tradeoffs() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    events, labels, _ = generate_synthetic_dataset(cfg.data_generation, seed=cfg.seed)
    train_report = train_calibrated_risk_model(events, labels, cfg)

    X_val = extract_feature_matrix(train_report.splits.val_events)
    val_probs, _ = train_report.model.predict_proba_matrix(X_val)

    opt_res = optimize_policy_thresholds(
        events=train_report.splits.val_events,
        labels=train_report.splits.val_labels,
        calibrated_probs=val_probs,
        cfg=cfg,
    )
    assert opt_res.evaluated_candidates > 50
    assert len(opt_res.pareto_frontier) >= 2
    assert opt_res.constraints_satisfied is True
    assert opt_res.selected_point.challenge_rate <= cfg.policy_optimization.max_challenge_rate + 1e-9
    assert opt_res.selected_point.fraud_loss_rate <= cfg.policy_optimization.max_fraud_loss_rate + 1e-9
    assert 0.5 <= opt_res.selected_point.acceptance_rate <= 1.0


def test_mandatory_challenge_rules_cannot_be_bypassed_even_with_extreme_thresholds() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    # Most permissive possible thresholds (trying to approve almost everything)
    permissive = PolicyThresholds(challenge_threshold=0.98, decline_threshold=0.999)

    mandatory_events = [
        PaymentEvent(
            payment_id="pay_mand_amt",
            account_id="a1",
            merchant_id="m1",
            merchant_category="retail",
            amount_minor=50000,  # >= 50000
            currency="USD",
            timestamp=datetime(2026, 2, 10, 0, 0, 0, tzinfo=timezone.utc),
            billing_country="US",
            card_bin_country="US",
            account_age_days=500.0,
            prior_chargebacks_90d=0,
            tx_count_24h=1,
            tx_amount_24h_minor=50000,
            device=DeviceSignals(
                device_age_days=200.0,
                is_emulator=False,
                is_tor_or_vpn=False,
                ip_country="US",
                ip_velocity_1h=1,
                ip_velocity_24h=1,
                device_trust_score=0.99,
                biometric_enrolled=True,
                browser_fingerprint_mismatch=False,
            ),
        ),
        PaymentEvent(
            payment_id="pay_mand_crypto",
            account_id="a2",
            merchant_id="m2",
            merchant_category="crypto_exchange",
            amount_minor=1500,
            currency="USD",
            timestamp=datetime(2026, 2, 10, 0, 1, 0, tzinfo=timezone.utc),
            billing_country="US",
            card_bin_country="US",
            account_age_days=500.0,
            prior_chargebacks_90d=0,
            tx_count_24h=1,
            tx_amount_24h_minor=1500,
            device=DeviceSignals(
                device_age_days=200.0,
                is_emulator=False,
                is_tor_or_vpn=False,
                ip_country="US",
                ip_velocity_1h=1,
                ip_velocity_24h=1,
                device_trust_score=0.99,
                biometric_enrolled=True,
                browser_fingerprint_mismatch=False,
            ),
        ),
        PaymentEvent(
            payment_id="pay_mand_emulator",
            account_id="a3",
            merchant_id="m3",
            merchant_category="groceries",
            amount_minor=1200,
            currency="USD",
            timestamp=datetime(2026, 2, 10, 0, 2, 0, tzinfo=timezone.utc),
            billing_country="US",
            card_bin_country="US",
            account_age_days=500.0,
            prior_chargebacks_90d=0,
            tx_count_24h=1,
            tx_amount_24h_minor=1200,
            device=DeviceSignals(
                device_age_days=200.0,
                is_emulator=True,
                is_tor_or_vpn=False,
                ip_country="US",
                ip_velocity_1h=1,
                ip_velocity_24h=1,
                device_trust_score=0.99,
                biometric_enrolled=False,
                browser_fingerprint_mismatch=False,
            ),
        ),
        PaymentEvent(
            payment_id="pay_mand_cross_border",
            account_id="a4",
            merchant_id="m4",
            merchant_category="retail",
            amount_minor=25000,
            currency="USD",
            timestamp=datetime(2026, 2, 10, 0, 3, 0, tzinfo=timezone.utc),
            billing_country="US",
            card_bin_country="US",
            account_age_days=500.0,
            prior_chargebacks_90d=0,
            tx_count_24h=1,
            tx_amount_24h_minor=25000,
            device=DeviceSignals(
                device_age_days=200.0,
                is_emulator=False,
                is_tor_or_vpn=False,
                ip_country="GB",
                ip_velocity_1h=1,
                ip_velocity_24h=1,
                device_trust_score=0.99,
                biometric_enrolled=True,
                browser_fingerprint_mismatch=False,
            ),
        ),
    ]

    for ev in mandatory_events:
        ext = extract_features(ev)
        # Even with a near-zero fraud probability of 0.0001, mandatory challenge MUST hold!
        dec = decide_authentication_action(ev, ext, permissive, cfg, "test_model", 0.0001, 0.0001)
        assert dec.mandatory_rule_triggered is True
        assert dec.action == AuthAction.CHALLENGE, f"Mandatory rule bypassed for {ev.payment_id}"


def test_high_risk_action_overrides_integration_scenario() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    permissive = PolicyThresholds(challenge_threshold=0.95, decline_threshold=0.999)
    engine = AuthenticationEngine(cfg, permissive, model=ReferenceOracleScorer(), model_available=True)

    override_scenarios = [
        (
            "prior_chargebacks_override",
            PaymentEvent(
                payment_id="pay_ov_cb",
                account_id="a1",
                merchant_id="m1",
                merchant_category="groceries",
                amount_minor=2000,
                currency="USD",
                timestamp=datetime(2026, 2, 11, 10, 0, 0, tzinfo=timezone.utc),
                billing_country="US",
                card_bin_country="US",
                account_age_days=500.0,
                prior_chargebacks_90d=2,
                tx_count_24h=1,
                tx_amount_24h_minor=2000,
                device=DeviceSignals(
                    device_age_days=200.0,
                    is_emulator=False,
                    is_tor_or_vpn=False,
                    ip_country="US",
                    ip_velocity_1h=1,
                    ip_velocity_24h=1,
                    device_trust_score=0.95,
                    biometric_enrolled=True,
                    browser_fingerprint_mismatch=False,
                ),
            ),
            "OVERRIDE_PRIOR_CHARGEBACKS",
        ),
        (
            "extreme_ip_velocity_override",
            PaymentEvent(
                payment_id="pay_ov_vel",
                account_id="a2",
                merchant_id="m2",
                merchant_category="retail",
                amount_minor=3000,
                currency="USD",
                timestamp=datetime(2026, 2, 11, 10, 1, 0, tzinfo=timezone.utc),
                billing_country="US",
                card_bin_country="US",
                account_age_days=300.0,
                prior_chargebacks_90d=0,
                tx_count_24h=2,
                tx_amount_24h_minor=6000,
                device=DeviceSignals(
                    device_age_days=100.0,
                    is_emulator=False,
                    is_tor_or_vpn=False,
                    ip_country="US",
                    ip_velocity_1h=22,
                    ip_velocity_24h=25,
                    device_trust_score=0.80,
                    biometric_enrolled=True,
                    browser_fingerprint_mismatch=False,
                ),
            ),
            "OVERRIDE_EXTREME_IP_VELOCITY",
        ),
        (
            "emulator_and_tor_override",
            PaymentEvent(
                payment_id="pay_ov_emul_tor",
                account_id="a3",
                merchant_id="m3",
                merchant_category="retail",
                amount_minor=1500,
                currency="USD",
                timestamp=datetime(2026, 2, 11, 10, 2, 0, tzinfo=timezone.utc),
                billing_country="US",
                card_bin_country="US",
                account_age_days=300.0,
                prior_chargebacks_90d=0,
                tx_count_24h=1,
                tx_amount_24h_minor=1500,
                device=DeviceSignals(
                    device_age_days=10.0,
                    is_emulator=True,
                    is_tor_or_vpn=True,
                    ip_country="US",
                    ip_velocity_1h=1,
                    ip_velocity_24h=1,
                    device_trust_score=0.60,
                    biometric_enrolled=False,
                    browser_fingerprint_mismatch=False,
                ),
            ),
            "OVERRIDE_EMULATOR_AND_TOR",
        ),
    ]

    for name, ev, expected_reason in override_scenarios:
        res = engine.evaluate_and_simulate_step_up(ev, actual_is_fraud=False, record_store=True)
        assert res.decision.action == AuthAction.DECLINE, f"Override failed for {name}"
        assert res.decision.override_triggered is True
        assert expected_reason in res.decision.reason_codes
        assert res.final_outcome.value == "declined"

    assert engine.counters.override_hits == len(override_scenarios)
