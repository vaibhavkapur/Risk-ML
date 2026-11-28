"""M2 Domain Correctness tests: Missing features, model outage, corrupted scores, deterministic fallback."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

from smart_auth_engine.config import load_config
from smart_auth_engine.contracts import AuthAction, DeviceSignals, FeatureContribution, PaymentEvent
from smart_auth_engine.features import extract_features
from smart_auth_engine.inference import AuthenticationEngine
from smart_auth_engine.policy import PolicyThresholds, decide_authentication_action
from smart_auth_engine.training import ReferenceOracleScorer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(PROJECT_ROOT / "config" / "default.yaml")
THRESHOLDS = PolicyThresholds(challenge_threshold=0.12, decline_threshold=0.55)


def _make_event(**overrides: Any) -> PaymentEvent:
    device_kwargs = overrides.pop(
        "device",
        {
            "device_id": "dev_1",
            "device_age_days": 120.0,
            "is_emulator": False,
            "is_tor_or_vpn": False,
            "ip_country": "US",
            "ip_velocity_1h": 1,
            "ip_velocity_24h": 2,
            "device_trust_score": 0.90,
            "biometric_enrolled": True,
            "browser_fingerprint_mismatch": False,
        },
    )
    base = {
        "payment_id": "pay_dom_001",
        "account_id": "acct_dom_001",
        "merchant_id": "merch_dom_001",
        "merchant_category": "retail",
        "amount_minor": 15000,
        "currency": "USD",
        "timestamp": datetime(2026, 2, 10, 10, 0, 0, tzinfo=timezone.utc),
        "billing_country": "US",
        "card_bin_country": "US",
        "account_age_days": 200.0,
        "prior_chargebacks_90d": 0,
        "tx_count_24h": 1,
        "tx_amount_24h_minor": 15000,
        "device": DeviceSignals(**device_kwargs) if isinstance(device_kwargs, dict) else device_kwargs,
    }
    base.update(overrides)
    return PaymentEvent(**base)


class _BrokenModel:
    model_version = "broken_model_v1"
    feature_version = "1.0.0"

    def predict_proba_event(self, event: PaymentEvent) -> Tuple[float, float, List[FeatureContribution]]:
        raise RuntimeError("Simulated inference backend timeout")


class _CorruptedScoreModel:
    model_version = "corrupted_score_v1"
    feature_version = "1.0.0"

    def predict_proba_event(self, event: PaymentEvent) -> Tuple[float, float, List[FeatureContribution]]:
        return float("nan"), float("nan"), []


def test_missing_critical_feature_triggers_deterministic_fallback() -> None:
    ev = _make_event(
        amount_minor=15000,
        device={
            "device_id": "dev_1",
            "device_age_days": 90.0,
            "is_emulator": False,
            "is_tor_or_vpn": False,
            "ip_country": "US",
            "ip_velocity_1h": 1,
            "ip_velocity_24h": 2,
            "device_trust_score": None,  # Critical feature missing
            "biometric_enrolled": True,
            "browser_fingerprint_mismatch": False,
        },
    )
    ext = extract_features(ev)
    assert "device_trust_score" in ext.missing_fields

    dec1 = decide_authentication_action(ev, ext, THRESHOLDS, CFG, "m_v1", 0.03, 0.03)
    dec2 = decide_authentication_action(ev, ext, THRESHOLDS, CFG, "m_v1", 0.03, 0.03)

    assert dec1.fallback_used is True
    assert dec1.action == AuthAction.CHALLENGE
    assert dec1.calibrated_fraud_prob is None
    assert any("FALLBACK_MISSING_CRITICAL_FEATURES" in r for r in dec1.reason_codes)
    assert dec1 == dec2  # Strictly deterministic


def test_excessive_missing_features_yields_abstain_unless_mandatory() -> None:
    # Low amount + almost all telemetry missing -> ABSTAIN
    ev_low = _make_event(
        payment_id="pay_abstain_1",
        amount_minor=4000,
        account_age_days=None,
        prior_chargebacks_90d=None,
        tx_count_24h=None,
        tx_amount_24h_minor=None,
        device=DeviceSignals(),
    )
    ext_low = extract_features(ev_low)
    assert ext_low.missing_ratio >= CFG.fallback.abstain_missing_feature_ratio
    dec_low = decide_authentication_action(ev_low, ext_low, THRESHOLDS, CFG, "m_v1", 0.02, 0.02)
    assert dec_low.action == AuthAction.ABSTAIN
    assert dec_low.fallback_used is True

    # High amount (>= mandatory threshold) + missing telemetry -> MUST still CHALLENGE (never abstain/bypass)
    ev_high = _make_event(
        payment_id="pay_abstain_high",
        amount_minor=60000,
        account_age_days=None,
        prior_chargebacks_90d=None,
        tx_count_24h=None,
        tx_amount_24h_minor=None,
        device=DeviceSignals(),
    )
    ext_high = extract_features(ev_high)
    dec_high = decide_authentication_action(ev_high, ext_high, THRESHOLDS, CFG, "m_v1", 0.02, 0.02)
    assert dec_high.action == AuthAction.CHALLENGE
    assert dec_high.mandatory_rule_triggered is True
    assert dec_high.fallback_used is True


def test_model_outage_and_corrupted_scores_fallback_deterministically() -> None:
    low_amt = _make_event(payment_id="pay_out_low", amount_minor=3500)
    med_amt = _make_event(payment_id="pay_out_med", amount_minor=15000)
    high_cap_amt = _make_event(payment_id="pay_out_cap", amount_minor=300000)
    mandatory_cat = _make_event(
        payment_id="pay_out_crypto", amount_minor=2500, merchant_category="crypto_exchange"
    )

    for engine in (
        AuthenticationEngine(CFG, THRESHOLDS, model=None, model_available=False),
        AuthenticationEngine(CFG, THRESHOLDS, model=_BrokenModel(), model_available=True),
        AuthenticationEngine(CFG, THRESHOLDS, model=_CorruptedScoreModel(), model_available=True),
    ):
        d_low_1 = engine.evaluate(low_amt, record_store=False)
        d_low_2 = engine.evaluate(low_amt, record_store=False)
        assert d_low_1.action == AuthAction.APPROVE
        assert d_low_1.fallback_used is True
        assert d_low_1 == d_low_2

        d_med = engine.evaluate(med_amt, record_store=False)
        assert d_med.action == AuthAction.CHALLENGE
        assert d_med.fallback_used is True

        d_cap = engine.evaluate(high_cap_amt, record_store=False)
        assert d_cap.action == AuthAction.DECLINE
        assert d_cap.fallback_used is True

        d_crypto = engine.evaluate(mandatory_cat, record_store=False)
        assert d_crypto.action == AuthAction.CHALLENGE
        assert d_crypto.mandatory_rule_triggered is True
        assert d_crypto.fallback_used is True


def test_toggle_outage_on_healthy_engine_preserves_determinism() -> None:
    engine = AuthenticationEngine(CFG, THRESHOLDS, model=ReferenceOracleScorer(), model_available=True)
    ev = _make_event(payment_id="pay_toggle_1", amount_minor=15000)

    dec_healthy = engine.evaluate(ev, record_store=False)
    assert dec_healthy.fallback_used is False
    assert dec_healthy.action == AuthAction.APPROVE

    engine.set_model_outage(True)
    dec_outage_a = engine.evaluate(ev, record_store=False)
    dec_outage_b = engine.evaluate(ev, record_store=False)
    assert dec_outage_a.fallback_used is True
    assert dec_outage_a.action == AuthAction.CHALLENGE
    assert dec_outage_a == dec_outage_b
    assert engine.counters.model_outage_hits == 2
