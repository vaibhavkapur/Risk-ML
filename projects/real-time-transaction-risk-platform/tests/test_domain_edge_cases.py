"""Deterministic unit tests for duplicate/late events, stale cache, and model timeout fallback."""

from __future__ import annotations

from pathlib import Path
import time
import pytest

from risk_platform.audit import AuditLogger
from risk_platform.config import load_config
from risk_platform.errors import DuplicateConflictError
from risk_platform.features import FEATURE_NAMES, OfflineFeatureExtractor, check_feature_parity
from risk_platform.model import CalibratedLinearRiskModel
from risk_platform.policy import FALLBACK_MODEL_VERSION
from risk_platform.schemas import FeatureVector, TransactionEvent
from risk_platform.service import RiskScoringService
from risk_platform.store import InMemoryFeatureStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def _make_event(
    event_id: str,
    event_ts: str,
    amount_cents: int = 5000,
    account_id: str = "acct_syn_edge_01",
    country_code: str = "US",
    merchant_category: str = "grocery",
    channel: str = "card_present",
    ip_risk_score: float = 0.05,
) -> TransactionEvent:
    return TransactionEvent.from_dict(
        {
            "schema_version": "tx.v1",
            "event_id": event_id,
            "account_id": account_id,
            "card_token": "card_tok_edge_01",
            "merchant_id": "merch_syn_101",
            "merchant_category": merchant_category,
            "amount_cents": amount_cents,
            "currency": "USD",
            "country_code": country_code,
            "channel": channel,
            "device_id": "dev_edge_01",
            "event_ts": event_ts,
            "ip_risk_score": ip_risk_score,
        }
    )


def _dummy_low_risk_model() -> CalibratedLinearRiskModel:
    return CalibratedLinearRiskModel(
        model_version="model.dummy_low.v1",
        feature_version="feat.v1",
        feature_names=FEATURE_NAMES,
        scaler_mean=tuple(0.0 for _ in FEATURE_NAMES),
        scaler_scale=tuple(1.0 for _ in FEATURE_NAMES),
        weights=tuple(0.0 for _ in FEATURE_NAMES),
        intercept=-3.0,  # sigmoid(-3) ~ 0.047 -> APPROVE
        platt_a=1.0,
        platt_b=0.0,
        training_metadata={},
    )


class _SlowModel:
    model_version = "model.slow.v1"
    feature_version = "feat.v1"

    def __init__(self, sleep_sec: float = 0.08) -> None:
        self._sleep_sec = sleep_sec

    def predict(self, feature_vector: FeatureVector):
        time.sleep(self._sleep_sec)
        return 0.02, 0.02, []


class _ExplodingModel:
    model_version = "model.broken.v1"
    feature_version = "feat.v1"

    def predict(self, feature_vector: FeatureVector):
        raise RuntimeError("Simulated inference runtime failure")


def test_duplicate_idempotency_and_conflicting_duplicate_rejection() -> None:
    store = InMemoryFeatureStore()
    ev1 = _make_event("evt_dup_001", "2026-01-15T10:00:00Z", amount_cents=5000)

    r1 = store.ingest(ev1)
    assert r1.status == "ACCEPTED"
    assert r1.account_event_count == 1

    # Exact duplicate is idempotently ignored and does NOT double-count velocity or amount sum
    r2 = store.ingest(ev1)
    assert r2.status == "DUPLICATE_IGNORED"
    assert r2.account_event_count == 1

    ev_next = _make_event("evt_dup_002", "2026-01-15T10:01:00Z", amount_cents=2500)
    fv = store.get_features(ev_next)
    assert fv.values["tx_count_5m"] == 1.0
    assert fv.values["tx_amount_sum_1h_usd"] == 50.0

    # Same event_id with tampered amount raises explicit DuplicateConflictError
    ev_conflict = _make_event("evt_dup_001", "2026-01-15T10:00:00Z", amount_cents=99999)
    with pytest.raises(DuplicateConflictError):
        store.ingest(ev_conflict)
    assert store.stats()["counters"]["duplicate_conflict"] == 1


def test_late_events_accepted_within_window_and_dropped_beyond_window() -> None:
    # 5-minute allowed lateness
    store = InMemoryFeatureStore(allowed_lateness_ms=300_000)
    e1 = _make_event("evt_late_001", "2026-01-15T10:10:00Z", amount_cents=4000)
    e_late_ok = _make_event("evt_late_002", "2026-01-15T10:07:00Z", amount_cents=6000)  # 3m late -> OK
    e_late_drop = _make_event("evt_late_003", "2026-01-15T10:02:00Z", amount_cents=8000)  # 8m late -> DROP
    e_query = _make_event("evt_late_004", "2026-01-15T10:11:00Z", amount_cents=3000)

    assert store.ingest(e1).status == "ACCEPTED"
    assert store.ingest(e_late_ok).status == "LATE_ACCEPTED"
    assert store.ingest(e_late_drop).status == "LATE_DROPPED"

    # Online store includes e1 + e_late_ok (sorted) and excludes dropped e_late_drop
    fv_online = store.get_features(e_query)
    assert fv_online.values["tx_count_5m"] == 2.0
    assert fv_online.values["tx_amount_sum_1h_usd"] == 100.0  # $40 + $60

    # Exact parity against offline extraction of accepted events
    offline_map = OfflineFeatureExtractor().extract_batch([e1, e_late_ok, e_query])
    parity = check_feature_parity(
        {e_query.event_id: offline_map[e_query.event_id]},
        {e_query.event_id: fv_online},
    )
    assert parity.passed is True


def test_stale_cache_triggers_conservative_escalation() -> None:
    cfg = load_config(DEFAULT_CFG_PATH)
    store = InMemoryFeatureStore(max_staleness_ms=600_000)  # 10 min staleness limit

    ev_hist = _make_event("evt_stale_001", "2026-01-15T10:00:00Z", amount_cents=5000)
    store.ingest(ev_hist)

    # New transaction arrives 30 minutes later with NO intermediate updates -> stale cache (30m > 10m)
    ev_high = _make_event("evt_stale_002", "2026-01-15T10:30:00Z", amount_cents=25000)  # $250 >= $150

    service = RiskScoringService(
        config=cfg,
        store=store,
        model=_dummy_low_risk_model(),
        audit_logger=AuditLogger(path=None),
    )
    try:
        decision, fv = service.score_transaction(ev_high)
    finally:
        service.close()

    assert fv.feature_status == "STALE"
    assert fv.feature_freshness_ms == 1_800_000
    assert decision.feature_status == "STALE"
    assert decision.action == "REVIEW"  # Escalated from APPROVE because cache is stale & amount >= $150
    assert decision.fallback_used is True
    assert decision.fallback_reason == "STALE_FEATURES"
    assert "STALE_CACHE_CONSERVATIVE_ESCALATION" in decision.reason_codes


def test_model_timeout_and_model_error_fallbacks() -> None:
    cfg = load_config(DEFAULT_CFG_PATH)
    store = InMemoryFeatureStore()
    ev_hist = _make_event("evt_to_001", "2026-01-15T10:00:00Z", amount_cents=3000)
    store.ingest(ev_hist)

    # High-value transaction ($1500) during model timeout -> conservative DECLINE fallback
    ev_big = _make_event(
        "evt_to_002",
        "2026-01-15T10:01:00Z",
        amount_cents=150000,
        merchant_category="electronics",
        channel="ecommerce",
        ip_risk_score=0.40,
    )
    slow_service = RiskScoringService(
        config=cfg,
        store=store,
        model=_SlowModel(sleep_sec=0.08),  # > 25ms timeout budget
        audit_logger=AuditLogger(path=None),
    )
    try:
        dec_timeout, _ = slow_service.score_transaction(ev_big)
        metrics_snap = slow_service.metrics.snapshot()
    finally:
        slow_service.close()

    assert dec_timeout.fallback_used is True
    assert dec_timeout.fallback_reason == "MODEL_TIMEOUT"
    assert dec_timeout.model_version == FALLBACK_MODEL_VERSION
    assert dec_timeout.action == "DECLINE"
    assert dec_timeout.raw_score is None
    assert "FALLBACK:MODEL_TIMEOUT" in dec_timeout.reason_codes
    assert metrics_snap["counters"]["model_timeout_fallbacks"] == 1

    # Runtime error in model -> conservative REVIEW fallback for mid-tier amount ($300)
    ev_mid = _make_event("evt_to_003", "2026-01-15T10:02:00Z", amount_cents=30000)
    err_service = RiskScoringService(
        config=cfg,
        store=store,
        model=_ExplodingModel(),
        audit_logger=AuditLogger(path=None),
    )
    try:
        dec_err, _ = err_service.score_transaction(ev_mid)
    finally:
        err_service.close()

    assert dec_err.fallback_used is True
    assert dec_err.fallback_reason == "MODEL_ERROR"
    assert dec_err.model_version == FALLBACK_MODEL_VERSION
    assert dec_err.action == "REVIEW"
