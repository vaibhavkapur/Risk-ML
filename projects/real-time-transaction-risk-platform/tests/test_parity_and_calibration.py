"""Offline/online feature parity, calibration, reproducibility, and training-serving skew tests."""

from __future__ import annotations

from pathlib import Path

from risk_platform.config import load_config
from risk_platform.features import OfflineFeatureExtractor, check_feature_parity
from risk_platform.model import CalibratedLinearRiskModel
from risk_platform.monitoring import detect_training_serving_skew
from risk_platform.store import InMemoryFeatureStore
from risk_platform.synthetic import generate_synthetic_dataset
from risk_platform.training import train_calibrated_baseline_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def test_reproducible_training_calibration_and_offline_online_parity(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CFG_PATH)
    events_a, labels_a = generate_synthetic_dataset(
        num_events=1800,
        num_accounts=90,
        seed=cfg.training.seed,
        label_maturity_hours=cfg.training.label_maturity_hours,
        fx_to_usd=cfg.schema.fx_to_usd,
    )
    events_b, labels_b = generate_synthetic_dataset(
        num_events=1800,
        num_accounts=90,
        seed=cfg.training.seed,
        label_maturity_hours=cfg.training.label_maturity_hours,
        fx_to_usd=cfg.schema.fx_to_usd,
    )

    model_a, meta_a = train_calibrated_baseline_model(events_a, labels_a, cfg)
    model_b, _ = train_calibrated_baseline_model(events_b, labels_b, cfg)

    # Exact reproducibility with fixed seed
    assert model_a.artifact_sha256 == model_b.artifact_sha256
    assert model_a.weights == model_b.weights
    assert model_a.platt_a == model_b.platt_a
    assert model_a.platt_b == model_b.platt_b

    # Model artifact round-trip integrity
    art_path = tmp_path / "calibrated_model.json"
    model_a.save(art_path)
    reloaded = CalibratedLinearRiskModel.load(art_path)
    assert reloaded.artifact_sha256 == model_a.artifact_sha256

    # Temporal evaluation quality sanity checks
    ev_metrics = meta_a["evaluation"]
    assert ev_metrics["calibrated_roc_auc"] >= 0.78
    assert 0.0 <= ev_metrics["calibrated_brier"] <= 0.25
    assert 0.0 <= ev_metrics["calibrated_ece"] <= 0.15

    # Offline/online parity across chronological held-out stream
    offline_ext = OfflineFeatureExtractor(fx_to_usd=cfg.schema.fx_to_usd)
    offline_all = offline_ext.extract_batch(events_a)

    store = InMemoryFeatureStore(
        allowed_lateness_ms=cfg.ingestion.allowed_lateness_ms,
        retention_window_ms=cfg.ingestion.retention_window_ms,
        max_events_per_account=cfg.ingestion.max_events_per_account,
        max_staleness_ms=cfg.serving.max_feature_staleness_ms,
        fx_to_usd=cfg.schema.fx_to_usd,
    )
    online_all = {}
    for ev in events_a:
        online_all[ev.event_id] = store.get_features(ev)
        store.ingest(ev)

    parity = check_feature_parity(offline_all, online_all, atol=1e-9)
    assert parity.passed is True
    assert parity.events_compared == len(events_a)
    assert parity.max_abs_diff <= 1e-9


def test_training_serving_skew_integration_clean_vs_drifted() -> None:
    cfg = load_config(DEFAULT_CFG_PATH)
    events, labels = generate_synthetic_dataset(
        num_events=2400,
        num_accounts=100,
        seed=cfg.training.seed,
        label_maturity_hours=cfg.training.label_maturity_hours,
        fx_to_usd=cfg.schema.fx_to_usd,
    )
    model, _ = train_calibrated_baseline_model(events, labels, cfg)

    offline_ext = OfflineFeatureExtractor(fx_to_usd=cfg.schema.fx_to_usd)
    offline_map = offline_ext.extract_batch(events)
    test_events = events[int(len(events) * 0.80) :]
    clean_vectors = [offline_map[e.event_id] for e in test_events]

    clean_report = detect_training_serving_skew(
        model=model,
        serving_vectors=clean_vectors,
        psi_warning=cfg.monitoring.skew_psi_warning,
        psi_critical=cfg.monitoring.skew_psi_critical,
        mean_shift_sigma_threshold=cfg.monitoring.skew_mean_shift_sigma,
    )
    assert clean_report["status"] == "OK"
    assert clean_report["alert_triggered"] is False
    assert clean_report["max_psi"] < cfg.monitoring.skew_psi_warning

    # Drifted serving stream (higher ticket size, elevated IP risk, more e-commerce/crypto)
    drift_events, _ = generate_synthetic_dataset(
        num_events=700,
        num_accounts=60,
        seed=cfg.training.seed + 777,
        label_maturity_hours=cfg.training.label_maturity_hours,
        fx_to_usd=cfg.schema.fx_to_usd,
        drift_mode=True,
        id_prefix="skew",
    )
    drift_vectors = list(offline_ext.extract_batch(drift_events).values())
    drifted_report = detect_training_serving_skew(
        model=model,
        serving_vectors=drift_vectors,
        psi_warning=cfg.monitoring.skew_psi_warning,
        psi_critical=cfg.monitoring.skew_psi_critical,
        mean_shift_sigma_threshold=cfg.monitoring.skew_mean_shift_sigma,
    )
    assert drifted_report["status"] == "CRITICAL_SKEW"
    assert drifted_report["alert_triggered"] is True
    assert drifted_report["max_psi"] >= cfg.monitoring.skew_psi_critical
    assert "amount_usd_log1p" in drifted_report["critical_features"]
