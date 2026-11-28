"""Typed configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union
import yaml

from risk_platform.errors import ConfigurationError


@dataclass(frozen=True)
class SchemaConfig:
    supported_versions: List[str]
    feature_version: str
    policy_version: str
    supported_currencies: List[str]
    fx_to_usd: Dict[str, float]
    max_amount_cents: int


@dataclass(frozen=True)
class IngestionConfig:
    allowed_lateness_ms: int
    retention_window_ms: int
    max_events_per_account: int


@dataclass(frozen=True)
class ServingConfig:
    max_feature_staleness_ms: int
    inference_timeout_ms: float
    review_threshold: float
    decline_threshold: float
    stale_review_amount_usd: float
    fallback_decline_amount_usd: float
    fallback_review_amount_usd: float


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    num_synthetic_events: int
    num_accounts: int
    label_maturity_hours: int
    train_fraction: float
    calibration_fraction: float
    test_fraction: float
    logreg_c: float
    max_iter: int


@dataclass(frozen=True)
class MonitoringConfig:
    skew_psi_warning: float
    skew_psi_critical: float
    skew_mean_shift_sigma: float


@dataclass(frozen=True)
class StorageConfig:
    model_artifact_path: str
    checkpoint_path: str
    audit_log_path: str
    redis_enabled: bool
    kafka_enabled: bool


@dataclass(frozen=True)
class PlatformConfig:
    name: str
    environment: str
    seed: int
    schema: SchemaConfig
    ingestion: IngestionConfig
    serving: ServingConfig
    training: TrainingConfig
    monitoring: MonitoringConfig
    storage: StorageConfig


def _require_section(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    val = data.get(key)
    if not isinstance(val, dict):
        raise ConfigurationError(f"Missing or invalid required config section: '{key}'")
    return val


def _require_key(section: Dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section or section[key] is None:
        raise ConfigurationError(f"Missing required config key: '{section_name}.{key}'")
    return section[key]


def load_config(path: Union[str, Path]) -> PlatformConfig:
    """Load and strictly validate PlatformConfig from a YAML file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigurationError(f"Configuration file not found: {cfg_path}")

    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigurationError(f"Failed to parse YAML config '{cfg_path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration file '{cfg_path}' must decode to a mapping")

    plat = _require_section(raw, "platform")
    sch = _require_section(raw, "schema")
    ing = _require_section(raw, "ingestion")
    srv = _require_section(raw, "serving")
    trn = _require_section(raw, "training")
    mon = _require_section(raw, "monitoring")
    stg = _require_section(raw, "storage")

    review_th = float(_require_key(srv, "review_threshold", "serving"))
    decline_th = float(_require_key(srv, "decline_threshold", "serving"))
    if not (0.0 < review_th < decline_th < 1.0):
        raise ConfigurationError(
            f"Thresholds must satisfy 0 < review_threshold ({review_th}) < decline_threshold ({decline_th}) < 1"
        )

    train_f = float(_require_key(trn, "train_fraction", "training"))
    cal_f = float(_require_key(trn, "calibration_fraction", "training"))
    test_f = float(_require_key(trn, "test_fraction", "training"))
    if abs((train_f + cal_f + test_f) - 1.0) > 1e-6:
        raise ConfigurationError(
            f"Training split fractions must sum to 1.0, got {train_f + cal_f + test_f}"
        )

    retention_ms = int(_require_key(ing, "retention_window_ms", "ingestion"))
    if retention_ms < 86_400_000:
        raise ConfigurationError(
            f"ingestion.retention_window_ms ({retention_ms}) must be >= 86400000 (24h)"
        )

    fx_raw = _require_key(sch, "fx_to_usd", "schema")
    if not isinstance(fx_raw, dict):
        raise ConfigurationError("schema.fx_to_usd must be a mapping")
    fx_to_usd = {str(k): float(v) for k, v in fx_raw.items()}
    supported_currencies = [str(c) for c in _require_key(sch, "supported_currencies", "schema")]
    for curr in supported_currencies:
        if curr not in fx_to_usd or fx_to_usd[curr] <= 0:
            raise ConfigurationError(f"Missing or non-positive FX rate for currency '{curr}'")

    return PlatformConfig(
        name=str(_require_key(plat, "name", "platform")),
        environment=str(_require_key(plat, "environment", "platform")),
        seed=int(_require_key(plat, "seed", "platform")),
        schema=SchemaConfig(
            supported_versions=[str(v) for v in _require_key(sch, "supported_versions", "schema")],
            feature_version=str(_require_key(sch, "feature_version", "schema")),
            policy_version=str(_require_key(sch, "policy_version", "schema")),
            supported_currencies=supported_currencies,
            fx_to_usd=fx_to_usd,
            max_amount_cents=int(_require_key(sch, "max_amount_cents", "schema")),
        ),
        ingestion=IngestionConfig(
            allowed_lateness_ms=int(_require_key(ing, "allowed_lateness_ms", "ingestion")),
            retention_window_ms=retention_ms,
            max_events_per_account=int(_require_key(ing, "max_events_per_account", "ingestion")),
        ),
        serving=ServingConfig(
            max_feature_staleness_ms=int(_require_key(srv, "max_feature_staleness_ms", "serving")),
            inference_timeout_ms=float(_require_key(srv, "inference_timeout_ms", "serving")),
            review_threshold=review_th,
            decline_threshold=decline_th,
            stale_review_amount_usd=float(_require_key(srv, "stale_review_amount_usd", "serving")),
            fallback_decline_amount_usd=float(
                _require_key(srv, "fallback_decline_amount_usd", "serving")
            ),
            fallback_review_amount_usd=float(
                _require_key(srv, "fallback_review_amount_usd", "serving")
            ),
        ),
        training=TrainingConfig(
            seed=int(_require_key(trn, "seed", "training")),
            num_synthetic_events=int(_require_key(trn, "num_synthetic_events", "training")),
            num_accounts=int(_require_key(trn, "num_accounts", "training")),
            label_maturity_hours=int(_require_key(trn, "label_maturity_hours", "training")),
            train_fraction=train_f,
            calibration_fraction=cal_f,
            test_fraction=test_f,
            logreg_c=float(_require_key(trn, "logreg_c", "training")),
            max_iter=int(_require_key(trn, "max_iter", "training")),
        ),
        monitoring=MonitoringConfig(
            skew_psi_warning=float(_require_key(mon, "skew_psi_warning", "monitoring")),
            skew_psi_critical=float(_require_key(mon, "skew_psi_critical", "monitoring")),
            skew_mean_shift_sigma=float(_require_key(mon, "skew_mean_shift_sigma", "monitoring")),
        ),
        storage=StorageConfig(
            model_artifact_path=str(_require_key(stg, "model_artifact_path", "storage")),
            checkpoint_path=str(_require_key(stg, "checkpoint_path", "storage")),
            audit_log_path=str(_require_key(stg, "audit_log_path", "storage")),
            redis_enabled=bool(_require_key(stg, "redis_enabled", "storage")),
            kafka_enabled=bool(_require_key(stg, "kafka_enabled", "storage")),
        ),
    )
