"""Typed configuration loading and validation for the ATO detection engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import yaml


class ConfigurationError(ValueError):
    """Raised when engine configuration is missing, malformed, or invalid."""


@dataclass(frozen=True)
class VersionsConfig:
    schema_version: str = "1.0.0"
    feature_version: str = "1.0.0"
    policy_version: str = "1.0.0"
    model_version: str = "1.0.0"


@dataclass(frozen=True)
class FeaturesConfig:
    short_window_seconds: int = 3600
    medium_window_seconds: int = 86400
    long_window_seconds: int = 2592000
    feature_ttl_seconds: int = 2592000
    max_plausible_velocity_kmh: float = 900.0
    shared_network_user_threshold: int = 4
    burst_failed_attempts_1h: int = 5
    missing_feature_fallback_ratio: float = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    train_fraction: float = 0.60
    calibration_fraction: float = 0.20
    test_fraction: float = 0.20
    label_maturity_hours: float = 48.0
    isolation_forest_contamination: float = 0.08
    isolation_forest_estimators: int = 120
    gb_max_iter: int = 120
    gb_max_depth: int = 4
    gb_learning_rate: float = 0.08
    anomaly_blend_weight: float = 0.20
    calibration_method: str = "sigmoid"


@dataclass(frozen=True)
class ThresholdsConfig:
    challenge_otp: float = 0.25
    challenge_webauthn: float = 0.55
    block_and_reset: float = 0.82


@dataclass(frozen=True)
class PolicyConfig:
    target_fprs: Tuple[float, ...] = (0.01, 0.02, 0.05)
    base_thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    high_value_account_delta: float = -0.08
    shared_network_dampening_delta: float = 0.06
    drift_tightening_delta: float = -0.05
    repeated_attempt_hard_step_up: int = 5
    repeated_attempt_hard_block: int = 10
    stale_model_max_age_days: int = 90


@dataclass(frozen=True)
class DriftConfig:
    psi_warning_threshold: float = 0.15
    psi_critical_threshold: float = 0.25
    missingness_increase_warning: float = 0.10
    missingness_increase_critical: float = 0.25
    expiry_rate_warning: float = 0.20
    min_sample_size: int = 20


@dataclass(frozen=True)
class AuditConfig:
    log_path: str = "reports/decision_audit.jsonl"
    strict_logging: bool = True


@dataclass(frozen=True)
class StoreConfig:
    backend: str = "memory"
    redis_url: Optional[str] = None
    redis_key_prefix: str = "ato_engine:"


@dataclass(frozen=True)
class EngineConfig:
    config_version: str = "1.0.0"
    seed: int = 42
    versions: VersionsConfig = field(default_factory=VersionsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    store: StoreConfig = field(default_factory=StoreConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_version": self.config_version,
            "seed": self.seed,
            "versions": self.versions.__dict__,
            "features": self.features.__dict__,
            "training": self.training.__dict__,
            "policy": {
                **{k: v for k, v in self.policy.__dict__.items() if k != "base_thresholds"},
                "target_fprs": list(self.policy.target_fprs),
                "base_thresholds": self.policy.base_thresholds.__dict__,
            },
            "drift": self.drift.__dict__,
            "audit": self.audit.__dict__,
            "store": self.store.__dict__,
        }


def validate_config(cfg: EngineConfig) -> None:
    """Validate structural invariants and numerical bounds of EngineConfig."""
    if cfg.seed < 0:
        raise ConfigurationError("seed must be non-negative")
    f = cfg.features
    if not (0 < f.short_window_seconds <= f.medium_window_seconds <= f.long_window_seconds):
        raise ConfigurationError(
            "feature windows must satisfy 0 < short <= medium <= long"
        )
    if f.feature_ttl_seconds <= 0:
        raise ConfigurationError("feature_ttl_seconds must be positive")
    if f.max_plausible_velocity_kmh <= 0:
        raise ConfigurationError("max_plausible_velocity_kmh must be positive")
    if not (0.0 < f.missing_feature_fallback_ratio <= 1.0):
        raise ConfigurationError("missing_feature_fallback_ratio must be in (0, 1]")

    tr = cfg.training
    total_frac = tr.train_fraction + tr.calibration_fraction + tr.test_fraction
    if abs(total_frac - 1.0) > 1e-6 or min(tr.train_fraction, tr.calibration_fraction, tr.test_fraction) <= 0:
        raise ConfigurationError("train/calibration/test fractions must be > 0 and sum to 1.0")
    if tr.label_maturity_hours < 0:
        raise ConfigurationError("label_maturity_hours must be >= 0")
    if not (0.0 <= tr.anomaly_blend_weight <= 1.0):
        raise ConfigurationError("anomaly_blend_weight must be in [0, 1]")
    if tr.calibration_method not in ("sigmoid", "isotonic"):
        raise ConfigurationError(f"Unsupported calibration_method: {tr.calibration_method}")

    th = cfg.policy.base_thresholds
    if not (0.0 < th.challenge_otp < th.challenge_webauthn < th.block_and_reset < 1.0):
        raise ConfigurationError(
            "base_thresholds must satisfy 0 < challenge_otp < challenge_webauthn < block_and_reset < 1"
        )
    if cfg.policy.repeated_attempt_hard_step_up >= cfg.policy.repeated_attempt_hard_block:
        raise ConfigurationError(
            "repeated_attempt_hard_step_up must be < repeated_attempt_hard_block"
        )
    if not cfg.policy.target_fprs or any(not (0.0 < fpr < 1.0) for fpr in cfg.policy.target_fprs):
        raise ConfigurationError("target_fprs must be non-empty probabilities in (0, 1)")

    dr = cfg.drift
    if not (0.0 < dr.psi_warning_threshold < dr.psi_critical_threshold):
        raise ConfigurationError("psi thresholds must satisfy 0 < warning < critical")
    if cfg.store.backend not in ("memory", "redis"):
        raise ConfigurationError(f"Unsupported store backend: {cfg.store.backend}")
    if cfg.store.backend == "redis" and not cfg.store.redis_url:
        raise ConfigurationError("store.redis_url is required when store.backend == 'redis'")


def load_config(path: str | Path) -> EngineConfig:
    """Load and validate an EngineConfig from a YAML or JSON file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigurationError(f"Configuration file not found: {cfg_path}")
    try:
        raw_text = cfg_path.read_text(encoding="utf-8")
        raw: Dict[str, Any] = (
            json.loads(raw_text) if cfg_path.suffix.lower() == ".json" else yaml.safe_load(raw_text)
        )
    except Exception as exc:
        raise ConfigurationError(f"Failed to parse configuration file {cfg_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration root must be a mapping in {cfg_path}")

    try:
        versions = VersionsConfig(**raw.get("versions", {}))
        features = FeaturesConfig(**raw.get("features", {}))
        training = TrainingConfig(**raw.get("training", {}))

        policy_raw = dict(raw.get("policy", {}))
        thresholds_raw = policy_raw.pop("base_thresholds", {})
        if "target_fprs" in policy_raw:
            policy_raw["target_fprs"] = tuple(float(x) for x in policy_raw["target_fprs"])
        policy = PolicyConfig(
            base_thresholds=ThresholdsConfig(**thresholds_raw),
            **policy_raw,
        )
        drift = DriftConfig(**raw.get("drift", {}))
        audit = AuditConfig(**raw.get("audit", {}))
        store = StoreConfig(**raw.get("store", {}))

        cfg = EngineConfig(
            config_version=str(raw.get("config_version", "1.0.0")),
            seed=int(raw.get("seed", 42)),
            versions=versions,
            features=features,
            training=training,
            policy=policy,
            drift=drift,
            audit=audit,
            store=store,
        )
    except TypeError as exc:
        raise ConfigurationError(f"Invalid configuration key or type: {exc}") from exc

    validate_config(cfg)
    return cfg
