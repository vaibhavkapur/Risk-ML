"""Configuration loading and validation with fail-fast diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class ConfigurationError(ValueError):
    """Raised when configuration is missing, unreadable, or invalid."""


class PrivacyConfig(BaseModel):
    salt: str = Field(min_length=4)
    digest_length: int = Field(default=12, ge=8, le=32)


class SyntheticDataConfig(BaseModel):
    n_accounts: int = Field(ge=20)
    n_devices: int = Field(ge=10)
    n_merchants: int = Field(ge=5)
    n_transactions: int = Field(ge=50)
    start_time_utc: str
    horizon_days: int = Field(ge=10)
    fraud_ring_count: int = Field(ge=1)
    accounts_per_ring: int = Field(ge=2)
    devices_per_ring: int = Field(ge=1)
    legitimate_shared_device_groups: int = Field(ge=1)
    accounts_per_shared_group: int = Field(ge=2)
    target_fraud_rate: float = Field(gt=0.0, lt=0.5)
    label_maturity_days_min: int = Field(ge=1)
    label_maturity_days_max: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_maturity_bounds(self) -> "SyntheticDataConfig":
        if self.label_maturity_days_max < self.label_maturity_days_min:
            raise ValueError("label_maturity_days_max must be >= label_maturity_days_min")
        return self


class TemporalSplitConfig(BaseModel):
    train_end_day: int = Field(ge=5)
    val_end_day: int = Field(ge=10)
    test_end_day: int = Field(ge=15)
    enforce_label_maturity: bool = True

    @model_validator(mode="after")
    def _validate_monotonic_splits(self) -> "TemporalSplitConfig":
        if not (self.train_end_day < self.val_end_day < self.test_end_day):
            raise ValueError("Temporal splits must satisfy train_end_day < val_end_day < test_end_day")
        return self


class FeatureConfig(BaseModel):
    short_window_hours: int = Field(default=24, ge=1)
    medium_window_days: int = Field(default=7, ge=1)
    long_window_days: int = Field(default=30, ge=1)
    max_hops: int = Field(default=2, ge=1, le=3)
    high_risk_merchant_min_tx: int = Field(default=5, ge=1)


class LogisticModelConfig(BaseModel):
    C: float = Field(default=1.0, gt=0.0)
    max_iter: int = Field(default=1000, ge=50)
    class_weight: str = Field(default="balanced")


class BoostedModelConfig(BaseModel):
    max_iter: int = Field(default=120, ge=10)
    max_depth: int = Field(default=5, ge=2, le=12)
    learning_rate: float = Field(default=0.06, gt=0.0, le=1.0)
    l2_regularization: float = Field(default=1.0, ge=0.0)
    min_samples_leaf: int = Field(default=15, ge=2)


class GNNModelConfig(BaseModel):
    enabled: bool = True
    hidden_dim: int = Field(default=24, ge=4, le=128)
    epochs: int = Field(default=75, ge=5, le=500)
    learning_rate: float = Field(default=0.025, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.001, ge=0.0)
    pos_weight: float = Field(default=8.0, gt=0.0)
    aggregation: str = Field(default="mean")
    neighbor_window_days: int = Field(default=30, ge=1)
    use_pyg_if_available: bool = False

    @model_validator(mode="after")
    def _validate_aggregation(self) -> "GNNModelConfig":
        if self.aggregation not in {"mean", "max"}:
            raise ValueError("GNN aggregation must be 'mean' or 'max'")
        return self


class ModelsConfig(BaseModel):
    logistic: LogisticModelConfig = Field(default_factory=LogisticModelConfig)
    boosted: BoostedModelConfig = Field(default_factory=BoostedModelConfig)
    gnn: GNNModelConfig = Field(default_factory=GNNModelConfig)


class CalibrationConfig(BaseModel):
    method: str = Field(default="platt")
    n_bins: int = Field(default=10, ge=3, le=30)

    @model_validator(mode="after")
    def _validate_method(self) -> "CalibrationConfig":
        if self.method not in {"platt", "isotonic", "none"}:
            raise ValueError("Calibration method must be one of: platt, isotonic, none")
        return self


class PolicyConfig(BaseModel):
    review_threshold: float = Field(default=0.35, gt=0.0, lt=1.0)
    block_threshold: float = Field(default=0.72, gt=0.0, lt=1.0)
    cold_start_review_threshold: float = Field(default=0.25, gt=0.0, lt=1.0)
    max_evidence_paths: int = Field(default=5, ge=1, le=20)
    fixed_fpr_targets: List[float] = Field(default_factory=lambda: [0.01, 0.05])

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "PolicyConfig":
        if not (self.review_threshold < self.block_threshold):
            raise ValueError("review_threshold must be strictly less than block_threshold")
        for fpr in self.fixed_fpr_targets:
            if not (0.0 < fpr < 1.0):
                raise ValueError("fixed_fpr_targets values must be in (0, 1)")
        return self


class ServingConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1024, le=65535)
    default_model: str = "gnn"

    @model_validator(mode="after")
    def _validate_default_model(self) -> "ServingConfig":
        if self.default_model not in {"logistic", "boosted", "gnn"}:
            raise ValueError("default_model must be one of: logistic, boosted, gnn")
        return self


class EngineConfig(BaseModel):
    config_version: str = "1.0.0"
    schema_version: str = "1.0.0"
    feature_version: str = "1.0.0"
    model_version: str = "0.1.0"
    seed: int = 42
    privacy: PrivacyConfig
    synthetic_data: SyntheticDataConfig
    temporal_split: TemporalSplitConfig
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)


def load_config(path: str | Path) -> EngineConfig:
    """Load and validate an EngineConfig YAML file.

    Raises ConfigurationError if the file does not exist, is empty, cannot be parsed,
    or violates schema constraints. Never silently falls back to defaults.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {config_path}")
    try:
        raw_text = config_path.read_text(encoding="utf-8")
        raw_data: Any = yaml.safe_load(raw_text)
    except Exception as exc:
        raise ConfigurationError(f"Failed to parse YAML configuration {config_path}: {exc}") from exc

    if not isinstance(raw_data, dict) or not raw_data:
        raise ConfigurationError(f"Configuration file {config_path} must contain a non-empty YAML mapping")

    try:
        return EngineConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration in {config_path}: {exc}") from exc


def config_to_dict(config: EngineConfig) -> Dict[str, Any]:
    """Serialize EngineConfig to a plain JSON-safe dictionary."""
    return config.model_dump()
