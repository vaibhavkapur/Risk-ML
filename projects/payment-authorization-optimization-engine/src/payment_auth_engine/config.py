"""Configuration loading, validation, and deterministic hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Union

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class ConfigurationError(Exception):
    """Raised when engine configuration is missing, malformed, or invalid."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class DataConfig(BaseModel):
    n_samples: int = Field(..., ge=50, le=500_000)
    start_timestamp_utc: str = "2025-01-01T00:00:00Z"
    span_days: int = Field(..., ge=10, le=3650)
    min_label_maturity_days: int = Field(..., ge=1, le=365)
    train_fraction: float = Field(..., gt=0.0, lt=1.0)
    calibration_fraction: float = Field(..., gt=0.0, lt=1.0)
    test_fraction: float = Field(..., gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate_fractions(self) -> "DataConfig":
        total = self.train_fraction + self.calibration_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_fraction + calibration_fraction + test_fraction must sum to 1.0, got {total:.6f}"
            )
        return self


class EconomicsConfig(BaseModel):
    merchant_margin_bps: float = Field(..., gt=0.0, le=5000.0)
    chargeback_fee_cents: int = Field(..., ge=0, le=100_000)
    loss_given_fraud_multiplier: float = Field(..., gt=0.0, le=5.0)
    step_up_3ds_cost_cents: int = Field(..., ge=0, le=10_000)
    secondary_route_cost_cents: int = Field(..., ge=0, le=10_000)


class RetryConfig(BaseModel):
    max_retries: int = Field(..., ge=0, le=5)
    allow_soft_decline_retry: bool = True
    allow_technical_failover: bool = True


class PolicyConfig(BaseModel):
    baseline_risk_threshold: float = Field(..., ge=0.0, le=1.0)
    max_expected_loss_bps: float = Field(..., gt=0.0, le=10_000.0)
    max_p95_latency_ms: float = Field(..., gt=0.0, le=30_000.0)
    missing_feature_risk_penalty: float = Field(..., ge=0.0, le=1.0)
    missing_feature_uncertainty_multiplier: float = Field(..., ge=1.0, le=5.0)
    dual_lambda_loss_grid: List[float] = Field(..., min_length=1)
    dual_lambda_latency_grid: List[float] = Field(..., min_length=1)


class CounterfactualConfig(BaseModel):
    min_propensity_threshold: float = Field(..., gt=0.0, lt=1.0)
    ips_clip_min: float = Field(..., gt=0.0, lt=1.0)
    min_ess_ratio: float = Field(..., gt=0.0, le=1.0)
    bootstrap_resamples: int = Field(..., ge=10, le=5000)
    confidence_level: float = Field(..., gt=0.5, lt=1.0)


class ProcessorConfig(BaseModel):
    base_latency_ms: float = Field(..., gt=0.0, le=10_000.0)
    approval_bias: float = Field(..., ge=-0.5, le=0.5)
    cost_cents: int = Field(..., ge=0, le=10_000)


class EngineConfig(BaseModel):
    config_version: str = "1.0.0"
    seed: int = Field(..., ge=0, le=2**31 - 1)
    data: DataConfig
    economics: EconomicsConfig
    retry: RetryConfig
    policy: PolicyConfig
    counterfactual: CounterfactualConfig
    processors: Dict[str, ProcessorConfig] = Field(..., min_length=1)

    def config_hash(self) -> str:
        """Return deterministic SHA-256 prefix over canonical config JSON."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_config(path: Union[str, Path]) -> EngineConfig:
    """Load and validate an EngineConfig YAML file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigurationError("CONFIG_NOT_FOUND", f"Configuration file not found: {cfg_path}")
    if not cfg_path.is_file():
        raise ConfigurationError("CONFIG_NOT_FILE", f"Configuration path is not a file: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError("CONFIG_PARSE_ERROR", f"Invalid YAML in {cfg_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError("CONFIG_EMPTY_OR_INVALID", f"Config root must be a mapping in {cfg_path}")

    try:
        return EngineConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError("CONFIG_VALIDATION_ERROR", f"Invalid config schema: {exc}") from exc
