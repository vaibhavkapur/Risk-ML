"""Typed configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigurationError(Exception):
    """Raised when engine configuration is missing, unreadable, or invalid."""


class DataGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_samples: int = Field(default=4500, ge=100, le=500_000)
    start_date: str = Field(default="2026-01-01T00:00:00Z")
    horizon_days: int = Field(default=60, ge=10, le=365)
    base_fraud_rate: float = Field(default=0.05, gt=0.001, lt=0.5)
    fraud_label_delay_days_min: int = Field(default=3, ge=0, le=180)
    fraud_label_delay_days_max: int = Field(default=18, ge=1, le=180)
    non_fraud_maturity_days: int = Field(default=14, ge=1, le=180)

    @model_validator(mode="after")
    def check_delay_bounds(self) -> "DataGenerationConfig":
        if self.fraud_label_delay_days_max < self.fraud_label_delay_days_min:
            raise ValueError("fraud_label_delay_days_max must be >= fraud_label_delay_days_min.")
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_end_day: int = Field(default=30, ge=5)
    cal_end_day: int = Field(default=40, ge=6)
    val_end_day: int = Field(default=50, ge=7)
    label_maturity_days: int = Field(default=14, ge=1, le=180)
    calibration_method: Literal["sigmoid", "isotonic", "none"] = Field(default="sigmoid")
    n_estimators: int = Field(default=120, ge=10, le=1000)
    max_depth: int = Field(default=3, ge=1, le=10)
    learning_rate: float = Field(default=0.08, gt=0.001, le=1.0)

    @model_validator(mode="after")
    def check_temporal_order(self) -> "TrainingConfig":
        if not (self.train_end_day < self.cal_end_day < self.val_end_day):
            raise ValueError("Temporal splits must satisfy train_end_day < cal_end_day < val_end_day.")
        return self


class MandatoryChallengeRulesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high_amount_minor: int = Field(default=50000, gt=0)
    high_risk_categories: List[str] = Field(default_factory=lambda: ["crypto_exchange", "gambling"])
    require_challenge_on_emulator: bool = Field(default=True)
    require_challenge_on_cross_border_high_amount_minor: int = Field(default=20000, gt=0)
    max_ip_velocity_1h_without_challenge: int = Field(default=8, ge=1)


class HighRiskOverridesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decline_prior_chargebacks_90d_gte: int = Field(default=2, ge=1)
    decline_ip_velocity_1h_gte: int = Field(default=20, ge=2)
    decline_emulator_and_tor: bool = Field(default=True)
    decline_amount_minor_with_chargeback: int = Field(default=10000, gt=0)


class PolicyOptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_challenge_rate: float = Field(default=0.22, gt=0.0, le=1.0)
    max_fraud_loss_rate: float = Field(default=0.015, gt=0.0, le=1.0)
    friction_cost_minor: int = Field(default=150, ge=0)
    decline_opportunity_cost_bps: int = Field(default=250, ge=0, le=10000)
    challenge_fraud_catch_rate: float = Field(default=0.80, ge=0.0, le=1.0)
    challenge_genuine_pass_rate: float = Field(default=0.88, ge=0.0, le=1.0)
    threshold_grid_steps: int = Field(default=45, ge=5, le=200)


class FallbackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    critical_features: List[str] = Field(
        default_factory=lambda: ["device_trust_score", "account_age_days", "ip_velocity_1h"]
    )
    max_missing_feature_ratio_for_model: float = Field(default=0.35, ge=0.0, le=1.0)
    abstain_missing_feature_ratio: float = Field(default=0.65, ge=0.0, le=1.0)
    outage_challenge_amount_minor: int = Field(default=10000, gt=0)
    outage_decline_amount_minor: int = Field(default=250000, gt=0)

    @model_validator(mode="after")
    def check_fallback_bounds(self) -> "FallbackConfig":
        if self.abstain_missing_feature_ratio < self.max_missing_feature_ratio_for_model:
            raise ValueError(
                "abstain_missing_feature_ratio must be >= max_missing_feature_ratio_for_model."
            )
        if self.outage_decline_amount_minor <= self.outage_challenge_amount_minor:
            raise ValueError(
                "outage_decline_amount_minor must be > outage_challenge_amount_minor."
            )
        return self


class StepUpSimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    genuine_pass_prob: float = Field(default=0.88, ge=0.0, le=1.0)
    genuine_abandon_prob: float = Field(default=0.08, ge=0.0, le=1.0)
    genuine_timeout_prob: float = Field(default=0.02, ge=0.0, le=1.0)
    genuine_fail_prob: float = Field(default=0.02, ge=0.0, le=1.0)
    fraud_pass_prob: float = Field(default=0.14, ge=0.0, le=1.0)
    fraud_abandon_prob: float = Field(default=0.34, ge=0.0, le=1.0)
    fraud_timeout_prob: float = Field(default=0.06, ge=0.0, le=1.0)
    fraud_fail_prob: float = Field(default=0.46, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_probability_sums(self) -> "StepUpSimulationConfig":
        g_sum = (
            self.genuine_pass_prob
            + self.genuine_abandon_prob
            + self.genuine_timeout_prob
            + self.genuine_fail_prob
        )
        f_sum = (
            self.fraud_pass_prob
            + self.fraud_abandon_prob
            + self.fraud_timeout_prob
            + self.fraud_fail_prob
        )
        if abs(g_sum - 1.0) > 1e-6:
            raise ValueError(f"Genuine challenge probabilities must sum to 1.0, got {g_sum:.6f}.")
        if abs(f_sum - 1.0) > 1e-6:
            raise ValueError(f"Fraud challenge probabilities must sum to 1.0, got {f_sum:.6f}.")
        return self


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    oracle_probability_tolerance: float = Field(default=1e-6, gt=0.0)
    reproducibility_tolerance: float = Field(default=1e-9, gt=0.0)
    max_acceptable_ece: float = Field(default=0.035, gt=0.0, le=1.0)
    max_acceptable_brier: float = Field(default=0.06, gt=0.0, le=1.0)


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["memory", "jsonl", "redis"] = Field(default="memory")
    jsonl_path: str = Field(default="reports/decision_log.jsonl")
    redis_url: str = Field(default="redis://localhost:6379/0")


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, ge=0)
    feature_version: str = Field(default="1.0.0")
    policy_version: str = Field(default="1.0.0")
    data_generation: DataGenerationConfig = Field(default_factory=DataGenerationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mandatory_challenge_rules: MandatoryChallengeRulesConfig = Field(
        default_factory=MandatoryChallengeRulesConfig
    )
    high_risk_overrides: HighRiskOverridesConfig = Field(default_factory=HighRiskOverridesConfig)
    policy_optimization: PolicyOptimizationConfig = Field(default_factory=PolicyOptimizationConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    step_up_simulation: StepUpSimulationConfig = Field(default_factory=StepUpSimulationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @model_validator(mode="after")
    def check_split_within_horizon(self) -> "EngineConfig":
        if self.training.val_end_day >= self.data_generation.horizon_days:
            raise ValueError(
                "training.val_end_day must be strictly less than data_generation.horizon_days "
                "so a chronological holdout test window exists."
            )
        return self

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> EngineConfig:
    """Load and validate an EngineConfig from a YAML or JSON file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigurationError(f"Configuration file not found: {cfg_path}")
    if not cfg_path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {cfg_path}")
    try:
        raw_text = cfg_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text)
    except Exception as exc:
        raise ConfigurationError(f"Failed to parse configuration file {cfg_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration file {cfg_path} must deserialize to a mapping, got {type(data).__name__}."
        )
    try:
        return EngineConfig.model_validate(data)
    except Exception as exc:
        raise ConfigurationError(f"Invalid configuration in {cfg_path}: {exc}") from exc
