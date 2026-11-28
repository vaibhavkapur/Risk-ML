"""Typed configuration loading, validation, and deterministic hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from chargeback_engine.contracts import ConfigError


@dataclass(frozen=True)
class MaturityConfig:
    default_maturity_days: int = 90
    max_dispute_delay_days: int = 120
    duplicate_dispute_policy: str = "raise"  # "raise" | "deduplicate_earliest"
    maturity_mode: str = "cohort_mature"  # "cohort_mature" | "exclude_immature_negatives_only"

    def validate(self) -> None:
        if self.default_maturity_days <= 0:
            raise ConfigError("maturity.default_maturity_days must be > 0")
        if self.max_dispute_delay_days < self.default_maturity_days:
            raise ConfigError(
                "maturity.max_dispute_delay_days must be >= default_maturity_days"
            )
        if self.duplicate_dispute_policy not in {"raise", "deduplicate_earliest"}:
            raise ConfigError(
                f"Unsupported duplicate_dispute_policy: {self.duplicate_dispute_policy!r}"
            )
        if self.maturity_mode not in {
            "cohort_mature",
            "exclude_immature_negatives_only",
        }:
            raise ConfigError(f"Unsupported maturity_mode: {self.maturity_mode!r}")


@dataclass(frozen=True)
class FeatureConfig:
    merchant_windows_days: Tuple[int, ...] = (7, 30, 90)
    cold_start_prior_weight: float = 20.0
    fallback_prior_dispute_rate: float = 0.02
    cold_start_min_mature_txns: int = 10
    missing_reason_code_label: str = "UNKNOWN"

    def validate(self) -> None:
        if not self.merchant_windows_days:
            raise ConfigError("features.merchant_windows_days must not be empty")
        if any(w <= 0 for w in self.merchant_windows_days):
            raise ConfigError("All features.merchant_windows_days must be > 0")
        if self.cold_start_prior_weight <= 0:
            raise ConfigError("features.cold_start_prior_weight must be > 0")
        if not (0.0 < self.fallback_prior_dispute_rate < 1.0):
            raise ConfigError(
                "features.fallback_prior_dispute_rate must be in (0, 1)"
            )
        if self.cold_start_min_mature_txns < 0:
            raise ConfigError("features.cold_start_min_mature_txns must be >= 0")


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.60
    calibration_fraction: float = 0.20
    test_fraction: float = 0.20
    embargo_days: int = 0

    def validate(self) -> None:
        total = self.train_fraction + self.calibration_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"Split fractions must sum to 1.0, got {total:.6f}"
            )
        for name, val in [
            ("train_fraction", self.train_fraction),
            ("calibration_fraction", self.calibration_fraction),
            ("test_fraction", self.test_fraction),
        ]:
            if not (0.0 < val < 1.0):
                raise ConfigError(f"split.{name} must be in (0, 1), got {val}")
        if self.embargo_days < 0:
            raise ConfigError("split.embargo_days must be >= 0")


@dataclass(frozen=True)
class ModelConfig:
    primary_model: str = "hist_gbdt"  # "hist_gbdt" | "logistic"
    calibration_method: str = "sigmoid"  # "sigmoid" | "isotonic"
    class_weight: Optional[str] = "balanced"
    hist_gbdt_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_iter": 120,
            "max_depth": 4,
            "learning_rate": 0.06,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        }
    )
    logistic_params: Dict[str, Any] = field(
        default_factory=lambda: {"C": 1.0, "max_iter": 500}
    )

    def validate(self) -> None:
        if self.primary_model not in {"hist_gbdt", "logistic"}:
            raise ConfigError(f"Unsupported primary_model: {self.primary_model!r}")
        if self.calibration_method not in {"sigmoid", "isotonic"}:
            raise ConfigError(
                f"Unsupported calibration_method: {self.calibration_method!r}"
            )
        if self.class_weight not in {None, "balanced"}:
            raise ConfigError(f"Unsupported class_weight: {self.class_weight!r}")


@dataclass(frozen=True)
class EconomicsConfig:
    chargeback_fee_usd: float = 15.0
    expected_recovery_rate: float = 0.15
    review_cost_usd: float = 4.50
    review_prevention_effectiveness: float = 0.55

    def validate(self) -> None:
        if self.chargeback_fee_usd < 0:
            raise ConfigError("economics.chargeback_fee_usd must be >= 0")
        if not (0.0 <= self.expected_recovery_rate <= 1.0):
            raise ConfigError(
                "economics.expected_recovery_rate must be in [0, 1]"
            )
        if self.review_cost_usd < 0:
            raise ConfigError("economics.review_cost_usd must be >= 0")
        if not (0.0 <= self.review_prevention_effectiveness <= 1.0):
            raise ConfigError(
                "economics.review_prevention_effectiveness must be in [0, 1]"
            )


@dataclass(frozen=True)
class PolicyConfig:
    review_budget_fraction: float = 0.05
    min_expected_net_savings_usd: float = 0.0
    allow_automated_merchant_punishment: bool = False
    merchant_alert_rate_cap: float = 0.25

    def validate(self) -> None:
        if not (0.0 < self.review_budget_fraction <= 1.0):
            raise ConfigError("policy.review_budget_fraction must be in (0, 1]")
        if self.allow_automated_merchant_punishment:
            raise ConfigError(
                "Punitive automated merchant decisions are prohibited by design "
                "(policy.allow_automated_merchant_punishment must be false)."
            )
        if not (0.0 < self.merchant_alert_rate_cap <= 1.0):
            raise ConfigError("policy.merchant_alert_rate_cap must be in (0, 1]")


@dataclass(frozen=True)
class SyntheticConfig:
    n_merchants: int = 60
    n_transactions: int = 4500
    cold_start_merchant_fraction: float = 0.20
    base_dispute_rate: float = 0.028
    missing_reason_code_rate: float = 0.12
    start_date: str = "2025-01-01T00:00:00Z"
    span_days: int = 240
    as_of_date: str = "2025-10-15T00:00:00Z"

    def validate(self) -> None:
        if self.n_merchants <= 0 or self.n_transactions <= 0:
            raise ConfigError("synthetic counts must be > 0")
        if not (0.0 <= self.cold_start_merchant_fraction <= 1.0):
            raise ConfigError("cold_start_merchant_fraction must be in [0, 1]")
        if not (0.0 < self.base_dispute_rate < 0.5):
            raise ConfigError("base_dispute_rate must be in (0, 0.5)")
        if not (0.0 <= self.missing_reason_code_rate <= 1.0):
            raise ConfigError("missing_reason_code_rate must be in [0, 1]")
        if self.span_days <= 0:
            raise ConfigError("synthetic.span_days must be > 0")


@dataclass(frozen=True)
class EngineConfig:
    schema_version: str = "1.0.0"
    feature_set_version: str = "1.0.0"
    seed: int = 42
    maturity: MaturityConfig = field(default_factory=MaturityConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigError("seed must be >= 0")
        self.maturity.validate()
        self.features.validate()
        self.split.validate()
        self.model.validate()
        self.economics.validate()
        self.policy.validate()
        self.synthetic.validate()

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        raw["features"]["merchant_windows_days"] = list(
            self.features.merchant_windows_days
        )
        return raw

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _coerce_section(raw: Dict[str, Any], key: str, cls: Any) -> Any:
    section = raw.get(key)
    if section is None:
        return cls()
    if not isinstance(section, dict):
        raise ConfigError(f"Configuration section '{key}' must be a mapping")
    kwargs = dict(section)
    if cls is FeatureConfig and "merchant_windows_days" in kwargs:
        windows = kwargs["merchant_windows_days"]
        if not isinstance(windows, (list, tuple)):
            raise ConfigError("features.merchant_windows_days must be a list of ints")
        kwargs["merchant_windows_days"] = tuple(int(w) for w in windows)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"Invalid keys in section '{key}': {exc}") from exc


def load_config(path: str | Path) -> EngineConfig:
    """Load and validate an EngineConfig from a YAML file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigError(f"Configuration file not found: {cfg_path}")
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {cfg_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration root in {cfg_path} must be a mapping")

    cfg = EngineConfig(
        schema_version=str(raw.get("schema_version", "1.0.0")),
        feature_set_version=str(raw.get("feature_set_version", "1.0.0")),
        seed=int(raw.get("seed", 42)),
        maturity=_coerce_section(raw, "maturity", MaturityConfig),
        features=_coerce_section(raw, "features", FeatureConfig),
        split=_coerce_section(raw, "split", SplitConfig),
        model=_coerce_section(raw, "model", ModelConfig),
        economics=_coerce_section(raw, "economics", EconomicsConfig),
        policy=_coerce_section(raw, "policy", PolicyConfig),
        synthetic=_coerce_section(raw, "synthetic", SyntheticConfig),
    )
    cfg.validate()
    return cfg
