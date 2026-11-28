"""Versioned configuration loading, validation, and deterministic hashing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Set

from aml_engine.validation import ConfigurationError


REQUIRED_TOP_LEVEL_KEYS = (
    "config_version",
    "schema_version",
    "feature_version",
    "policy_version",
    "base_currency",
    "seed",
    "fx_rates_to_usd",
    "supported_channels",
    "rolling_windows_hours",
    "scenarios",
    "ml",
    "policy",
)


@dataclass(frozen=True)
class EngineConfig:
    raw: Dict[str, Any]
    config_hash: str

    @property
    def config_version(self) -> str:
        return str(self.raw["config_version"])

    @property
    def schema_version(self) -> str:
        return str(self.raw["schema_version"])

    @property
    def feature_version(self) -> str:
        return str(self.raw["feature_version"])

    @property
    def policy_version(self) -> str:
        return str(self.raw["policy_version"])

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def fx_rates_to_usd(self) -> Dict[str, float]:
        return {k.upper(): float(v) for k, v in self.raw["fx_rates_to_usd"].items()}

    @property
    def supported_currencies(self) -> Set[str]:
        return set(self.fx_rates_to_usd.keys())

    @property
    def supported_channels(self) -> Set[str]:
        return {str(c).upper() for c in self.raw["supported_channels"]}

    @property
    def rolling_windows_hours(self) -> Dict[str, float]:
        return {str(k): float(v) for k, v in self.raw["rolling_windows_hours"].items()}

    @property
    def scenarios(self) -> Dict[str, Dict[str, Any]]:
        return self.raw["scenarios"]

    @property
    def ml(self) -> Dict[str, Any]:
        return self.raw["ml"]

    @property
    def policy(self) -> Dict[str, Any]:
        return self.raw["policy"]


def validate_config_dict(raw: Dict[str, Any]) -> EngineConfig:
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration payload must be a JSON object.")
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in raw]
    if missing:
        raise ConfigurationError(f"Configuration missing required keys: {missing}")

    fx = raw.get("fx_rates_to_usd")
    if not isinstance(fx, dict) or "USD" not in fx:
        raise ConfigurationError("Configuration 'fx_rates_to_usd' must be a dict containing 'USD'.")
    for curr, rate in fx.items():
        if not isinstance(rate, (int, float)) or rate <= 0:
            raise ConfigurationError(f"FX rate for {curr!r} must be a positive number.")

    channels = raw.get("supported_channels")
    if not isinstance(channels, list) or not channels:
        raise ConfigurationError("Configuration 'supported_channels' must be a non-empty list.")

    canonical_json = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    config_hash = hashlib.sha256(canonical_json).hexdigest()[:16]
    return EngineConfig(raw=raw, config_hash=config_hash)


def load_config(path: str | Path) -> EngineConfig:
    config_path = Path(path)
    if not config_path.exists() or not config_path.is_file():
        raise ConfigurationError(
            f"Required configuration file not found: {config_path}. "
            "Provide a valid JSON config path (e.g., config/default_config.json)."
        )
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in configuration file {config_path}: {exc}") from exc
    return validate_config_dict(raw)


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "default_config.json"
