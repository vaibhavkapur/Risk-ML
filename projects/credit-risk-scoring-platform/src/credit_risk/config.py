"""Typed configuration loader and validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from credit_risk.errors import ConfigurationError, UnsupportedInputError

SUPPORTED_BOOSTED_BACKENDS = ("sklearn_hist_gb", "xgboost")
SUPPORTED_CALIBRATION_METHODS = ("sigmoid", "isotonic")
SUPPORTED_PRIMARY_MODELS = ("logistic_regression", "boosted_trees")


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str
    schema_version: str
    feature_version: str
    policy_version: str
    seed: int


@dataclass(frozen=True)
class DataConfig:
    n_applicants: int
    start_date: str
    end_date: str
    observation_window_days: int
    as_of_cutoff_date: str
    missing_rate: float
    shift_start_date: str
    shift_magnitude: float


@dataclass(frozen=True)
class TemporalSplitConfig:
    train_end_date: str
    calibration_end_date: str
    test_end_date: str
    require_label_maturity: bool = True


@dataclass(frozen=True)
class LogisticConfig:
    C: float = 1.0
    max_iter: int = 1000
    class_weight: Optional[str] = None


@dataclass(frozen=True)
class BoostedConfig:
    backend: str = "sklearn_hist_gb"
    max_iter: int = 180
    max_depth: int = 4
    learning_rate: float = 0.05
    min_samples_leaf: int = 30
    l2_regularization: float = 1.0


@dataclass(frozen=True)
class ModelsConfig:
    logistic_regression: LogisticConfig
    boosted_trees: BoostedConfig
    primary_model: str = "boosted_trees"


@dataclass(frozen=True)
class CalibrationConfig:
    method: str = "isotonic"
    comparison_methods: List[str] = field(default_factory=lambda: ["sigmoid", "isotonic"])
    n_bins: int = 10
    clip_epsilon: float = 1e-4
    monotonicity_tolerance: float = 1e-6


@dataclass(frozen=True)
class BaselineComparisonConfig:
    min_auroc_improvement_or_parity: float = -0.02
    max_brier_degradation: float = 0.015
    max_ece_calibrated: float = 0.05


@dataclass(frozen=True)
class EvaluationConfig:
    min_subgroup_sample_size: int
    subgroup_columns: List[str]
    psi_bins: int
    psi_warning_threshold: float
    psi_alert_threshold: float
    decision_threshold: float
    baseline_comparison: BaselineComparisonConfig


@dataclass(frozen=True)
class RiskBandConfig:
    name: str
    max_pd: float
    queue_priority: str
    review_sla_hours: int
    recommended_action: str


@dataclass(frozen=True)
class PolicyConfig:
    automated_decision_allowed: bool
    risk_bands: List[RiskBandConfig]


@dataclass(frozen=True)
class ExplainabilityConfig:
    top_k_factors: int = 5
    use_shap_if_available: bool = False


@dataclass(frozen=True)
class PathsConfig:
    artifacts_dir: str
    reports_dir: str
    review_store_path: str


@dataclass(frozen=True)
class PlatformConfig:
    project: ProjectConfig
    data: DataConfig
    temporal_split: TemporalSplitConfig
    models: ModelsConfig
    calibration: CalibrationConfig
    evaluation: EvaluationConfig
    policy: PolicyConfig
    explainability: ExplainabilityConfig
    paths: PathsConfig


def _require_section(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in raw or not isinstance(raw[key], dict):
        raise ConfigurationError(
            f"Missing required configuration section '{key}'.",
            details={"missing_section": key},
        )
    return raw[key]


def _require_key(section: Dict[str, Any], section_name: str, key: str) -> Any:
    if key not in section:
        raise ConfigurationError(
            f"Missing required key '{section_name}.{key}'.",
            details={"section": section_name, "key": key},
        )
    return section[key]


def load_config(config_path: Union[str, Path]) -> PlatformConfig:
    """Load and validate a YAML configuration file into a typed PlatformConfig."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {path}",
            details={"path": str(path)},
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Malformed YAML in configuration file: {path}",
            details={"path": str(path), "error": str(exc)},
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Configuration root must be a YAML mapping: {path}",
            details={"path": str(path)},
        )

    return parse_config_dict(raw)


def parse_config_dict(raw: Dict[str, Any]) -> PlatformConfig:
    """Parse and strictly validate a raw config mapping."""
    proj_raw = _require_section(raw, "project")
    data_raw = _require_section(raw, "data")
    split_raw = _require_section(raw, "temporal_split")
    models_raw = _require_section(raw, "models")
    cal_raw = _require_section(raw, "calibration")
    eval_raw = _require_section(raw, "evaluation")
    policy_raw = _require_section(raw, "policy")
    exp_raw = _require_section(raw, "explainability")
    paths_raw = _require_section(raw, "paths")

    project = ProjectConfig(
        name=str(_require_key(proj_raw, "project", "name")),
        version=str(_require_key(proj_raw, "project", "version")),
        schema_version=str(_require_key(proj_raw, "project", "schema_version")),
        feature_version=str(_require_key(proj_raw, "project", "feature_version")),
        policy_version=str(_require_key(proj_raw, "project", "policy_version")),
        seed=int(_require_key(proj_raw, "project", "seed")),
    )

    data = DataConfig(
        n_applicants=int(_require_key(data_raw, "data", "n_applicants")),
        start_date=str(_require_key(data_raw, "data", "start_date")),
        end_date=str(_require_key(data_raw, "data", "end_date")),
        observation_window_days=int(_require_key(data_raw, "data", "observation_window_days")),
        as_of_cutoff_date=str(_require_key(data_raw, "data", "as_of_cutoff_date")),
        missing_rate=float(_require_key(data_raw, "data", "missing_rate")),
        shift_start_date=str(_require_key(data_raw, "data", "shift_start_date")),
        shift_magnitude=float(_require_key(data_raw, "data", "shift_magnitude")),
    )
    if data.n_applicants < 20:
        raise ConfigurationError("data.n_applicants must be >= 20.")
    if not (0.0 <= data.missing_rate < 0.8):
        raise ConfigurationError("data.missing_rate must be in [0.0, 0.8).")
    if data.observation_window_days <= 0:
        raise ConfigurationError("data.observation_window_days must be positive.")

    temporal_split = TemporalSplitConfig(
        train_end_date=str(_require_key(split_raw, "temporal_split", "train_end_date")),
        calibration_end_date=str(_require_key(split_raw, "temporal_split", "calibration_end_date")),
        test_end_date=str(_require_key(split_raw, "temporal_split", "test_end_date")),
        require_label_maturity=bool(split_raw.get("require_label_maturity", True)),
    )
    if not (
        temporal_split.train_end_date
        < temporal_split.calibration_end_date
        < temporal_split.test_end_date
    ):
        raise ConfigurationError(
            "Temporal splits must satisfy train_end_date < calibration_end_date < test_end_date.",
            details={
                "train_end_date": temporal_split.train_end_date,
                "calibration_end_date": temporal_split.calibration_end_date,
                "test_end_date": temporal_split.test_end_date,
            },
        )

    lr_raw = _require_section(models_raw, "logistic_regression")
    bt_raw = _require_section(models_raw, "boosted_trees")
    backend = str(bt_raw.get("backend", "sklearn_hist_gb"))
    if backend not in SUPPORTED_BOOSTED_BACKENDS:
        raise UnsupportedInputError(
            f"Unsupported boosted_trees backend '{backend}'. Supported: {SUPPORTED_BOOSTED_BACKENDS}",
            details={"backend": backend},
        )
    primary_model = str(models_raw.get("primary_model", "boosted_trees"))
    if primary_model not in SUPPORTED_PRIMARY_MODELS:
        raise UnsupportedInputError(
            f"Unsupported primary_model '{primary_model}'. Supported: {SUPPORTED_PRIMARY_MODELS}",
            details={"primary_model": primary_model},
        )

    models = ModelsConfig(
        logistic_regression=LogisticConfig(
            C=float(lr_raw.get("C", 1.0)),
            max_iter=int(lr_raw.get("max_iter", 1000)),
            class_weight=lr_raw.get("class_weight"),
        ),
        boosted_trees=BoostedConfig(
            backend=backend,
            max_iter=int(bt_raw.get("max_iter", 180)),
            max_depth=int(bt_raw.get("max_depth", 4)),
            learning_rate=float(bt_raw.get("learning_rate", 0.05)),
            min_samples_leaf=int(bt_raw.get("min_samples_leaf", 30)),
            l2_regularization=float(bt_raw.get("l2_regularization", 1.0)),
        ),
        primary_model=primary_model,
    )

    cal_method = str(cal_raw.get("method", "isotonic"))
    if cal_method not in SUPPORTED_CALIBRATION_METHODS:
        raise UnsupportedInputError(
            f"Unsupported calibration method '{cal_method}'. Supported: {SUPPORTED_CALIBRATION_METHODS}",
            details={"method": cal_method},
        )
    comp_methods = [str(m) for m in cal_raw.get("comparison_methods", ["sigmoid", "isotonic"])]
    for m in comp_methods:
        if m not in SUPPORTED_CALIBRATION_METHODS:
            raise UnsupportedInputError(
                f"Unsupported calibration comparison method '{m}'.",
                details={"method": m},
            )
    calibration = CalibrationConfig(
        method=cal_method,
        comparison_methods=comp_methods,
        n_bins=int(cal_raw.get("n_bins", 10)),
        clip_epsilon=float(cal_raw.get("clip_epsilon", 1e-4)),
        monotonicity_tolerance=float(cal_raw.get("monotonicity_tolerance", 1e-6)),
    )

    bc_raw = _require_section(eval_raw, "baseline_comparison")
    evaluation = EvaluationConfig(
        min_subgroup_sample_size=int(_require_key(eval_raw, "evaluation", "min_subgroup_sample_size")),
        subgroup_columns=[str(c) for c in _require_key(eval_raw, "evaluation", "subgroup_columns")],
        psi_bins=int(_require_key(eval_raw, "evaluation", "psi_bins")),
        psi_warning_threshold=float(_require_key(eval_raw, "evaluation", "psi_warning_threshold")),
        psi_alert_threshold=float(_require_key(eval_raw, "evaluation", "psi_alert_threshold")),
        decision_threshold=float(_require_key(eval_raw, "evaluation", "decision_threshold")),
        baseline_comparison=BaselineComparisonConfig(
            min_auroc_improvement_or_parity=float(bc_raw.get("min_auroc_improvement_or_parity", -0.02)),
            max_brier_degradation=float(bc_raw.get("max_brier_degradation", 0.015)),
            max_ece_calibrated=float(bc_raw.get("max_ece_calibrated", 0.05)),
        ),
    )
    if evaluation.min_subgroup_sample_size < 1:
        raise ConfigurationError("evaluation.min_subgroup_sample_size must be >= 1.")

    automated_allowed = bool(_require_key(policy_raw, "policy", "automated_decision_allowed"))
    if automated_allowed:
        raise UnsupportedInputError(
            "Automated lending decisions are strictly prohibited by platform policy "
            "(policy.automated_decision_allowed must be false).",
            details={"automated_decision_allowed": True},
        )
    raw_bands = _require_key(policy_raw, "policy", "risk_bands")
    if not isinstance(raw_bands, list) or len(raw_bands) < 2:
        raise ConfigurationError("policy.risk_bands must be a list with at least 2 bands.")

    bands: List[RiskBandConfig] = []
    prev_max = 0.0
    for idx, b in enumerate(raw_bands):
        max_pd = float(b["max_pd"])
        if max_pd <= prev_max or max_pd > 1.0:
            raise ConfigurationError(
                f"policy.risk_bands[{idx}].max_pd ({max_pd}) must be strictly increasing in (0, 1].",
                details={"index": idx, "max_pd": max_pd, "previous_max_pd": prev_max},
            )
        prev_max = max_pd
        bands.append(
            RiskBandConfig(
                name=str(b["name"]),
                max_pd=max_pd,
                queue_priority=str(b["queue_priority"]),
                review_sla_hours=int(b["review_sla_hours"]),
                recommended_action=str(b["recommended_action"]),
            )
        )
    if abs(bands[-1].max_pd - 1.0) > 1e-9:
        raise ConfigurationError("The final risk band in policy.risk_bands must have max_pd == 1.0.")

    policy = PolicyConfig(
        automated_decision_allowed=False,
        risk_bands=bands,
    )

    explainability = ExplainabilityConfig(
        top_k_factors=int(exp_raw.get("top_k_factors", 5)),
        use_shap_if_available=bool(exp_raw.get("use_shap_if_available", False)),
    )

    paths = PathsConfig(
        artifacts_dir=str(_require_key(paths_raw, "paths", "artifacts_dir")),
        reports_dir=str(_require_key(paths_raw, "paths", "reports_dir")),
        review_store_path=str(_require_key(paths_raw, "paths", "review_store_path")),
    )

    return PlatformConfig(
        project=project,
        data=data,
        temporal_split=temporal_split,
        models=models,
        calibration=calibration,
        evaluation=evaluation,
        policy=policy,
        explainability=explainability,
        paths=paths,
    )
