"""Credit risk probability-of-default baseline models.

Implements:
- LogisticRegressionPDModel (interpretable linear log-odds reference baseline)
- BoostedTreesPDModel (scikit-learn HistGradientBoostingClassifier by default,
  with optional XGBoost backend when installed)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from credit_risk.config import BoostedConfig, LogisticConfig, ModelsConfig
from credit_risk.errors import UnsupportedInputError


def _prob_to_log_odds(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probs, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


class BasePDModel(ABC):
    """Abstract typed interface for probability-of-default estimators."""

    model_id: str
    model_type: str
    backend: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BasePDModel":
        raise NotImplementedError

    @abstractmethod
    def predict_raw_proba(self, X: np.ndarray) -> np.ndarray:
        """Return uncalibrated 1-D default probability estimates in (0, 1)."""
        raise NotImplementedError

    def predict_log_odds(self, X: np.ndarray) -> np.ndarray:
        return _prob_to_log_odds(self.predict_raw_proba(X))

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError


class LogisticRegressionPDModel(BasePDModel):
    """Regularized logistic regression reference baseline."""

    def __init__(self, cfg: LogisticConfig, seed: int = 42, model_id: str = "lr_baseline_v1") -> None:
        self.cfg = cfg
        self.seed = seed
        self.model_id = model_id
        self.model_type = "logistic_regression"
        self.backend = "sklearn_logistic_regression"
        self.estimator = LogisticRegression(
            C=cfg.C,
            max_iter=cfg.max_iter,
            class_weight=cfg.class_weight,
            random_state=seed,
            solver="lbfgs",
        )
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionPDModel":
        self.estimator.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict_raw_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("LogisticRegressionPDModel must be fitted before prediction.")
        return np.asarray(self.estimator.predict_proba(X)[:, 1], dtype=np.float64)

    def predict_log_odds(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("LogisticRegressionPDModel must be fitted before prediction.")
        return np.asarray(self.estimator.decision_function(X), dtype=np.float64)

    def coefficients(self) -> Tuple[float, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted.")
        return float(self.estimator.intercept_[0]), np.asarray(self.estimator.coef_[0], dtype=np.float64)

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "backend": self.backend,
            "C": self.cfg.C,
            "max_iter": self.cfg.max_iter,
            "class_weight": self.cfg.class_weight,
            "seed": self.seed,
        }


class BoostedTreesPDModel(BasePDModel):
    """Gradient-boosted decision tree baseline (sklearn default, optional XGBoost)."""

    def __init__(self, cfg: BoostedConfig, seed: int = 42, model_id: str = "gbdt_baseline_v1") -> None:
        self.cfg = cfg
        self.seed = seed
        self.model_id = model_id
        self.model_type = "boosted_trees"
        self.backend = cfg.backend
        self.is_fitted_ = False

        if cfg.backend == "sklearn_hist_gb":
            self.estimator: Any = HistGradientBoostingClassifier(
                max_iter=cfg.max_iter,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                min_samples_leaf=cfg.min_samples_leaf,
                l2_regularization=cfg.l2_regularization,
                random_state=seed,
            )
        elif cfg.backend == "xgboost":
            try:
                import xgboost as xgb  # type: ignore
            except ImportError as exc:
                raise UnsupportedInputError(
                    "XGBoost backend requested but optional dependency 'xgboost' is not installed. "
                    "Install with `pip install xgboost` or set models.boosted_trees.backend='sklearn_hist_gb'.",
                    details={"backend": "xgboost", "optional_dependency_missing": True},
                ) from exc
            self.estimator = xgb.XGBClassifier(
                n_estimators=cfg.max_iter,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                min_child_weight=max(1, cfg.min_samples_leaf // 5),
                reg_lambda=cfg.l2_regularization,
                random_state=seed,
                eval_metric="logloss",
            )
        else:
            raise UnsupportedInputError(
                f"Unsupported boosted trees backend '{cfg.backend}'.",
                details={"backend": cfg.backend},
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BoostedTreesPDModel":
        self.estimator.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict_raw_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("BoostedTreesPDModel must be fitted before prediction.")
        return np.asarray(self.estimator.predict_proba(X)[:, 1], dtype=np.float64)

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "backend": self.backend,
            "max_iter": self.cfg.max_iter,
            "max_depth": self.cfg.max_depth,
            "learning_rate": self.cfg.learning_rate,
            "min_samples_leaf": self.cfg.min_samples_leaf,
            "l2_regularization": self.cfg.l2_regularization,
            "seed": self.seed,
        }


def train_baseline_models(
    X_train_std: np.ndarray,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    models_cfg: ModelsConfig,
    seed: int = 42,
) -> Dict[str, BasePDModel]:
    """Train both logistic regression (standardized) and boosted tree baselines."""
    lr_model = LogisticRegressionPDModel(models_cfg.logistic_regression, seed=seed)
    lr_model.fit(X_train_std, y_train)

    gb_model = BoostedTreesPDModel(models_cfg.boosted_trees, seed=seed)
    gb_model.fit(X_train_raw, y_train)

    return {
        "logistic_regression": lr_model,
        "boosted_trees": gb_model,
    }
