"""Calibrated dispute classifiers, expected loss economics, and decision explanations."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chargeback_engine.config import EconomicsConfig, ModelConfig


def compute_loss_given_dispute(
    amount_usd: float, economics: EconomicsConfig
) -> float:
    """Loss incurred if a transaction results in a dispute (USD)."""
    return (
        float(amount_usd) * (1.0 - economics.expected_recovery_rate)
        + economics.chargeback_fee_usd
    )


def compute_expected_loss(
    prob: float, amount_usd: float, economics: EconomicsConfig
) -> float:
    """Expected dispute cost = calibrated_prob * LGD (USD)."""
    return float(prob) * compute_loss_given_dispute(amount_usd, economics)


def compute_expected_review_savings(
    prob: float, amount_usd: float, economics: EconomicsConfig
) -> Tuple[float, float]:
    """Return (expected_gross_savings_usd, expected_net_savings_usd) of manual review."""
    el = compute_expected_loss(prob, amount_usd, economics)
    gross = el * economics.review_prevention_effectiveness
    net = gross - economics.review_cost_usd
    return gross, net


def compute_expected_loss_vectorized(
    probs: np.ndarray, amounts_usd: np.ndarray, economics: EconomicsConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized (lgd, expected_loss, expected_gross_savings, expected_net_savings)."""
    lgd = amounts_usd * (1.0 - economics.expected_recovery_rate) + economics.chargeback_fee_usd
    el = probs * lgd
    gross = el * economics.review_prevention_effectiveness
    net = gross - economics.review_cost_usd
    return lgd, el, gross, net


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins in [0, 1]."""
    if len(y_true) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(y_true))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            avg_conf = float(y_prob[mask].mean())
            avg_acc = float(y_true[mask].mean())
            ece += (cnt / n) * abs(avg_acc - avg_conf)
    return float(ece)


def compute_precision_recall_at_budget(
    y_true: np.ndarray,
    ranking_scores: np.ndarray,
    amounts_usd: np.ndarray,
    economics: EconomicsConfig,
    budget_fraction: float,
) -> Dict[str, float]:
    """Evaluate precision, recall, and realized loss capture at top-K review budget."""
    n = len(y_true)
    if n == 0:
        return {
            "reviewed_count": 0.0,
            "precision_at_budget": 0.0,
            "recall_at_budget": 0.0,
            "loss_capture_at_budget": 0.0,
        }
    k = max(1, int(math.floor(n * budget_fraction)))
    order = np.argsort(-ranking_scores, kind="mergesort")
    top_idx = order[:k]

    positives_in_top = int(y_true[top_idx].sum())
    total_positives = int(y_true.sum())

    lgd = amounts_usd * (1.0 - economics.expected_recovery_rate) + economics.chargeback_fee_usd
    realized_loss = y_true * lgd
    total_realized_loss = float(realized_loss.sum())
    captured_loss = float(realized_loss[top_idx].sum())

    return {
        "reviewed_count": float(k),
        "precision_at_budget": float(positives_in_top / k) if k > 0 else 0.0,
        "recall_at_budget": (
            float(positives_in_top / total_positives) if total_positives > 0 else 0.0
        ),
        "loss_capture_at_budget": (
            float(captured_loss / total_realized_loss)
            if total_realized_loss > 0.0
            else 0.0
        ),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts_usd: np.ndarray,
    economics: EconomicsConfig,
    budget_fraction: float,
) -> Dict[str, float]:
    """Compute discrimination, calibration, expected-loss, and review-budget metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    amounts_usd = np.asarray(amounts_usd, dtype=np.float64)

    n = len(y_true)
    n_pos = int(y_true.sum())
    has_both_classes = (n_pos > 0) and (n_pos < n)

    pr_auc = float(average_precision_score(y_true, y_prob)) if n_pos > 0 else 0.0
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else 0.5
    brier = float(brier_score_loss(y_true, y_prob)) if n > 0 else 0.0
    ll = float(log_loss(y_true, y_prob, labels=[0, 1])) if n > 0 else 0.0
    ece = compute_ece(y_true, y_prob)

    lgd, exp_loss, _, exp_net_savings = compute_expected_loss_vectorized(
        y_prob, amounts_usd, economics
    )
    realized_loss = y_true * lgd
    total_exp_loss = float(exp_loss.sum())
    total_realized_loss = float(realized_loss.sum())
    loss_ratio = (
        float(total_exp_loss / total_realized_loss)
        if total_realized_loss > 0.0
        else 0.0
    )

    obs_rate = float(n_pos / n) if n > 0 else 0.0
    mean_pred = float(y_prob.mean()) if n > 0 else 0.0

    prob_budget = compute_precision_recall_at_budget(
        y_true, y_prob, amounts_usd, economics, budget_fraction
    )
    value_budget = compute_precision_recall_at_budget(
        y_true, exp_net_savings, amounts_usd, economics, budget_fraction
    )

    return {
        "n_samples": float(n),
        "n_positives": float(n_pos),
        "observed_dispute_rate": round(obs_rate, 6),
        "mean_predicted_prob": round(mean_pred, 6),
        "calibration_ratio": round(mean_pred / obs_rate, 6) if obs_rate > 0 else 0.0,
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc_auc, 6),
        "brier_score": round(brier, 6),
        "log_loss": round(ll, 6),
        "ece": round(ece, 6),
        "total_expected_loss_usd": round(total_exp_loss, 2),
        "total_realized_loss_usd": round(total_realized_loss, 2),
        "expected_to_realized_loss_ratio": round(loss_ratio, 6),
        "precision_at_review_budget_by_prob": round(
            prob_budget["precision_at_budget"], 6
        ),
        "recall_at_review_budget_by_prob": round(prob_budget["recall_at_budget"], 6),
        "loss_capture_at_review_budget_by_prob": round(
            prob_budget["loss_capture_at_budget"], 6
        ),
        "precision_at_review_budget_by_value": round(
            value_budget["precision_at_budget"], 6
        ),
        "recall_at_review_budget_by_value": round(value_budget["recall_at_budget"], 6),
        "loss_capture_at_review_budget_by_value": round(
            value_budget["loss_capture_at_budget"], 6
        ),
    }


@dataclass
class CalibratedDisputeModel:
    """Wrapper combining a base binary classifier with chronological post-hoc calibration."""

    model_id: str
    model_family: str
    calibration_method: str
    feature_columns: Tuple[str, ...]
    base_estimator: Any
    calibrator: Any
    calibration_fallback_used: bool
    feature_medians: np.ndarray
    feature_importances: Dict[str, float]
    train_positive_rate: float
    calibration_positive_rate: float

    def predict_raw_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base_estimator.predict_proba(X)[:, 1]
        return np.clip(raw.astype(np.float64), 1e-6, 1.0 - 1e-6)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict_raw_proba(X)
        if self.calibrator is None:
            return raw
        if self.calibration_method == "sigmoid":
            logits = _logit(raw).reshape(-1, 1)
            cal = self.calibrator.predict_proba(logits)[:, 1]
            return np.clip(cal.astype(np.float64), 1e-6, 1.0 - 1e-6)
        if self.calibration_method == "isotonic":
            cal = self.calibrator.predict(raw)
            return np.clip(cal.astype(np.float64), 1e-6, 1.0 - 1e-6)
        return raw

    def explain_row(self, x_row: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Generate deterministic local feature attribution for a single transaction."""
        x_row = np.asarray(x_row, dtype=np.float64).reshape(1, -1)
        base_prob = float(self.predict_proba(x_row)[0])

        # For logistic pipeline, compute exact standardized linear log-odds contributions.
        if isinstance(self.base_estimator, Pipeline) and "clf" in self.base_estimator.named_steps:
            scaler = self.base_estimator.named_steps["scaler"]
            clf = self.base_estimator.named_steps["clf"]
            z = scaler.transform(x_row)[0]
            contribs = z * clf.coef_[0]
            ranked = np.argsort(-np.abs(contribs))[:top_k]
            return [
                {
                    "feature": self.feature_columns[int(idx)],
                    "value": round(float(x_row[0, int(idx)]), 4),
                    "contribution_log_odds": round(float(contribs[int(idx)]), 5),
                    "direction": "increases_risk" if contribs[int(idx)] >= 0 else "reduces_risk",
                }
                for idx in ranked
            ]

        # For tree ensembles, perturb top candidate features to training median.
        ordered_feats = sorted(
            self.feature_importances.items(), key=lambda kv: (-abs(kv[1]), kv[0])
        )
        candidate_names = [name for name, _ in ordered_feats[: max(12, top_k)]]
        name_to_idx = {name: i for i, name in enumerate(self.feature_columns)}

        deltas: List[Tuple[str, float, float]] = []
        for name in candidate_names:
            idx = name_to_idx[name]
            val = float(x_row[0, idx])
            med = float(self.feature_medians[idx])
            if abs(val - med) < 1e-9:
                deltas.append((name, val, 0.0))
                continue
            perturbed = x_row.copy()
            perturbed[0, idx] = med
            pert_prob = float(self.predict_proba(perturbed)[0])
            deltas.append((name, val, base_prob - pert_prob))

        deltas.sort(key=lambda item: (-abs(item[2]), item[0]))
        return [
            {
                "feature": name,
                "value": round(val, 4),
                "prob_delta_vs_median": round(delta, 6),
                "direction": "increases_risk" if delta >= 0 else "reduces_risk",
            }
            for name, val, delta in deltas[:top_k]
        ]

    def metadata_dict(self) -> Dict[str, Any]:
        top_imp = sorted(
            self.feature_importances.items(), key=lambda kv: (-abs(kv[1]), kv[0])
        )[:10]
        return {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "calibration_method": self.calibration_method,
            "calibration_fallback_used": self.calibration_fallback_used,
            "n_features": len(self.feature_columns),
            "train_positive_rate": round(self.train_positive_rate, 6),
            "calibration_positive_rate": round(self.calibration_positive_rate, 6),
            "top_feature_importances": [
                {"feature": k, "importance": round(float(v), 6)} for k, v in top_imp
            ],
        }


def _build_base_estimator(
    model_family: str, model_cfg: ModelConfig, seed: int
) -> Any:
    if model_family == "logistic":
        params = dict(model_cfg.logistic_params)
        clf = LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 500)),
            class_weight=model_cfg.class_weight,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    params = dict(model_cfg.hist_gbdt_params)
    return HistGradientBoostingClassifier(
        max_iter=int(params.get("max_iter", 120)),
        max_depth=int(params.get("max_depth", 4)),
        learning_rate=float(params.get("learning_rate", 0.06)),
        min_samples_leaf=int(params.get("min_samples_leaf", 20)),
        l2_regularization=float(params.get("l2_regularization", 1.0)),
        class_weight=model_cfg.class_weight,
        random_state=seed,
    )


def train_calibrated_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    feature_columns: Sequence[str],
    model_cfg: ModelConfig,
    *,
    model_family_override: Optional[str] = None,
    apply_calibration: bool = True,
    seed: int = 42,
    config_hash: str = "",
    feature_schema_hash: str = "",
) -> CalibratedDisputeModel:
    """Train a base classifier on train split and calibrate strictly on chronological cal split."""
    family = model_family_override or model_cfg.primary_model
    estimator = _build_base_estimator(family, model_cfg, seed=seed)
    estimator.fit(X_train, y_train)

    calibrator: Any = None
    fallback_used = False
    cal_method = model_cfg.calibration_method if apply_calibration else "none"

    if apply_calibration:
        raw_cal = np.clip(estimator.predict_proba(X_cal)[:, 1], 1e-6, 1.0 - 1e-6)
        n_pos_cal = int(y_cal.sum())
        n_neg_cal = int(len(y_cal) - n_pos_cal)

        if n_pos_cal < 2 or n_neg_cal < 2:
            fallback_used = True
            calibrator = None
        elif model_cfg.calibration_method == "sigmoid":
            platt = LogisticRegression(C=2.0, max_iter=500, random_state=seed)
            platt.fit(_logit(raw_cal).reshape(-1, 1), y_cal)
            calibrator = platt
        elif model_cfg.calibration_method == "isotonic":
            iso = IsotonicRegression(y_min=1e-5, y_max=1.0 - 1e-5, out_of_bounds="clip")
            iso.fit(raw_cal, y_cal)
            calibrator = iso

    feature_medians = np.median(X_train, axis=0)

    if family == "logistic" and isinstance(estimator, Pipeline):
        coefs = np.abs(estimator.named_steps["clf"].coef_[0])
        importances = {
            col: float(coefs[i]) for i, col in enumerate(feature_columns)
        }
    else:
        perm = permutation_importance(
            estimator,
            X_cal,
            y_cal,
            n_repeats=2,
            random_state=seed,
            scoring="average_precision" if int(y_cal.sum()) > 0 else "neg_brier_score",
        )
        importances = {
            col: float(perm.importances_mean[i])
            for i, col in enumerate(feature_columns)
        }

    id_payload = (
        f"{family}:{cal_method}:{seed}:{config_hash}:{feature_schema_hash}:"
        f"{len(y_train)}:{len(y_cal)}"
    )
    model_id = f"cbm_{hashlib.sha256(id_payload.encode('utf-8')).hexdigest()[:12]}"

    return CalibratedDisputeModel(
        model_id=model_id,
        model_family=family,
        calibration_method=cal_method,
        feature_columns=tuple(feature_columns),
        base_estimator=estimator,
        calibrator=calibrator,
        calibration_fallback_used=fallback_used,
        feature_medians=feature_medians,
        feature_importances=importances,
        train_positive_rate=float(y_train.mean()) if len(y_train) > 0 else 0.0,
        calibration_positive_rate=float(y_cal.mean()) if len(y_cal) > 0 else 0.0,
    )
