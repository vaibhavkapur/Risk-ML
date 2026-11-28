"""Anomaly, supervised, and calibrated ATO risk models with chronological validation.

Boundaries:
- Feature vectors are passed in as strictly causal arrays matching FEATURE_NAMES.
- Chronological split enforces label maturity: training uses ONLY events whose
  `label_available_at <= train_cutoff`.
- Probability calibration (Platt sigmoid or isotonic) is fitted strictly on the
  chronological calibration split after the training cutoff and before test evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import EngineConfig
from .features import FEATURE_NAMES, vector_to_array_row
from .schemas import FeatureVector, LoginSessionEvent, ModelArtifactError, ModelScore


def rule_oracle_score(features: Dict[str, float]) -> float:
    """Simple, transparent, hand-calculable reference baseline score in [0, 1].

    Established as the domain reference oracle before ML complexity is added.
    """
    score = 0.04
    score += 0.12 * float(features["is_new_device_for_user"])
    score += 0.28 * float(features["is_impossible_travel"])
    score += 0.24 * float(features["ip_reputation_risk"])
    score += 0.18 * (1.0 - float(features["device_trust_score"]))
    score += 0.16 * min(float(features["failed_attempts_1h"]) / 5.0, 1.0)
    score += 0.20 * min(float(features["sensitive_action_count"]) / 3.0, 1.0)
    score += 0.12 * float(features["auth_factor_downgrade"])
    score += 0.14 * float(features["credential_stuffing_burst"])
    score += 0.06 * min(float(features["hour_of_day_deviation"]) / 12.0, 1.0)
    score += 0.05 * float(features["profile_expired"])
    if float(features["is_shared_network_ip"]) >= 1.0 and float(features["credential_stuffing_burst"]) == 0.0:
        score -= 0.06
    return float(max(0.0, min(1.0, round(score, 6))))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) over uniform probability bins."""
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
        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(y_prob[mask]))
            avg_acc = float(np.mean(y_true[mask]))
            ece += (count / n) * abs(avg_acc - avg_conf)
    return float(round(ece, 6))


def recall_at_fixed_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fprs: Sequence[float] = (0.01, 0.02, 0.05),
) -> Dict[str, Dict[str, float]]:
    """Compute recall and operating threshold at each target false positive rate."""
    if len(np.unique(y_true)) < 2:
        return {
            f"{fpr:.2%}": {"target_fpr": float(fpr), "achieved_fpr": 0.0, "recall": 0.0, "threshold": 1.0}
            for fpr in target_fprs
        }
    fprs, tprs, thresholds = roc_curve(y_true, y_score)
    results: Dict[str, Dict[str, float]] = {}
    for target in target_fprs:
        valid_idx = np.where(fprs <= target)[0]
        if len(valid_idx) == 0:
            best_idx = 0
        else:
            best_idx = int(valid_idx[-1])
        thresh = float(thresholds[best_idx])
        if not math.isfinite(thresh):
            thresh = 1.0
        results[f"{target:.2%}"] = {
            "target_fpr": float(target),
            "achieved_fpr": round(float(fprs[best_idx]), 6),
            "recall": round(float(tprs[best_idx]), 6),
            "threshold": round(float(max(0.0, min(1.0, thresh))), 6),
        }
    return results


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fprs: Sequence[float] = (0.01, 0.02, 0.05),
) -> Dict[str, Any]:
    """Compute discrimination, calibration, and fixed-FPR recall metrics."""
    has_both = len(np.unique(y_true)) >= 2
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if has_both else 0.0
    brier = float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0))) if len(y_true) > 0 else 0.0
    ece = expected_calibration_error(y_true, np.clip(y_prob, 0.0, 1.0))
    return {
        "n_samples": int(len(y_true)),
        "positive_count": int(np.sum(y_true)),
        "positive_rate": round(float(np.mean(y_true)) if len(y_true) > 0 else 0.0, 6),
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "brier_score": round(brier, 6),
        "ece": round(ece, 6),
        "recall_at_fpr": recall_at_fixed_fpr(y_true, y_prob, target_fprs=target_fprs),
    }


@dataclass(frozen=True)
class ChronologicalSplitResult:
    train_indices: List[int]
    excluded_immature_train_indices: List[int]
    calibration_indices: List[int]
    test_indices: List[int]
    train_cutoff: datetime
    calibration_cutoff: datetime


def chronological_split_with_maturity(
    events: Sequence[LoginSessionEvent],
    train_fraction: float,
    calibration_fraction: float,
    label_maturity_hours: float,
) -> ChronologicalSplitResult:
    """Split chronologically sorted events into train, calibration, and test sets.

    Enforces label maturity on training data: an event in the training window is
    included as a labeled training row ONLY if its label is non-null AND its
    `label_available_at` (or `timestamp + label_maturity_hours` if not explicit)
    is `<= train_cutoff`. This prevents delayed-label leakage.
    """
    n = len(events)
    if n < 10:
        raise ValueError(f"Need at least 10 events for chronological split, got {n}")
    for i in range(1, n):
        if events[i].timestamp < events[i - 1].timestamp:
            raise ValueError("Events must be chronologically sorted before splitting")

    train_end = max(1, int(n * train_fraction))
    cal_end = max(train_end + 1, int(n * (train_fraction + calibration_fraction)))
    cal_end = min(cal_end, n - 1)

    train_cutoff = events[train_end - 1].timestamp.astimezone(timezone.utc)
    calibration_cutoff = events[cal_end - 1].timestamp.astimezone(timezone.utc)
    default_delay = timedelta(hours=label_maturity_hours)

    train_indices: List[int] = []
    excluded_immature: List[int] = []
    for idx in range(train_end):
        ev = events[idx]
        if ev.label is None:
            excluded_immature.append(idx)
            continue
        avail_at = (
            ev.label_available_at.astimezone(timezone.utc)
            if ev.label_available_at is not None
            else ev.timestamp.astimezone(timezone.utc) + default_delay
        )
        if avail_at <= train_cutoff:
            train_indices.append(idx)
        else:
            excluded_immature.append(idx)

    calibration_indices = [
        idx for idx in range(train_end, cal_end) if events[idx].label is not None
    ]
    test_indices = [idx for idx in range(cal_end, n) if events[idx].label is not None]

    return ChronologicalSplitResult(
        train_indices=train_indices,
        excluded_immature_train_indices=excluded_immature,
        calibration_indices=calibration_indices,
        test_indices=test_indices,
        train_cutoff=train_cutoff,
        calibration_cutoff=calibration_cutoff,
    )


class ATORiskModelBundle:
    """Versioned ensemble of IsolationForest anomaly baseline + LogisticRegression +
    HistGradientBoostingClassifier + post-hoc probability calibrator.
    """

    def __init__(
        self,
        model_version: str,
        feature_version: str,
        feature_names: Tuple[str, ...],
        anomaly_pipeline: Pipeline,
        anomaly_min: float,
        anomaly_max: float,
        logistic_pipeline: Pipeline,
        gb_model: HistGradientBoostingClassifier,
        calibrator: Any,
        calibration_method: str,
        anomaly_blend_weight: float,
        trained_at: datetime,
        train_cutoff: datetime,
        calibration_cutoff: datetime,
        reference_score_distribution: List[float],
        reference_feature_stats: Dict[str, Dict[str, float]],
        reference_missingness_rate: float,
        linear_feature_weights: Dict[str, float],
        metadata: Dict[str, Any],
    ) -> None:
        self.model_version = model_version
        self.feature_version = feature_version
        self.feature_names = feature_names
        self.anomaly_pipeline = anomaly_pipeline
        self.anomaly_min = anomaly_min
        self.anomaly_max = anomaly_max
        self.logistic_pipeline = logistic_pipeline
        self.gb_model = gb_model
        self.calibrator = calibrator
        self.calibration_method = calibration_method
        self.anomaly_blend_weight = anomaly_blend_weight
        self.trained_at = trained_at
        self.train_cutoff = train_cutoff
        self.calibration_cutoff = calibration_cutoff
        self.reference_score_distribution = reference_score_distribution
        self.reference_feature_stats = reference_feature_stats
        self.reference_missingness_rate = reference_missingness_rate
        self.linear_feature_weights = linear_feature_weights
        self.metadata = metadata

    def _raw_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        # IsolationForest.decision_function: higher = more normal; invert so higher = more anomalous
        raw = -self.anomaly_pipeline.decision_function(X)
        span = max(self.anomaly_max - self.anomaly_min, 1e-6)
        scaled = (raw - self.anomaly_min) / span
        return np.clip(scaled, 0.0, 1.0)

    def _raw_blended_score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        anom = self._raw_anomaly_score(X)
        lr_prob = self.logistic_pipeline.predict_proba(X)[:, 1]
        gb_prob = self.gb_model.predict_proba(X)[:, 1]
        sup_prob = 0.35 * lr_prob + 0.65 * gb_prob
        blended = (1.0 - self.anomaly_blend_weight) * sup_prob + self.anomaly_blend_weight * anom
        return np.clip(blended, 0.0, 1.0), sup_prob, lr_prob, anom

    def _apply_calibrator(self, raw_scores: np.ndarray) -> np.ndarray:
        eps = 1e-6
        clipped = np.clip(raw_scores, eps, 1.0 - eps)
        if self.calibration_method == "isotonic":
            return np.clip(self.calibrator.predict(clipped), 0.0, 1.0)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return np.clip(self.calibrator.predict_proba(logits)[:, 1], 0.0, 1.0)

    def predict_batch(self, feature_vectors: Sequence[FeatureVector]) -> List[ModelScore]:
        if not feature_vectors:
            return []
        X = np.asarray([vector_to_array_row(fv) for fv in feature_vectors], dtype=float)
        blended, sup_prob, lr_prob, anom = self._raw_blended_score(X)
        calibrated = self._apply_calibrator(blended)
        return [
            ModelScore(
                calibrated_risk_score=round(float(calibrated[i]), 6),
                supervised_probability=round(float(sup_prob[i]), 6),
                logistic_probability=round(float(lr_prob[i]), 6),
                anomaly_score=round(float(anom[i]), 6),
                model_version=self.model_version,
            )
            for i in range(len(feature_vectors))
        ]

    def predict_one(self, fv: FeatureVector) -> ModelScore:
        return self.predict_batch([fv])[0]

    def is_stale(self, now: datetime, max_age_days: int) -> bool:
        age = (now.astimezone(timezone.utc) - self.trained_at.astimezone(timezone.utc)).total_seconds()
        return age > max_age_days * 86400.0

    def save(self, artifact_path: str | Path) -> str:
        """Persist model bundle plus a JSON manifest containing its SHA-256 digest."""
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "artifact_file": path.name,
            "sha256": digest,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "feature_names": list(self.feature_names),
            "calibration_method": self.calibration_method,
            "trained_at": self.trained_at.isoformat(),
            "train_cutoff": self.train_cutoff.isoformat(),
            "calibration_cutoff": self.calibration_cutoff.isoformat(),
            "metadata": self.metadata,
        }
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return digest

    @classmethod
    def load(cls, artifact_path: str | Path, verify_manifest: bool = True) -> "ATORiskModelBundle":
        path = Path(artifact_path)
        if not path.exists():
            raise ModelArtifactError(f"Model artifact not found: {path}")
        if verify_manifest:
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
            if not manifest_path.exists():
                raise ModelArtifactError(f"Model manifest not found: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != manifest.get("sha256"):
                raise ModelArtifactError(
                    f"Model artifact checksum mismatch for {path}: expected {manifest.get('sha256')}, got {actual_hash}"
                )
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            raise ModelArtifactError(f"Failed to load model artifact {path}: {exc}") from exc
        if not isinstance(bundle, cls):
            raise ModelArtifactError(f"Artifact {path} is not a valid ATORiskModelBundle")
        if tuple(bundle.feature_names) != FEATURE_NAMES:
            raise ModelArtifactError("Model artifact feature_names do not match current FEATURE_NAMES")
        return bundle


def train_model_bundle(
    events: Sequence[LoginSessionEvent],
    feature_vectors: Sequence[FeatureVector],
    config: EngineConfig,
    trained_at: datetime | None = None,
) -> Tuple[ATORiskModelBundle, ChronologicalSplitResult, Dict[str, Any]]:
    """Train anomaly and supervised baselines with chronological validation and calibration."""
    if len(events) != len(feature_vectors):
        raise ValueError("events and feature_vectors must have identical lengths")

    split = chronological_split_with_maturity(
        events=events,
        train_fraction=config.training.train_fraction,
        calibration_fraction=config.training.calibration_fraction,
        label_maturity_hours=config.training.label_maturity_hours,
    )
    if len(split.train_indices) < 20:
        raise ValueError(
            f"Insufficient mature labeled training rows ({len(split.train_indices)}); check label_maturity_hours"
        )

    X_all = np.asarray([vector_to_array_row(fv) for fv in feature_vectors], dtype=float)
    y_all = np.asarray([ev.label if ev.label is not None else -1 for ev in events], dtype=int)

    X_train = X_all[split.train_indices]
    y_train = y_all[split.train_indices]
    X_cal = X_all[split.calibration_indices]
    y_cal = y_all[split.calibration_indices]
    X_test = X_all[split.test_indices]
    y_test = y_all[split.test_indices]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_cal)) < 2:
        raise ValueError("Both train and calibration splits must contain benign and ATO examples")

    # 1. Anomaly baseline (unsupervised on all training-window events including benign majority)
    anomaly_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "iforest",
                IsolationForest(
                    n_estimators=config.training.isolation_forest_estimators,
                    contamination=config.training.isolation_forest_contamination,
                    random_state=config.seed,
                ),
            ),
        ]
    )
    anomaly_pipeline.fit(X_train)
    raw_train_anom = -anomaly_pipeline.decision_function(X_train)
    anom_min = float(np.percentile(raw_train_anom, 1))
    anom_max = float(np.percentile(raw_train_anom, 99))

    # 2. Supervised linear interpretable baseline (LogisticRegression)
    logistic_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=0.8,
                    max_iter=500,
                    class_weight="balanced",
                    random_state=config.seed,
                ),
            ),
        ]
    )
    logistic_pipeline.fit(X_train, y_train)
    lr_coefs = logistic_pipeline.named_steps["lr"].coef_[0]
    linear_weights = {
        FEATURE_NAMES[i]: round(float(lr_coefs[i]), 6) for i in range(len(FEATURE_NAMES))
    }

    # 3. Supervised nonlinear baseline (HistGradientBoostingClassifier)
    gb_model = HistGradientBoostingClassifier(
        max_iter=config.training.gb_max_iter,
        max_depth=config.training.gb_max_depth,
        learning_rate=config.training.gb_learning_rate,
        l2_regularization=1.0,
        random_state=config.seed,
    )
    gb_model.fit(X_train, y_train)

    # 4. Fit post-hoc calibrator strictly on chronological calibration split
    def _blend(X_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_a = -anomaly_pipeline.decision_function(X_mat)
        a_scaled = np.clip((raw_a - anom_min) / max(anom_max - anom_min, 1e-6), 0.0, 1.0)
        lr_p = logistic_pipeline.predict_proba(X_mat)[:, 1]
        gb_p = gb_model.predict_proba(X_mat)[:, 1]
        sup_p = 0.35 * lr_p + 0.65 * gb_p
        w = config.training.anomaly_blend_weight
        return np.clip((1.0 - w) * sup_p + w * a_scaled, 0.0, 1.0), sup_p, lr_p, a_scaled

    cal_blended, _, _, _ = _blend(X_cal)
    eps = 1e-6
    if config.training.calibration_method == "isotonic":
        calibrator: Any = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(np.clip(cal_blended, eps, 1.0 - eps), y_cal)
    else:
        cal_clipped = np.clip(cal_blended, eps, 1.0 - eps)
        cal_logits = np.log(cal_clipped / (1.0 - cal_clipped)).reshape(-1, 1)
        calibrator = LogisticRegression(C=10.0, max_iter=300, random_state=config.seed)
        calibrator.fit(cal_logits, y_cal)

    # Reference baseline distributions from calibration split (for drift monitoring)
    if config.training.calibration_method == "isotonic":
        cal_probs = np.clip(calibrator.predict(np.clip(cal_blended, eps, 1.0 - eps)), 0.0, 1.0)
    else:
        cal_clipped = np.clip(cal_blended, eps, 1.0 - eps)
        cal_logits = np.log(cal_clipped / (1.0 - cal_clipped)).reshape(-1, 1)
        cal_probs = np.clip(calibrator.predict_proba(cal_logits)[:, 1], 0.0, 1.0)

    benign_train = X_train[y_train == 0]
    lr_scaler: StandardScaler = logistic_pipeline.named_steps["scaler"]
    ref_feature_stats: Dict[str, Dict[str, float]] = {}
    for idx, fname in enumerate(FEATURE_NAMES):
        col = X_cal[:, idx]
        b_col = benign_train[:, idx] if len(benign_train) > 0 else X_train[:, idx]
        ref_feature_stats[fname] = {
            "mean": round(float(np.mean(b_col)), 6),
            "std": round(float(max(np.std(b_col), 0.10)), 6),
            "lr_mean": round(float(lr_scaler.mean_[idx]), 6),
            "lr_scale": round(float(max(lr_scaler.scale_[idx], 0.10)), 6),
            "cal_mean": round(float(np.mean(col)), 6),
            "cal_p10": round(float(np.percentile(col, 10)), 6),
            "cal_p50": round(float(np.percentile(col, 50)), 6),
            "cal_p90": round(float(np.percentile(col, 90)), 6),
        }

    ref_missingness = float(
        np.mean([feature_vectors[i].values["missing_feature_ratio"] for i in split.calibration_indices])
    )

    now_ts = (trained_at or events[-1].timestamp).astimezone(timezone.utc)
    bundle = ATORiskModelBundle(
        model_version=config.versions.model_version,
        feature_version=config.versions.feature_version,
        feature_names=FEATURE_NAMES,
        anomaly_pipeline=anomaly_pipeline,
        anomaly_min=anom_min,
        anomaly_max=anom_max,
        logistic_pipeline=logistic_pipeline,
        gb_model=gb_model,
        calibrator=calibrator,
        calibration_method=config.training.calibration_method,
        anomaly_blend_weight=config.training.anomaly_blend_weight,
        trained_at=now_ts,
        train_cutoff=split.train_cutoff,
        calibration_cutoff=split.calibration_cutoff,
        reference_score_distribution=[round(float(x), 6) for x in cal_probs.tolist()],
        reference_feature_stats=ref_feature_stats,
        reference_missingness_rate=round(ref_missingness, 6),
        linear_feature_weights=linear_weights,
        metadata={
            "seed": config.seed,
            "train_rows": len(split.train_indices),
            "excluded_immature_train_rows": len(split.excluded_immature_train_indices),
            "calibration_rows": len(split.calibration_indices),
            "test_rows": len(split.test_indices),
            "label_maturity_hours": config.training.label_maturity_hours,
        },
    )

    # Evaluate all baselines on chronological test split
    test_fvs = [feature_vectors[i] for i in split.test_indices]
    test_scores = bundle.predict_batch(test_fvs)
    test_cal_probs = np.asarray([s.calibrated_risk_score for s in test_scores], dtype=float)
    test_sup_probs = np.asarray([s.supervised_probability for s in test_scores], dtype=float)
    test_lr_probs = np.asarray([s.logistic_probability for s in test_scores], dtype=float)
    test_anom_scores = np.asarray([s.anomaly_score for s in test_scores], dtype=float)
    test_oracle_scores = np.asarray([rule_oracle_score(fv.values) for fv in test_fvs], dtype=float)

    # Also compute uncalibrated blended score on test for calibration improvement comparison
    test_raw_blended, _, _, _ = bundle._raw_blended_score(X_test)

    eval_summary = {
        "split": {
            "train_rows": len(split.train_indices),
            "excluded_immature_train_rows": len(split.excluded_immature_train_indices),
            "calibration_rows": len(split.calibration_indices),
            "test_rows": len(split.test_indices),
            "train_cutoff": split.train_cutoff.isoformat(),
            "calibration_cutoff": split.calibration_cutoff.isoformat(),
        },
        "test_metrics": {
            "rule_oracle_baseline": compute_classification_metrics(
                y_test, test_oracle_scores, config.policy.target_fprs
            ),
            "anomaly_isolation_forest": compute_classification_metrics(
                y_test, test_anom_scores, config.policy.target_fprs
            ),
            "logistic_regression": compute_classification_metrics(
                y_test, test_lr_probs, config.policy.target_fprs
            ),
            "supervised_uncalibrated_blend": compute_classification_metrics(
                y_test, test_raw_blended, config.policy.target_fprs
            ),
            "calibrated_ensemble": compute_classification_metrics(
                y_test, test_cal_probs, config.policy.target_fprs
            ),
            "supervised_combined": compute_classification_metrics(
                y_test, test_sup_probs, config.policy.target_fprs
            ),
        },
    }
    return bundle, split, eval_summary
