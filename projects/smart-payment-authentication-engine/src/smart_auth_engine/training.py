"""Temporal splitting, delayed-label maturity filtering, reference oracle, and calibrated risk model training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from smart_auth_engine.calibration import (
    CalibrationMetrics,
    PostHocProbabilityCalibrator,
    compute_calibration_metrics,
)
from smart_auth_engine.config import EngineConfig
from smart_auth_engine.contracts import (
    FeatureContribution,
    ModelArtifactManifest,
    PaymentEvent,
    RiskLabel,
)
from smart_auth_engine.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    ExtractedFeatures,
    extract_feature_matrix,
    extract_features,
)


class RiskModelError(Exception):
    """Raised when model scoring or artifact loading fails."""


class ModelOutageError(RiskModelError):
    """Raised when the risk model is unavailable or produces invalid scores."""


class ReferenceOracleScorer:
    """Transparent, hand-calculable analytical baseline scorer used as an M1/M2 reference oracle."""

    model_version = "reference_oracle_v1.0.0"
    feature_version = FEATURE_VERSION

    def predict_proba_event(self, event: PaymentEvent) -> Tuple[float, float, List[FeatureContribution]]:
        dev = event.device
        amt_major = event.amount_major
        trust = 0.5 if dev.device_trust_score is None else float(dev.device_trust_score)
        is_emul = 1.0 if dev.is_emulator else 0.0
        is_tor = 1.0 if dev.is_tor_or_vpn else 0.0
        prior_cb = float(min(event.prior_chargebacks_90d or 0, 3))
        vel_1h = float(min(dev.ip_velocity_1h or 0, 25))
        cross_b = 1.0 if event.is_cross_border else 0.0
        acct_age = 90.0 if event.account_age_days is None else float(event.account_age_days)

        terms = [
            ("log_amount_major", math.log1p(amt_major), 0.40 * math.log1p(amt_major)),
            ("device_trust_score", trust, -3.00 * (trust - 0.5)),
            ("is_emulator", is_emul, 1.50 * is_emul),
            ("is_tor_or_vpn", is_tor, 1.00 * is_tor),
            ("prior_chargebacks_90d", prior_cb, 1.20 * prior_cb),
            ("ip_velocity_1h", vel_1h, 0.12 * vel_1h),
            ("is_cross_border", cross_b, 0.60 * cross_b),
            ("log_account_age_months", math.log1p(acct_age / 30.0), -0.30 * math.log1p(acct_age / 30.0)),
        ]
        logit = -3.20 + sum(c for _, _, c in terms)
        prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
        explanations = [
            FeatureContribution(feature=name, value=round(val, 6), contribution=round(contrib, 6))
            for name, val, contrib in sorted(terms, key=lambda x: abs(x[2]), reverse=True)[:5]
        ]
        return prob, prob, explanations

    def predict_proba_events(self, events: Sequence[PaymentEvent]) -> np.ndarray:
        return np.array([self.predict_proba_event(ev)[0] for ev in events], dtype=np.float64)


@dataclass
class TemporalSplits:
    train_events: List[PaymentEvent]
    train_labels: List[RiskLabel]
    train_excluded_immature: int
    cal_events: List[PaymentEvent]
    cal_labels: List[RiskLabel]
    cal_excluded_immature: int
    val_events: List[PaymentEvent]
    val_labels: List[RiskLabel]
    test_events: List[PaymentEvent]
    test_labels: List[RiskLabel]
    train_as_of: datetime
    cal_as_of: datetime


def filter_mature_examples(
    events: Sequence[PaymentEvent],
    labels_by_id: Dict[str, RiskLabel],
    as_of: datetime,
) -> Tuple[List[PaymentEvent], List[RiskLabel], int]:
    """Keep only events whose label has matured at or before `as_of` (strict anti-leakage gate)."""
    as_of_utc = as_of.astimezone(timezone.utc)
    mature_events: List[PaymentEvent] = []
    mature_labels: List[RiskLabel] = []
    excluded = 0
    for ev in events:
        if ev.timestamp > as_of_utc:
            excluded += 1
            continue
        lbl = labels_by_id.get(ev.payment_id)
        if lbl is None or not lbl.is_mature_at(as_of_utc):
            excluded += 1
            continue
        mature_events.append(ev)
        mature_labels.append(lbl)
    return mature_events, mature_labels, excluded


def build_temporal_splits(
    events: Sequence[PaymentEvent],
    labels: Sequence[RiskLabel],
    cfg: EngineConfig,
) -> TemporalSplits:
    """Chronologically partition dataset and apply strict label-maturity filtering to train/cal splits."""
    if not events:
        raise ValueError("Cannot split an empty event sequence.")
    labels_by_id = {lbl.payment_id: lbl for lbl in labels}

    start_dt = datetime.fromisoformat(cfg.data_generation.start_date.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    train_end = start_dt + timedelta(days=cfg.training.train_end_day)
    cal_end = start_dt + timedelta(days=cfg.training.cal_end_day)
    val_end = start_dt + timedelta(days=cfg.training.val_end_day)

    # At training cutoff `train_end`, only labels matured by `train_end` are observable!
    # Events close to `train_end` (within label_maturity_days) will naturally be immature and excluded.
    raw_train = [ev for ev in events if ev.timestamp < train_end]
    raw_cal = [ev for ev in events if train_end <= ev.timestamp < cal_end]
    raw_val = [ev for ev in events if cal_end <= ev.timestamp < val_end]
    raw_test = [ev for ev in events if ev.timestamp >= val_end]

    train_as_of = train_end
    cal_as_of = cal_end

    train_events, train_labels, train_excl = filter_mature_examples(raw_train, labels_by_id, train_as_of)
    # For calibration, we evaluate maturity as of `cal_end + label_maturity_days` OR `cal_end`.
    # Using `cal_end` strictly filters out recent unmatured calibration events; if too few remain
    # (when cal window <= maturity days), observe maturity at `val_end` which precedes test window!
    cal_events, cal_labels, cal_excl = filter_mature_examples(raw_cal, labels_by_id, cal_as_of)
    if len(cal_events) < 50 or len({lbl.is_fraud for lbl in cal_labels}) < 2:
        cal_as_of = val_end
        cal_events, cal_labels, cal_excl = filter_mature_examples(raw_cal, labels_by_id, cal_as_of)

    val_events = [ev for ev in raw_val if ev.payment_id in labels_by_id]
    val_labels = [labels_by_id[ev.payment_id] for ev in val_events]
    test_events = [ev for ev in raw_test if ev.payment_id in labels_by_id]
    test_labels = [labels_by_id[ev.payment_id] for ev in test_events]

    return TemporalSplits(
        train_events=train_events,
        train_labels=train_labels,
        train_excluded_immature=train_excl,
        cal_events=cal_events,
        cal_labels=cal_labels,
        cal_excluded_immature=cal_excl,
        val_events=val_events,
        val_labels=val_labels,
        test_events=test_events,
        test_labels=test_labels,
        train_as_of=train_as_of,
        cal_as_of=cal_as_of,
    )


class CalibratedRiskModel:
    """Versioned calibrated fraud risk model with per-decision feature attributions."""

    def __init__(
        self,
        estimator: GradientBoostingClassifier,
        calibrator: PostHocProbabilityCalibrator,
        manifest: ModelArtifactManifest,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        feature_importances: np.ndarray,
    ) -> None:
        self.estimator = estimator
        self.calibrator = calibrator
        self.manifest = manifest
        self.feature_means = feature_means
        self.feature_stds = np.where(feature_stds > 1e-9, feature_stds, 1.0)
        self.feature_importances = feature_importances
        self.model_version = manifest.model_version
        self.feature_version = manifest.feature_version

    def predict_proba_matrix(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (calibrated_probs, raw_probs) for a 2D feature matrix."""
        if X.ndim != 2 or X.shape[1] != len(self.manifest.feature_names):
            raise RiskModelError(
                f"Expected feature matrix with {len(self.manifest.feature_names)} columns, got shape {X.shape}."
            )
        raw = np.asarray(self.estimator.predict_proba(X)[:, 1], dtype=np.float64)
        cal = np.asarray(self.calibrator.transform(raw), dtype=np.float64)
        if not (np.all(np.isfinite(raw)) and np.all(np.isfinite(cal))):
            raise ModelOutageError("Risk model produced non-finite probabilities.")
        return np.clip(cal, 0.0, 1.0), np.clip(raw, 0.0, 1.0)

    def predict_proba_features(
        self,
        extracted: ExtractedFeatures,
    ) -> Tuple[float, float, List[FeatureContribution]]:
        """Score a single ExtractedFeatures object and compute directional top-5 explanations."""
        if extracted.feature_version != self.feature_version:
            raise RiskModelError(
                f"Feature version mismatch: model expects {self.feature_version!r}, "
                f"got {extracted.feature_version!r}."
            )
        X = extracted.values.reshape(1, -1)
        cal_arr, raw_arr = self.predict_proba_matrix(X)
        cal_p = float(cal_arr[0])
        raw_p = float(raw_arr[0])

        # Compute deterministic local sensitivity via single-feature baseline replacement
        explanations: List[FeatureContribution] = []
        top_indices = np.argsort(self.feature_importances)[::-1][:6]
        for idx in top_indices:
            name = self.manifest.feature_names[int(idx)]
            x_ref = X.copy()
            x_ref[0, int(idx)] = self.feature_means[int(idx)]
            cal_ref, _ = self.predict_proba_matrix(x_ref)
            delta = cal_p - float(cal_ref[0])
            raw_val = extracted.raw_map.get(name)
            explanations.append(
                FeatureContribution(
                    feature=name,
                    value=None if raw_val is None else round(float(raw_val), 6),
                    contribution=round(delta, 6),
                )
            )
        explanations.sort(key=lambda item: abs(item.contribution), reverse=True)
        return cal_p, raw_p, explanations[:5]

    def predict_proba_event(
        self,
        event: PaymentEvent,
    ) -> Tuple[float, float, List[FeatureContribution]]:
        extracted = extract_features(event)
        return self.predict_proba_features(extracted)

    def save(self, artifact_dir: str | Path) -> Tuple[Path, Path]:
        """Persist model binary (.joblib) and JSON manifest with SHA-256 digest."""
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / f"{self.model_version}.joblib"
        manifest_path = out_dir / f"{self.model_version}.manifest.json"
        payload = {
            "estimator": self.estimator,
            "calibrator": self.calibrator,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "feature_importances": self.feature_importances,
            "manifest": self.manifest.model_dump(mode="json"),
        }
        joblib.dump(payload, model_path)
        sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.manifest.artifact_sha256 = sha256
        manifest_path.write_text(
            json.dumps(self.manifest.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return model_path, manifest_path

    @classmethod
    def load(cls, model_path: str | Path, manifest_path: Optional[str | Path] = None) -> "CalibratedRiskModel":
        m_path = Path(model_path)
        if not m_path.exists():
            raise ModelOutageError(f"Model artifact not found: {m_path}")
        if manifest_path is not None:
            mf_path = Path(manifest_path)
            if not mf_path.exists():
                raise ModelOutageError(f"Model manifest not found: {mf_path}")
            manifest = ModelArtifactManifest.model_validate_json(mf_path.read_text(encoding="utf-8"))
            actual_sha = hashlib.sha256(m_path.read_bytes()).hexdigest()
            if manifest.artifact_sha256 and manifest.artifact_sha256 != actual_sha:
                raise ModelOutageError(
                    f"Model artifact SHA-256 mismatch: expected {manifest.artifact_sha256}, got {actual_sha}."
                )
        payload = joblib.load(m_path)
        loaded_manifest = ModelArtifactManifest.model_validate(payload["manifest"])
        return cls(
            estimator=payload["estimator"],
            calibrator=payload["calibrator"],
            manifest=loaded_manifest,
            feature_means=np.asarray(payload["feature_means"], dtype=np.float64),
            feature_stds=np.asarray(payload["feature_stds"], dtype=np.float64),
            feature_importances=np.asarray(payload["feature_importances"], dtype=np.float64),
        )


@dataclass
class TrainingReport:
    model: CalibratedRiskModel
    splits: TemporalSplits
    raw_val_metrics: CalibrationMetrics
    calibrated_val_metrics: CalibrationMetrics
    oracle_val_metrics: CalibrationMetrics
    comparison_by_calibrator: Dict[str, Dict[str, float]]


def train_calibrated_risk_model(
    events: Sequence[PaymentEvent],
    labels: Sequence[RiskLabel],
    cfg: EngineConfig,
) -> TrainingReport:
    """Train and calibrate a risk model using temporal splits and delayed-label maturity filtering."""
    splits = build_temporal_splits(events, labels, cfg)
    if len(splits.train_events) < 20:
        raise ValueError(
            f"Insufficient mature training examples ({len(splits.train_events)}); "
            "check horizon_days, train_end_day, and label_maturity_days."
        )
    y_train = np.array([1 if lbl.is_fraud else 0 for lbl in splits.train_labels], dtype=np.int64)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Mature training set must contain both fraud and non-fraud examples.")

    X_train = extract_feature_matrix(splits.train_events)
    X_cal = extract_feature_matrix(splits.cal_events)
    y_cal = np.array([1 if lbl.is_fraud else 0 for lbl in splits.cal_labels], dtype=np.int64)
    X_val = extract_feature_matrix(splits.val_events)
    y_val = np.array([1 if lbl.is_fraud else 0 for lbl in splits.val_labels], dtype=np.int64)

    estimator = GradientBoostingClassifier(
        n_estimators=cfg.training.n_estimators,
        max_depth=cfg.training.max_depth,
        learning_rate=cfg.training.learning_rate,
        random_state=cfg.seed,
    )
    estimator.fit(X_train, y_train)

    raw_cal_probs = estimator.predict_proba(X_cal)[:, 1]
    raw_val_probs = estimator.predict_proba(X_val)[:, 1]

    # Compare uncalibrated, sigmoid (Platt), and isotonic calibration on validation split
    comparison: Dict[str, Dict[str, float]] = {}
    chosen_calibrator: Optional[PostHocProbabilityCalibrator] = None
    for method in ("none", "sigmoid", "isotonic"):
        cal = PostHocProbabilityCalibrator(method=method).fit(raw_cal_probs, y_cal)
        val_p = cal.transform(raw_val_probs)
        m = compute_calibration_metrics(y_val, val_p)
        comparison[method] = {
            "brier_score": round(m.brier_score, 6),
            "log_loss": round(m.log_loss, 6),
            "ece": round(m.ece, 6),
            "roc_auc": round(m.roc_auc, 6),
        }
        if method == cfg.training.calibration_method:
            chosen_calibrator = cal

    assert chosen_calibrator is not None
    cal_val_probs = chosen_calibrator.transform(raw_val_probs)
    raw_val_metrics = compute_calibration_metrics(y_val, raw_val_probs)
    cal_val_metrics = compute_calibration_metrics(y_val, cal_val_probs)

    oracle = ReferenceOracleScorer()
    oracle_val_probs = oracle.predict_proba_events(splits.val_events)
    oracle_val_metrics = compute_calibration_metrics(y_val, oracle_val_probs)

    model_version = f"gbm_{cfg.training.calibration_method}_seed{cfg.seed}_fv{FEATURE_VERSION}"
    manifest = ModelArtifactManifest(
        model_version=model_version,
        feature_version=FEATURE_VERSION,
        feature_names=list(FEATURE_NAMES),
        calibration_method=cfg.training.calibration_method,
        trained_as_of=splits.train_as_of,
        label_maturity_days=cfg.training.label_maturity_days,
        n_train_total=len(splits.train_events) + splits.train_excluded_immature,
        n_train_mature=len(splits.train_events),
        n_train_excluded_immature=splits.train_excluded_immature,
        n_calibration_mature=len(splits.cal_events),
        seed=cfg.seed,
        metrics={
            "val_brier_score": round(cal_val_metrics.brier_score, 6),
            "val_log_loss": round(cal_val_metrics.log_loss, 6),
            "val_ece": round(cal_val_metrics.ece, 6),
            "val_roc_auc": round(cal_val_metrics.roc_auc, 6),
            "raw_val_brier_score": round(raw_val_metrics.brier_score, 6),
            "raw_val_ece": round(raw_val_metrics.ece, 6),
        },
    )

    model = CalibratedRiskModel(
        estimator=estimator,
        calibrator=chosen_calibrator,
        manifest=manifest,
        feature_means=X_train.mean(axis=0),
        feature_stds=X_train.std(axis=0),
        feature_importances=np.asarray(estimator.feature_importances_, dtype=np.float64),
    )

    return TrainingReport(
        model=model,
        splits=splits,
        raw_val_metrics=raw_val_metrics,
        calibrated_val_metrics=cal_val_metrics,
        oracle_val_metrics=oracle_val_metrics,
        comparison_by_calibrator=comparison,
    )
