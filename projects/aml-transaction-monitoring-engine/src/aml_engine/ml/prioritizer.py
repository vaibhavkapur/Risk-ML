"""ML alert prioritization with strict separation of features, training, calibration, policy, and inference.

Implements a deterministic, pure-Python L2-regularized Logistic Regression model
with Platt probability calibration, temporal train/calibration/test splitting,
label maturity enforcement, exact per-feature log-odds explainability, and
versioned JSON model artifact persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aml_engine.config import EngineConfig
from aml_engine.models import (
    Account,
    Entity,
    GraphFeatureVector,
    NormalizedTransaction,
    PrioritizationResult,
    ReasonContribution,
    ReviewPriority,
    RollingFeatureVector,
    ScenarioHit,
)
from aml_engine.validation import parse_utc_timestamp


FEATURE_NAMES: Tuple[str, ...] = (
    "rule_score",
    "num_scenarios_triggered",
    "is_layering_or_cycle",
    "is_cash_or_wire_ratio",
    "log_total_involved_usd",
    "entity_risk_tier_num",
    "is_high_risk_jurisdiction",
    "structuring_band_count_24h",
    "passthrough_ratio_24h",
    "velocity_ratio_vs_baseline",
    "fan_in_fan_out_ratio",
    "ego_density",
    "dormancy_days_log",
)

HIGH_RISK_JURISDICTIONS = {"PA", "CY", "VG", "HK"}
RISK_TIER_NUM = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0, "UNKNOWN": 0.6}


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    subject_account_id: str
    trigger_timestamp_utc: datetime
    trigger_timestamp_iso: str
    features: Dict[str, float]
    label: int  # 1 = true synthetic suspicious scenario, 0 = benign false positive
    rule_score: float


@dataclass(frozen=True)
class TemporalSplitResult:
    train: List[CandidateRecord]
    calibration: List[CandidateRecord]
    test: List[CandidateRecord]
    excluded_immature_train: int
    excluded_immature_cal: int
    train_end_iso: str
    cal_end_iso: str
    maturity_days: int


def extract_candidate_features(
    hits: Sequence[ScenarioHit],
    tx_by_id: Dict[str, NormalizedTransaction],
    accounts_by_id: Dict[str, Account],
    entities_by_id: Dict[str, Entity],
    rolling_features: Dict[Tuple[str, str], RollingFeatureVector],
    graph_features: Dict[str, GraphFeatureVector],
) -> Dict[str, float]:
    """Extract versioned numeric feature vector for a grouped candidate alert."""
    subject_acc_id = hits[0].subject_account_id
    rule_score = max(h.rule_score for h in hits)
    scenario_names = {h.scenario_name for h in hits}
    is_layering_or_cycle = 1.0 if (
        "RAPID_LAYERING_CHAIN" in scenario_names or "ROUND_TRIP_CYCLE" in scenario_names
    ) else 0.0

    involved_txs = [
        tx_by_id[tid]
        for h in hits
        for tid in h.involved_tx_ids
        if tid in tx_by_id
    ]
    unique_txs = list({t.tx_id: t for t in involved_txs}.values())
    total_usd = sum(t.amount_usd for t in unique_txs)
    cash_wire_count = sum(1 for t in unique_txs if t.channel in {"CASH_DEPOSIT", "WIRE"})
    cash_wire_ratio = (cash_wire_count / len(unique_txs)) if unique_txs else 0.0

    acc = accounts_by_id.get(subject_acc_id)
    ent = entities_by_id.get(acc.entity_id) if acc else None
    risk_num = RISK_TIER_NUM.get(ent.risk_tier if ent else "UNKNOWN", 0.5)
    high_risk_juris = 1.0 if (ent and ent.jurisdiction in HIGH_RISK_JURISDICTIONS) else 0.0

    # Take max rolling indicators across involved transactions for the subject account
    str_count = 0.0
    passthrough = 0.0
    velocity = 0.0
    dormancy = 0.0
    for tx in unique_txs:
        for role in ("SOURCE", "TARGET"):
            fv = rolling_features.get((tx.tx_id, role))
            if fv and fv.account_id == subject_acc_id:
                str_count = max(str_count, float(fv.structuring_band_count_24h))
                passthrough = max(passthrough, fv.passthrough_ratio_24h)
                velocity = max(velocity, min(fv.velocity_ratio_vs_baseline, 50.0))
                dormancy = max(dormancy, fv.dormancy_days_prior)

    gf = graph_features.get(subject_acc_id)
    fan_ratio = gf.fan_in_fan_out_ratio if gf else 0.0
    ego_density = gf.ego_density if gf else 0.0

    return {
        "rule_score": round(rule_score, 6),
        "num_scenarios_triggered": float(len(scenario_names)),
        "is_layering_or_cycle": is_layering_or_cycle,
        "is_cash_or_wire_ratio": round(cash_wire_ratio, 6),
        "log_total_involved_usd": round(math.log1p(max(0.0, total_usd)), 6),
        "entity_risk_tier_num": risk_num,
        "is_high_risk_jurisdiction": high_risk_juris,
        "structuring_band_count_24h": str_count,
        "passthrough_ratio_24h": round(min(passthrough, 3.0), 6),
        "velocity_ratio_vs_baseline": round(velocity, 6),
        "fan_in_fan_out_ratio": round(min(fan_ratio, 10.0), 6),
        "ego_density": round(ego_density, 6),
        "dormancy_days_log": round(math.log1p(max(0.0, dormancy)), 6),
    }


def temporal_maturity_split(
    records: Sequence[CandidateRecord],
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    maturity_days: int = 7,
) -> TemporalSplitResult:
    """Split candidate alerts strictly in chronological order with label maturity buffer.

    Records falling within `maturity_days` prior to the train/cal or cal/test
    boundaries are excluded from training/calibration to prevent immature-label leakage.
    """
    ordered = sorted(records, key=lambda r: (r.trigger_timestamp_utc, r.candidate_id))
    if not ordered:
        return TemporalSplitResult([], [], [], 0, 0, "", "", maturity_days)

    t_min = ordered[0].trigger_timestamp_utc
    t_max = ordered[-1].trigger_timestamp_utc
    span = max(timedelta(days=1), t_max - t_min)

    train_end = t_min + timedelta(seconds=span.total_seconds() * train_fraction)
    cal_end = t_min + timedelta(
        seconds=span.total_seconds() * (train_fraction + calibration_fraction)
    )
    maturity_delta = timedelta(days=maturity_days)

    train: List[CandidateRecord] = []
    cal: List[CandidateRecord] = []
    test: List[CandidateRecord] = []
    excl_train = 0
    excl_cal = 0

    for rec in ordered:
        ts = rec.trigger_timestamp_utc
        if ts < train_end:
            if ts <= train_end - maturity_delta:
                train.append(rec)
            else:
                excl_train += 1
        elif ts < cal_end:
            if ts <= cal_end - maturity_delta:
                cal.append(rec)
            else:
                excl_cal += 1
        else:
            test.append(rec)

    return TemporalSplitResult(
        train=train,
        calibration=cal,
        test=test,
        excluded_immature_train=excl_train,
        excluded_immature_cal=excl_cal,
        train_end_iso=train_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        cal_end_iso=cal_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        maturity_days=maturity_days,
    )


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-min(z, 60.0))
        return 1.0 / (1.0 + ez)
    ez = math.exp(max(z, -60.0))
    return ez / (1.0 + ez)


def compute_brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def compute_ece(probs: Sequence[float], labels: Sequence[int], n_bins: int = 5) -> float:
    """Expected Calibration Error across equal-width probability bins."""
    if not probs:
        return 0.0
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idxs = [
            i
            for i, p in enumerate(probs)
            if (lo <= p < hi) or (b == n_bins - 1 and p == 1.0)
        ]
        if not idxs:
            continue
        avg_p = sum(probs[i] for i in idxs) / len(idxs)
        avg_y = sum(labels[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / n) * abs(avg_p - avg_y)
    return ece


class MLPrioritizer:
    """L2-regularized logistic prioritizer with Platt calibration and exact attributions."""

    def __init__(
        self,
        feature_names: Tuple[str, ...],
        means: List[float],
        stds: List[float],
        weights: List[float],
        bias: float,
        platt_a: float,
        platt_b: float,
        model_version: str,
        feature_version: str,
        training_metadata: Dict[str, Any],
    ) -> None:
        self.feature_names = feature_names
        self.means = means
        self.stds = stds
        self.weights = weights
        self.bias = bias
        self.platt_a = platt_a
        self.platt_b = platt_b
        self.model_version = model_version
        self.feature_version = feature_version
        self.training_metadata = training_metadata

    @classmethod
    def fit(
        cls,
        train_records: Sequence[CandidateRecord],
        calibration_records: Sequence[CandidateRecord],
        config: EngineConfig,
    ) -> "MLPrioritizer":
        """Train logistic regression on train_records and calibrate via Platt scaling."""
        ml_cfg = config.ml
        lr = float(ml_cfg.get("learning_rate", 0.15))
        l2 = float(ml_cfg.get("l2_regularization", 0.05))
        epochs = int(ml_cfg.get("epochs", 400))

        n = len(train_records)
        d = len(FEATURE_NAMES)
        if n == 0:
            raise ValueError("Cannot train MLPrioritizer on an empty training set.")

        X_raw = [[rec.features.get(f, 0.0) for f in FEATURE_NAMES] for rec in train_records]
        y = [float(rec.label) for rec in train_records]

        means = [sum(row[j] for row in X_raw) / n for j in range(d)]
        stds = []
        for j in range(d):
            var = sum((row[j] - means[j]) ** 2 for row in X_raw) / max(1, n)
            sd = math.sqrt(var)
            stds.append(sd if sd > 1e-8 else 1.0)

        X = [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in X_raw]

        weights = [0.0] * d
        bias = 0.0

        for _ in range(epochs):
            grad_w = [0.0] * d
            grad_b = 0.0
            for i in range(n):
                z = bias + sum(weights[j] * X[i][j] for j in range(d))
                pred = _sigmoid(z)
                err = pred - y[i]
                grad_b += err
                for j in range(d):
                    grad_w[j] += err * X[i][j]
            bias -= lr * (grad_b / n)
            for j in range(d):
                weights[j] -= lr * ((grad_w[j] / n) + l2 * weights[j])

        # Platt scaling calibration on held-out temporal calibration window
        platt_a = 1.0
        platt_b = 0.0
        cal_metrics: Dict[str, Any] = {}
        cal_source = list(calibration_records) if len(calibration_records) >= 4 else list(train_records)
        cal_logits: List[float] = []
        cal_labels: List[int] = []
        for rec in cal_source:
            x_std = [(rec.features.get(f, 0.0) - means[j]) / stds[j] for j, f in enumerate(FEATURE_NAMES)]
            logit = bias + sum(weights[j] * x_std[j] for j in range(d))
            cal_logits.append(logit)
            cal_labels.append(rec.label)

        raw_cal_probs = [_sigmoid(z) for z in cal_logits]
        if len(set(cal_labels)) == 2:
            a = 1.0
            b = 0.0
            cal_lr = 0.10
            m = len(cal_logits)
            for _ in range(300):
                ga = 0.0
                gb = 0.0
                for z, lbl in zip(cal_logits, cal_labels):
                    p = _sigmoid(a * z + b)
                    diff = p - lbl
                    ga += diff * z
                    gb += diff
                a -= cal_lr * ((ga / m) + 0.02 * (a - 1.0))
                b -= cal_lr * ((gb / m) + 0.02 * b)
            platt_a = max(0.1, a)
            platt_b = b

        cal_probs = [_sigmoid(platt_a * z + platt_b) for z in cal_logits]
        cal_metrics = {
            "calibration_samples": len(cal_source),
            "raw_brier": round(compute_brier_score(raw_cal_probs, cal_labels), 6),
            "calibrated_brier": round(compute_brier_score(cal_probs, cal_labels), 6),
            "raw_ece": round(compute_ece(raw_cal_probs, cal_labels), 6),
            "calibrated_ece": round(compute_ece(cal_probs, cal_labels), 6),
        }

        fingerprint_payload = json.dumps(
            {
                "feature_version": config.feature_version,
                "config_hash": config.config_hash,
                "weights": [round(w, 6) for w in weights],
                "bias": round(bias, 6),
                "platt_a": round(platt_a, 6),
                "platt_b": round(platt_b, 6),
            },
            sort_keys=True,
        ).encode("utf-8")
        model_hash = hashlib.sha256(fingerprint_payload).hexdigest()[:12]
        model_version = f"logreg-platt-{config.feature_version}-{model_hash}"

        metadata = {
            "seed": config.seed,
            "config_hash": config.config_hash,
            "train_samples": n,
            "train_positives": int(sum(y)),
            "calibration": cal_metrics,
            "python_version": platform.python_version(),
        }
        return cls(
            feature_names=FEATURE_NAMES,
            means=means,
            stds=stds,
            weights=weights,
            bias=bias,
            platt_a=platt_a,
            platt_b=platt_b,
            model_version=model_version,
            feature_version=config.feature_version,
            training_metadata=metadata,
        )

    def predict_and_explain(
        self,
        features: Dict[str, float],
        top_k: int = 4,
    ) -> Tuple[float, float, Tuple[ReasonContribution, ...]]:
        """Return (raw_probability, calibrated_probability, top_k_feature_contributions)."""
        contribs: List[Tuple[float, str, float]] = []
        z = self.bias
        for j, fname in enumerate(self.feature_names):
            val = float(features.get(fname, 0.0))
            z_j = self.weights[j] * ((val - self.means[j]) / self.stds[j])
            z += z_j
            contribs.append((z_j, fname, val))

        raw_prob = _sigmoid(z)
        cal_prob = _sigmoid(self.platt_a * z + self.platt_b)

        contribs.sort(key=lambda item: (-abs(item[0]), item[1]))
        reasons: List[ReasonContribution] = []
        for delta_logit, fname, val in contribs[:top_k]:
            direction = "increased" if delta_logit >= 0 else "decreased"
            reasons.append(
                ReasonContribution(
                    code=f"ML_DRIVER_{fname.upper()}",
                    category="ML_FEATURE",
                    weight=round(delta_logit, 6),
                    description=(
                        f"Feature '{fname}'={val:.3f} {direction} prioritization log-odds "
                        f"by {delta_logit:+.3f}."
                    ),
                    evidence_metrics={"feature": fname, "value": round(val, 6), "log_odds_delta": round(delta_logit, 6)},
                )
            )
        return round(raw_prob, 6), round(cal_prob, 6), tuple(reasons)

    def to_artifact_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "feature_names": list(self.feature_names),
            "means": [round(m, 8) for m in self.means],
            "stds": [round(s, 8) for s in self.stds],
            "weights": [round(w, 8) for w in self.weights],
            "bias": round(self.bias, 8),
            "platt_a": round(self.platt_a, 8),
            "platt_b": round(self.platt_b, 8),
            "training_metadata": self.training_metadata,
        }

    def save_artifact(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_artifact_dict(), fh, indent=2, sort_keys=True)
        return out_path

    @classmethod
    def load_artifact(cls, path: str | Path) -> "MLPrioritizer":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            feature_names=tuple(data["feature_names"]),
            means=[float(x) for x in data["means"]],
            stds=[float(x) for x in data["stds"]],
            weights=[float(x) for x in data["weights"]],
            bias=float(data["bias"]),
            platt_a=float(data["platt_a"]),
            platt_b=float(data["platt_b"]),
            model_version=str(data["model_version"]),
            feature_version=str(data["feature_version"]),
            training_metadata=dict(data.get("training_metadata", {})),
        )


def apply_prioritization_policy(
    rule_score: float,
    features: Dict[str, float],
    config: EngineConfig,
    prioritizer: Optional[MLPrioritizer] = None,
) -> PrioritizationResult:
    """Policy layer combining rule score and optional calibrated ML score into review priority."""
    policy_cfg = config.policy
    thresholds = policy_cfg.get("priority_thresholds", {})
    p1_th = float(thresholds.get("P1_HIGH", 0.75))
    p2_th = float(thresholds.get("P2_MEDIUM", 0.45))
    p3_th = float(thresholds.get("P3_LOW", 0.25))

    if prioritizer is not None:
        raw_prob, cal_prob, ml_contribs = prioritizer.predict_and_explain(features)
        w_rule = float(policy_cfg.get("hybrid_rule_weight", 0.45))
        w_ml = float(policy_cfg.get("hybrid_ml_weight", 0.55))
        combined = min(1.0, max(0.0, w_rule * rule_score + w_ml * cal_prob))
        model_version = prioritizer.model_version
    else:
        raw_prob = None
        cal_prob = None
        ml_contribs = ()
        combined = min(1.0, max(0.0, rule_score))
        model_version = None

    if combined >= p1_th:
        priority = ReviewPriority.P1_HIGH
    elif combined >= p2_th:
        priority = ReviewPriority.P2_MEDIUM
    elif combined >= p3_th:
        priority = ReviewPriority.P3_LOW
    else:
        priority = ReviewPriority.SUPPRESSED_BELOW_THRESHOLD

    return PrioritizationResult(
        rule_score=round(rule_score, 6),
        ml_raw_score=raw_prob,
        ml_calibrated_probability=cal_prob,
        combined_score=round(combined, 6),
        priority=priority,
        policy_version=config.policy_version,
        model_version=model_version,
        feature_contributions=ml_contribs,
    )
