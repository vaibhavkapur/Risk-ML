"""Calibrated acceptance and fraud/loss probability models with temporal validation."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Suppress benign Apple Silicon Accelerate BLAS FPU flag warnings in sklearn matmul
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning, module="sklearn")

from payment_auth_engine import FEATURE_SCHEMA_VERSION
from payment_auth_engine.config import EngineConfig
from payment_auth_engine.features import (
    FeatureExtractor,
    filter_matured_label_indices,
    temporal_train_cal_test_split,
)
from payment_auth_engine.schemas import (
    AuthorizationAttempt,
    HistoricalOutcome,
    PolicyAction,
    ProcessorHealthState,
    ProcessorHealthStatus,
)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) using equal-width bins."""
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    n = len(y_true_arr)
    if n == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob_arr >= lo) & (y_prob_arr <= hi)
        else:
            mask = (y_prob_arr >= lo) & (y_prob_arr < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        avg_conf = float(np.mean(y_prob_arr[mask]))
        avg_acc = float(np.mean(y_true_arr[mask]))
        ece += (count / float(n)) * abs(avg_acc - avg_conf)
    return float(ece)


@dataclass(frozen=True)
class CalibrationMetrics:
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    log_loss_before: float
    log_loss_after: float
    roc_auc: float
    n_calibration_samples: int
    positive_rate: float


class PlattCalibratedClassifier:
    """Standardized logistic classifier calibrated on a strictly later temporal fold."""

    def __init__(self, *, seed: int = 42, c_reg: float = 1.0) -> None:
        self.seed = seed
        self.c_reg = c_reg
        self.scaler = StandardScaler()
        self.base_model = LogisticRegression(
            C=c_reg,
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        )
        self.calibrator = LogisticRegression(
            C=10.0,
            max_iter=500,
            random_state=seed,
            solver="lbfgs",
        )
        self.is_fitted = False
        self.uses_constant_prob: Optional[float] = None
        self.calibration_metrics: Optional[CalibrationMetrics] = None

    @staticmethod
    def _safe_logit(probs: np.ndarray) -> np.ndarray:
        clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> CalibrationMetrics:
        y_tr = np.asarray(y_train, dtype=np.int64)
        y_ca = np.asarray(y_cal, dtype=np.int64)
        if len(y_tr) == 0:
            raise ValueError("Cannot train classifier on empty training fold")

        x_tr_scaled = self.scaler.fit_transform(x_train)
        if len(np.unique(y_tr)) < 2:
            self.uses_constant_prob = float(np.mean(y_tr))
            self.is_fitted = True
            probs = np.full(len(y_ca), self.uses_constant_prob, dtype=np.float64)
            metrics = CalibrationMetrics(
                brier_before=float(brier_score_loss(y_ca, probs)) if len(y_ca) else 0.0,
                brier_after=float(brier_score_loss(y_ca, probs)) if len(y_ca) else 0.0,
                ece_before=compute_ece(y_ca, probs) if len(y_ca) else 0.0,
                ece_after=compute_ece(y_ca, probs) if len(y_ca) else 0.0,
                log_loss_before=0.0,
                log_loss_after=0.0,
                roc_auc=0.5,
                n_calibration_samples=len(y_ca),
                positive_rate=float(np.mean(y_ca)) if len(y_ca) else self.uses_constant_prob,
            )
            self.calibration_metrics = metrics
            return metrics

        self.uses_constant_prob = None
        self.base_model.fit(x_tr_scaled, y_tr)

        x_ca_scaled = self.scaler.transform(x_cal)
        raw_cal_probs = np.clip(self.base_model.predict_proba(x_ca_scaled)[:, 1], 1e-6, 1.0 - 1e-6)

        if len(np.unique(y_ca)) >= 2:
            self.calibrator.fit(self._safe_logit(raw_cal_probs), y_ca)
            cal_probs = np.clip(
                self.calibrator.predict_proba(self._safe_logit(raw_cal_probs))[:, 1],
                1e-6,
                1.0 - 1e-6,
            )
            auc = float(roc_auc_score(y_ca, cal_probs))
        else:
            cal_probs = raw_cal_probs
            auc = 0.5

        metrics = CalibrationMetrics(
            brier_before=float(brier_score_loss(y_ca, raw_cal_probs)),
            brier_after=float(brier_score_loss(y_ca, cal_probs)),
            ece_before=compute_ece(y_ca, raw_cal_probs),
            ece_after=compute_ece(y_ca, cal_probs),
            log_loss_before=float(log_loss(y_ca, raw_cal_probs, labels=[0, 1])),
            log_loss_after=float(log_loss(y_ca, cal_probs, labels=[0, 1])),
            roc_auc=auc,
            n_calibration_samples=len(y_ca),
            positive_rate=float(np.mean(y_ca)),
        )
        self.calibration_metrics = metrics
        self.is_fitted = True
        return metrics

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba")
        if len(x) == 0:
            return np.empty(0, dtype=np.float64)
        if self.uses_constant_prob is not None:
            return np.full(len(x), self.uses_constant_prob, dtype=np.float64)
        x_scaled = self.scaler.transform(x)
        raw = np.clip(self.base_model.predict_proba(x_scaled)[:, 1], 1e-6, 1.0 - 1e-6)
        if hasattr(self.calibrator, "coef_"):
            cal = self.calibrator.predict_proba(self._safe_logit(raw))[:, 1]
            return np.clip(cal, 1e-6, 1.0 - 1e-6)
        return raw

    def explain_contributions(
        self, x_row: np.ndarray, feature_names: Sequence[str], top_k: int = 4
    ) -> Dict[str, float]:
        """Return top standardized linear feature contributions for transparency."""
        if not self.is_fitted or self.uses_constant_prob is not None:
            return {}
        x_scaled = self.scaler.transform(x_row.reshape(1, -1))[0]
        weights = self.base_model.coef_[0]
        contribs = x_scaled * weights
        order = np.argsort(np.abs(contribs))[::-1][:top_k]
        return {str(feature_names[int(i)]): round(float(contribs[int(i)]), 4) for i in order}


@dataclass(frozen=True)
class ActionPrediction:
    """Calibrated model predictions for a single (attempt, action) pair."""

    action: PolicyAction
    processor_selected: Optional[str]
    feasible: bool
    infeasible_reason: Optional[str]
    acceptance_prob: float
    fraud_prob_given_approved: float
    expected_loss_cents: float
    expected_net_value_cents: float
    estimated_latency_ms: float
    missing_features: List[str]
    top_factors: Dict[str, float]


@dataclass
class TrainingSummary:
    model_version: str
    feature_schema_version: str
    seed: int
    config_hash: str
    trained_at_utc: str
    n_total: int
    n_train: int
    n_cal: int
    n_test: int
    n_loss_train_matured: int
    n_loss_cal_matured: int
    train_end_utc: str
    cal_end_utc: str
    test_start_utc: str
    test_end_utc: str
    acceptance_calibration: Dict[str, Any]
    loss_calibration: Dict[str, Any]


class AuthorizationModelBundle:
    """Bundled feature extractor + calibrated acceptance and loss models."""

    def __init__(
        self,
        *,
        config: EngineConfig,
        feature_extractor: FeatureExtractor,
        acceptance_model: PlattCalibratedClassifier,
        loss_model: PlattCalibratedClassifier,
        summary: TrainingSummary,
    ) -> None:
        self.config = config
        self.feature_extractor = feature_extractor
        self.acceptance_model = acceptance_model
        self.loss_model = loss_model
        self.summary = summary
        self.model_version = summary.model_version

    @classmethod
    def train(
        cls,
        attempts: Sequence[AuthorizationAttempt],
        outcomes: Sequence[HistoricalOutcome],
        config: EngineConfig,
        as_of_utc: datetime,
    ) -> Tuple["AuthorizationModelBundle", np.ndarray, np.ndarray, np.ndarray]:
        """Train and calibrate models using strict temporal splits and matured labels."""
        timestamps = [a.timestamp_utc for a in attempts]
        train_idx, cal_idx, test_idx = temporal_train_cal_test_split(
            timestamps,
            config.data.train_fraction,
            config.data.calibration_fraction,
            config.data.test_fraction,
        )

        # Fit feature imputation strictly on training split
        extractor = FeatureExtractor(schema_version=FEATURE_SCHEMA_VERSION)
        extractor.fit([attempts[int(i)] for i in train_idx])

        # Filter out DECLINE_POLICY rows for acceptance model (acceptance is trivially 0 on decline)
        acc_train_idx = [int(i) for i in train_idx if outcomes[int(i)].logged_action != PolicyAction.DECLINE_POLICY]
        acc_cal_idx = [int(i) for i in cal_idx if outcomes[int(i)].logged_action != PolicyAction.DECLINE_POLICY]

        x_acc_train = extractor.transform(
            [attempts[i] for i in acc_train_idx],
            [outcomes[i].logged_action for i in acc_train_idx],
        ).matrix
        y_acc_train = np.asarray([1 if outcomes[i].authorized else 0 for i in acc_train_idx], dtype=np.int64)

        x_acc_cal = extractor.transform(
            [attempts[i] for i in acc_cal_idx],
            [outcomes[i].logged_action for i in acc_cal_idx],
        ).matrix
        y_acc_cal = np.asarray([1 if outcomes[i].authorized else 0 for i in acc_cal_idx], dtype=np.int64)

        acceptance_model = PlattCalibratedClassifier(seed=config.seed, c_reg=1.0)
        acc_metrics = acceptance_model.fit(x_acc_train, y_acc_train, x_acc_cal, y_acc_cal)

        # Loss/fraud model trained strictly on matured approved attempts
        loss_train_idx = filter_matured_label_indices(
            outcomes, train_idx, as_of_utc, config.data.min_label_maturity_days
        )
        loss_cal_idx = filter_matured_label_indices(
            outcomes, cal_idx, as_of_utc, config.data.min_label_maturity_days
        )
        # If calibration fold sits completely within maturity window (e.g. short span), fall back to matured tail of train
        if len(loss_cal_idx) < 20:
            split_pt = max(1, int(len(loss_train_idx) * 0.75))
            loss_cal_idx = loss_train_idx[split_pt:]
            loss_train_idx = loss_train_idx[:split_pt]

        x_loss_train = extractor.transform(
            [attempts[int(i)] for i in loss_train_idx],
            [outcomes[int(i)].logged_action for i in loss_train_idx],
        ).matrix
        y_loss_train = np.asarray(
            [1 if outcomes[int(i)].fraud_confirmed else 0 for i in loss_train_idx], dtype=np.int64
        )

        x_loss_cal = extractor.transform(
            [attempts[int(i)] for i in loss_cal_idx],
            [outcomes[int(i)].logged_action for i in loss_cal_idx],
        ).matrix
        y_loss_cal = np.asarray(
            [1 if outcomes[int(i)].fraud_confirmed else 0 for i in loss_cal_idx], dtype=np.int64
        )

        loss_model = PlattCalibratedClassifier(seed=config.seed + 1, c_reg=0.8)
        loss_metrics = loss_model.fit(x_loss_train, y_loss_train, x_loss_cal, y_loss_cal)

        version_seed = f"{config.seed}:{config.config_hash()}:{FEATURE_SCHEMA_VERSION}:{len(attempts)}"
        model_hash = hashlib.sha256(version_seed.encode("utf-8")).hexdigest()[:10]
        model_version = f"v1.0.0-{model_hash}"

        summary = TrainingSummary(
            model_version=model_version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            seed=config.seed,
            config_hash=config.config_hash(),
            trained_at_utc=as_of_utc.astimezone(timezone.utc).isoformat(),
            n_total=len(attempts),
            n_train=len(train_idx),
            n_cal=len(cal_idx),
            n_test=len(test_idx),
            n_loss_train_matured=int(len(loss_train_idx)),
            n_loss_cal_matured=int(len(loss_cal_idx)),
            train_end_utc=attempts[int(train_idx[-1])].timestamp_utc.isoformat(),
            cal_end_utc=attempts[int(cal_idx[-1])].timestamp_utc.isoformat(),
            test_start_utc=attempts[int(test_idx[0])].timestamp_utc.isoformat(),
            test_end_utc=attempts[int(test_idx[-1])].timestamp_utc.isoformat(),
            acceptance_calibration=asdict(acc_metrics),
            loss_calibration=asdict(loss_metrics),
        )

        bundle = cls(
            config=config,
            feature_extractor=extractor,
            acceptance_model=acceptance_model,
            loss_model=loss_model,
            summary=summary,
        )
        return bundle, train_idx, cal_idx, test_idx

    def _select_secondary_processor(
        self,
        primary_id: str,
        health_states: Dict[str, ProcessorHealthState],
    ) -> Optional[str]:
        """Choose best available alternative processor (prefer HEALTHY over DEGRADED, never OUTAGE)."""
        candidates = [pid for pid in self.config.processors if pid != primary_id]
        healthy = [
            pid
            for pid in candidates
            if health_states.get(pid, ProcessorHealthState(processor_id=pid)).status
            == ProcessorHealthStatus.HEALTHY
        ]
        if healthy:
            # Prefer proc_b when healthy, otherwise deterministic sorted order
            return "proc_b" if "proc_b" in healthy else sorted(healthy)[0]
        degraded = [
            pid
            for pid in candidates
            if health_states.get(pid, ProcessorHealthState(processor_id=pid)).status
            == ProcessorHealthStatus.DEGRADED
        ]
        if degraded:
            return sorted(degraded)[0]
        return None

    def predict_action(
        self,
        attempt: AuthorizationAttempt,
        action: PolicyAction,
        health_states: Optional[Dict[str, ProcessorHealthState]] = None,
    ) -> ActionPrediction:
        """Predict calibrated acceptance, fraud/loss, net value, and latency for one (attempt, action)."""
        states = health_states or {}
        extracted = self.feature_extractor.transform([attempt], [action])
        missing = extracted.missing_fields_per_row[0]

        if action == PolicyAction.DECLINE_POLICY:
            return ActionPrediction(
                action=action,
                processor_selected=None,
                feasible=True,
                infeasible_reason=None,
                acceptance_prob=0.0,
                fraud_prob_given_approved=0.0,
                expected_loss_cents=0.0,
                expected_net_value_cents=0.0,
                estimated_latency_ms=25.0,
                missing_features=missing,
                top_factors=self.loss_model.explain_contributions(
                    extracted.matrix[0], extracted.feature_names, top_k=4
                ),
            )

        primary_id = attempt.processor_id if attempt.processor_id in self.config.processors else sorted(self.config.processors)[0]
        primary_state = states.get(primary_id, ProcessorHealthState(processor_id=primary_id))

        if action in (PolicyAction.APPROVE_PRIMARY, PolicyAction.STEP_UP_3DS):
            if primary_state.status == ProcessorHealthStatus.OUTAGE:
                return ActionPrediction(
                    action=action,
                    processor_selected=primary_id,
                    feasible=False,
                    infeasible_reason=f"PROCESSOR_OUTAGE:{primary_id}",
                    acceptance_prob=0.0,
                    fraud_prob_given_approved=0.0,
                    expected_loss_cents=0.0,
                    expected_net_value_cents=-1e9,
                    estimated_latency_ms=0.0,
                    missing_features=missing,
                    top_factors={},
                )
            selected_proc = primary_id
            proc_state = primary_state
        elif action == PolicyAction.ROUTE_SECONDARY:
            secondary_id = self._select_secondary_processor(primary_id, states)
            if secondary_id is None:
                return ActionPrediction(
                    action=action,
                    processor_selected=None,
                    feasible=False,
                    infeasible_reason="NO_HEALTHY_SECONDARY_PROCESSOR",
                    acceptance_prob=0.0,
                    fraud_prob_given_approved=0.0,
                    expected_loss_cents=0.0,
                    expected_net_value_cents=-1e9,
                    estimated_latency_ms=0.0,
                    missing_features=missing,
                    top_factors={},
                )
            selected_proc = secondary_id
            proc_state = states.get(secondary_id, ProcessorHealthState(processor_id=secondary_id))
        else:
            raise ValueError(f"predict_action does not score non-evaluable action {action}")

        p_acc = float(self.acceptance_model.predict_proba(extracted.matrix)[0])
        p_fraud = float(self.loss_model.predict_proba(extracted.matrix)[0])

        # Apply processor health adjustments
        if proc_state.status == ProcessorHealthStatus.DEGRADED:
            p_acc = max(0.0, p_acc - proc_state.approval_penalty)

        proc_cfg = self.config.processors[selected_proc]
        cross_border = 1.0 if attempt.issuer_country != attempt.merchant_country else 0.0
        if action == PolicyAction.STEP_UP_3DS:
            base_lat = proc_cfg.base_latency_ms + 260.0 + 30.0 * cross_border
            action_cost = float(self.config.economics.step_up_3ds_cost_cents)
        elif action == PolicyAction.ROUTE_SECONDARY:
            base_lat = proc_cfg.base_latency_ms + 25.0 * cross_border
            action_cost = float(self.config.economics.secondary_route_cost_cents)
        else:
            base_lat = proc_cfg.base_latency_ms + 18.0 * cross_border
            action_cost = float(proc_cfg.cost_cents)

        latency_ms = base_lat * proc_state.latency_multiplier

        lgd_cents = (
            attempt.amount_cents * self.config.economics.loss_given_fraud_multiplier
            + self.config.economics.chargeback_fee_cents
        )
        margin_rate = self.config.economics.merchant_margin_bps / 10_000.0
        exp_loss = p_acc * p_fraud * lgd_cents
        exp_rev = p_acc * (attempt.amount_cents * margin_rate)
        exp_net = exp_rev - exp_loss - action_cost

        top_factors = self.loss_model.explain_contributions(
            extracted.matrix[0], extracted.feature_names, top_k=4
        )

        return ActionPrediction(
            action=action,
            processor_selected=selected_proc,
            feasible=True,
            infeasible_reason=None,
            acceptance_prob=p_acc,
            fraud_prob_given_approved=p_fraud,
            expected_loss_cents=exp_loss,
            expected_net_value_cents=exp_net,
            estimated_latency_ms=latency_ms,
            missing_features=missing,
            top_factors=top_factors,
        )

    def predict_all_actions(
        self,
        attempt: AuthorizationAttempt,
        health_states: Optional[Dict[str, ProcessorHealthState]] = None,
    ) -> Dict[PolicyAction, ActionPrediction]:
        return self.predict_batch_all_actions([attempt], health_states=health_states)[0]

    def predict_batch_all_actions(
        self,
        attempts: Sequence[AuthorizationAttempt],
        health_states: Optional[Dict[str, ProcessorHealthState]] = None,
    ) -> List[Dict[PolicyAction, ActionPrediction]]:
        """Vectorized calibrated prediction across all evaluable actions for a batch of attempts."""
        from payment_auth_engine.schemas import EVALUABLE_ACTIONS

        n = len(attempts)
        if n == 0:
            return []

        states = health_states or {}
        results: List[Dict[PolicyAction, ActionPrediction]] = [{} for _ in range(n)]
        margin_rate = self.config.economics.merchant_margin_bps / 10_000.0
        lgd_mult = self.config.economics.loss_given_fraud_multiplier
        cb_fee = self.config.economics.chargeback_fee_cents
        default_proc = sorted(self.config.processors)[0]

        for action in EVALUABLE_ACTIONS:
            extracted = self.feature_extractor.transform(attempts, [action] * n)
            if action == PolicyAction.DECLINE_POLICY:
                for i in range(n):
                    results[i][action] = ActionPrediction(
                        action=action,
                        processor_selected=None,
                        feasible=True,
                        infeasible_reason=None,
                        acceptance_prob=0.0,
                        fraud_prob_given_approved=0.0,
                        expected_loss_cents=0.0,
                        expected_net_value_cents=0.0,
                        estimated_latency_ms=25.0,
                        missing_features=extracted.missing_fields_per_row[i],
                        top_factors={},
                    )
                continue

            acc_probs = self.acceptance_model.predict_proba(extracted.matrix)
            fraud_probs = self.loss_model.predict_proba(extracted.matrix)

            for i, attempt in enumerate(attempts):
                missing = extracted.missing_fields_per_row[i]
                primary_id = attempt.processor_id if attempt.processor_id in self.config.processors else default_proc
                primary_state = states.get(primary_id, ProcessorHealthState(processor_id=primary_id))

                if action in (PolicyAction.APPROVE_PRIMARY, PolicyAction.STEP_UP_3DS):
                    if primary_state.status == ProcessorHealthStatus.OUTAGE:
                        results[i][action] = ActionPrediction(
                            action=action,
                            processor_selected=primary_id,
                            feasible=False,
                            infeasible_reason=f"PROCESSOR_OUTAGE:{primary_id}",
                            acceptance_prob=0.0,
                            fraud_prob_given_approved=0.0,
                            expected_loss_cents=0.0,
                            expected_net_value_cents=-1e9,
                            estimated_latency_ms=0.0,
                            missing_features=missing,
                            top_factors={},
                        )
                        continue
                    selected_proc = primary_id
                    proc_state = primary_state
                else:
                    secondary_id = self._select_secondary_processor(primary_id, states)
                    if secondary_id is None:
                        results[i][action] = ActionPrediction(
                            action=action,
                            processor_selected=None,
                            feasible=False,
                            infeasible_reason="NO_HEALTHY_SECONDARY_PROCESSOR",
                            acceptance_prob=0.0,
                            fraud_prob_given_approved=0.0,
                            expected_loss_cents=0.0,
                            expected_net_value_cents=-1e9,
                            estimated_latency_ms=0.0,
                            missing_features=missing,
                            top_factors={},
                        )
                        continue
                    selected_proc = secondary_id
                    proc_state = states.get(secondary_id, ProcessorHealthState(processor_id=secondary_id))

                p_acc = float(acc_probs[i])
                p_fraud = float(fraud_probs[i])
                if proc_state.status == ProcessorHealthStatus.DEGRADED:
                    p_acc = max(0.0, p_acc - proc_state.approval_penalty)

                proc_cfg = self.config.processors[selected_proc]
                cross_border = 1.0 if attempt.issuer_country != attempt.merchant_country else 0.0
                if action == PolicyAction.STEP_UP_3DS:
                    base_lat = proc_cfg.base_latency_ms + 260.0 + 30.0 * cross_border
                    action_cost = float(self.config.economics.step_up_3ds_cost_cents)
                elif action == PolicyAction.ROUTE_SECONDARY:
                    base_lat = proc_cfg.base_latency_ms + 25.0 * cross_border
                    action_cost = float(self.config.economics.secondary_route_cost_cents)
                else:
                    base_lat = proc_cfg.base_latency_ms + 18.0 * cross_border
                    action_cost = float(proc_cfg.cost_cents)

                latency_ms = base_lat * proc_state.latency_multiplier
                lgd_cents = attempt.amount_cents * lgd_mult + cb_fee
                exp_loss = p_acc * p_fraud * lgd_cents
                exp_rev = p_acc * (attempt.amount_cents * margin_rate)
                exp_net = exp_rev - exp_loss - action_cost

                top_factors = (
                    self.loss_model.explain_contributions(extracted.matrix[i], extracted.feature_names, top_k=4)
                    if n <= 20
                    else {}
                )
                results[i][action] = ActionPrediction(
                    action=action,
                    processor_selected=selected_proc,
                    feasible=True,
                    infeasible_reason=None,
                    acceptance_prob=p_acc,
                    fraud_prob_given_approved=p_fraud,
                    expected_loss_cents=exp_loss,
                    expected_net_value_cents=exp_net,
                    estimated_latency_ms=latency_ms,
                    missing_features=missing,
                    top_factors=top_factors,
                )

        return results

    def save_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self.summary)
        payload["imputation_stats"] = asdict(self.feature_extractor.imputation)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
