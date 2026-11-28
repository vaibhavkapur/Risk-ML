"""Off-policy counterfactual evaluation (DM, IPS, SNIPS, DR) with explicit assumption guards.

Evaluates candidate authorization policies on logged synthetic data generated under a
known stochastic logging policy, reports bootstrap confidence bounds for net acceptance
and expected loss, and fails explicitly on positivity/overlap/ESS assumption violations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from payment_auth_engine.config import CounterfactualConfig, EngineConfig
from payment_auth_engine.models import ActionPrediction
from payment_auth_engine.schemas import AuthorizationAttempt, HistoricalOutcome, PolicyAction
from payment_auth_engine.synthetic import ActionOracle


class CounterfactualAssumptionError(Exception):
    """Raised when counterfactual identification assumptions are violated."""

    def __init__(self, code: str, message: str, diagnostics: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class ConfidenceBound:
    estimate: float
    lower: float
    upper: float
    confidence_level: float


@dataclass(frozen=True)
class PolicyCounterfactualReport:
    policy_name: str
    n_samples: int
    matched_samples: int
    match_rate: float
    effective_sample_size: float
    ess_ratio: float
    min_propensity_observed: float
    action_distribution: Dict[str, int]
    # Direct Method (model-based)
    dm_net_acceptance_rate: float
    dm_expected_loss_bps: float
    dm_expected_net_value_cents: float
    # IPS / SNIPS / DR off-policy estimators
    ips_net_acceptance_rate: float
    snips_net_acceptance_rate: float
    dr_net_acceptance_rate: float
    dr_expected_loss_bps: float
    dr_expected_net_value_cents: float
    # Bootstrap confidence bounds
    net_acceptance_bounds: ConfidenceBound
    expected_loss_bps_bounds: ConfidenceBound
    # Latency
    p50_latency_ms: float
    p95_latency_ms: float
    # Optional synthetic oracle comparison
    oracle_net_acceptance_rate: Optional[float] = None
    oracle_expected_loss_bps: Optional[float] = None
    oracle_expected_net_value_cents: Optional[float] = None
    oracle_p95_latency_ms: Optional[float] = None
    dr_vs_oracle_acceptance_abs_err: Optional[float] = None
    dr_vs_oracle_loss_bps_abs_err: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_exact_estimators(
    *,
    logged_actions: Sequence[PolicyAction],
    candidate_actions: Sequence[PolicyAction],
    propensities: Sequence[float],
    observed_rewards: Sequence[float],
    dm_candidate_rewards: Sequence[float],
    dm_logged_rewards: Sequence[float],
    ips_clip_min: float = 0.05,
    min_propensity_threshold: float = 0.02,
    min_ess_ratio: float = 0.10,
    strict_ess: bool = True,
) -> Dict[str, float]:
    """Pure, verifiable DM / IPS / SNIPS / DR computation for a scalar reward vector.

    Exposed directly so hand-calculated reference fixtures can verify exact arithmetic
    to tight numerical tolerance.
    """
    n = len(logged_actions)
    if n == 0:
        raise CounterfactualAssumptionError("EMPTY_EVALUATION_SET", "Cannot evaluate counterfactuals on 0 samples.")
    if not (
        len(candidate_actions) == n
        and len(propensities) == n
        and len(observed_rewards) == n
        and len(dm_candidate_rewards) == n
        and len(dm_logged_rewards) == n
    ):
        raise CounterfactualAssumptionError("DIMENSION_MISMATCH", "All counterfactual input sequences must have equal length.")

    prop = np.asarray(propensities, dtype=np.float64)
    y_obs = np.asarray(observed_rewards, dtype=np.float64)
    mu_cand = np.asarray(dm_candidate_rewards, dtype=np.float64)
    mu_log = np.asarray(dm_logged_rewards, dtype=np.float64)

    if not np.all(np.isfinite(prop)):
        raise CounterfactualAssumptionError("INVALID_PROPENSITY", "Propensity scores contain NaN or Inf.")
    if np.any(prop <= 0.0) or np.any(prop > 1.0):
        raise CounterfactualAssumptionError(
            "POSITIVITY_VIOLATION",
            f"All propensities must lie in (0, 1]; observed min={float(np.min(prop)):.6f}, max={float(np.max(prop)):.6f}.",
            diagnostics={"min_propensity": float(np.min(prop)), "max_propensity": float(np.max(prop))},
        )
    if float(np.min(prop)) < min_propensity_threshold:
        raise CounterfactualAssumptionError(
            "POSITIVITY_VIOLATION",
            f"Minimum propensity {float(np.min(prop)):.6f} is below required support threshold {min_propensity_threshold:.6f}.",
            diagnostics={"min_propensity": float(np.min(prop)), "threshold": min_propensity_threshold},
        )

    if not (np.all(np.isfinite(y_obs)) and np.all(np.isfinite(mu_cand)) and np.all(np.isfinite(mu_log))):
        raise CounterfactualAssumptionError("NON_FINITE_REWARDS", "Observed or predicted rewards contain NaN or Inf.")

    # Check logged support for every candidate action
    logged_set = set(logged_actions)
    missing_support = sorted({a.value for a in candidate_actions if a not in logged_set})
    if missing_support:
        raise CounterfactualAssumptionError(
            "INSUFFICIENT_SUPPORT",
            f"Candidate policy selects actions with zero support in logged data: {missing_support}",
            diagnostics={"unsupported_actions": missing_support},
        )

    matched = np.asarray([1.0 if la == ca else 0.0 for la, ca in zip(logged_actions, candidate_actions)], dtype=np.float64)
    if float(np.sum(matched)) == 0.0:
        raise CounterfactualAssumptionError(
            "ZERO_OVERLAP",
            "Zero logged observations matched the candidate policy actions.",
            diagnostics={"matched_samples": 0},
        )

    clipped_prop = np.maximum(prop, ips_clip_min)
    weights = matched / clipped_prop

    sum_w = float(np.sum(weights))
    sum_w2 = float(np.sum(weights * weights))
    ess = (sum_w * sum_w) / sum_w2 if sum_w2 > 0.0 else 0.0
    ess_ratio = ess / float(n)
    if strict_ess and ess_ratio < min_ess_ratio:
        raise CounterfactualAssumptionError(
            "ESS_COLLAPSE",
            f"Effective sample size ratio {ess_ratio:.4f} (ESS={ess:.2f}/{n}) fell below min_ess_ratio={min_ess_ratio:.4f}.",
            diagnostics={"ess": ess, "ess_ratio": ess_ratio, "min_ess_ratio": min_ess_ratio},
        )

    dm = float(np.mean(mu_cand))
    ips = float(np.mean(weights * y_obs))
    snips = float(np.sum(weights * y_obs) / sum_w)
    dr = float(np.mean(mu_cand + weights * (y_obs - mu_log)))

    return {
        "dm": dm,
        "ips": ips,
        "snips": snips,
        "dr": dr,
        "ess": ess,
        "ess_ratio": ess_ratio,
        "matched_samples": int(np.sum(matched)),
        "min_propensity": float(np.min(prop)),
    }


def _bootstrap_bounds(
    values: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
    confidence_level: float,
    ratio_denominator: Optional[np.ndarray] = None,
    scale: float = 1.0,
) -> ConfidenceBound:
    """Deterministic seeded percentile bootstrap confidence bounds."""
    n = len(values)
    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence_level) / 2.0

    if ratio_denominator is None:
        point = float(np.mean(values)) * scale
        samples = np.empty(n_resamples, dtype=np.float64)
        for b in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            samples[b] = float(np.mean(values[idx])) * scale
    else:
        denom_sum = float(np.sum(ratio_denominator))
        point = scale * (float(np.sum(values)) / max(1.0, denom_sum))
        samples = np.empty(n_resamples, dtype=np.float64)
        for b in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            d = float(np.sum(ratio_denominator[idx]))
            samples[b] = scale * (float(np.sum(values[idx])) / max(1.0, d))

    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    # Ensure bounds bracket point estimate cleanly in finite samples
    lo = min(lo, point)
    hi = max(hi, point)
    return ConfidenceBound(
        estimate=round(point, 6),
        lower=round(lo, 6),
        upper=round(hi, 6),
        confidence_level=confidence_level,
    )


def evaluate_policy_counterfactual(
    *,
    policy_name: str,
    attempts: Sequence[AuthorizationAttempt],
    outcomes: Sequence[HistoricalOutcome],
    predictions_per_attempt: Sequence[Dict[PolicyAction, ActionPrediction]],
    candidate_actions: Sequence[PolicyAction],
    config: EngineConfig,
    oracles: Optional[Sequence[ActionOracle]] = None,
    strict_ess: bool = True,
) -> PolicyCounterfactualReport:
    """Evaluate a candidate policy on logged outcomes using DM, IPS, SNIPS, DR, and optional oracle."""
    cf_cfg: CounterfactualConfig = config.counterfactual
    n = len(attempts)
    if n == 0:
        raise CounterfactualAssumptionError("EMPTY_EVALUATION_SET", "Cannot evaluate policy on 0 samples.")

    logged_actions = [o.logged_action for o in outcomes]
    propensities = [o.propensity_score for o in outcomes]

    # Observed & model-predicted acceptance
    obs_acc = [1.0 if o.authorized else 0.0 for o in outcomes]
    dm_cand_acc = [preds[ca].acceptance_prob for preds, ca in zip(predictions_per_attempt, candidate_actions)]
    dm_log_acc = [preds[la].acceptance_prob for preds, la in zip(predictions_per_attempt, logged_actions)]

    acc_est = compute_exact_estimators(
        logged_actions=logged_actions,
        candidate_actions=candidate_actions,
        propensities=propensities,
        observed_rewards=obs_acc,
        dm_candidate_rewards=dm_cand_acc,
        dm_logged_rewards=dm_log_acc,
        ips_clip_min=cf_cfg.ips_clip_min,
        min_propensity_threshold=cf_cfg.min_propensity_threshold,
        min_ess_ratio=cf_cfg.min_ess_ratio,
        strict_ess=strict_ess,
    )

    # Observed & predicted loss and net value (for immature outcomes in test set, fall back to model expectation
    # on unobserved fraud while flagging clearly, or use oracle/matured loss when available)
    margin_rate = config.economics.merchant_margin_bps / 10_000.0
    obs_loss_cents: List[float] = []
    obs_approved_amt: List[float] = []
    obs_net_val_cents: List[float] = []
    dm_cand_loss: List[float] = []
    dm_log_loss: List[float] = []
    dm_cand_amt: List[float] = []
    dm_log_amt: List[float] = []
    dm_cand_net: List[float] = []
    dm_log_net: List[float] = []
    latencies: List[float] = []

    for i, (att, out, preds, ca) in enumerate(zip(attempts, outcomes, predictions_per_attempt, candidate_actions)):
        la = out.logged_action
        cand_pred = preds[ca]
        log_pred = preds[la]

        if out.fraud_confirmed is not None:
            o_loss = float(out.loss_cents)
        else:
            # Right-censored outcome: substitute calibrated loss expectation conditioned on authorization
            if out.authorized:
                lgd = att.amount_cents * config.economics.loss_given_fraud_multiplier + config.economics.chargeback_fee_cents
                o_loss = float(log_pred.fraud_prob_given_approved * lgd)
            else:
                o_loss = 0.0

        o_amt = float(att.amount_cents) if out.authorized else 0.0
        if la == PolicyAction.STEP_UP_3DS:
            la_cost = float(config.economics.step_up_3ds_cost_cents)
        elif la == PolicyAction.ROUTE_SECONDARY:
            la_cost = float(config.economics.secondary_route_cost_cents)
        elif la == PolicyAction.APPROVE_PRIMARY:
            pcfg = config.processors.get(att.processor_id)
            la_cost = float(pcfg.cost_cents if pcfg else 0)
        else:
            la_cost = 0.0
        o_net = (o_amt * margin_rate) - o_loss - la_cost

        obs_loss_cents.append(o_loss)
        obs_approved_amt.append(o_amt)
        obs_net_val_cents.append(o_net)

        dm_cand_loss.append(cand_pred.expected_loss_cents)
        dm_log_loss.append(log_pred.expected_loss_cents)
        dm_cand_amt.append(cand_pred.acceptance_prob * att.amount_cents)
        dm_log_amt.append(log_pred.acceptance_prob * att.amount_cents)
        dm_cand_net.append(cand_pred.expected_net_value_cents)
        dm_log_net.append(log_pred.expected_net_value_cents)
        latencies.append(cand_pred.estimated_latency_ms)

    # DR per-sample influence terms for bootstrap confidence intervals
    prop_arr = np.maximum(np.asarray(propensities, dtype=np.float64), cf_cfg.ips_clip_min)
    matched_arr = np.asarray([1.0 if la == ca else 0.0 for la, ca in zip(logged_actions, candidate_actions)], dtype=np.float64)
    w_arr = matched_arr / prop_arr

    dr_acc_terms = np.asarray(dm_cand_acc) + w_arr * (np.asarray(obs_acc) - np.asarray(dm_log_acc))
    dr_loss_terms = np.asarray(dm_cand_loss) + w_arr * (np.asarray(obs_loss_cents) - np.asarray(dm_log_loss))
    dr_amt_terms = np.asarray(dm_cand_amt) + w_arr * (np.asarray(obs_approved_amt) - np.asarray(dm_log_amt))
    dr_net_terms = np.asarray(dm_cand_net) + w_arr * (np.asarray(obs_net_val_cents) - np.asarray(dm_log_net))

    # Clamp negative DR loss tail only at aggregate reporting floor of 0 bps
    dm_loss_bps = 10_000.0 * (float(np.sum(dm_cand_loss)) / max(1.0, float(np.sum(dm_cand_amt))))
    dr_loss_bps_raw = 10_000.0 * (float(np.sum(dr_loss_terms)) / max(1.0, float(np.sum(dr_amt_terms))))
    dr_loss_bps = max(0.0, dr_loss_bps_raw)

    acc_bounds = _bootstrap_bounds(
        np.clip(dr_acc_terms, 0.0, 1.0),
        seed=config.seed + 101,
        n_resamples=cf_cfg.bootstrap_resamples,
        confidence_level=cf_cfg.confidence_level,
    )
    loss_bounds_raw = _bootstrap_bounds(
        np.maximum(0.0, dr_loss_terms),
        seed=config.seed + 202,
        n_resamples=cf_cfg.bootstrap_resamples,
        confidence_level=cf_cfg.confidence_level,
        ratio_denominator=np.maximum(1.0, dr_amt_terms),
        scale=10_000.0,
    )
    loss_bounds = ConfidenceBound(
        estimate=round(dr_loss_bps, 4),
        lower=round(max(0.0, min(loss_bounds_raw.lower, dr_loss_bps)), 4),
        upper=round(max(loss_bounds_raw.upper, dr_loss_bps), 4),
        confidence_level=cf_cfg.confidence_level,
    )

    action_dist: Dict[str, int] = {}
    for ca in candidate_actions:
        action_dist[ca.value] = action_dist.get(ca.value, 0) + 1

    lat_arr = np.asarray(latencies, dtype=np.float64)
    p50_lat = float(np.percentile(lat_arr, 50))
    p95_lat = float(np.percentile(lat_arr, 95))

    oracle_acc: Optional[float] = None
    oracle_loss_bps: Optional[float] = None
    oracle_net: Optional[float] = None
    oracle_p95: Optional[float] = None
    acc_err: Optional[float] = None
    loss_err: Optional[float] = None

    if oracles is not None:
        o_accs = [o.acceptance_prob[ca] for o, ca in zip(oracles, candidate_actions)]
        o_losses = [o.expected_loss_cents[ca] for o, ca in zip(oracles, candidate_actions)]
        o_amts = [o.acceptance_prob[ca] * a.amount_cents for a, o, ca in zip(attempts, oracles, candidate_actions)]
        o_nets = [o.expected_net_value_cents[ca] for o, ca in zip(oracles, candidate_actions)]
        o_lats = [o.latency_ms[ca] for o, ca in zip(oracles, candidate_actions)]
        oracle_acc = round(float(np.mean(o_accs)), 6)
        oracle_loss_bps = round(10_000.0 * (float(np.sum(o_losses)) / max(1.0, float(np.sum(o_amts)))), 4)
        oracle_net = round(float(np.mean(o_nets)), 4)
        oracle_p95 = round(float(np.percentile(np.asarray(o_lats, dtype=np.float64), 95)), 2)
        acc_err = round(abs(acc_est["dr"] - oracle_acc), 6)
        loss_err = round(abs(dr_loss_bps - oracle_loss_bps), 4)

    return PolicyCounterfactualReport(
        policy_name=policy_name,
        n_samples=n,
        matched_samples=int(acc_est["matched_samples"]),
        match_rate=round(acc_est["matched_samples"] / float(n), 4),
        effective_sample_size=round(acc_est["ess"], 2),
        ess_ratio=round(acc_est["ess_ratio"], 4),
        min_propensity_observed=round(acc_est["min_propensity"], 6),
        action_distribution=action_dist,
        dm_net_acceptance_rate=round(acc_est["dm"], 6),
        dm_expected_loss_bps=round(dm_loss_bps, 4),
        dm_expected_net_value_cents=round(float(np.mean(dm_cand_net)), 4),
        ips_net_acceptance_rate=round(acc_est["ips"], 6),
        snips_net_acceptance_rate=round(acc_est["snips"], 6),
        dr_net_acceptance_rate=round(acc_est["dr"], 6),
        dr_expected_loss_bps=round(dr_loss_bps, 4),
        dr_expected_net_value_cents=round(float(np.mean(dr_net_terms)), 4),
        net_acceptance_bounds=acc_bounds,
        expected_loss_bps_bounds=loss_bounds,
        p50_latency_ms=round(p50_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        oracle_net_acceptance_rate=oracle_acc,
        oracle_expected_loss_bps=oracle_loss_bps,
        oracle_expected_net_value_cents=oracle_net,
        oracle_p95_latency_ms=oracle_p95,
        dr_vs_oracle_acceptance_abs_err=acc_err,
        dr_vs_oracle_loss_bps_abs_err=loss_err,
    )
