"""Deterministic synthetic authorization dataset and ground-truth oracle generator.

All data generated here is purely synthetic and contains zero real transactions,
cardholder data, or external credentials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np

from payment_auth_engine.config import EngineConfig
from payment_auth_engine.schemas import (
    EVALUABLE_ACTIONS,
    AuthorizationAttempt,
    HistoricalOutcome,
    PolicyAction,
)

MCCS = ("5411", "5812", "5732", "5999", "4111", "5967", "6051", "7995")
HIGH_RISK_MCCS = frozenset({"5967", "6051", "7995"})
NETWORKS = ("visa", "mastercard", "amex", "discover")
COUNTRIES = ("US", "GB", "DE", "CA", "AU")
CURRENCIES = ("USD", "GBP", "EUR", "CAD", "AUD")
ENTRY_MODES = ("ecommerce", "chip", "contactless", "recurring", "moto")
PROCESSORS = ("proc_a", "proc_b", "proc_c")


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class ActionOracle:
    """True synthetic data-generating probabilities and latencies for one attempt across actions."""

    acceptance_prob: Dict[PolicyAction, float]
    fraud_prob_given_approved: Dict[PolicyAction, float]
    expected_loss_cents: Dict[PolicyAction, float]
    expected_net_value_cents: Dict[PolicyAction, float]
    latency_ms: Dict[PolicyAction, float]


@dataclass
class SyntheticDataset:
    """Bundle of synthetic attempts, logged outcomes, and oracle ground-truth values."""

    attempts: List[AuthorizationAttempt]
    outcomes: List[HistoricalOutcome]
    oracles: List[ActionOracle]
    as_of_utc: datetime
    seed: int


def compute_oracle_for_attempt(attempt: AuthorizationAttempt, config: EngineConfig) -> ActionOracle:
    """Deterministic latent synthetic ground-truth model for counterfactual verification."""
    risk = attempt.risk_score if attempt.risk_score is not None else 0.35
    bin_rate = attempt.bin_approval_rate_30d if attempt.bin_approval_rate_30d is not None else 0.85
    vel1 = float(attempt.velocity_1h if attempt.velocity_1h is not None else 2)
    vel24 = float(attempt.velocity_24h if attempt.velocity_24h is not None else 5)
    age = float(attempt.account_age_days if attempt.account_age_days is not None else 90)
    cross_border = 1.0 if attempt.issuer_country != attempt.merchant_country else 0.0
    high_risk_mcc = 1.0 if attempt.merchant_category_code in HIGH_RISK_MCCS else 0.0
    is_ecom = 1.0 if attempt.entry_mode == "ecommerce" else 0.0
    log_amt = math.log1p(attempt.amount_cents / 100.0)

    proc_cfg = config.processors.get(attempt.processor_id)
    primary_bias = proc_cfg.approval_bias if proc_cfg is not None else 0.0
    primary_lat = proc_cfg.base_latency_ms if proc_cfg is not None else 150.0
    secondary_cfg = config.processors.get("proc_b", proc_cfg)
    secondary_bias = secondary_cfg.approval_bias if secondary_cfg is not None else 0.04
    secondary_lat = secondary_cfg.base_latency_ms if secondary_cfg is not None else 210.0

    # Latent acceptance logit (realistic signal from BIN approval, risk, token, cross-border)
    base_acc_logit = (
        1.40
        + 5.2 * (bin_rate - 0.82)
        - 2.60 * risk
        - 0.75 * cross_border
        - 0.45 * high_risk_mcc
        - 0.14 * max(0.0, log_amt - 4.0)
        + 0.60 * (1.0 if attempt.has_network_token else 0.0)
        + 0.25 * (1.0 if attempt.has_3ds else 0.0)
        - 0.20 * math.log1p(vel1)
    )

    # Latent fraud logit given approved (strong realistic risk gradient: ~0.15% low-risk -> ~25%+ high-risk)
    base_fraud_logit = (
        -7.20
        + 8.20 * risk
        + 0.70 * cross_border
        + 0.85 * high_risk_mcc
        + 0.35 * is_ecom
        + 0.30 * math.log1p(vel1)
        + 0.15 * math.log1p(vel24)
        - 0.25 * math.log1p(age / 30.0)
        - 0.45 * (1.0 if attempt.has_network_token else 0.0)
        - 0.65 * (1.0 if attempt.has_3ds else 0.0)
    )

    acc_map: Dict[PolicyAction, float] = {}
    fraud_map: Dict[PolicyAction, float] = {}
    loss_map: Dict[PolicyAction, float] = {}
    net_val_map: Dict[PolicyAction, float] = {}
    lat_map: Dict[PolicyAction, float] = {}

    margin_rate = config.economics.merchant_margin_bps / 10_000.0
    lgd_cents = (
        attempt.amount_cents * config.economics.loss_given_fraud_multiplier
        + config.economics.chargeback_fee_cents
    )

    for action in EVALUABLE_ACTIONS:
        if action == PolicyAction.DECLINE_POLICY:
            p_acc = 0.0
            p_fraud = 0.0
            action_cost = 0.0
            latency = 25.0
        elif action == PolicyAction.APPROVE_PRIMARY:
            p_acc = _sigmoid(base_acc_logit + 2.5 * primary_bias)
            p_fraud = _sigmoid(base_fraud_logit)
            action_cost = float(proc_cfg.cost_cents if proc_cfg else 0)
            latency = primary_lat + 18.0 * cross_border
        elif action == PolicyAction.ROUTE_SECONDARY:
            # Secondary processor boosts acceptance on cross-border & low-BIN transactions; slightly lower on domestic clean
            low_bin_boost = 0.75 if bin_rate < 0.85 else 0.0
            p_acc = _sigmoid(base_acc_logit + 0.90 * cross_border + low_bin_boost - 0.12)
            p_fraud = _sigmoid(base_fraud_logit - 0.05)
            action_cost = float(config.economics.secondary_route_cost_cents)
            latency = secondary_lat + 25.0 * cross_border
        elif action == PolicyAction.STEP_UP_3DS:
            # 3DS introduces modest checkout friction (slightly lower acceptance) but strongly suppresses fraud
            p_acc = _sigmoid(base_acc_logit - (0.38 if is_ecom else 0.15))
            p_fraud = _sigmoid(base_fraud_logit - 2.10)
            action_cost = float(config.economics.step_up_3ds_cost_cents)
            latency = primary_lat + 260.0 + 30.0 * cross_border
        else:
            raise ValueError(f"Unexpected evaluable action: {action}")

        exp_loss = p_acc * p_fraud * lgd_cents
        exp_rev = p_acc * (attempt.amount_cents * margin_rate)
        exp_net = exp_rev - exp_loss - action_cost

        acc_map[action] = p_acc
        fraud_map[action] = p_fraud
        loss_map[action] = exp_loss
        net_val_map[action] = exp_net
        lat_map[action] = latency

    return ActionOracle(
        acceptance_prob=acc_map,
        fraud_prob_given_approved=fraud_map,
        expected_loss_cents=loss_map,
        expected_net_value_cents=net_val_map,
        latency_ms=lat_map,
    )


def _logging_policy_probs(attempt: AuthorizationAttempt) -> Dict[PolicyAction, float]:
    """Stochastic logging policy with guaranteed full support (positivity >= 0.08)."""
    risk = attempt.risk_score if attempt.risk_score is not None else 0.35
    cross_border = attempt.issuer_country != attempt.merchant_country
    if risk < 0.25:
        if cross_border:
            return {
                PolicyAction.APPROVE_PRIMARY: 0.42,
                PolicyAction.ROUTE_SECONDARY: 0.38,
                PolicyAction.STEP_UP_3DS: 0.12,
                PolicyAction.DECLINE_POLICY: 0.08,
            }
        return {
            PolicyAction.APPROVE_PRIMARY: 0.56,
            PolicyAction.ROUTE_SECONDARY: 0.24,
            PolicyAction.STEP_UP_3DS: 0.12,
            PolicyAction.DECLINE_POLICY: 0.08,
        }
    elif risk < 0.58:
        return {
            PolicyAction.APPROVE_PRIMARY: 0.34,
            PolicyAction.ROUTE_SECONDARY: 0.28,
            PolicyAction.STEP_UP_3DS: 0.24,
            PolicyAction.DECLINE_POLICY: 0.14,
        }
    else:
        return {
            PolicyAction.APPROVE_PRIMARY: 0.12,
            PolicyAction.ROUTE_SECONDARY: 0.12,
            PolicyAction.STEP_UP_3DS: 0.32,
            PolicyAction.DECLINE_POLICY: 0.44,
        }


def generate_synthetic_dataset(config: EngineConfig) -> SyntheticDataset:
    """Generate a reproducible synthetic payment authorization dataset with temporal ordering."""
    rng = np.random.default_rng(config.seed)
    n = config.data.n_samples
    start_dt = datetime.fromisoformat(config.data.start_timestamp_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    total_seconds = config.data.span_days * 86400
    as_of_utc = start_dt + timedelta(seconds=total_seconds)

    # Strictly non-decreasing timestamps across the span
    offsets = np.sort(rng.uniform(0.0, float(total_seconds), size=n))

    attempts: List[AuthorizationAttempt] = []
    outcomes: List[HistoricalOutcome] = []
    oracles: List[ActionOracle] = []

    lgd_mult = config.economics.loss_given_fraud_multiplier
    cb_fee = config.economics.chargeback_fee_cents

    for i in range(n):
        ts = start_dt + timedelta(seconds=float(offsets[i]))
        country_idx = int(rng.choice(len(COUNTRIES), p=[0.52, 0.16, 0.14, 0.10, 0.08]))
        merchant_country = COUNTRIES[country_idx]
        currency = CURRENCIES[country_idx]
        if rng.random() < 0.22:
            issuer_country = str(rng.choice(COUNTRIES))
        else:
            issuer_country = merchant_country

        mcc = str(rng.choice(MCCS, p=[0.24, 0.18, 0.16, 0.14, 0.10, 0.06, 0.06, 0.06]))
        network = str(rng.choice(NETWORKS, p=[0.48, 0.34, 0.12, 0.06]))
        entry_mode = str(rng.choice(ENTRY_MODES, p=[0.55, 0.15, 0.12, 0.12, 0.06]))
        is_recurring = entry_mode == "recurring"
        has_3ds = bool(entry_mode == "ecommerce" and rng.random() < 0.22)
        has_token = bool(rng.random() < 0.35)

        amount_dollars = float(np.clip(rng.lognormal(mean=4.0, sigma=0.95), 2.0, 4500.0))
        amount_cents = int(round(amount_dollars * 100))

        # Correlated risk score (realistic bimodal mixture: clean majority + elevated-risk tail)
        if rng.random() < 0.16:
            base_risk = float(rng.beta(4.0, 2.2))
        else:
            base_risk = float(rng.beta(1.3, 6.5))
        if mcc in HIGH_RISK_MCCS:
            base_risk = min(0.99, base_risk + float(rng.uniform(0.10, 0.25)))
        if issuer_country != merchant_country:
            base_risk = min(0.99, base_risk + float(rng.uniform(0.04, 0.12)))
        risk_score = round(base_risk, 4)

        velocity_1h = int(rng.poisson(lam=0.8 + 4.5 * risk_score))
        velocity_24h = velocity_1h + int(rng.poisson(lam=2.0 + 7.0 * risk_score))
        account_age_days = int(max(0, round(rng.exponential(scale=max(20.0, 320.0 * (1.0 - 0.7 * risk_score))))))
        bin_approval = round(float(np.clip(rng.normal(0.89 - 0.10 * risk_score, 0.06), 0.45, 0.99)), 4)

        # Inject controlled realistic missingness (approx 4% risk_score, 5% bin rate)
        miss_risk = None if rng.random() < 0.03 else risk_score
        miss_v1 = None if rng.random() < 0.04 else velocity_1h
        miss_v24 = None if rng.random() < 0.04 else velocity_24h
        miss_age = None if rng.random() < 0.05 else account_age_days
        miss_bin = None if rng.random() < 0.05 else bin_approval

        processor_id = str(rng.choice(PROCESSORS, p=[0.60, 0.25, 0.15]))

        attempt = AuthorizationAttempt(
            attempt_id=f"att_{i:06d}",
            transaction_id=f"txn_{i:06d}",
            idempotency_key=f"idem_{i:06d}",
            timestamp_utc=ts,
            amount_cents=amount_cents,
            currency=currency,
            merchant_id=f"merch_{int(rng.integers(1, 26)):03d}",
            merchant_category_code=mcc,
            card_network=network,  # type: ignore[arg-type]
            issuer_country=issuer_country,
            merchant_country=merchant_country,
            entry_mode=entry_mode,  # type: ignore[arg-type]
            is_recurring=is_recurring,
            has_3ds=has_3ds,
            has_network_token=has_token,
            attempt_number=1,
            prior_response_code=None,
            prior_outcome_known=True,
            risk_score=miss_risk,
            velocity_1h=miss_v1,
            velocity_24h=miss_v24,
            account_age_days=miss_age,
            bin_approval_rate_30d=miss_bin,
            processor_id=processor_id,
        )

        # Compute ground truth using the full latent risk (reconstruct proxy if missing)
        oracle_attempt = attempt.model_copy(
            update={
                "risk_score": risk_score,
                "velocity_1h": velocity_1h,
                "velocity_24h": velocity_24h,
                "account_age_days": account_age_days,
                "bin_approval_rate_30d": bin_approval,
            }
        )
        oracle = compute_oracle_for_attempt(oracle_attempt, config)

        # Sample logged action from stochastic logging policy
        probs_map = _logging_policy_probs(attempt)
        actions_list = list(EVALUABLE_ACTIONS)
        probs_arr = np.asarray([probs_map[a] for a in actions_list], dtype=np.float64)
        logged_action = actions_list[int(rng.choice(len(actions_list), p=probs_arr))]
        propensity = float(probs_map[logged_action])

        # Simulate outcome under logged action
        p_acc = oracle.acceptance_prob[logged_action]
        authorized = bool(rng.random() < p_acc) if logged_action != PolicyAction.DECLINE_POLICY else False
        if logged_action == PolicyAction.DECLINE_POLICY:
            resp_code = "59"
        elif authorized:
            resp_code = "00"
        else:
            resp_code = str(rng.choice(["05", "51", "65", "91", "14"], p=[0.45, 0.20, 0.15, 0.10, 0.10]))

        obs_latency = max(5.0, float(oracle.latency_ms[logged_action] + rng.normal(0.0, 12.0)))

        maturity_days = max(0, int((as_of_utc - ts).total_seconds() // 86400))
        is_matured = maturity_days >= config.data.min_label_maturity_days

        if authorized and is_matured:
            fraud = bool(rng.random() < oracle.fraud_prob_given_approved[logged_action])
            loss_cents = int(round(amount_cents * lgd_mult + cb_fee)) if fraud else 0
            matured_at = ts + timedelta(days=config.data.min_label_maturity_days)
        elif not authorized and is_matured:
            fraud = False
            loss_cents = 0
            matured_at = ts + timedelta(days=config.data.min_label_maturity_days)
        else:
            # Right-censored / immature outcome
            fraud = None
            loss_cents = 0
            matured_at = None

        outcome = HistoricalOutcome(
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            attempt_timestamp_utc=ts,
            logged_action=logged_action,
            propensity_score=propensity,
            authorized=authorized,
            response_code=resp_code,
            latency_ms=round(obs_latency, 2),
            fraud_confirmed=fraud,
            loss_cents=loss_cents,
            label_maturity_days=maturity_days,
            matured_at_utc=matured_at,
        )

        attempts.append(attempt)
        outcomes.append(outcome)
        oracles.append(oracle)

    return SyntheticDataset(
        attempts=attempts,
        outcomes=outcomes,
        oracles=oracles,
        as_of_utc=as_of_utc,
        seed=config.seed,
    )
