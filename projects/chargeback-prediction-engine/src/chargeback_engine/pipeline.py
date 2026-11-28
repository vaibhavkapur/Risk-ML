"""End-to-end orchestration, reference fixture verification, and structured diagnostics."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import sklearn

from chargeback_engine.config import EngineConfig
from chargeback_engine.contracts import (
    SCHEMA_VERSION,
    LabelStatus,
    parse_utc_timestamp,
    validate_dataset_contracts,
)
from chargeback_engine.features import (
    FeatureSchemaMetadata,
    build_causal_features,
    extract_feature_matrix,
)
from chargeback_engine.maturity import (
    MaturityDiagnostics,
    assign_maturity_labels,
    assign_temporal_splits,
)
from chargeback_engine.models import (
    CalibratedDisputeModel,
    compute_expected_loss,
    compute_expected_review_savings,
    compute_loss_given_dispute,
    evaluate_predictions,
    train_calibrated_model,
)
from chargeback_engine.policy import (
    fit_empirical_delay_cdf,
    prioritize_reviews,
    simulate_delayed_outcome_monitoring,
)


@dataclass
class PipelineResult:
    run_id: str
    schema_version: str
    config_hash: str
    feature_metadata: FeatureSchemaMetadata
    maturity_diagnostics: MaturityDiagnostics
    split_summary: Dict[str, Any]
    baseline_model_meta: Dict[str, Any]
    primary_model_meta: Dict[str, Any]
    metrics_baseline_logistic: Dict[str, float]
    metrics_primary_uncalibrated: Dict[str, float]
    metrics_primary_calibrated: Dict[str, float]
    cold_start_slice_metrics: Dict[str, Dict[str, float]]
    policy_comparison: Dict[str, Any]
    delayed_outcome_monitoring: List[Dict[str, Any]]
    sample_explanations: List[Dict[str, Any]]
    structured_events: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "feature_metadata": {
                "feature_set_version": self.feature_metadata.feature_set_version,
                "feature_schema_hash": self.feature_metadata.feature_schema_hash,
                "n_features": len(self.feature_metadata.feature_columns),
                "prior_dispute_rate": round(
                    self.feature_metadata.prior_dispute_rate, 6
                ),
            },
            "maturity_diagnostics": self.maturity_diagnostics.to_dict(),
            "split_summary": self.split_summary,
            "models": {
                "baseline_logistic": self.baseline_model_meta,
                "primary_calibrated": self.primary_model_meta,
            },
            "test_metrics": {
                "baseline_logistic_calibrated": self.metrics_baseline_logistic,
                "primary_uncalibrated": self.metrics_primary_uncalibrated,
                "primary_calibrated": self.metrics_primary_calibrated,
            },
            "cold_start_slice_metrics": self.cold_start_slice_metrics,
            "policy_comparison": self.policy_comparison,
            "delayed_outcome_monitoring": self.delayed_outcome_monitoring,
            "sample_explanations": self.sample_explanations,
            "structured_events": self.structured_events,
        }


def collect_runtime_metadata(cfg: EngineConfig) -> Dict[str, Any]:
    """Record runtime environment and dependency versions for reproducibility."""
    return {
        "schema_version": SCHEMA_VERSION,
        "config_hash": cfg.config_hash(),
        "seed": cfg.seed,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "polars": pl.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "config": cfg.to_dict(),
    }


def _compute_run_id(
    cfg_hash: str,
    as_of_ts: datetime,
    n_merchants: int,
    n_tx: int,
    n_disputes: int,
) -> str:
    payload = f"{SCHEMA_VERSION}:{cfg_hash}:{as_of_ts.isoformat()}:{n_merchants}:{n_tx}:{n_disputes}"
    return f"run_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:14]}"


def run_pipeline(
    cfg: EngineConfig,
    merchants_raw: Sequence[Dict[str, Any]],
    transactions_raw: Sequence[Dict[str, Any]],
    disputes_raw: Sequence[Dict[str, Any]],
    *,
    as_of_ts: Optional[datetime] = None,
) -> PipelineResult:
    """Execute full chargeback prediction pipeline from raw inputs to evaluated policy."""
    cfg.validate()
    snapshot_ts = (
        as_of_ts
        if as_of_ts is not None
        else parse_utc_timestamp(cfg.synthetic.as_of_date, "as_of_date")
    )
    events: List[Dict[str, Any]] = []

    # 1. Contract validation
    merchants, transactions, disputes = validate_dataset_contracts(
        merchants_raw, transactions_raw, disputes_raw
    )
    run_id = _compute_run_id(
        cfg.config_hash(), snapshot_ts, len(merchants), len(transactions), len(disputes)
    )
    events.append(
        {
            "stage": "contract_validation",
            "status": "ok",
            "merchants": len(merchants),
            "transactions": len(transactions),
            "disputes": len(disputes),
        }
    )

    # 2. Observation snapshot & label maturity assignment
    labeled_df, maturity_diag = assign_maturity_labels(
        merchants=merchants,
        transactions=transactions,
        disputes=disputes,
        as_of_ts=snapshot_ts,
        maturity_cfg=cfg.maturity,
        missing_reason_code_label=cfg.features.missing_reason_code_label,
    )
    events.append(
        {
            "stage": "maturity_labeling",
            "status": "ok",
            "mature_positive": maturity_diag.mature_positive_count,
            "mature_negative": maturity_diag.mature_negative_count,
            "immature_positive": maturity_diag.immature_positive_count,
            "immature_negative_excluded": maturity_diag.immature_negative_count,
            "disputes_masked_after_as_of": maturity_diag.disputes_after_as_of_masked,
            "missing_reason_codes": maturity_diag.missing_reason_codes,
        }
    )

    # 3. Chronological temporal split on trainable mature records
    train_raw, cal_raw, test_raw = assign_temporal_splits(labeled_df, cfg.split)
    train_prior = float(train_raw["label"].mean())
    if not (0.0 < train_prior < 1.0):
        train_prior = cfg.features.fallback_prior_dispute_rate

    # 4. Causal point-in-time feature generation using strictly training-set global prior
    featured_df, feat_meta = build_causal_features(
        labeled_df=labeled_df,
        feature_cfg=cfg.features,
        feature_set_version=cfg.feature_set_version,
        prior_dispute_rate=train_prior,
    )
    events.append(
        {
            "stage": "causal_feature_engineering",
            "status": "ok",
            "feature_set_version": feat_meta.feature_set_version,
            "feature_schema_hash": feat_meta.feature_schema_hash,
            "n_features": len(feat_meta.feature_columns),
            "prior_dispute_rate": round(feat_meta.prior_dispute_rate, 6),
        }
    )

    # Re-slice featured_df using exact split transaction IDs (preserves temporal boundaries)
    train_ids = set(train_raw["transaction_id"].to_list())
    cal_ids = set(cal_raw["transaction_id"].to_list())
    test_ids = set(test_raw["transaction_id"].to_list())

    train_df = featured_df.filter(pl.col("transaction_id").is_in(list(train_ids))).sort(
        ["transaction_ts", "transaction_id"]
    )
    cal_df = featured_df.filter(pl.col("transaction_id").is_in(list(cal_ids))).sort(
        ["transaction_ts", "transaction_id"]
    )
    test_df = featured_df.filter(pl.col("transaction_id").is_in(list(test_ids))).sort(
        ["transaction_ts", "transaction_id"]
    )
    immature_df = featured_df.filter(~pl.col("is_mature")).sort(
        ["transaction_ts", "transaction_id"]
    )

    split_summary = {
        "train_rows": train_df.height,
        "train_positives": int(train_df["label"].sum()),
        "train_min_ts": train_df["transaction_ts"].min().isoformat(),
        "train_max_ts": train_df["transaction_ts"].max().isoformat(),
        "calibration_rows": cal_df.height,
        "calibration_positives": int(cal_df["label"].sum()),
        "calibration_min_ts": cal_df["transaction_ts"].min().isoformat(),
        "calibration_max_ts": cal_df["transaction_ts"].max().isoformat(),
        "test_rows": test_df.height,
        "test_positives": int(test_df["label"].sum()),
        "test_min_ts": test_df["transaction_ts"].min().isoformat(),
        "test_max_ts": test_df["transaction_ts"].max().isoformat(),
        "immature_monitoring_rows": immature_df.height,
    }

    # 5. Train baseline (logistic) and primary (hist_gbdt / configured) calibrated models
    fcols = feat_meta.feature_columns
    X_train = extract_feature_matrix(train_df, fcols)
    y_train = train_df["label"].to_numpy().astype(np.int64)
    X_cal = extract_feature_matrix(cal_df, fcols)
    y_cal = cal_df["label"].to_numpy().astype(np.int64)
    X_test = extract_feature_matrix(test_df, fcols)
    y_test = test_df["label"].to_numpy().astype(np.int64)
    amounts_test = test_df["amount_usd"].to_numpy().astype(np.float64)

    baseline_model = train_calibrated_model(
        X_train,
        y_train,
        X_cal,
        y_cal,
        fcols,
        cfg.model,
        model_family_override="logistic",
        apply_calibration=True,
        seed=cfg.seed,
        config_hash=cfg.config_hash(),
        feature_schema_hash=feat_meta.feature_schema_hash,
    )
    primary_model = train_calibrated_model(
        X_train,
        y_train,
        X_cal,
        y_cal,
        fcols,
        cfg.model,
        apply_calibration=True,
        seed=cfg.seed,
        config_hash=cfg.config_hash(),
        feature_schema_hash=feat_meta.feature_schema_hash,
    )

    p_baseline = baseline_model.predict_proba(X_test)
    p_primary_raw = primary_model.predict_raw_proba(X_test)
    p_primary_cal = primary_model.predict_proba(X_test)

    metrics_baseline = evaluate_predictions(
        y_test, p_baseline, amounts_test, cfg.economics, cfg.policy.review_budget_fraction
    )
    metrics_raw = evaluate_predictions(
        y_test, p_primary_raw, amounts_test, cfg.economics, cfg.policy.review_budget_fraction
    )
    metrics_cal = evaluate_predictions(
        y_test, p_primary_cal, amounts_test, cfg.economics, cfg.policy.review_budget_fraction
    )

    events.append(
        {
            "stage": "model_evaluation",
            "status": "ok",
            "primary_model_id": primary_model.model_id,
            "pr_auc": metrics_cal["pr_auc"],
            "brier_score": metrics_cal["brier_score"],
            "ece": metrics_cal["ece"],
            "precision_at_review_budget": metrics_cal[
                "precision_at_review_budget_by_value"
            ],
        }
    )

    # 6. Cold-start vs established merchant slice evaluation on test holdout
    cold_mask = test_df["m_is_cold_start"].to_numpy() > 0.5
    est_mask = ~cold_mask
    cold_start_slices: Dict[str, Dict[str, float]] = {}
    if int(cold_mask.sum()) > 0:
        cold_start_slices["cold_start_merchants"] = evaluate_predictions(
            y_test[cold_mask],
            p_primary_cal[cold_mask],
            amounts_test[cold_mask],
            cfg.economics,
            cfg.policy.review_budget_fraction,
        )
    if int(est_mask.sum()) > 0:
        cold_start_slices["established_merchants"] = evaluate_predictions(
            y_test[est_mask],
            p_primary_cal[est_mask],
            amounts_test[est_mask],
            cfg.economics,
            cfg.policy.review_budget_fraction,
        )

    # 7. Review prioritization policy simulation (value-ranked vs prob-ranked vs baseline)
    decisions_value, summary_value = prioritize_reviews(
        test_df, p_primary_cal, cfg.economics, cfg.policy, rank_by="expected_net_savings"
    )
    _, summary_prob = prioritize_reviews(
        test_df, p_primary_cal, cfg.economics, cfg.policy, rank_by="probability"
    )
    _, summary_baseline = prioritize_reviews(
        test_df, p_baseline, cfg.economics, cfg.policy, rank_by="expected_net_savings"
    )

    policy_comparison = {
        "primary_expected_net_savings_policy": summary_value,
        "primary_probability_only_policy": summary_prob,
        "baseline_logistic_value_policy": summary_baseline,
    }

    # 8. Decision explanations for top 5 prioritized review transactions
    tx_id_to_row_idx = {
        tid: i for i, tid in enumerate(test_df["transaction_id"].to_list())
    }
    sample_explanations: List[Dict[str, Any]] = []
    for dec in decisions_value[:5]:
        ridx = tx_id_to_row_idx[dec.transaction_id]
        sample_explanations.append(
            {
                "transaction_id": dec.transaction_id,
                "merchant_id": dec.merchant_id,
                "action": dec.action,
                "calibrated_prob": dec.calibrated_prob,
                "amount_usd": dec.amount_usd,
                "expected_loss_usd": dec.expected_loss_usd,
                "expected_net_savings_usd": dec.expected_net_savings_usd,
                "top_drivers": primary_model.explain_row(X_test[ridx], top_k=5),
            }
        )

    # 9. Delayed-outcome monitoring simulation on immature cohort (or test cohort fallback)
    delay_cdf = fit_empirical_delay_cdf(
        train_df, max_days=cfg.maturity.max_dispute_delay_days
    )
    monitor_frame = immature_df if immature_df.height > 0 else test_df
    X_mon = extract_feature_matrix(monitor_frame, fcols)
    p_mon = primary_model.predict_proba(X_mon)
    mon_decisions, _ = prioritize_reviews(
        monitor_frame, p_mon, cfg.economics, cfg.policy, rank_by="expected_net_savings"
    )
    reviewed_mon_ids = [
        d.transaction_id for d in mon_decisions if d.action == "manual_review"
    ]
    monitoring_snapshots = simulate_delayed_outcome_monitoring(
        monitored_df=monitor_frame,
        predicted_probs=p_mon,
        reviewed_tx_ids=reviewed_mon_ids,
        all_disputes=disputes,
        delay_cdf=delay_cdf,
        start_as_of_ts=snapshot_ts,
        checkpoint_offsets_days=(0, 15, 30, 60, 90, 120),
    )

    events.append(
        {
            "stage": "delayed_outcome_monitoring",
            "status": "ok",
            "monitored_cohort_size": monitor_frame.height,
            "reviewed_queue_size": len(reviewed_mon_ids),
            "checkpoints": len(monitoring_snapshots),
        }
    )

    return PipelineResult(
        run_id=run_id,
        schema_version=SCHEMA_VERSION,
        config_hash=cfg.config_hash(),
        feature_metadata=feat_meta,
        maturity_diagnostics=maturity_diag,
        split_summary=split_summary,
        baseline_model_meta=baseline_model.metadata_dict(),
        primary_model_meta=primary_model.metadata_dict(),
        metrics_baseline_logistic=metrics_baseline,
        metrics_primary_uncalibrated=metrics_raw,
        metrics_primary_calibrated=metrics_cal,
        cold_start_slice_metrics=cold_start_slices,
        policy_comparison=policy_comparison,
        delayed_outcome_monitoring=monitoring_snapshots,
        sample_explanations=sample_explanations,
        structured_events=events,
    )


def verify_reference_fixture(fixtures_dir: str | Path) -> Dict[str, Any]:
    """Run hand-calculated reference fixture and verify all invariants within tolerance."""
    fdir = Path(fixtures_dir)
    merchants_raw = json.loads((fdir / "reference_merchants.json").read_text("utf-8"))
    tx_raw = json.loads((fdir / "reference_transactions.json").read_text("utf-8"))
    disputes_raw = json.loads((fdir / "reference_disputes.json").read_text("utf-8"))
    expected = json.loads((fdir / "reference_expected.json").read_text("utf-8"))

    tol = float(expected["tolerance"])
    as_of_ts = parse_utc_timestamp(expected["as_of_ts"], "as_of_ts")
    cfg = EngineConfig()

    merchants, transactions, disputes = validate_dataset_contracts(
        merchants_raw, tx_raw, disputes_raw
    )
    labeled_df, diag = assign_maturity_labels(
        merchants,
        transactions,
        disputes,
        as_of_ts=as_of_ts,
        maturity_cfg=cfg.maturity,
    )
    featured_df, feat_meta = build_causal_features(
        labeled_df,
        cfg.features,
        prior_dispute_rate=float(expected["prior_dispute_rate"]),
    )

    # Check diagnostics
    diag_dict = diag.to_dict()
    for k, exp_val in expected["diagnostics"].items():
        actual_val = diag_dict[k]
        assert actual_val == exp_val, f"Diagnostic mismatch for {k}: {actual_val} != {exp_val}"

    # Check per-transaction hand-calculated expectations
    by_id = {r["transaction_id"]: r for r in featured_df.to_dicts()}
    for tx_id, exp_fields in expected["transactions"].items():
        row = by_id[tx_id]
        for field_name, exp_val in exp_fields.items():
            actual = row[field_name]
            if isinstance(actual, datetime):
                actual = actual.isoformat()
            if isinstance(exp_val, float):
                assert abs(float(actual) - exp_val) <= tol, (
                    f"{tx_id}.{field_name}: actual={actual} expected={exp_val}"
                )
            else:
                assert actual == exp_val, (
                    f"{tx_id}.{field_name}: actual={actual!r} expected={exp_val!r}"
                )

    # Check expected-loss oracle
    oracle = expected["expected_loss_oracle"]
    lgd = compute_loss_given_dispute(oracle["amount_usd"], cfg.economics)
    el = compute_expected_loss(oracle["probability"], oracle["amount_usd"], cfg.economics)
    gross, net = compute_expected_review_savings(
        oracle["probability"], oracle["amount_usd"], cfg.economics
    )
    assert abs(lgd - oracle["loss_given_dispute_usd"]) <= tol
    assert abs(el - oracle["expected_loss_usd"]) <= tol
    assert abs(gross - oracle["expected_gross_savings_usd"]) <= tol
    assert abs(net - oracle["expected_net_savings_usd"]) <= tol

    return {
        "status": "passed",
        "checked_transactions": len(expected["transactions"]),
        "feature_schema_hash": feat_meta.feature_schema_hash,
        "diagnostics": diag_dict,
    }
