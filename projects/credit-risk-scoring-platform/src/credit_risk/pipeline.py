"""End-to-end reproducible training, calibration, evaluation, and manual-review pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from credit_risk.calibration import PDCalibrator, save_calibration_plots
from credit_risk.config import PlatformConfig, load_config
from credit_risk.contracts import ApplicantRecord, LabelRecord
from credit_risk.data import (
    generate_synthetic_dataset,
    save_json_file,
    temporal_split_dataset,
)
from credit_risk.evaluation import (
    compare_against_baseline,
    compute_discrimination_and_calibration_metrics,
    compute_feature_stability_csi,
    compute_psi,
    evaluate_subgroups,
)
from credit_risk.features import FeaturePipeline
from credit_risk.inference import CreditRiskScorer
from credit_risk.models import BasePDModel, LogisticRegressionPDModel, train_baseline_models
from credit_risk.policy import ManualReviewPolicy
from credit_risk.reporting import collect_runtime_metadata, write_model_card
from credit_risk.review import ManualReviewStore


def _labels_to_array(labels: List[LabelRecord]) -> np.ndarray:
    vals = []
    for lbl in labels:
        if lbl.default_flag is None:
            raise ValueError(f"Immature label with null default_flag encountered: {lbl.application_id}")
        vals.append(int(lbl.default_flag))
    return np.asarray(vals, dtype=np.int64)


def run_end_to_end_pipeline(
    cfg: PlatformConfig,
    base_dir: Union[str, Path],
    n_review_queue_samples: int = 8,
    deterministic_timestamp: str = "2025-07-01T12:00:00Z",
) -> Dict[str, Any]:
    """Execute full temporal training, calibration, evaluation, reporting, and review-queue pipeline."""
    root = Path(base_dir)
    reports_dir = root / cfg.paths.reports_dir
    artifacts_dir = root / cfg.paths.artifacts_dir
    review_store_path = root / cfg.paths.review_store_path
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate deterministic point-in-time synthetic dataset.
    dataset = generate_synthetic_dataset(cfg.data, seed=cfg.project.seed)

    # 2. Temporal split enforcing label maturity.
    splits = temporal_split_dataset(
        applicants=dataset.applicants,
        labels=dataset.labels,
        train_end_date=cfg.temporal_split.train_end_date,
        calibration_end_date=cfg.temporal_split.calibration_end_date,
        test_end_date=cfg.temporal_split.test_end_date,
        require_label_maturity=cfg.temporal_split.require_label_maturity,
    )
    train_apps, train_lbls = splits["train"]
    cal_apps, cal_lbls = splits["calibration"]
    test_apps, test_lbls = splits["test"]
    censored_apps, _ = splits["censored_excluded"]

    y_train = _labels_to_array(train_lbls)
    y_cal = _labels_to_array(cal_lbls)
    y_test = _labels_to_array(test_lbls)

    # 3. Fit point-in-time feature pipeline strictly on training split.
    feature_pipeline = FeaturePipeline(feature_version=cfg.project.feature_version)
    X_train_std = feature_pipeline.fit_transform(train_apps, standardized=True)
    X_train_raw = feature_pipeline.transform(train_apps, standardized=False)

    X_cal_std = feature_pipeline.transform(cal_apps, standardized=True)
    X_cal_raw = feature_pipeline.transform(cal_apps, standardized=False)

    X_test_std = feature_pipeline.transform(test_apps, standardized=True)
    X_test_raw = feature_pipeline.transform(test_apps, standardized=False)

    # 4. Train logistic regression and boosted tree baselines.
    models = train_baseline_models(
        X_train_std=X_train_std,
        X_train_raw=X_train_raw,
        y_train=y_train,
        models_cfg=cfg.models,
        seed=cfg.project.seed,
    )

    # 5. Calibrate each model on the held-out temporal calibration window.
    calibrators: Dict[str, Dict[str, PDCalibrator]] = {}
    monotonicity_checks: Dict[str, Dict[str, Any]] = {}
    model_evals: Dict[str, Dict[str, Any]] = {}
    plot_curves: Dict[str, np.ndarray] = {}

    for m_name, model in models.items():
        use_std = isinstance(model, LogisticRegressionPDModel)
        X_tr = X_train_std if use_std else X_train_raw
        X_ca = X_cal_std if use_std else X_cal_raw
        X_te = X_test_std if use_std else X_test_raw

        raw_train = model.predict_raw_proba(X_tr)
        raw_cal = model.predict_raw_proba(X_ca)
        raw_test = model.predict_raw_proba(X_te)

        calibrators[m_name] = {}
        monotonicity_checks[m_name] = {}
        cal_test_metrics: Dict[str, Any] = {}

        for cal_method in cfg.calibration.comparison_methods:
            calibrator = PDCalibrator(
                method=cal_method,
                clip_epsilon=cfg.calibration.clip_epsilon,
                monotonicity_tolerance=cfg.calibration.monotonicity_tolerance,
            )
            calibrator.fit(raw_cal, y_cal)
            calibrators[m_name][cal_method] = calibrator
            monotonicity_checks[m_name][cal_method] = calibrator.verify_monotonicity()

            cal_test_probs = calibrator.transform(raw_test)
            cal_test_metrics[cal_method] = compute_discrimination_and_calibration_metrics(
                y_test,
                cal_test_probs,
                n_bins=cfg.calibration.n_bins,
                decision_threshold=cfg.evaluation.decision_threshold,
            )

        selected_cal = calibrators[m_name][cfg.calibration.method]
        selected_test_probs = selected_cal.transform(raw_test)

        uncal_test = compute_discrimination_and_calibration_metrics(
            y_test,
            raw_test,
            n_bins=cfg.calibration.n_bins,
            decision_threshold=cfg.evaluation.decision_threshold,
        )

        model_evals[m_name] = {
            "metadata": model.metadata(),
            "uncalibrated_test": uncal_test,
            "calibrated_test": cal_test_metrics[cfg.calibration.method],
            "calibration_comparison_test": cal_test_metrics,
        }

        if m_name == "logistic_regression":
            plot_curves[f"LR ({cfg.calibration.method})"] = selected_test_probs
        else:
            plot_curves["Boosted (uncalibrated)"] = raw_test
            for cm in cfg.calibration.comparison_methods:
                plot_curves[f"Boosted ({cm})"] = calibrators[m_name][cm].transform(raw_test)

    # 6. Primary model stability (PSI/CSI) and subgroup diagnostics on temporal test split.
    primary_name = cfg.models.primary_model
    primary_model: BasePDModel = models[primary_name]
    primary_calibrator = calibrators[primary_name][cfg.calibration.method]
    primary_use_std = isinstance(primary_model, LogisticRegressionPDModel)

    primary_train_pd = primary_calibrator.transform(
        primary_model.predict_raw_proba(X_train_std if primary_use_std else X_train_raw)
    )
    primary_test_pd = primary_calibrator.transform(
        primary_model.predict_raw_proba(X_test_std if primary_use_std else X_test_raw)
    )

    score_psi = compute_psi(
        expected_values=primary_train_pd,
        actual_values=primary_test_pd,
        n_bins=cfg.evaluation.psi_bins,
        warning_threshold=cfg.evaluation.psi_warning_threshold,
        alert_threshold=cfg.evaluation.psi_alert_threshold,
    )
    feature_csi = compute_feature_stability_csi(
        X_expected=X_train_raw[:, :14],
        X_actual=X_test_raw[:, :14],
        feature_names=feature_pipeline.feature_names_[:14],
        n_bins=cfg.evaluation.psi_bins,
        warning_threshold=cfg.evaluation.psi_warning_threshold,
        alert_threshold=cfg.evaluation.psi_alert_threshold,
    )

    subgroup_report = evaluate_subgroups(
        applicants=test_apps,
        y_true=y_test,
        y_prob=primary_test_pd,
        subgroup_columns=cfg.evaluation.subgroup_columns,
        min_sample_size=cfg.evaluation.min_subgroup_sample_size,
        decision_threshold=cfg.evaluation.decision_threshold,
        strict=False,
    )

    baseline_cmp = compare_against_baseline(
        logistic_calibrated=model_evals["logistic_regression"]["calibrated_test"],
        boosted_calibrated=model_evals["boosted_trees"]["calibrated_test"],
        tolerances=cfg.evaluation.baseline_comparison,
    )

    # 7. Save calibration plots (PNG + SVG).
    plot_paths = save_calibration_plots(
        y_true=y_test,
        curves=plot_curves,
        output_dir=reports_dir,
        n_bins=cfg.calibration.n_bins,
    )

    # 8. Build manual review policy & inference scorer, verify monotonicity, save bundle.
    policy = ManualReviewPolicy(cfg=cfg.policy, policy_version=cfg.project.policy_version)
    policy_check = policy.verify_threshold_monotonicity()

    scorer = CreditRiskScorer(
        model=primary_model,
        feature_pipeline=feature_pipeline,
        calibrator=primary_calibrator,
        policy=policy,
        top_k_factors=cfg.explainability.top_k_factors,
        use_shap_if_available=cfg.explainability.use_shap_if_available,
    )
    bundle_path = scorer.save_bundle(artifacts_dir)

    # 9. Populate manual review queue with representative test cases across risk spectrum.
    if review_store_path.exists():
        review_store_path.unlink()
    review_store = ManualReviewStore(review_store_path)

    order = np.argsort(primary_test_pd)
    k = min(n_review_queue_samples, len(test_apps))
    sample_indices = np.linspace(0, len(test_apps) - 1, k, dtype=int).tolist()
    sample_decisions: List[Dict[str, Any]] = []
    for idx in sample_indices:
        app = test_apps[int(order[idx])]
        dec = scorer.score_applicant(app, scored_at_utc=deterministic_timestamp)
        review_store.enqueue(app, dec)
        sample_decisions.append(dec.to_dict())

    # Record one example human review on the highest-risk queued case to demonstrate audit log.
    if sample_decisions:
        highest_case = sample_decisions[-1]
        review_store.record_human_review(
            decision_id=highest_case["decision_id"],
            reviewer_id="demo-underwriter-01",
            disposition="HUMAN_REQUEST_DOCUMENTATION",
            rationale="Synthetic demo review: elevated PD driven by DTI/utilization; requesting income verification.",
            reviewed_at_utc=deterministic_timestamp,
        )

    # Verify deterministic replay on first queued sample.
    replay_check: Optional[Dict[str, Any]] = None
    if sample_decisions:
        first_app = test_apps[int(order[sample_indices[0]])]
        replay_check = scorer.replay_decision(
            first_app,
            expected_decision_id=sample_decisions[0]["decision_id"],
            scored_at_utc=deterministic_timestamp,
        )

    # 10. Persist machine-readable reports & model card.
    runtime_meta = collect_runtime_metadata(cfg)
    evaluation_report: Dict[str, Any] = {
        "project": {
            "name": cfg.project.name,
            "version": cfg.project.version,
            "schema_version": cfg.project.schema_version,
            "feature_version": cfg.project.feature_version,
            "policy_version": cfg.project.policy_version,
            "seed": cfg.project.seed,
        },
        "dataset_metadata": dataset.metadata,
        "splits": {
            "train": {
                "count": len(train_apps),
                "default_rate": round(float(np.mean(y_train)), 6),
                "end_date": cfg.temporal_split.train_end_date,
            },
            "calibration": {
                "count": len(cal_apps),
                "default_rate": round(float(np.mean(y_cal)), 6),
                "end_date": cfg.temporal_split.calibration_end_date,
            },
            "test": {
                "count": len(test_apps),
                "default_rate": round(float(np.mean(y_test)), 6),
                "end_date": cfg.temporal_split.test_end_date,
            },
            "censored_excluded": {
                "count": len(censored_apps),
            },
        },
        "primary_model": primary_name,
        "selected_calibration_method": cfg.calibration.method,
        "models": model_evals,
        "monotonicity_checks": {
            "calibration": monotonicity_checks,
            "policy": policy_check,
        },
        "stability": {
            "score_psi_train_vs_test": score_psi,
            "feature_csi_train_vs_test": feature_csi,
        },
        "subgroup_diagnostics": subgroup_report,
        "baseline_comparison": baseline_cmp,
        "inference_diagnostics": scorer.diagnostics.to_dict(),
        "replay_verification": {
            "decision_id_match": replay_check["decision_id_match"] if replay_check else None,
            "decision_id": replay_check["replayed_decision_id"] if replay_check else None,
        },
        "artifacts": {
            "scorer_bundle": str(bundle_path),
            "calibration_plots": plot_paths,
            "manual_review_queue": str(review_store_path),
        },
    }

    eval_path = save_json_file(reports_dir / "evaluation_report.json", evaluation_report)
    cmp_path = save_json_file(reports_dir / "baseline_comparison.json", baseline_cmp)
    meta_path = save_json_file(reports_dir / "runtime_metadata.json", runtime_meta)
    demo_out_path = save_json_file(
        reports_dir / "demo_output.json",
        {
            "automated_decision_allowed": False,
            "sample_scored_decisions": sample_decisions,
            "replay_verification": replay_check,
        },
    )
    card_path = write_model_card(
        cfg=cfg,
        evaluation_report=evaluation_report,
        runtime_metadata=runtime_meta,
        output_path=reports_dir / "model_card.md",
    )

    return {
        "evaluation_report": evaluation_report,
        "runtime_metadata": runtime_meta,
        "sample_decisions": sample_decisions,
        "outputs": {
            "evaluation_report": str(eval_path),
            "baseline_comparison": str(cmp_path),
            "runtime_metadata": str(meta_path),
            "demo_output": str(demo_out_path),
            "model_card": str(card_path),
            "calibration_plot_png": plot_paths["png"],
            "calibration_plot_svg": plot_paths["svg"],
            "manual_review_queue": str(review_store_path),
            "scorer_bundle": str(bundle_path),
        },
    }


def run_from_config_path(
    config_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    cfg_path = Path(config_path).resolve()
    root = Path(base_dir).resolve() if base_dir is not None else cfg_path.parent.parent
    cfg = load_config(cfg_path)
    return run_end_to_end_pipeline(cfg, base_dir=root)
