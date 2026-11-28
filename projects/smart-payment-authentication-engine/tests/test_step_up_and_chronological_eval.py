"""M4 Boundary & Failure tests: Step-up simulation, chronological evaluation, label delay, and reproducibility."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from smart_auth_engine.config import load_config
from smart_auth_engine.data_generator import generate_synthetic_dataset
from smart_auth_engine.evaluation import (
    ChronologicalOrderError,
    LabelAlignmentError,
    evaluate_chronologically,
)
from smart_auth_engine.features import extract_feature_matrix
from smart_auth_engine.inference import AuthenticationEngine
from smart_auth_engine.policy import optimize_policy_thresholds
from smart_auth_engine.training import train_calibrated_risk_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_chronological_evaluation_and_label_delay_boundary() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    events, labels, _ = generate_synthetic_dataset(cfg.data_generation, seed=cfg.seed)
    train_report = train_calibrated_risk_model(events, labels, cfg)
    X_val = extract_feature_matrix(train_report.splits.val_events)
    val_probs, _ = train_report.model.predict_proba_matrix(X_val)
    opt_res = optimize_policy_thresholds(
        train_report.splits.val_events, train_report.splits.val_labels, val_probs, cfg
    )

    test_events = train_report.splits.test_events
    test_labels = train_report.splits.test_labels

    # 1. Partial maturity cutoff: evaluating mid-horizon MUST mark recent labels immature
    early_as_of = test_events[-1].timestamp + timedelta(days=5)
    eng_early = AuthenticationEngine(cfg, opt_res.selected_thresholds, model=train_report.model)
    rep_early = evaluate_chronologically(
        test_events, test_labels, eng_early, cfg, evaluation_as_of=early_as_of
    )
    assert rep_early.immature_events > 0
    assert rep_early.matured_events + rep_early.immature_events == len(test_events)
    assert rep_early.mandatory_challenge_bypass_count == 0
    assert rep_early.override_bypass_count == 0

    # 2. Full maturity retrospective audit: all labels matured
    eng_full = AuthenticationEngine(cfg, opt_res.selected_thresholds, model=train_report.model)
    rep_full = evaluate_chronologically(test_events, test_labels, eng_full, cfg)
    assert rep_full.immature_events == 0
    assert rep_full.matured_events == len(test_events)
    assert rep_full.mandatory_challenge_bypass_count == 0
    assert rep_full.override_bypass_count == 0
    assert rep_full.matured_fraud_loss_rate is not None
    assert rep_full.matured_genuine_acceptance_rate is not None


def test_chronological_evaluation_rejects_out_of_order_and_missing_labels() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    events, labels, _ = generate_synthetic_dataset(cfg.data_generation, seed=cfg.seed)
    train_report = train_calibrated_risk_model(events, labels, cfg)
    X_val = extract_feature_matrix(train_report.splits.val_events)
    val_probs, _ = train_report.model.predict_proba_matrix(X_val)
    opt_res = optimize_policy_thresholds(
        train_report.splits.val_events, train_report.splits.val_labels, val_probs, cfg
    )
    engine = AuthenticationEngine(cfg, opt_res.selected_thresholds, model=train_report.model)

    # Swapping two events must raise ChronologicalOrderError
    swapped = list(train_report.splits.test_events[:10])
    swapped[3], swapped[4] = swapped[4], swapped[3]
    with pytest.raises(ChronologicalOrderError):
        evaluate_chronologically(swapped, train_report.splits.test_labels[:10], engine, cfg)

    # Missing a label must raise LabelAlignmentError
    with pytest.raises(LabelAlignmentError):
        evaluate_chronologically(
            train_report.splits.test_events[:10],
            train_report.splits.test_labels[:9],
            engine,
            cfg,
        )


def test_end_to_end_fixed_seed_reproducibility() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")

    def _run_once():
        events, labels, _ = generate_synthetic_dataset(cfg.data_generation, seed=cfg.seed)
        tr = train_calibrated_risk_model(events, labels, cfg)
        X_val = extract_feature_matrix(tr.splits.val_events)
        val_p, _ = tr.model.predict_proba_matrix(X_val)
        opt = optimize_policy_thresholds(tr.splits.val_events, tr.splits.val_labels, val_p, cfg)
        eng = AuthenticationEngine(cfg, opt.selected_thresholds, model=tr.model)
        rep = evaluate_chronologically(tr.splits.test_events, tr.splits.test_labels, eng, cfg)
        return opt.selected_point.to_dict(), rep.to_dict()

    opt1, rep1 = _run_once()
    opt2, rep2 = _run_once()
    assert opt1 == opt2
    assert rep1 == rep2
