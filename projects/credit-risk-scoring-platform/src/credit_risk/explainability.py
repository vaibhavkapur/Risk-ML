"""Transparent per-applicant decision explanations and standardized reason codes.

Supports:
- Exact additive log-odds attributions for LogisticRegressionPDModel
- Deterministic baseline-perturbation marginal log-odds attributions for BoostedTreesPDModel
- Optional TreeSHAP integration when `shap` is installed and explicitly enabled
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from credit_risk.contracts import ApplicantRecord, FeatureContribution
from credit_risk.features import FeaturePipeline
from credit_risk.models import BasePDModel, LogisticRegressionPDModel

REASON_CODE_MAP = {
    "credit_score": ("RC01_CREDIT_SCORE", "Bureau credit score"),
    "debt_to_income_ratio": ("RC02_DEBT_TO_INCOME", "Debt-to-income ratio"),
    "revolving_utilization_ratio": ("RC03_REVOLVING_UTILIZATION", "Revolving credit utilization"),
    "delinquencies_24m": ("RC04_RECENT_DELINQUENCIES", "Delinquencies in past 24 months"),
    "inquiries_6m": ("RC05_RECENT_INQUIRIES", "Credit inquiries in past 6 months"),
    "loan_to_income_ratio": ("RC06_LOAN_TO_INCOME", "Requested loan-to-income ratio"),
    "estimated_payment_to_income": ("RC07_PAYMENT_BURDEN", "Estimated monthly payment burden"),
    "annual_income_usd": ("RC08_ANNUAL_INCOME", "Annual income level"),
    "log_annual_income": ("RC08_ANNUAL_INCOME", "Annual income scale"),
    "loan_amount_usd": ("RC09_LOAN_AMOUNT", "Requested loan amount"),
    "log_loan_amount": ("RC09_LOAN_AMOUNT", "Requested loan scale"),
    "loan_term_months": ("RC10_LOAN_TERM", "Requested repayment term length"),
    "credit_history_months": ("RC11_CREDIT_HISTORY", "Length of credit history"),
    "employment_length_years": ("RC12_EMPLOYMENT_TENURE", "Employment length"),
}


def _reason_code_and_label(feature_name: str) -> Tuple[str, str]:
    if feature_name in REASON_CODE_MAP:
        return REASON_CODE_MAP[feature_name]
    if feature_name.startswith("missing_"):
        base = feature_name[len("missing_") :]
        return ("RC13_MISSING_ORIGINATION_DATA", f"Missing origination field ({base})")
    if feature_name.startswith("home_ownership_"):
        cat = feature_name[len("home_ownership_") :]
        return ("RC14_HOUSING_STATUS", f"Housing status ({cat})")
    if feature_name.startswith("loan_purpose_"):
        cat = feature_name[len("loan_purpose_") :]
        return ("RC15_LOAN_PURPOSE", f"Loan purpose ({cat})")
    return ("RC99_OTHER_FACTOR", feature_name)


class DecisionExplainer:
    """Computes local per-applicant feature attributions in log-odds space."""

    def __init__(
        self,
        model: BasePDModel,
        feature_pipeline: FeaturePipeline,
        top_k: int = 5,
        use_shap_if_available: bool = False,
    ) -> None:
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.top_k = top_k
        self.use_shap_if_available = use_shap_if_available
        self.explanation_method = self._resolve_method()

    def _resolve_method(self) -> str:
        if isinstance(self.model, LogisticRegressionPDModel):
            return "exact_linear_log_odds"
        if self.use_shap_if_available:
            try:
                import shap  # type: ignore  # noqa: F401

                return "shap_tree_explainer"
            except ImportError:
                pass
        return "marginal_baseline_log_odds"

    def _raw_attributions(self, record: ApplicantRecord) -> Tuple[np.ndarray, np.ndarray]:
        x_unscaled = self.feature_pipeline.transform([record], standardized=False)[0]
        x_std = self.feature_pipeline.transform([record], standardized=True)[0]

        if isinstance(self.model, LogisticRegressionPDModel):
            _, coefs = self.model.coefficients()
            contribs = coefs * x_std
            return contribs, x_unscaled

        if self.explanation_method == "shap_tree_explainer":
            import shap  # type: ignore

            explainer = shap.TreeExplainer(self.model.estimator)
            vals = explainer.shap_values(x_unscaled.reshape(1, -1))
            if isinstance(vals, list):
                vals = vals[-1]
            contribs = np.asarray(vals, dtype=np.float64).reshape(-1)
            return contribs, x_unscaled

        # Deterministic marginal baseline log-odds perturbation:
        # contribution_j = log_odds(x) - log_odds(x with feature j set to training mean).
        assert self.feature_pipeline.means_ is not None
        baseline_vec = self.feature_pipeline.means_
        n_features = len(x_unscaled)

        # Evaluate original + all single-feature baseline perturbations in one vectorized batch.
        batch = np.repeat(x_unscaled.reshape(1, -1), n_features + 1, axis=0)
        for j in range(n_features):
            batch[j + 1, j] = baseline_vec[j]

        logits = self.model.predict_log_odds(batch)
        base_logit = logits[0]
        contribs = base_logit - logits[1:]
        return contribs, x_unscaled

    def explain(self, record: ApplicantRecord) -> List[FeatureContribution]:
        """Return top-K local feature contributions ordered by absolute magnitude."""
        contribs, x_unscaled = self._raw_attributions(record)
        names = self.feature_pipeline.feature_names_

        indexed = list(enumerate(contribs))
        # Deterministic tie-breaking by feature name.
        indexed.sort(key=lambda item: (-abs(float(item[1])), names[item[0]]))

        top_items = indexed[: self.top_k]
        explanations: List[FeatureContribution] = []
        for idx, raw_val in top_items:
            cval = float(raw_val)
            fname = names[idx]
            fval: Optional[float] = round(float(x_unscaled[idx]), 4)
            direction = "INCREASES_RISK" if cval >= 0.0 else "DECREASES_RISK"
            code, label = _reason_code_and_label(fname)
            verb = "increases" if cval >= 0.0 else "reduces"
            summary = f"{label} ({fval}) {verb} estimated default risk (delta log-odds {cval:+.4f})."
            explanations.append(
                FeatureContribution(
                    feature=fname,
                    value=fval,
                    contribution_log_odds=round(cval, 6),
                    direction=direction,
                    reason_code=code,
                    summary=summary,
                )
            )
        return explanations
