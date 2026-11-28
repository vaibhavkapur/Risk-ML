"""Processor response taxonomy and deterministic retry eligibility state machine.

Core safety invariants:
1. NEVER retry a hard-declined payment.
2. NEVER retry an uncertain authorization outcome blindly (hold for reconciliation).
3. NEVER double-authorize a duplicate attempt (idempotency & already-approved guard).
4. Unknown or unmapped response codes fail closed as UNCERTAIN.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from payment_auth_engine.schemas import AuthorizationAttempt, PolicyAction, RetryEligibilityResult


class ResponseCategory(str, Enum):
    APPROVED = "APPROVED"
    HARD_DECLINE = "HARD_DECLINE"
    SOFT_DECLINE = "SOFT_DECLINE"
    TECHNICAL_RETRIABLE = "TECHNICAL_RETRIABLE"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class ResponseCodeSpec:
    code: str
    category: ResponseCategory
    description: str
    retry_eligible: bool
    requires_reconciliation: bool = False
    supports_3ds_stepup: bool = False
    supports_processor_failover: bool = False
    recurring_backoff_only: bool = False


RESPONSE_TAXONOMY: Dict[str, ResponseCodeSpec] = {
    "00": ResponseCodeSpec(
        code="00",
        category=ResponseCategory.APPROVED,
        description="Approved or completed successfully",
        retry_eligible=False,
    ),
    # Hard declines: NEVER retry
    "14": ResponseCodeSpec(
        code="14",
        category=ResponseCategory.HARD_DECLINE,
        description="Invalid card number",
        retry_eligible=False,
    ),
    "54": ResponseCodeSpec(
        code="54",
        category=ResponseCategory.HARD_DECLINE,
        description="Expired card",
        retry_eligible=False,
    ),
    "57": ResponseCodeSpec(
        code="57",
        category=ResponseCategory.HARD_DECLINE,
        description="Transaction not permitted to cardholder",
        retry_eligible=False,
    ),
    "59": ResponseCodeSpec(
        code="59",
        category=ResponseCategory.HARD_DECLINE,
        description="Suspected fraud",
        retry_eligible=False,
    ),
    "63": ResponseCodeSpec(
        code="63",
        category=ResponseCategory.HARD_DECLINE,
        description="Security violation",
        retry_eligible=False,
    ),
    # Soft declines: conditional, bounded eligibility
    "05": ResponseCodeSpec(
        code="05",
        category=ResponseCategory.SOFT_DECLINE,
        description="Do not honor (generic soft decline)",
        retry_eligible=True,
        supports_processor_failover=True,
    ),
    "51": ResponseCodeSpec(
        code="51",
        category=ResponseCategory.SOFT_DECLINE,
        description="Insufficient funds",
        retry_eligible=True,
        recurring_backoff_only=True,
    ),
    "61": ResponseCodeSpec(
        code="61",
        category=ResponseCategory.SOFT_DECLINE,
        description="Exceeds withdrawal amount limit",
        retry_eligible=True,
        recurring_backoff_only=True,
    ),
    "65": ResponseCodeSpec(
        code="65",
        category=ResponseCategory.SOFT_DECLINE,
        description="Additional customer authentication (SCA/3DS) required",
        retry_eligible=True,
        supports_3ds_stepup=True,
    ),
    # Technical retriable: processor/switch failure with confirmed non-authorization
    "91": ResponseCodeSpec(
        code="91",
        category=ResponseCategory.TECHNICAL_RETRIABLE,
        description="Issuer or switch inoperative (confirmed non-auth)",
        retry_eligible=True,
        supports_processor_failover=True,
    ),
    "96": ResponseCodeSpec(
        code="96",
        category=ResponseCategory.TECHNICAL_RETRIABLE,
        description="System malfunction (confirmed non-auth)",
        retry_eligible=True,
        supports_processor_failover=True,
    ),
    # Uncertain authorization outcomes: NEVER retry blindly; require reconciliation
    "TIMEOUT": ResponseCodeSpec(
        code="TIMEOUT",
        category=ResponseCategory.UNCERTAIN,
        description="Processor or issuer timeout — authorization state unknown",
        retry_eligible=False,
        requires_reconciliation=True,
    ),
    "NETWORK_ERROR": ResponseCodeSpec(
        code="NETWORK_ERROR",
        category=ResponseCategory.UNCERTAIN,
        description="Network transport failure mid-flight — authorization state unknown",
        retry_eligible=False,
        requires_reconciliation=True,
    ),
    "UNKNOWN": ResponseCodeSpec(
        code="UNKNOWN",
        category=ResponseCategory.UNCERTAIN,
        description="Unmapped or malformed processor response — fail closed",
        retry_eligible=False,
        requires_reconciliation=True,
    ),
}


def classify_response_code(code: Optional[str]) -> ResponseCodeSpec:
    """Map a processor response code to its canonical taxonomy spec.

    Unmapped, blank, or None codes fail closed as UNCERTAIN (UNKNOWN).
    """
    if code is None:
        return RESPONSE_TAXONOMY["UNKNOWN"]
    normalized = code.strip().upper()
    if not normalized:
        return RESPONSE_TAXONOMY["UNKNOWN"]
    return RESPONSE_TAXONOMY.get(normalized, RESPONSE_TAXONOMY["UNKNOWN"])


def evaluate_retry_eligibility(
    attempt: AuthorizationAttempt,
    *,
    idempotency_seen: bool = False,
    prior_approved: bool = False,
    max_retries: int = 2,
    allow_soft_decline_retry: bool = True,
    allow_technical_failover: bool = True,
) -> RetryEligibilityResult:
    """Evaluate whether an authorization attempt is safe and eligible to proceed or retry.

    This function is pure and isolated from I/O.
    """
    # 1. Duplicate / already-approved guard (always first to prevent double-charging)
    if idempotency_seen or prior_approved:
        reason = "ALREADY_APPROVED_TRANSACTION" if prior_approved else "DUPLICATE_IDEMPOTENCY_KEY"
        return RetryEligibilityResult(
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            eligible=False,
            recommended_action=PolicyAction.DEDUPLICATE_RETURN_PRIOR,
            response_category=ResponseCategory.APPROVED.value if prior_approved else None,
            reason_code=reason,
            explanation="Duplicate or already-approved payment detected; return cached prior outcome without re-submitting.",
            requires_reconciliation=False,
            is_hard_stop=True,
        )

    # 2. Initial attempt (attempt_number == 1)
    if attempt.attempt_number == 1:
        if not attempt.prior_outcome_known:
            return RetryEligibilityResult(
                attempt_id=attempt.attempt_id,
                transaction_id=attempt.transaction_id,
                eligible=False,
                recommended_action=PolicyAction.HOLD_RECONCILE,
                response_category=ResponseCategory.UNCERTAIN.value,
                reason_code="UNCERTAIN_STATE_ON_INITIAL",
                explanation="Prior outcome marked unknown; hold for reconciliation instead of blind submission.",
                requires_reconciliation=True,
                is_hard_stop=True,
            )
        return RetryEligibilityResult(
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            eligible=True,
            recommended_action=PolicyAction.APPROVE_PRIMARY,
            response_category=None,
            reason_code="INITIAL_ATTEMPT_ELIGIBLE",
            explanation="Fresh authorization attempt eligible for policy evaluation.",
        )

    # 3. Retry attempt (attempt_number > 1): inspect prior outcome certainty first
    spec = classify_response_code(attempt.prior_response_code)
    if not attempt.prior_outcome_known or spec.category == ResponseCategory.UNCERTAIN:
        return RetryEligibilityResult(
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            eligible=False,
            recommended_action=PolicyAction.HOLD_RECONCILE,
            response_category=ResponseCategory.UNCERTAIN.value,
            reason_code="UNCERTAIN_OUTCOME_NEVER_BLIND_RETRY",
            explanation=(
                f"Prior authorization outcome is uncertain (code={attempt.prior_response_code!r}, "
                f"known={attempt.prior_outcome_known}); blind retry is forbidden to prevent duplicate capture."
            ),
            requires_reconciliation=True,
            is_hard_stop=True,
        )

    # 4. Prior hard decline: NEVER retry
    if spec.category == ResponseCategory.HARD_DECLINE:
        return RetryEligibilityResult(
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            eligible=False,
            recommended_action=PolicyAction.BLOCK_HARD_DECLINE,
            response_category=spec.category.value,
            reason_code="HARD_DECLINE_NEVER_RETRY",
            explanation=f"Prior response {spec.code} ({spec.description}) is a hard decline; retries are prohibited.",
            requires_reconciliation=False,
            is_hard_stop=True,
        )

    # 5. Prior approved code sent on retry
    if spec.category == ResponseCategory.APPROVED:
        return RetryEligibilityResult(
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            eligible=False,
            recommended_action=PolicyAction.DEDUPLICATE_RETURN_PRIOR,
            response_category=spec.category.value,
            reason_code="PRIOR_ATTEMPT_ALREADY_APPROVED",
            explanation="Prior attempt already received response code 00 (Approved); do not retry.",
            requires_reconciliation=False,
            is_hard_stop=True,
        )

    # 6. Retry count cap
    retries_used = attempt.attempt_number - 1
    if retries_used > max_retries:
        return RetryEligibilityResult(
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            eligible=False,
            recommended_action=PolicyAction.BLOCK_MAX_RETRIES,
            response_category=spec.category.value,
            reason_code="MAX_RETRIES_EXCEEDED",
            explanation=f"Retry count {retries_used} exceeds configured max_retries={max_retries}.",
            requires_reconciliation=False,
            is_hard_stop=True,
        )

    # 7. Technical retriable (e.g. 91, 96)
    if spec.category == ResponseCategory.TECHNICAL_RETRIABLE:
        if not allow_technical_failover:
            return RetryEligibilityResult(
                attempt_id=attempt.attempt_id,
                transaction_id=attempt.transaction_id,
                eligible=False,
                recommended_action=PolicyAction.DECLINE_POLICY,
                response_category=spec.category.value,
                reason_code="TECHNICAL_FAILOVER_DISABLED",
                explanation=f"Technical code {spec.code} is retriable, but technical failover is disabled by config.",
                is_hard_stop=True,
            )
        return RetryEligibilityResult(
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            eligible=True,
            recommended_action=PolicyAction.ROUTE_SECONDARY,
            response_category=spec.category.value,
            reason_code="TECHNICAL_RETRIABLE_FAILOVER",
            explanation=f"Confirmed non-auth technical decline {spec.code} ({spec.description}); eligible for secondary failover.",
        )

    # 8. Soft decline (05, 51, 61, 65)
    if not allow_soft_decline_retry:
        return RetryEligibilityResult(
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            eligible=False,
            recommended_action=PolicyAction.DECLINE_POLICY,
            response_category=spec.category.value,
            reason_code="SOFT_DECLINE_RETRY_DISABLED",
            explanation=f"Soft decline {spec.code} retry is disabled by policy configuration.",
            is_hard_stop=True,
        )

    if spec.supports_3ds_stepup:
        if attempt.has_3ds:
            return RetryEligibilityResult(
                attempt_id=attempt.attempt_id,
                transaction_id=attempt.transaction_id,
                eligible=False,
                recommended_action=PolicyAction.DECLINE_POLICY,
                response_category=spec.category.value,
                reason_code="STEP_UP_ALREADY_APPLIED",
                explanation="Prior code 65 requested SCA/3DS, but 3DS is already present on this attempt.",
                is_hard_stop=True,
            )
        return RetryEligibilityResult(
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            eligible=True,
            recommended_action=PolicyAction.STEP_UP_3DS,
            response_category=spec.category.value,
            reason_code="SOFT_DECLINE_STEP_UP_3DS",
            explanation="Prior code 65 requires SCA/3DS; eligible for step-up authentication retry.",
        )

    if spec.recurring_backoff_only:
        if not attempt.is_recurring:
            return RetryEligibilityResult(
                attempt_id=attempt.attempt_id,
                transaction_id=attempt.transaction_id,
                eligible=False,
                recommended_action=PolicyAction.DECLINE_POLICY,
                response_category=spec.category.value,
                reason_code="SOFT_DECLINE_NO_IMMEDIATE_RETRY",
                explanation=(
                    f"Soft decline {spec.code} ({spec.description}) is not eligible for immediate "
                    "non-recurring retry."
                ),
                is_hard_stop=True,
            )
        return RetryEligibilityResult(
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            eligible=True,
            recommended_action=PolicyAction.APPROVE_PRIMARY,
            response_category=spec.category.value,
            reason_code="RECURRING_SOFT_DECLINE_RETRY",
            explanation=f"Recurring payment after {spec.code}; eligible for scheduled retry evaluation.",
        )

    # Generic soft decline (05): allow at most one smart secondary failover retry
    if attempt.attempt_number > 2:
        return RetryEligibilityResult(
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            eligible=False,
            recommended_action=PolicyAction.BLOCK_MAX_RETRIES,
            response_category=spec.category.value,
            reason_code="GENERIC_SOFT_DECLINE_RETRY_LIMIT",
            explanation="Generic soft decline 05 allows at most one retry attempt.",
            is_hard_stop=True,
        )

    return RetryEligibilityResult(
        attempt_id=attempt.attempt_id,
        transaction_id=attempt.transaction_id,
        eligible=True,
        recommended_action=PolicyAction.ROUTE_SECONDARY,
        response_category=spec.category.value,
        reason_code="SOFT_DECLINE_SMART_FAILOVER",
        explanation="Generic soft decline 05 eligible for single smart failover evaluation.",
    )
