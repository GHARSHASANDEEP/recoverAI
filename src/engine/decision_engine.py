import pandas as pd


# ============================================================
# DECISION ENGINE
# ============================================================
# Purpose:
#   Select the safest and most economically valuable recovery
#   action after ML scoring.
#
# Architecture:
#   ML scores actions
#        ↓
#   Decision Engine
#        ↓
#   Deterministic Guardrails
#        ↓
#   Best permitted action
#
# IMPORTANT:
#   The ML model suggests actions.
#   Guardrails always have final authority.
# ============================================================


ACTIONS = [
    "retry",
    "reminder",
    "escalate",
    "stop",
]


# Failure types for which retrying is unsafe or ineffective.
RETRY_BLOCKED_FAILURES = {
    "risk_decline",
    "blocked_instrument",
    "expired_instrument",
}


# Maximum retry attempts.
MAX_RETRY_ATTEMPTS = 3


# Maximum total automated recovery attempts.
MAX_TOTAL_RECOVERY_ATTEMPTS = 3


# High-value transaction threshold.
HIGH_VALUE_THRESHOLD = 50000.0


# Recovery action must have positive ERV.
MIN_ERV_TO_ACT = 0.0


# ============================================================
# GUARDRAILS
# ============================================================

def evaluate_guardrails(
    case: dict,
    action: str,
) -> dict:
    """
    Evaluate whether a proposed recovery action
    is permitted by deterministic business rules.

    ML predictions never override these guardrails.
    """

    failure_category = case.get(
        "failure_category",
        "unknown_failure",
    )

    communication_opt_in = bool(
        case.get(
            "communication_opt_in",
            False,
        )
    )

    attempt_number = int(
        case.get(
            "attempt_number",
            1,
        )
    )

    recovery_amount = float(
        case.get(
            "recovery_amount",
            0.0,
        )
    )

    erv = float(
        case.get(
            "erv",
            0.0,
        )
    )

    # --------------------------------------------------------
    # VALIDATE ACTION
    # --------------------------------------------------------

    if action not in ACTIONS:
        return {
            "allowed": False,
            "reason": f"Unknown recovery action: {action}.",
        }

    # --------------------------------------------------------
    # STOP IS ALWAYS SAFE
    # --------------------------------------------------------

    if action == "stop":
        return {
            "allowed": True,
            "reason": "Recovery stopped by policy.",
        }

    # --------------------------------------------------------
    # TOTAL RECOVERY ATTEMPT LIMIT
    # --------------------------------------------------------
    #
    # Once the automated recovery budget is exhausted:
    #   - retry/reminder are blocked
    #   - escalation remains available
    #
    # This prevents infinite automated recovery attempts.
    # --------------------------------------------------------

    if attempt_number > MAX_TOTAL_RECOVERY_ATTEMPTS:

        if action == "escalate":
            return {
                "allowed": True,
                "reason": (
                    "Maximum recovery attempt limit reached. "
                    "Case escalated to manual review."
                ),
            }

        return {
            "allowed": False,
            "reason": (
                "Maximum recovery attempt limit reached. "
                "Further automated recovery actions are blocked."
            ),
        }

    # --------------------------------------------------------
    # ECONOMIC GUARDRAIL
    # --------------------------------------------------------
    #
    # Do not spend recovery effort when the expected recovery
    # value is zero or negative.
    #
    # Escalation is still allowed because it may require
    # human intervention.
    # --------------------------------------------------------

    if erv <= MIN_ERV_TO_ACT:

        if action == "escalate":
            return {
                "allowed": True,
                "reason": (
                    "Automated recovery has non-positive ERV. "
                    "Case may be escalated for manual review."
                ),
            }

        return {
            "allowed": False,
            "reason": (
                "ERV is not economically positive "
                "for automated recovery."
            ),
        }

    # --------------------------------------------------------
    # RETRY GUARDRAILS
    # --------------------------------------------------------

    if action == "retry":

        # Certain failure categories should never be retried.
        if failure_category in RETRY_BLOCKED_FAILURES:
            return {
                "allowed": False,
                "reason": (
                    f"Retry blocked for failure category "
                    f"'{failure_category}'."
                ),
            }

        # Prevent excessive retries.
        if attempt_number >= MAX_RETRY_ATTEMPTS:
            return {
                "allowed": False,
                "reason": (
                    "Maximum retry limit reached."
                ),
            }

    # --------------------------------------------------------
    # REMINDER GUARDRAILS
    # --------------------------------------------------------

    if action == "reminder":

        # Communication must be permitted.
        if not communication_opt_in:
            return {
                "allowed": False,
                "reason": (
                    "Customer has not opted into communication."
                ),
            }

    # --------------------------------------------------------
    # ESCALATION
    # --------------------------------------------------------

    if action == "escalate":
        return {
            "allowed": True,
            "reason": (
                "Case routed to manual review."
            ),
        }

    # --------------------------------------------------------
    # HIGH-VALUE CASE
    # --------------------------------------------------------
    #
    # High-value transactions are allowed to proceed through
    # controlled recovery after passing the basic guardrails.
    # --------------------------------------------------------

    if recovery_amount >= HIGH_VALUE_THRESHOLD:
        return {
            "allowed": True,
            "reason": (
                "High-value case permitted for "
                "controlled recovery."
            ),
        }

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return {
        "allowed": True,
        "reason": (
            "Action passed all guardrails."
        ),
    }


# ============================================================
# DECISION APPLICATION
# ============================================================

def apply_decision(
    case: dict,
    action_scores: pd.DataFrame,
) -> dict:
    """
    Select the highest-ERV action that passes all guardrails.

    Expected action_scores columns:

        action
        recovery_probability
        expected_recovery
        action_cost
        erv

    Returns a structured decision containing:
        - final action
        - guardrail status
        - ERV
        - recovery probability
        - candidate action evaluation
    """

    # --------------------------------------------------------
    # BASIC INPUT VALIDATION
    # --------------------------------------------------------

    if "case_id" not in case:
        raise ValueError(
            "case must contain 'case_id'."
        )

    required_columns = {
        "action",
        "recovery_probability",
        "expected_recovery",
        "action_cost",
        "erv",
    }

    missing_columns = (
        required_columns
        - set(action_scores.columns)
    )

    if missing_columns:
        raise ValueError(
            "action_scores is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if action_scores.empty:
        return {
            "case_id": case["case_id"],
            "final_action": "stop",
            "decision_reason": (
                "No candidate recovery actions were provided."
            ),
            "guardrail_status": "blocked",
            "candidate_actions": [],
        }

    # --------------------------------------------------------
    # EVALUATE EVERY CANDIDATE ACTION
    # --------------------------------------------------------

    evaluated = []

    for row in action_scores.to_dict(
        orient="records"
    ):

        action = row["action"]

        # ERV from the scoring engine is evaluated by the
        # deterministic guardrail layer.
        case_with_erv = {
            **case,
            "erv": row["erv"],
        }

        guardrail = evaluate_guardrails(
            case_with_erv,
            action,
        )

        evaluated.append(
            {
                **row,
                "guardrail_allowed": (
                    guardrail["allowed"]
                ),
                "guardrail_reason": (
                    guardrail["reason"]
                ),
            }
        )

    evaluated_df = pd.DataFrame(
        evaluated
    )

    # --------------------------------------------------------
    # FILTER PERMITTED ACTIONS
    # --------------------------------------------------------

    allowed = evaluated_df[
        evaluated_df["guardrail_allowed"] == True
    ].copy()

    # --------------------------------------------------------
    # NO PERMITTED ACTION
    # --------------------------------------------------------

    if allowed.empty:

        return {
            "case_id": case["case_id"],
            "final_action": "stop",
            "decision_reason": (
                "No permitted recovery action remains."
            ),
            "guardrail_status": "blocked",
            "candidate_actions": evaluated,
        }

    # --------------------------------------------------------
    # SELECT BEST ACTION
    # --------------------------------------------------------
    #
    # Priority:
    #   1. Highest ERV
    #   2. Highest recovery probability
    #
    # This makes the decision economically driven while using
    # probability as a secondary tie-breaker.
    # --------------------------------------------------------

    best = allowed.sort_values(
        [
            "erv",
            "recovery_probability",
        ],
        ascending=[
            False,
            False,
        ],
    ).iloc[0]

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    return {
        "case_id": case["case_id"],
        "final_action": best["action"],
        "decision_reason": best[
            "guardrail_reason"
        ],
        "guardrail_status": "passed",
        "recovery_probability": float(
            best["recovery_probability"]
        ),
        "expected_recovery": float(
            best["expected_recovery"]
        ),
        "action_cost": float(
            best["action_cost"]
        ),
        "erv": float(
            best["erv"]
        ),
        "candidate_actions": evaluated,
    }