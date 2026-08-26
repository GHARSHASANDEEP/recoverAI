import pandas as pd


ACTIONS = [
    "retry",
    "reminder",
    "escalate",
    "stop",
]


RETRY_BLOCKED_FAILURES = {
    "risk_decline",
    "blocked_instrument",
    "expired_instrument",
}


MAX_RETRY_ATTEMPTS = 3

# Maximum number of actual recovery attempts.
# Once the next attempt would exceed this limit,
# automated recovery is stopped and the case is escalated.
MAX_TOTAL_RECOVERY_ATTEMPTS = 3

HIGH_VALUE_THRESHOLD = 50000.0

MIN_ERV_TO_ACT = 0.0


def evaluate_guardrails(case: dict, action: str) -> dict:
    """
    Determine whether an action is permitted.

    Guardrails are deterministic policy rules.
    The ML model does not override them.
    """

    failure_category = (
        case.get(
            "failure_category",
            "unknown_failure",
        )
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

    # --------------------------------------------------
    # TOTAL ATTEMPT LIMIT
    # --------------------------------------------------
    # attempt_number represents the next action attempt.
    # At attempt 4+, the automated recovery budget is exhausted.
    # Escalation remains permitted so the case is not silently
    # stopped without a human-review path.
    # --------------------------------------------------

    if (
        attempt_number
        > MAX_TOTAL_RECOVERY_ATTEMPTS
    ):

        if action == "escalate":
            return {
                "allowed": True,
                "reason": (
                    "Maximum recovery attempt limit "
                    "reached. Case escalated to manual review."
                ),
            }

        return {
            "allowed": False,
            "reason": (
                "Maximum recovery attempt limit reached. "
                "Further automated recovery actions are blocked."
            ),
        }

    # --------------------------------------------------
    # STOP CONDITIONS
    # --------------------------------------------------

    if (
        action != "stop"
        and erv <= MIN_ERV_TO_ACT
    ):
        return {
            "allowed": False,
            "reason": (
                "ERV is not economically positive."
            ),
        }

    # --------------------------------------------------
    # RETRY GUARDRAILS
    # --------------------------------------------------

    if action == "retry":

        if failure_category in (
            RETRY_BLOCKED_FAILURES
        ):
            return {
                "allowed": False,
                "reason": (
                    f"Retry blocked for "
                    f"{failure_category}."
                ),
            }

        if attempt_number >= (
            MAX_RETRY_ATTEMPTS
        ):
            return {
                "allowed": False,
                "reason": (
                    "Maximum retry limit reached."
                ),
            }

    # --------------------------------------------------
    # REMINDER GUARDRAIL
    # --------------------------------------------------

    if action == "reminder":

        if not communication_opt_in:
            return {
                "allowed": False,
                "reason": (
                    "Customer has not opted "
                    "into communication."
                ),
            }

    # --------------------------------------------------
    # ESCALATION
    # --------------------------------------------------

    if action == "escalate":

        return {
            "allowed": True,
            "reason": (
                "Case routed to manual review."
            ),
        }

    # --------------------------------------------------
    # HIGH-VALUE CASE INFORMATION
    # --------------------------------------------------

    if (
        recovery_amount
        >= HIGH_VALUE_THRESHOLD
    ):
        return {
            "allowed": True,
            "reason": (
                "High-value case permitted "
                "for controlled recovery."
            ),
        }

    return {
        "allowed": True,
        "reason": (
            "Action passed all guardrails."
        ),
    }


def apply_decision(
    case: dict,
    action_scores: pd.DataFrame,
) -> dict:
    """
    Select the highest-ERV action that passes
    all guardrails.
    """

    evaluated = []

    for row in action_scores.to_dict(
        orient="records"
    ):

        action = row["action"]

        case_with_erv = {
            **case,
            "erv": row["erv"],
        }

        guardrail = (
            evaluate_guardrails(
                case_with_erv,
                action,
            )
        )

        evaluated.append(
            {
                **row,
                "guardrail_allowed": (
                    guardrail[
                        "allowed"
                    ]
                ),
                "guardrail_reason": (
                    guardrail[
                        "reason"
                    ]
                ),
            }
        )

    evaluated_df = pd.DataFrame(
        evaluated
    )

    allowed = evaluated_df[
        evaluated_df[
            "guardrail_allowed"
        ]
    ].copy()

    # --------------------------------------------------
    # No permitted automated action.
    # --------------------------------------------------

    if allowed.empty:

        return {
            "case_id": case[
                "case_id"
            ],
            "final_action": "stop",
            "decision_reason": (
                "No permitted recovery "
                "action remains."
            ),
            "guardrail_status": "blocked",
            "candidate_actions": evaluated,
        }

    # --------------------------------------------------
    # Highest ERV among permitted actions.
    # --------------------------------------------------

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

    return {
        "case_id": case[
            "case_id"
        ],
        "final_action": best[
            "action"
        ],
        "decision_reason": best[
            "guardrail_reason"
        ],
        "guardrail_status": "passed",
        "recovery_probability": best[
            "recovery_probability"
        ],
        "expected_recovery": best[
            "expected_recovery"
        ],
        "action_cost": best[
            "action_cost"
        ],
        "erv": best[
            "erv"
        ],
        "candidate_actions": evaluated,
    }
