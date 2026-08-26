from datetime import datetime

import pandas as pd

from src.engine.decision_engine import (
    evaluate_guardrails,
)
from src.engine.executor import (
    execute_and_verify,
)


MAX_AUTOMATED_ATTEMPTS = 3


def select_next_action(
    case: dict,
    action_scores: pd.DataFrame,
    attempted_actions: set,
):
    """
    Select the highest-ERV action that:

    1. Has not already been attempted.
    2. Passes deterministic guardrails.
    """

    candidates = action_scores[
        ~action_scores["action"].isin(
            attempted_actions
        )
    ].copy()

    if candidates.empty:
        return {
            "final_action": "stop",
            "decision_reason": (
                "No untried recovery action remains."
            ),
            "guardrail_status": "blocked",
            "erv": 0.0,
            "recovery_probability": 0.0,
            "expected_recovery": 0.0,
            "action_cost": 0.0,
        }

    evaluated = []

    for row in candidates.to_dict(
        orient="records"
    ):

        action = row["action"]

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

    allowed = evaluated_df[
        evaluated_df[
            "guardrail_allowed"
        ]
    ].copy()

    if allowed.empty:
        return {
            "final_action": "stop",
            "decision_reason": (
                "No remaining action "
                "passes guardrails."
            ),
            "guardrail_status": "blocked",
            "erv": 0.0,
            "recovery_probability": 0.0,
            "expected_recovery": 0.0,
            "action_cost": 0.0,
        }

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
        "final_action": best[
            "action"
        ],
        "decision_reason": best[
            "guardrail_reason"
        ],
        "guardrail_status": "passed",
        "erv": float(
            best["erv"]
        ),
        "recovery_probability": float(
            best[
                "recovery_probability"
            ]
        ),
        "expected_recovery": float(
            best[
                "expected_recovery"
            ]
        ),
        "action_cost": float(
            best["action_cost"]
        ),
    }


def run_recovery_case(
    case: dict,
    action_scores: pd.DataFrame,
) -> dict:
    """
    Run the adaptive recovery loop.

    The agent:
      1. Executes the selected action.
      2. Verifies the result.
      3. Stops on successful recovery.
      4. Removes failed actions.
      5. Selects the best remaining permitted action.
      6. Stops or escalates when no useful action remains.
    """

    audit_events = []

    attempted_actions = set()

    attempt_count = 0

    total_recovered = 0.0

    current_case = dict(case)

    # --------------------------------------------------
    # Initial action
    # --------------------------------------------------

    initial_action = current_case.get(
        "final_action",
        "stop",
    )

    current_case[
        "final_action"
    ] = initial_action

    while True:

        action = current_case.get(
            "final_action",
            "stop",
        )

        # --------------------------------------------------
        # STOP
        # --------------------------------------------------

        if action == "stop":

            audit_events.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "stopping_decision",
                    "action": "stop",
                    "reason": (
                        current_case.get(
                            "decision_reason",
                            "Recovery stopped.",
                        )
                    ),
                }
            )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "stopped",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
            }

        # --------------------------------------------------
        # ESCALATE
        # --------------------------------------------------

        if action == "escalate":

            audit_events.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "escalation",
                    "action": "escalate",
                    "reason": (
                        current_case.get(
                            "decision_reason",
                            "Case routed to manual review.",
                        )
                    ),
                }
            )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "escalated",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
            }

        # --------------------------------------------------
        # AUTOMATED ACTION
        # --------------------------------------------------

        if attempt_count >= (
            MAX_AUTOMATED_ATTEMPTS
        ):

            audit_events.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "stopping_decision",
                    "action": "stop",
                    "reason": (
                        "Maximum automated "
                        "attempt limit reached."
                    ),
                }
            )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "stopped",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
            }

        # --------------------------------------------------
        # Execute
        # --------------------------------------------------

        attempt_count += 1

        current_case[
            "attempt_number"
        ] = attempt_count

        attempted_actions.add(
            action
        )

        audit_events.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": "action_selected",
                "attempt_number": attempt_count,
                "action": action,
                "reason": current_case.get(
                    "decision_reason",
                    "",
                ),
            }
        )

        result = execute_and_verify(
            current_case
        )

        recovered = bool(
            result.get(
                "verified_recovered",
                False,
            )
        )

        recovered_amount = float(
            result.get(
                "verified_amount",
                0.0,
            )
        )

        total_recovered += (
            recovered_amount
        )

        audit_events.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": "action_executed",
                "attempt_number": attempt_count,
                "action": action,
                "execution_status": result.get(
                    "execution_status"
                ),
                "verification_status": result.get(
                    "verification_status"
                ),
                "verified_recovered": recovered,
                "verified_amount": recovered_amount,
                "verification_reason": result.get(
                    "verification_reason"
                ),
            }
        )

        # --------------------------------------------------
        # SUCCESS
        # --------------------------------------------------

        if recovered:

            audit_events.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "recovery_verified",
                    "attempt_number": attempt_count,
                    "action": action,
                    "amount": recovered_amount,
                    "reason": (
                        "Recovery successfully "
                        "verified."
                    ),
                }
            )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "recovered",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
            }

        # --------------------------------------------------
        # FAILED ACTION
        # --------------------------------------------------

        audit_events.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": "recovery_failed",
                "attempt_number": attempt_count,
                "action": action,
                "reason": (
                    "Action executed but "
                    "recovery was not verified."
                ),
            }
        )

        # --------------------------------------------------
        # Customer opt-out
        # --------------------------------------------------

        if not bool(
            current_case.get(
                "communication_opt_in",
                False,
            )
        ):

            audit_events.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "stopping_decision",
                    "reason": (
                        "Customer communication "
                        "opt-out."
                    ),
                }
            )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "stopped",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
            }

        # --------------------------------------------------
        # Attempt limit
        # --------------------------------------------------

        if attempt_count >= (
            MAX_AUTOMATED_ATTEMPTS
        ):

            audit_events.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "escalation",
                    "reason": (
                        "Maximum automated "
                        "attempts reached."
                    ),
                }
            )

            return {
                "case_id": case[
                    "case_id"
                ],
                "final_status": "escalated",
                "total_recovered": (
                    total_recovered
                ),
                "attempts": attempt_count,
                "audit_events": audit_events,
            }

        # --------------------------------------------------
        # Select next action
        # --------------------------------------------------

        next_decision = select_next_action(
            current_case,
            action_scores,
            attempted_actions,
        )

        audit_events.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": "next_action_evaluation",
                "previous_action": action,
                "next_action": next_decision[
                    "final_action"
                ],
                "erv": next_decision[
                    "erv"
                ],
                "reason": next_decision[
                    "decision_reason"
                ],
            }
        )

        current_case.update(
            next_decision
        )


def main():
    """
    This module is intentionally not a CLI runner.

    Use agent_batch.py for batch execution.
    """
    print(
        "Recovery agent module loaded."
    )


if __name__ == "__main__":
    main()