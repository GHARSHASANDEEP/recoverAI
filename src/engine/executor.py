from datetime import datetime

from src.data.outcome_simulator import (
    simulate_outcome,
)


def execute_action(case: dict) -> dict:
    """
    Execute one approved recovery action.

    This is a simulation adapter for the hackathon.
    No real payment is processed.
    """

    action = case["final_action"]

    if action == "stop":
        return {
            "execution_status": "stopped",
            "action_executed": "stop",
            "execution_message": (
                "Recovery stopped by policy."
            ),
        }

    if action == "escalate":
        return {
            "execution_status": "escalated",
            "action_executed": "escalate",
            "execution_message": (
                "Case routed to manual review."
            ),
        }

    outcome = simulate_outcome(
        case_id=case["case_id"],
        failure_category=case[
            "failure_category"
        ],
        action=action,
        customer_segment=case[
            "customer_segment"
        ],
        amount=float(
            case["recovery_amount"]
        ),
        attempt_number=int(
            case.get(
                "attempt_number",
                1,
            )
        ),
    )

    return {
        "execution_status": "executed",
        "action_executed": action,
        "execution_message": (
            f"Simulated {action} executed."
        ),
        "simulated_recovered": (
            outcome["recovered"]
        ),
        "simulated_recovered_amount": (
            outcome[
                "recovered_amount"
            ]
        ),
    }


def verify_recovery(
    execution_result: dict,
) -> dict:
    """
    Verify whether the recovery actually occurred.

    Verification is explicit and never inferred
    merely because an action was executed.
    """

    if (
        execution_result[
            "execution_status"
        ]
        != "executed"
    ):
        return {
            "verification_status": "not_applicable",
            "verified_recovered": False,
            "verified_amount": 0.0,
            "verification_reason": (
                "No payment recovery execution "
                "occurred."
            ),
        }

    recovered = bool(
        execution_result.get(
            "simulated_recovered",
            False,
        )
    )

    amount = float(
        execution_result.get(
            "simulated_recovered_amount",
            0.0,
        )
    )

    if recovered and amount > 0:
        return {
            "verification_status": "verified",
            "verified_recovered": True,
            "verified_amount": amount,
            "verification_reason": (
                "Recovery confirmed by "
                "simulated payment verification."
            ),
        }

    return {
        "verification_status": "not_recovered",
        "verified_recovered": False,
        "verified_amount": 0.0,
        "verification_reason": (
            "Action executed but no successful "
            "payment was verified."
        ),
    }


def execute_and_verify(
    case: dict,
) -> dict:
    """
    Execute an approved action and immediately
    verify the resulting recovery.
    """

    executed_at = datetime.now().isoformat()

    execution = execute_action(
        case
    )

    verification = verify_recovery(
        execution
    )

    return {
        **case,
        **execution,
        **verification,
        "executed_at": executed_at,
    }   