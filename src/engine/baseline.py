from src.data.outcome_simulator import (
    simulate_outcome,
)


def run_baseline_case(case: dict) -> dict:
    """
    Simple benchmark baseline.

    Policy:
    - Attempt one retry.
    - No ML probability.
    - No ERV optimization.
    - No intelligent action selection.
    - No second attempt.
    """

    action = "retry"

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
        attempt_number=1,
    )

    recovered = bool(
        outcome["recovered"]
    )

    recovered_amount = float(
        outcome["recovered_amount"]
    )

    return {
        "case_id": case[
            "case_id"
        ],
        "customer_id": case[
            "customer_id"
        ],
        "action": action,
        "recovered": recovered,
        "recovered_amount": recovered_amount,
        "attempts": 1,
    }