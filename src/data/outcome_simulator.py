import pandas as pd

from src.data.outcome_rules import (
    get_hidden_recovery_probability,
    stable_random_value,
)


def simulate_outcome(
    case_id: str,
    failure_category: str | None,
    action: str,
    customer_segment: str,
    amount: float,
    attempt_number: int = 1,
) -> dict:
    """
    Simulate the hidden outcome for one recovery action.

    The caller receives the outcome but does not receive
    the hidden probability as an input to decision making.
    """

    hidden_probability = (
        get_hidden_recovery_probability(
            failure_category=failure_category,
            action=action,
            customer_segment=customer_segment,
            amount=amount,
            attempt_number=attempt_number,
        )
    )

    random_value = stable_random_value(
        case_id
    )

    recovered = (
        random_value < hidden_probability
    )

    return {
        "case_id": case_id,
        "action": action,
        "recovered": recovered,
        "recovered_amount": (
            float(amount)
            if recovered
            else 0.0
        ),
    }