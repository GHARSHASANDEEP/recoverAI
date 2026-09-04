import pandas as pd

from src.data.outcome_rules import (
    ACTION_COSTS,
    get_amount_bucket,
)
from src.model.action_recommender import (
    predict_action_probability,
)


ACTIONS = [
    "retry",
    "reminder",
    "escalate",
]


def calculate_erv(
    recovery_amount: float,
    probability: float,
    action_cost: float,
) -> float:
    """
    Expected Recovery Value.

    Expected recovery = probability × amount

    ERV = expected recovery - action cost
    """

    expected_recovery = (
        probability
        * recovery_amount
    )

    return (
        expected_recovery
        - action_cost
    )


def score_case_actions(
    case: dict,
) -> pd.DataFrame:
    """
    Score every permitted action for a recovery case.
    """

    results = []

    for action in ACTIONS:

        probability = (
            predict_action_probability(
                case,
                action,
            )
        )

        action_cost = ACTION_COSTS[
            action
        ]

        expected_recovery = (
            probability
            * float(
                case[
                    "recovery_amount"
                ]
            )
        )

        erv = calculate_erv(
            recovery_amount=float(
                case[
                    "recovery_amount"
                ]
            ),
            probability=probability,
            action_cost=action_cost,
        )

        results.append(
            {
                "case_id": case[
                    "case_id"
                ],
                "action": action,
                "recovery_amount": float(
                    case[
                        "recovery_amount"
                    ]
                ),
                "recovery_probability": (
                    probability
                ),
                "expected_recovery": (
                    expected_recovery
                ),
                "action_cost": (
                    action_cost
                ),
                "erv": erv,
            }
        )

    return pd.DataFrame(
        results
    )


def select_best_action(
    scores: pd.DataFrame,
) -> dict:
    """Select the action with the highest ERV."""

    ranked = scores.sort_values(
        [
            "erv",
            "recovery_probability",
        ],
        ascending=[
            False,
            False,
        ],
    )

    best = ranked.iloc[0]

    return best.to_dict()