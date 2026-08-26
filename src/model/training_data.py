import os

import pandas as pd

from src.data.outcome_rules import (
    get_hidden_recovery_probability,
    get_amount_bucket,
    stable_random_value,
)


CASES_PATH = "data/processed/recovery_cases.csv"
CUSTOMERS_PATH = "data/raw/customers.csv"

OUTPUT_PATH = (
    "data/processed/training_data.csv"
)


ACTIONS = [
    "retry",
    "reminder",
    "escalate",
]


def load_data():
    """Load recovery cases and customer information."""

    cases = pd.read_csv(
        CASES_PATH
    )

    customers = pd.read_csv(
        CUSTOMERS_PATH
    )

    return cases, customers


def get_primary_failure_category(
    value,
) -> str:

    if pd.isna(value):
        return "unknown_failure"

    value = str(value)

    if "|" in value:
        return value.split("|")[0]

    return value


def get_attempt_number(
    event_ids: str,
) -> int:
    """
    Estimate attempt number from event structure.

    For now, recovery cases start at attempt 1.
    Later the execution engine will maintain the
    actual retry count.
    """

    return 1


def build_training_dataset():
    """
    Create action-level training examples.

    Each recovery case produces one example for each
    possible action.

    The hidden probability is used ONLY to simulate
    the synthetic outcome. It is never stored as a
    model feature.
    """

    cases, customers = load_data()

    df = cases.merge(
        customers[
            [
                "customer_id",
                "customer_segment",
                "customer_lifetime_value",
                "successful_payment_count",
                "failed_payment_count",
                "total_payment_attempts",
                "communication_opt_in",
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    rows = []

    for case in df.itertuples(
        index=False
    ):

        failure_category = (
            get_primary_failure_category(
                case.failure_categories
            )
        )

        amount_bucket = (
            get_amount_bucket(
                case.recovery_amount
            )
        )

        attempt_number = (
            get_attempt_number(
                case.event_ids
            )
        )

        historical_attempts = (
            int(
                case.total_payment_attempts
            )
        )

        historical_failures = (
            int(
                case.failed_payment_count
            )
        )

        if historical_attempts > 0:
            historical_failure_rate = (
                historical_failures
                / historical_attempts
            )
        else:
            historical_failure_rate = 0.0

        for action in ACTIONS:

            # -------------------------------------------------
            # Hidden synthetic world.
            #
            # The model NEVER receives this probability.
            # -------------------------------------------------

            hidden_probability = (
                get_hidden_recovery_probability(
                    failure_category=(
                        failure_category
                    ),
                    action=action,
                    customer_segment=(
                        case.customer_segment
                    ),
                    amount=(
                        case.recovery_amount
                    ),
                    attempt_number=(
                        attempt_number
                    ),
                )
            )

            # -------------------------------------------------
            # Deterministic synthetic outcome.
            # -------------------------------------------------

            outcome_key = (
                f"{case.case_id}|{action}"
            )

            random_value = (
                stable_random_value(
                    outcome_key
                )
            )

            recovered = int(
                random_value
                < hidden_probability
            )

            rows.append(
                {
                    # -----------------------------------------
                    # Identifiers
                    # -----------------------------------------

                    "case_id": (
                        case.case_id
                    ),

                    # -----------------------------------------
                    # Observable features
                    # -----------------------------------------

                    "failure_category": (
                        failure_category
                    ),

                    "customer_segment": (
                        case.customer_segment
                    ),

                    "amount_bucket": (
                        amount_bucket
                    ),

                    "action": action,

                    "attempt_number": (
                        attempt_number
                    ),

                    "recovery_amount": (
                        float(
                            case.recovery_amount
                        )
                    ),

                    "customer_lifetime_value": (
                        float(
                            case.customer_lifetime_value
                        )
                    ),

                    "successful_payment_count": (
                        int(
                            case.successful_payment_count
                        )
                    ),

                    "failed_payment_count": (
                        int(
                            case.failed_payment_count
                        )
                    ),

                    "total_payment_attempts": (
                        historical_attempts
                    ),

                    "historical_failure_rate": (
                        historical_failure_rate
                    ),

                    "communication_opt_in": (
                        int(
                            bool(
                                case.communication_opt_in
                            )
                        )
                    ),

                    # -----------------------------------------
                    # Training target
                    # -----------------------------------------

                    "recovered": recovered,
                }
            )

    training_df = pd.DataFrame(
        rows
    )

    return training_df


def validate_training_data(
    training_df: pd.DataFrame,
) -> None:
    """Run basic training-data validation."""

    required_columns = {
        "case_id",
        "failure_category",
        "customer_segment",
        "amount_bucket",
        "action",
        "attempt_number",
        "recovery_amount",
        "customer_lifetime_value",
        "successful_payment_count",
        "failed_payment_count",
        "total_payment_attempts",
        "historical_failure_rate",
        "communication_opt_in",
        "recovered",
    }

    missing_columns = (
        required_columns
        - set(training_df.columns)
    )

    assert not missing_columns, (
        f"Missing columns: "
        f"{missing_columns}"
    )

    assert (
        training_df["recovered"]
        .isin([0, 1])
        .all()
    ), (
        "Target must be binary."
    )

    assert (
        training_df["case_id"]
        .notna()
        .all()
    )

    assert (
        training_df["action"]
        .isin(ACTIONS)
        .all()
    )

    print(
        "✓ Training-data schema "
        "validation passed."
    )


def save_training_data(
    training_df: pd.DataFrame,
) -> None:

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True,
    )

    training_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )


def main():

    training_df = (
        build_training_dataset()
    )

    validate_training_data(
        training_df
    )

    save_training_data(
        training_df
    )

    print()

    print(
        f"Training examples: "
        f"{len(training_df):,}"
    )

    print()

    print(
        "Actions:"
    )

    print(
        training_df[
            "action"
        ].value_counts()
    )

    print()

    print(
        "Target distribution:"
    )

    print(
        training_df[
            "recovered"
        ].value_counts()
    )

    print()

    print(
        "Recovery rate:"
    )

    print(
        round(
            training_df[
                "recovered"
            ].mean(),
            4,
        )
    )

    print()

    print(
        "Feature columns:"
    )

    print(
        [
            column
            for column in training_df.columns
            if column
            not in {
                "case_id",
                "recovered",
            }
        ]
    )

    print()

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()