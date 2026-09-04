import os

import joblib
import pandas as pd

from src.data.outcome_rules import (
    ACTION_COSTS,
    get_amount_bucket,
)


CASES_PATH = (
    "data/unseen/processed/recovery_cases.csv"
)

CUSTOMERS_PATH = (
    "data/unseen/raw/customers.csv"
)

MODEL_PATH = (
    "models/recovery_probability_model_v3.joblib"
)

OUTPUT_PATH = (
    "data/unseen/processed/erv_scores.csv"
)


ACTIONS = [
    "retry",
    "reminder",
    "escalate",
]


CUSTOMER_COLUMNS = [
    "customer_id",
    "customer_segment",
    "customer_lifetime_value",
    "successful_payment_count",
    "failed_payment_count",
    "total_payment_attempts",
    "communication_opt_in",
]


def load_data():
    """Load unseen recovery cases and unseen customers."""

    cases = pd.read_csv(
        CASES_PATH
    )

    customers = pd.read_csv(
        CUSTOMERS_PATH
    )

    cases = cases.merge(
        customers[
            CUSTOMER_COLUMNS
        ],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    if cases[
        "customer_segment"
    ].isna().any():
        raise ValueError(
            "Some unseen cases could not be matched "
            "to customer records."
        )

    cases["failure_category"] = (
        cases[
            "failure_categories"
        ]
        .fillna("unknown_failure")
        .astype(str)
        .str.split("|")
        .str[0]
    )

    cases["amount_bucket"] = (
        cases[
            "recovery_amount"
        ].apply(
            get_amount_bucket
        )
    )

    cases["attempt_number"] = 1

    return cases


def build_prediction_rows(
    cases,
):
    """
    Convert every unseen case into three action rows.

    3 actions are evaluated for every case:
    retry, reminder, escalate.
    """

    rows = []

    for case in cases.to_dict(
        orient="records"
    ):

        for action in ACTIONS:

            successful_payments = max(
                int(
                    case[
                        "successful_payment_count"
                    ]
                ),
                0,
            )

            failed_payments = max(
                int(
                    case[
                        "failed_payment_count"
                    ]
                ),
                0,
            )

            total_attempts = max(
                int(
                    case[
                        "total_payment_attempts"
                    ]
                ),
                1,
            )

            recovery_amount = float(
                case[
                    "recovery_amount"
                ]
            )

            lifetime_value = max(
                float(
                    case[
                        "customer_lifetime_value"
                    ]
                ),
                1.0,
            )

            row = {
                "case_id": case[
                    "case_id"
                ],

                "customer_id": case[
                    "customer_id"
                ],

                "customer_segment": case[
                    "customer_segment"
                ],

                "customer_lifetime_value": (
                    case[
                        "customer_lifetime_value"
                    ]
                ),

                "successful_payment_count": (
                    case[
                        "successful_payment_count"
                    ]
                ),

                "failed_payment_count": (
                    case[
                        "failed_payment_count"
                    ]
                ),

                "total_payment_attempts": (
                    case[
                        "total_payment_attempts"
                    ]
                ),

                "communication_opt_in": (
                    case[
                        "communication_opt_in"
                    ]
                ),

                "action": action,

                "recovery_amount": (
                    recovery_amount
                ),

                "action_cost": (
                    ACTION_COSTS[
                        action
                    ]
                ),

                "failure_category": (
                    case[
                        "failure_category"
                    ]
                ),

                "amount_bucket": (
                    case[
                        "amount_bucket"
                    ]
                ),

                "attempt_number": 1,

                "historical_failure_rate": (
                    failed_payments
                    / total_attempts
                ),

                "customer_success_rate": (
                    successful_payments
                    / total_attempts
                ),

                "failure_to_success_ratio": (
                    failed_payments
                    / max(
                        successful_payments,
                        1,
                    )
                ),

                "amount_to_lifetime_value_ratio": (
                    recovery_amount
                    / lifetime_value
                ),

                "payment_history_volume": (
                    successful_payments
                    + failed_payments
                ),
            }

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def validate_prediction_data(
    prediction_df,
    cases,
):
    """Validate the unseen inference dataset."""

    expected_rows = (
        len(cases)
        * len(ACTIONS)
    )

    assert len(
        prediction_df
    ) == expected_rows, (
        "Unexpected number of prediction rows."
    )

    assert (
        prediction_df[
            "case_id"
        ].nunique()
        == len(cases)
    )

    assert (
        prediction_df[
            "customer_id"
        ]
        .astype(str)
        .str.startswith("UNSEEN_")
        .all()
    )

    assert (
        prediction_df[
            "recovery_amount"
        ]
        .ge(0)
        .all()
    )

    print(
        "✓ Unseen inference-data validation passed."
    )


def main():

    print(
        "Loading unseen recovery cases..."
    )

    cases = load_data()

    print(
        f"Cases loaded: "
        f"{len(cases):,}"
    )

    print()

    print(
        "Building unseen batch features..."
    )

    prediction_df = (
        build_prediction_rows(
            cases
        )
    )

    print(
        f"Prediction rows: "
        f"{len(prediction_df):,}"
    )

    validate_prediction_data(
        prediction_df,
        cases,
    )

    feature_columns = [
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
        "customer_success_rate",
        "failure_to_success_ratio",
        "amount_to_lifetime_value_ratio",
        "payment_history_volume",
    ]

    X = prediction_df[
        feature_columns
    ]

    print(
        "Loading frozen model..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Running unseen batch inference..."
    )

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    prediction_df[
        "recovery_probability"
    ] = probabilities

    prediction_df[
        "expected_recovery"
    ] = (
        prediction_df[
            "recovery_probability"
        ]
        *
        prediction_df[
            "recovery_amount"
        ]
    )

    prediction_df[
        "erv"
    ] = (
        prediction_df[
            "expected_recovery"
        ]
        -
        prediction_df[
            "action_cost"
        ]
    )

    # -----------------------------------------------------
    # Select best action by ERV.
    # Guardrails will make the final decision later.
    # -----------------------------------------------------

    best_indices = (
        prediction_df
        .groupby(
            "case_id"
        )[
            "erv"
        ]
        .idxmax()
    )

    best_df = (
        prediction_df.loc[
            best_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # -----------------------------------------------------
    # Save every action score.
    # -----------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True,
    )

    prediction_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()

    print(
        "================================"
    )
    print(
        "UNSEEN ERV SCORING"
    )
    print(
        "================================"
    )

    print(
        "Model retrained: NO"
    )

    print(
        f"Cases scored: "
        f"{len(best_df):,}"
    )

    print(
        f"Action evaluations: "
        f"{len(prediction_df):,}"
    )

    print()

    print(
        "Recommended actions before guardrails:"
    )

    print(
        best_df[
            "action"
        ].value_counts()
    )

    print()

    print(
        "Total revenue at risk:"
    )

    print(
        f"₹{best_df['recovery_amount'].sum():,.2f}"
    )

    print()

    print(
        "Total expected recovery:"
    )

    print(
        f"₹{best_df['expected_recovery'].sum():,.2f}"
    )

    print()

    print(
        "Total action cost:"
    )

    print(
        f"₹{best_df['action_cost'].sum():,.2f}"
    )

    print()

    print(
        "Total ERV:"
    )

    print(
        f"₹{best_df['erv'].sum():,.2f}"
    )

    print()

    print(
        "Sample unseen scores:"
    )

    print(
        best_df[
            [
                "case_id",
                "customer_id",
                "action",
                "recovery_amount",
                "recovery_probability",
                "expected_recovery",
                "action_cost",
                "erv",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()