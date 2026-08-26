import os

import pandas as pd

from src.engine.decision_engine import (
    apply_decision,
)


ERV_PATH = (
    "data/processed/erv_scores.csv"
)

OUTPUT_PATH = (
    "data/processed/decisions.csv"
)


def load_data():
    """Load the already-enriched ERV scores."""

    return pd.read_csv(
        ERV_PATH
    )


def build_case(group):
    """Build one case context from its ERV rows."""

    row = group.iloc[0]

    return {
        "case_id": row["case_id"],
        "customer_id": row["customer_id"],
        "failure_category": row[
            "failure_category"
        ],
        "customer_segment": row[
            "customer_segment"
        ],
        "customer_lifetime_value": row[
            "customer_lifetime_value"
        ],
        "successful_payment_count": row[
            "successful_payment_count"
        ],
        "failed_payment_count": row[
            "failed_payment_count"
        ],
        "total_payment_attempts": row[
            "total_payment_attempts"
        ],
        "communication_opt_in": row[
            "communication_opt_in"
        ],
        "recovery_amount": row[
            "recovery_amount"
        ],
        "attempt_number": row[
            "attempt_number"
        ],
    }


def main():

    erv = load_data()

    required_columns = [
        "case_id",
        "customer_id",
        "customer_segment",
        "customer_lifetime_value",
        "successful_payment_count",
        "failed_payment_count",
        "total_payment_attempts",
        "communication_opt_in",
        "failure_category",
        "recovery_amount",
        "attempt_number",
        "action",
        "recovery_probability",
        "expected_recovery",
        "action_cost",
        "erv",
    ]

    missing = [
        column
        for column in required_columns
        if column not in erv.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    decisions = []

    for case_id, group in erv.groupby(
        "case_id",
        sort=False,
    ):

        case = build_case(
            group
        )

        decision = apply_decision(
            case,
            group,
        )

        decisions.append(
            {
                "case_id": case[
                    "case_id"
                ],

                "customer_id": case[
                    "customer_id"
                ],

                "customer_segment": case[
                    "customer_segment"
                ],

                "communication_opt_in": case[
                    "communication_opt_in"
                ],

                "failure_category": case[
                    "failure_category"
                ],

                "recovery_amount": case[
                    "recovery_amount"
                ],

                "final_action": decision[
                    "final_action"
                ],

                "decision_reason": decision[
                    "decision_reason"
                ],

                "guardrail_status": decision[
                    "guardrail_status"
                ],

                "recovery_probability": decision.get(
                    "recovery_probability",
                    0.0,
                ),

                "expected_recovery": decision.get(
                    "expected_recovery",
                    0.0,
                ),

                "action_cost": decision.get(
                    "action_cost",
                    0.0,
                ),

                "erv": decision.get(
                    "erv",
                    0.0,
                ),
            }
        )

    decisions_df = pd.DataFrame(
        decisions
    )

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True,
    )

    decisions_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "✓ Decision engine completed."
    )

    print()

    print(
        f"Cases evaluated: "
        f"{len(decisions_df):,}"
    )

    print()

    print(
        "Final actions:"
    )

    print(
        decisions_df[
            "final_action"
        ].value_counts()
    )

    print()

    print(
        "Guardrail status:"
    )

    print(
        decisions_df[
            "guardrail_status"
        ].value_counts()
    )

    print()

    print(
        "Expected recovery:"
    )

    print(
        f"₹{decisions_df['expected_recovery'].sum():,.2f}"
    )

    print()

    print(
        "ERV:"
    )

    print(
        f"₹{decisions_df['erv'].sum():,.2f}"
    )

    print()

    print(
        "Sample decisions:"
    )

    print(
        decisions_df[
            [
                "case_id",
                "customer_id",
                "final_action",
                "recovery_amount",
                "recovery_probability",
                "expected_recovery",
                "erv",
                "guardrail_status",
                "decision_reason",
            ]
        ]
        .head(15)
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