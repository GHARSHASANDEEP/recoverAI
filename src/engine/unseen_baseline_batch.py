import os

import pandas as pd

from src.engine.baseline import (
    run_baseline_case,
)


DECISIONS_PATH = (
    "data/unseen/processed/decisions.csv"
)

CASES_PATH = (
    "data/unseen/processed/recovery_cases.csv"
)

CUSTOMERS_PATH = (
    "data/raw/customers.csv"
)

OUTPUT_PATH = (
    "data/unseen/processed/baseline_results.csv"
)


def normalize_failure_category(value):

    if pd.isna(value):
        return "unknown_failure"

    value = str(value).strip()

    if value.startswith("[") and value.endswith("]"):
        value = value.strip("[]").strip("'\" ")

    return value or "unknown_failure"


def safe_float(value, default=0.0):

    if pd.isna(value):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):

    if pd.isna(value):
        return default

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def main():

    print("Loading unseen baseline inputs...")

    decisions = pd.read_csv(
        DECISIONS_PATH
    )

    cases = pd.read_csv(
        CASES_PATH
    )

    customers = pd.read_csv(
        CUSTOMERS_PATH
    )

    print(
        f"Decisions loaded: {len(decisions):,}"
    )

    print(
        f"Cases loaded: {len(cases):,}"
    )

    # ---------------------------------------------------------
    # Validate unseen data
    # ---------------------------------------------------------

    if not decisions[
        "customer_id"
    ].astype(str).str.startswith(
        "UNSEEN_"
    ).all():

        raise ValueError(
            "Non-unseen customer found in "
            "unseen baseline evaluation."
        )

    # ---------------------------------------------------------
    # Build complete customer context
    # ---------------------------------------------------------

    customer_columns = [
        "customer_id",
        "customer_segment",
        "communication_opt_in",
        "customer_lifetime_value",
        "successful_payment_count",
        "failed_payment_count",
        "total_payment_attempts",
    ]

    customer_columns = [
        c for c in customer_columns
        if c in customers.columns
    ]

    context = cases.merge(
        customers[customer_columns],
        on="customer_id",
        how="left",
    )

    context["failure_category"] = (
        context[
            "failure_categories"
        ].apply(
            normalize_failure_category
        )
    )

    context["communication_opt_in"] = (
        context[
            "communication_opt_in"
        ]
        .fillna(False)
        .astype(bool)
    )

    context["recovery_amount"] = (
        pd.to_numeric(
            context[
                "recovery_amount"
            ],
            errors="coerce",
        )
        .fillna(0.0)
    )

    # ---------------------------------------------------------
    # Build lookup by case_id
    # ---------------------------------------------------------

    context_lookup = {
        str(row["case_id"]): row
        for _, row in context.iterrows()
    }

    results = []

    # ---------------------------------------------------------
    # Run baseline on EXACT SAME unseen cases
    # ---------------------------------------------------------

    for decision in decisions.to_dict(
        orient="records"
    ):

        case_id = str(
            decision["case_id"]
        )

        if case_id not in context_lookup:

            raise ValueError(
                f"Missing case context for "
                f"{case_id}"
            )

        row = context_lookup[
            case_id
        ]

        case = {
            "case_id": case_id,

            "customer_id": decision[
                "customer_id"
            ],

            "customer_segment": (
                row.get(
                    "customer_segment",
                    "unknown",
                )
                if not pd.isna(
                    row.get(
                        "customer_segment",
                        "unknown",
                    )
                )
                else "unknown"
            ),

            "communication_opt_in": bool(
                row.get(
                    "communication_opt_in",
                    False,
                )
            ),

            "customer_lifetime_value": safe_float(
                row.get(
                    "customer_lifetime_value",
                    0.0,
                )
            ),

            "successful_payment_count": safe_int(
                row.get(
                    "successful_payment_count",
                    0,
                )
            ),

            "failed_payment_count": safe_int(
                row.get(
                    "failed_payment_count",
                    0,
                )
            ),

            "total_payment_attempts": safe_int(
                row.get(
                    "total_payment_attempts",
                    0,
                )
            ),

            "failure_category": (
                decision[
                    "failure_category"
                ]
            ),

            "recovery_amount": safe_float(
                row.get(
                    "recovery_amount",
                    0.0,
                )
            ),
        }

        result = run_baseline_case(
            case
        )

        results.append(
            result
        )

    output = pd.DataFrame(
        results
    )

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    total_cases = len(output)

    recovered_cases = int(
        output[
            "recovered"
        ].sum()
    )

    total_recovered = float(
        output[
            "recovered_amount"
        ].sum()
    )

    recovery_rate = (
        recovered_cases / total_cases
        if total_cases
        else 0.0
    )

    print()
    print(
        "✓ Unseen baseline evaluation completed."
    )

    print(
        f"Cases evaluated: "
        f"{total_cases:,}"
    )

    print(
        f"Recovered cases: "
        f"{recovered_cases:,}"
    )

    print(
        f"Recovery rate: "
        f"{recovery_rate:.2%}"
    )

    print(
        f"Baseline recovered money: "
        f"₹{total_recovered:,.2f}"
    )

    # ---------------------------------------------------------
    # Category breakdown
    # ---------------------------------------------------------

    category_rates = (
        decisions[
            [
                "case_id",
                "failure_category",
            ]
        ]
        .merge(
            output[
                [
                    "case_id",
                    "recovered",
                    "recovered_amount",
                ]
            ],
            on="case_id",
        )
        .groupby(
            "failure_category"
        )
        .agg(
            cases=(
                "case_id",
                "count",
            ),
            recovered=(
                "recovered",
                "sum",
            ),
            recovered_amount=(
                "recovered_amount",
                "sum",
            ),
        )
    )

    category_rates[
        "recovery_rate"
    ] = (
        category_rates[
            "recovered"
        ]
        / category_rates[
            "cases"
        ]
    )

    print()
    print(
        "Recovery by failure category:"
    )

    print(
        category_rates.sort_values(
            "recovery_rate",
            ascending=False,
        ).to_string()
    )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()