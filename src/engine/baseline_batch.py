import os

import pandas as pd

from src.engine.baseline import (
    run_baseline_case,
)


DECISIONS_PATH = (
    "data/processed/decisions.csv"
)

OUTPUT_PATH = (
    "data/processed/baseline_results.csv"
)


def main():

    decisions = pd.read_csv(
        DECISIONS_PATH
    )

    results = []

    for case in decisions.to_dict(
        orient="records"
    ):

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

    total_cases = len(output)

    recovered_cases = int(
        output["recovered"].sum()
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

    print(
        "✓ Baseline evaluation completed."
    )

    print()

    print(
        f"Cases evaluated: "
        f"{total_cases:,}"
    )

    print()

    print(
        f"Recovered cases: "
        f"{recovered_cases:,}"
    )

    print()

    print(
        f"Recovery rate: "
        f"{recovery_rate:.4f}"
    )

    print()

    print(
        "Baseline recovered money:"
    )

    print(
        f"₹{total_recovered:,.2f}"
    )

    print()

    print(
        "Recovery by failure category:"
    )

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
        category_rates["recovered"]
        / category_rates["cases"]
    )

    print(
        category_rates.sort_values(
            "recovery_rate",
            ascending=False,
        ).to_string()
    )

    print()

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()