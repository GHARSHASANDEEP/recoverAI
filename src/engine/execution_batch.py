import os

import pandas as pd

from src.engine.executor import (
    execute_and_verify,
)


DECISIONS_PATH = (
    "data/processed/decisions.csv"
)

OUTPUT_PATH = (
    "data/processed/execution_results.csv"
)


def main():

    decisions = pd.read_csv(
        DECISIONS_PATH
    )

    results = []

    for case in decisions.to_dict(
        orient="records"
    ):

        result = execute_and_verify(
            case
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "✓ Execution + verification completed."
    )

    print()

    print(
        f"Cases processed: "
        f"{len(results_df):,}"
    )

    print()

    print(
        "Execution status:"
    )

    print(
        results_df[
            "execution_status"
        ].value_counts()
    )

    print()

    print(
        "Verification status:"
    )

    print(
        results_df[
            "verification_status"
        ].value_counts()
    )

    print()

    recovered = results_df[
        "verified_recovered"
    ]

    print(
        f"Verified recoveries: "
        f"{recovered.sum():,}"
    )

    print()

    print(
        "Verified money recovered:"
    )

    print(
        f"₹{results_df['verified_amount'].sum():,.2f}"
    )

    print()

    print(
        "Unrecovered cases:"
    )

    print(
        f"{(~recovered).sum():,}"
    )

    print()

    print(
        "Recovery rate by action:"
    )

    executed = results_df[
        results_df[
            "execution_status"
        ] == "executed"
    ]

    if not executed.empty:

        print(
            executed.groupby(
                "action_executed"
            )[
                "verified_recovered"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

    print()

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()