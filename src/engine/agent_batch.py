import json
import os

import pandas as pd

from src.engine.recovery_agent import (
    run_recovery_case,
)


DECISIONS_PATH = (
    "data/processed/decisions.csv"
)

ERV_PATH = (
    "data/processed/erv_scores.csv"
)

OUTPUT_PATH = (
    "data/processed/agent_results.csv"
)

def main():

    decisions = pd.read_csv(
        DECISIONS_PATH
    )

    erv = pd.read_csv(
        ERV_PATH
    )

    results = []

    for case in decisions.to_dict(
        orient="records"
    ):

        action_scores = erv[
            erv["case_id"]
            == case["case_id"]
        ].copy()

        if action_scores.empty:
            raise ValueError(
                f"No ERV scores found for "
                f"{case['case_id']}"
            )

        result = run_recovery_case(
            case,
            action_scores,
        )

        results.append(
            result
        )

    rows = []

    for result in results:

        rows.append(
            {
                "case_id": result[
                    "case_id"
                ],
                "final_status": result[
                    "final_status"
                ],
                "attempts": result[
                    "attempts"
                ],
                "total_recovered": result[
                    "total_recovered"
                ],
                "audit_event_count": len(
                    result[
                        "audit_events"
                    ]
                ),
                "audit_trail": json.dumps(
                    result[
                        "audit_events"
                    ]
                ),
            }
        )

    output = pd.DataFrame(
        rows
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

    print(
        "✓ Recovery agent batch completed."
    )

    print()

    print(
        f"Cases processed: "
        f"{len(output):,}"
    )

    print()

    print(
        "Final status:"
    )

    print(
        output[
            "final_status"
        ].value_counts()
    )

    print()

    print(
        "Total verified money recovered:"
    )

    print(
        f"₹{output['total_recovered'].sum():,.2f}"
    )

    print()

    print(
        "Total recovery attempts:"
    )

    print(
        output[
            "attempts"
        ].sum()
    )

    print()

    print(
        "Cases requiring more than one attempt:"
    )

    print(
        (
            output[
                "attempts"
            ] > 1
        ).sum()
    )

    print()

    print(
        "Audit events:"
    )

    print(
        output[
            "audit_event_count"
        ].sum()
    )

    print()

    print(
        "Sample agent results:"
    )

    print(
        output[
            [
                "case_id",
                "final_status",
                "attempts",
                "total_recovered",
                "audit_event_count",
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