import pandas as pd

from src.engine.normalizer import build_unified_events
from src.engine.deduplicator import (
    find_candidate_matches,
    select_one_to_one_matches,
)


OUTPUT_PATH = "data/processed/recovery_cases.csv"


def build_recovery_cases(
    events: pd.DataFrame,
    selected_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert normalized revenue-risk events into
    deduplicated recovery opportunities.

    A payment + checkout match becomes ONE recovery case.

    All unmatched events remain independent recovery cases.
    """

    cases = []

    matched_event_ids = set()

    # -----------------------------------------------------
    # Create cases from matched payment + checkout events.
    # -----------------------------------------------------

    if not selected_matches.empty:

        for index, match in enumerate(
            selected_matches.itertuples(
                index=False
            ),
            start=1,
        ):

            payment_id = (
                match.payment_event_id
            )

            checkout_id = (
                match.checkout_event_id
            )

            payment_event = events[
                events["event_id"]
                == payment_id
            ].iloc[0]

            checkout_event = events[
                events["event_id"]
                == checkout_id
            ].iloc[0]

            matched_event_ids.add(
                payment_id
            )

            matched_event_ids.add(
                checkout_id
            )

            # Use the larger amount so that we don't
            # understate the opportunity while avoiding
            # double counting.
            recovery_amount = max(
                float(
                    payment_event[
                        "event_amount"
                    ]
                ),
                float(
                    checkout_event[
                        "event_amount"
                    ]
                ),
            )

            failure_categories = []

            if pd.notna(
                payment_event[
                    "failure_category"
                ]
            ):
                failure_categories.append(
                    payment_event[
                        "failure_category"
                    ]
                )

            case = {
                "case_id": (
                    f"CASE_{len(cases) + 1:06d}"
                ),
                "customer_id": (
                    match.customer_id
                ),
                "event_ids": (
                    f"{payment_id}|"
                    f"{checkout_id}"
                ),
                "surfaces": (
                    "payment|checkout"
                ),
                "event_types": (
                    "payment_failed|"
                    "checkout_abandoned"
                ),
                "recovery_amount": (
                    recovery_amount
                ),
                "failure_categories": (
                    "|".join(
                        failure_categories
                    )
                    if failure_categories
                    else None
                ),
                "dedup_status": "matched",
                "dedup_score": (
                    float(
                        match.dedup_score
                    )
                ),
                "case_created_at": min(
                    payment_event[
                        "event_timestamp"
                    ],
                    checkout_event[
                        "event_timestamp"
                    ],
                ),
            }

            cases.append(case)

    # -----------------------------------------------------
    # Create independent cases from all unmatched events.
    # -----------------------------------------------------

    unmatched_events = events[
        ~events["event_id"].isin(
            matched_event_ids
        )
    ].copy()

    for event in unmatched_events.itertuples(
        index=False
    ):

        failure_category = (
            event.failure_category
        )

        if pd.isna(
            failure_category
        ):
            failure_category = None

        case = {
            "case_id": (
                f"CASE_{len(cases) + 1:06d}"
            ),
            "customer_id": (
                event.customer_id
            ),
            "event_ids": event.event_id,
            "surfaces": event.surface,
            "event_types": event.event_type,
            "recovery_amount": float(
                event.event_amount
            ),
            "failure_categories": (
                failure_category
            ),
            "dedup_status": "independent",
            "dedup_score": None,
            "case_created_at": (
                event.event_timestamp
            ),
        }

        cases.append(case)

    return pd.DataFrame(cases)


def validate_recovery_cases(
    events: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    """
    Validate that every actionable event belongs to
    exactly one recovery case.
    """

    case_event_ids = []

    for event_ids in cases[
        "event_ids"
    ]:

        case_event_ids.extend(
            event_ids.split("|")
        )

    event_id_counts = pd.Series(
        case_event_ids
    ).value_counts()

    # Every event must appear exactly once.
    duplicate_events = (
        event_id_counts[
            event_id_counts > 1
        ]
    )

    missing_events = (
        set(events["event_id"])
        - set(event_id_counts.index)
    )

    assert duplicate_events.empty, (
        "Some events belong to multiple "
        "recovery cases."
    )

    assert not missing_events, (
        "Some actionable events were not "
        "assigned to a recovery case."
    )

    print(
        "✓ Recovery-case event coverage "
        "validation passed."
    )


def save_recovery_cases(
    cases: pd.DataFrame,
) -> None:
    """Save recovery cases to the processed dataset."""

    output_path = OUTPUT_PATH

    import os

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    cases.to_csv(
        output_path,
        index=False,
    )


def main() -> None:
    """Build and validate recovery cases."""

    events = build_unified_events()

    candidates = find_candidate_matches(
        events
    )

    selected_matches = (
        select_one_to_one_matches(
            candidates
        )
    )

    cases = build_recovery_cases(
        events,
        selected_matches,
    )

    validate_recovery_cases(
        events,
        cases,
    )

    save_recovery_cases(
        cases
    )

    print()

    print(
        f"Actionable events: "
        f"{len(events):,}"
    )

    print(
        f"Candidate matches: "
        f"{len(candidates):,}"
    )

    print(
        f"Selected deduplicated matches: "
        f"{len(selected_matches):,}"
    )

    print(
        f"Recovery cases: "
        f"{len(cases):,}"
    )

    print()

    print(
        "Cases by deduplication status:"
    )

    print(
        cases[
            "dedup_status"
        ].value_counts()
    )

    print()

    print(
        "Recovery amount:"
    )

    print(
        f"₹{cases['recovery_amount'].sum():,.2f}"
    )

    print()

    print(
        "Top recovery cases:"
    )

    print(
        cases.sort_values(
            "recovery_amount",
            ascending=False,
        )[
            [
                "case_id",
                "customer_id",
                "surfaces",
                "recovery_amount",
                "dedup_status",
                "dedup_score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()