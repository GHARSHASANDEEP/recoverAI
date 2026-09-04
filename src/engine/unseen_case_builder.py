from pathlib import Path

import pandas as pd

from src.engine.normalizer import (
    set_data_directory,
    build_unified_events,
)
from src.engine.deduplicator import (
    find_candidate_matches,
    select_one_to_one_matches,
)


UNSEEN_DATA_DIRECTORY = Path(
    "data/unseen/raw"
)

UNSEEN_OUTPUT_PATH = (
    "data/unseen/processed/recovery_cases.csv"
)


def build_unseen_recovery_cases() -> pd.DataFrame:
    """
    Build recovery cases from the fresh unseen dataset.

    This uses the same normalization and deduplication
    logic as the production pipeline, but reads from
    data/unseen/raw and writes to data/unseen/processed.
    """

    # -----------------------------------------------------
    # Point the normalizer at the unseen dataset.
    # -----------------------------------------------------

    set_data_directory(
        UNSEEN_DATA_DIRECTORY
    )

    # -----------------------------------------------------
    # Normalize unseen events.
    # -----------------------------------------------------

    events = build_unified_events()

    # -----------------------------------------------------
    # Find and select deduplication matches.
    # -----------------------------------------------------

    candidates = find_candidate_matches(
        events
    )

    selected_matches = (
        select_one_to_one_matches(
            candidates
        )
    )

    # -----------------------------------------------------
    # Build cases using the same case-construction logic.
    #
    # Import here so we don't accidentally execute its
    # normal main() pipeline.
    # -----------------------------------------------------

    from src.engine.case_builder import (
        build_recovery_cases,
    )

    cases = build_recovery_cases(
        events,
        selected_matches,
    )

    return (
        events,
        candidates,
        selected_matches,
        cases,
    )


def validate_unseen_cases(
    events: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    """
    Validate unseen recovery-case coverage.
    """

    case_event_ids = []

    for event_ids in cases[
        "event_ids"
    ]:

        case_event_ids.extend(
            str(event_ids).split("|")
        )

    event_id_counts = pd.Series(
        case_event_ids
    ).value_counts()

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
        "Some unseen events belong to "
        "multiple recovery cases."
    )

    assert not missing_events, (
        "Some unseen actionable events were "
        "not assigned to a recovery case."
    )

    assert (
        cases["case_id"].is_unique
    ), (
        "Unseen case IDs must be unique."
    )

    assert (
        cases["customer_id"]
        .astype(str)
        .str.startswith("UNSEEN_")
        .all()
    ), (
        "Unseen cases contain non-unseen "
        "customer IDs."
    )

    print(
        "✓ Unseen recovery-case validation passed."
    )


def save_unseen_cases(
    cases: pd.DataFrame,
) -> None:
    """Save unseen recovery cases separately."""

    output_path = Path(
        UNSEEN_OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases.to_csv(
        output_path,
        index=False,
    )


def main() -> None:

    (
        events,
        candidates,
        selected_matches,
        cases,
    ) = build_unseen_recovery_cases()

    validate_unseen_cases(
        events,
        cases,
    )

    save_unseen_cases(
        cases
    )

    print()

    print(
        "================================"
    )
    print(
        "UNSEEN RECOVERY CASES"
    )
    print(
        "================================"
    )

    print(
        f"Actionable events: "
        f"{len(events):,}"
    )

    print(
        f"Candidate matches: "
        f"{len(candidates):,}"
    )

    print(
        f"Selected matches: "
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
        "Revenue at risk:"
    )

    print(
        f"₹{cases['recovery_amount'].sum():,.2f}"
    )

    print()

    print(
        "Sample unseen cases:"
    )

    print(
        cases[
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
        f"Saved to: "
        f"{UNSEEN_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()