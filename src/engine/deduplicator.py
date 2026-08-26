import pandas as pd

from src.engine.normalizer import build_unified_events


MAX_TIME_HOURS = 24
MAX_AMOUNT_DIFFERENCE = 0.10
MIN_DEDUP_SCORE = 0.70


def calculate_amount_similarity(
    amount_a: float,
    amount_b: float,
) -> float:
    """Calculate normalized monetary similarity."""

    denominator = max(
        float(amount_a),
        float(amount_b),
    )

    if denominator == 0:
        return 0.0

    difference = abs(
        float(amount_a)
        - float(amount_b)
    )

    return max(
        0.0,
        1.0 - (
            difference / denominator
        ),
    )


def calculate_time_similarity(
    timestamp_a: pd.Timestamp,
    timestamp_b: pd.Timestamp,
) -> float:
    """
    Calculate temporal similarity.

    Events at the same time receive 1.0.
    Events 24+ hours apart receive 0.0.
    """

    hours = abs(
        (
            timestamp_a
            - timestamp_b
        ).total_seconds()
    ) / 3600

    if hours >= MAX_TIME_HOURS:
        return 0.0

    return 1.0 - (
        hours / MAX_TIME_HOURS
    )


def calculate_dedup_score(
    amount_similarity: float,
    time_similarity: float,
) -> float:
    """Combine amount and time similarity."""

    return (
        0.60 * amount_similarity
        + 0.40 * time_similarity
    )


def find_candidate_matches(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find payment ↔ checkout candidates.

    Only same-customer payment failures and checkout
    abandonments are considered.
    """

    payments = events[
        events["event_type"]
        == "payment_failed"
    ].copy()

    checkouts = events[
        events["event_type"]
        == "checkout_abandoned"
    ].copy()

    candidates = []

    for payment in payments.itertuples(
        index=False
    ):

        for checkout in checkouts.itertuples(
            index=False
        ):

            # ---------------------------------------------
            # Same customer is mandatory.
            # ---------------------------------------------

            if (
                payment.customer_id
                != checkout.customer_id
            ):
                continue

            # ---------------------------------------------
            # Amount must be within 10%.
            # ---------------------------------------------

            amount_difference = (
                abs(
                    payment.event_amount
                    - checkout.event_amount
                )
                / max(
                    payment.event_amount,
                    checkout.event_amount,
                )
            )

            if (
                amount_difference
                > MAX_AMOUNT_DIFFERENCE
            ):
                continue

            # ---------------------------------------------
            # Time must be within 24 hours.
            # ---------------------------------------------

            time_difference_hours = (
                abs(
                    (
                        payment.event_timestamp
                        - checkout.event_timestamp
                    ).total_seconds()
                )
                / 3600
            )

            if (
                time_difference_hours
                > MAX_TIME_HOURS
            ):
                continue

            amount_similarity = (
                calculate_amount_similarity(
                    payment.event_amount,
                    checkout.event_amount,
                )
            )

            time_similarity = (
                calculate_time_similarity(
                    payment.event_timestamp,
                    checkout.event_timestamp,
                )
            )

            dedup_score = (
                calculate_dedup_score(
                    amount_similarity,
                    time_similarity,
                )
            )

            if (
                dedup_score
                < MIN_DEDUP_SCORE
            ):
                continue

            candidates.append(
                {
                    "payment_event_id": (
                        payment.event_id
                    ),
                    "checkout_event_id": (
                        checkout.event_id
                    ),
                    "customer_id": (
                        payment.customer_id
                    ),
                    "payment_amount": (
                        payment.event_amount
                    ),
                    "checkout_amount": (
                        checkout.event_amount
                    ),
                    "amount_similarity": (
                        amount_similarity
                    ),
                    "time_similarity": (
                        time_similarity
                    ),
                    "time_difference_hours": (
                        time_difference_hours
                    ),
                    "dedup_score": (
                        dedup_score
                    ),
                }
            )

    return pd.DataFrame(
        candidates
    )


def select_one_to_one_matches(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the strongest non-conflicting matches.

    Each payment and checkout can belong to at most
    one deduplicated recovery opportunity.
    """

    if candidates.empty:
        return candidates.copy()

    ranked = candidates.sort_values(
        "dedup_score",
        ascending=False,
    ).reset_index(drop=True)

    used_payments = set()
    used_checkouts = set()

    selected = []

    for candidate in ranked.itertuples(
        index=False
    ):

        if (
            candidate.payment_event_id
            in used_payments
        ):
            continue

        if (
            candidate.checkout_event_id
            in used_checkouts
        ):
            continue

        used_payments.add(
            candidate.payment_event_id
        )

        used_checkouts.add(
            candidate.checkout_event_id
        )

        selected.append(
            candidate._asdict()
        )

    return pd.DataFrame(
        selected
    )


def build_deduplication_result(
    events: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build the final selected matches and unmatched events.
    """

    candidates = find_candidate_matches(
        events
    )

    selected_matches = (
        select_one_to_one_matches(
            candidates
        )
    )

    matched_event_ids = set()

    if not selected_matches.empty:

        matched_event_ids.update(
            selected_matches[
                "payment_event_id"
            ]
        )

        matched_event_ids.update(
            selected_matches[
                "checkout_event_id"
            ]
        )

    unmatched_events = events[
        ~events["event_id"].isin(
            matched_event_ids
        )
    ].copy()

    return (
        selected_matches,
        unmatched_events,
    )


def main() -> None:
    """Run the deduplication pipeline."""

    events = build_unified_events()

    candidates = find_candidate_matches(
        events
    )

    selected_matches = (
        select_one_to_one_matches(
            candidates
        )
    )

    (
        _selected,
        unmatched_events,
    ) = build_deduplication_result(
        events
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
        f"Unmatched events: "
        f"{len(unmatched_events):,}"
    )

    print()

    if not selected_matches.empty:

        print(
            "Selected deduplication matches:"
        )

        print(
            selected_matches[
                [
                    "payment_event_id",
                    "checkout_event_id",
                    "customer_id",
                    "payment_amount",
                    "checkout_amount",
                    "amount_similarity",
                    "time_difference_hours",
                    "dedup_score",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

        print()

        print(
            "Average dedup score:",
            round(
                selected_matches[
                    "dedup_score"
                ].mean(),
                4,
            ),
        )

    else:

        print(
            "No matches selected."
        )


if __name__ == "__main__":
    main()