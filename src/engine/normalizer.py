from pathlib import Path

import pandas as pd


DATA_DIRECTORY = Path("data/raw")


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load all four revenue surfaces."""

    customers = pd.read_csv(
        DATA_DIRECTORY / "customers.csv"
    )

    payments = pd.read_csv(
        DATA_DIRECTORY / "payments.csv"
    )

    subscriptions = pd.read_csv(
        DATA_DIRECTORY / "subscriptions.csv"
    )

    checkouts = pd.read_csv(
        DATA_DIRECTORY / "checkouts.csv"
    )

    invoices = pd.read_csv(
        DATA_DIRECTORY / "invoices.csv"
    )

    return (
        customers,
        payments,
        subscriptions,
        checkouts,
        invoices,
    )


def normalize_payments(
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize failed payment events."""

    failed = payments[
        payments["status"] == "failed"
    ].copy()

    failed["event_id"] = failed[
        "payment_id"
    ]

    failed["surface"] = "payment"

    failed["event_type"] = "payment_failed"

    failed["event_amount"] = failed[
        "amount"
    ]

    failed["event_status"] = failed[
        "status"
    ]

    failed["event_timestamp"] = pd.to_datetime(
        failed["created_at"]
    )

    return failed[
        [
            "customer_id",
            "event_id",
            "surface",
            "event_type",
            "event_amount",
            "event_status",
            "failure_category",
            "event_timestamp",
        ]
    ]


def normalize_subscriptions(
    subscriptions: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize failed subscription events."""

    failed = subscriptions[
        subscriptions["status"] == "failed"
    ].copy()

    failed["event_id"] = failed[
        "subscription_id"
    ]

    failed["surface"] = "subscription"

    failed["event_type"] = (
        "subscription_failed"
    )

    failed["event_amount"] = failed[
        "amount"
    ]

    failed["event_status"] = failed[
        "status"
    ]

    failed["event_timestamp"] = pd.to_datetime(
        failed["created_at"]
    )

    return failed[
        [
            "customer_id",
            "event_id",
            "surface",
            "event_type",
            "event_amount",
            "event_status",
            "failure_category",
            "event_timestamp",
        ]
    ]


def normalize_checkouts(
    checkouts: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize abandoned checkout events."""

    abandoned = checkouts[
        checkouts["status"] == "abandoned"
    ].copy()

    abandoned["event_id"] = abandoned[
        "checkout_id"
    ]

    abandoned["surface"] = "checkout"

    abandoned["event_type"] = (
        "checkout_abandoned"
    )

    abandoned["event_amount"] = abandoned[
        "amount"
    ]

    abandoned["event_status"] = abandoned[
        "status"
    ]

    abandoned["failure_category"] = None

    abandoned["event_timestamp"] = pd.to_datetime(
        abandoned["created_at"]
    )

    return abandoned[
        [
            "customer_id",
            "event_id",
            "surface",
            "event_type",
            "event_amount",
            "event_status",
            "failure_category",
            "event_timestamp",
        ]
    ]


def normalize_invoices(
    invoices: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize overdue invoice events."""

    overdue = invoices[
        invoices["status"] == "overdue"
    ].copy()

    overdue["event_id"] = overdue[
        "invoice_id"
    ]

    overdue["surface"] = "invoice"

    overdue["event_type"] = (
        "invoice_overdue"
    )

    overdue["event_amount"] = overdue[
        "amount"
    ]

    overdue["event_status"] = overdue[
        "status"
    ]

    overdue["event_timestamp"] = pd.to_datetime(
        overdue["created_at"]
    )

    return overdue[
        [
            "customer_id",
            "event_id",
            "surface",
            "event_type",
            "event_amount",
            "event_status",
            "failure_category",
            "event_timestamp",
        ]
    ]


def build_unified_events() -> pd.DataFrame:
    """
    Load all revenue surfaces and combine their
    actionable revenue-risk events.
    """

    (
        _customers,
        payments,
        subscriptions,
        checkouts,
        invoices,
    ) = load_data()

    normalized_events = [
        normalize_payments(payments),
        normalize_subscriptions(
            subscriptions
        ),
        normalize_checkouts(checkouts),
        normalize_invoices(invoices),
    ]

    unified_events = pd.concat(
        normalized_events,
        ignore_index=True,
    )

    unified_events = unified_events.sort_values(
        [
            "customer_id",
            "event_timestamp",
        ]
    ).reset_index(drop=True)

    return unified_events


def main() -> None:
    """Run the event normalization pipeline."""

    events = build_unified_events()

    print(
        f"Unified events: {len(events):,}"
    )

    print()

    print("Events by surface:")

    print(
        events["surface"].value_counts()
    )

    print()

    print("Events by type:")

    print(
        events["event_type"].value_counts()
    )

    print()

    print("Customers with actionable events:")

    print(
        events["customer_id"].nunique()
    )

    print()

    print(
        events.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()