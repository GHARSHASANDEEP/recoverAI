from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker


from src.data.config import (
    RANDOM_SEED,
    NUM_CUSTOMERS,
    NUM_PAYMENTS,
    CUSTOMER_OPT_OUT_RATE,
    NEW_CUSTOMER_RATE,
    HIGH_VALUE_CUSTOMER_RATE,
    MIN_PAYMENT_AMOUNT,
    MAX_PAYMENT_AMOUNT,
    SUBSCRIPTION_ACTIVE_RATE,
    SUBSCRIPTION_FAILED_RATE,
    SUBSCRIPTION_CANCELLED_RATE,
    MIN_SUBSCRIPTION_AMOUNT,
    MAX_SUBSCRIPTION_AMOUNT,
    CHECKOUT_COMPLETED_RATE,
    CHECKOUT_ABANDONED_RATE,
    CHECKOUT_EXPIRED_RATE,
    CHECKOUT_STAGES,
    MIN_CHECKOUT_AMOUNT,
    MAX_CHECKOUT_AMOUNT,
    INVOICE_PAID_RATE,
    INVOICE_OVERDUE_RATE,
    INVOICE_CANCELLED_RATE,
    MIN_INVOICE_AMOUNT,
    MAX_INVOICE_AMOUNT,
)

from src.data.taxonomy import RAW_FAILURE_CODE_MAP


# ---------------------------------------------------------
# Reproducible random generators
# ---------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

fake = Faker("en_IN")
fake.seed_instance(RANDOM_SEED)


# ---------------------------------------------------------
# Payment configuration
# ---------------------------------------------------------

PAYMENT_METHODS = [
    "upi",
    "card",
    "netbanking",
]


FAILURE_CATEGORY_WEIGHTS = {
    "temporary_bank_failure": 0.20,
    "timeout": 0.12,
    "insufficient_funds": 0.22,
    "authentication_failed": 0.10,
    "expired_instrument": 0.08,
    "blocked_instrument": 0.05,
    "risk_decline": 0.08,
    "limit_exceeded": 0.06,
    "unknown_failure": 0.09,
}


# Map normalized category → possible raw codes.
CATEGORY_TO_RAW_CODES = {}

for raw_code, category in RAW_FAILURE_CODE_MAP.items():
    CATEGORY_TO_RAW_CODES.setdefault(
        category,
        [],
    ).append(raw_code)


# ---------------------------------------------------------
# Customer generation
# ---------------------------------------------------------

def generate_customers() -> pd.DataFrame:
    """
    Generate the initial customer population.

    Payment-history counters are initially zero.
    They will be populated from actual payment events later.
    """

    customers = []

    for i in range(NUM_CUSTOMERS):

        customer_id = f"CUST_{i + 1:05d}"

        segment_probability = rng.random()

        if segment_probability < NEW_CUSTOMER_RATE:
            segment = "new"

        elif segment_probability < (
            NEW_CUSTOMER_RATE
            + HIGH_VALUE_CUSTOMER_RATE
        ):
            segment = "high_value"

        else:
            segment = "regular"

        preferred_payment_method = str(
            rng.choice(
                PAYMENT_METHODS,
                p=[0.55, 0.30, 0.15],
            )
        )

        communication_opt_in = bool(
            rng.random() >= CUSTOMER_OPT_OUT_RATE
        )

        if segment == "high_value":

            customer_lifetime_value = round(
                float(rng.uniform(100000, 500000)),
                2,
            )

        elif segment == "regular":

            customer_lifetime_value = round(
                float(rng.uniform(20000, 100000)),
                2,
            )

        else:

            customer_lifetime_value = round(
                float(rng.uniform(1000, 20000)),
                2,
            )

        customers.append(
            {
                "customer_id": customer_id,
                "customer_name": fake.name(),
                "customer_segment": segment,
                "preferred_payment_method": (
                    preferred_payment_method
                ),
                "communication_opt_in": (
                    communication_opt_in
                ),

                # These are deliberately zero here.
                # They will be calculated from actual
                # payment events.
                "successful_payment_count": 0,
                "failed_payment_count": 0,
                "total_payment_attempts": 0,

                "customer_lifetime_value": (
                    customer_lifetime_value
                ),
            }
        )

    return pd.DataFrame(customers)


# ---------------------------------------------------------
# Payment helpers
# ---------------------------------------------------------

def generate_payment_amount(
    customer_segment: str,
) -> float:
    """Generate an amount based on customer segment."""

    if customer_segment == "high_value":

        low = 5000
        high = MAX_PAYMENT_AMOUNT

    elif customer_segment == "regular":

        low = 1000
        high = 50000

    else:

        low = MIN_PAYMENT_AMOUNT
        high = 20000

    return round(
        float(rng.uniform(low, high)),
        2,
    )


def choose_failure_category() -> str:
    """Choose a normalized synthetic failure category."""

    categories = list(
        FAILURE_CATEGORY_WEIGHTS.keys()
    )

    weights = list(
        FAILURE_CATEGORY_WEIGHTS.values()
    )

    return str(
        rng.choice(
            categories,
            p=weights,
        )
    )


def choose_raw_failure_code(
    failure_category: str,
) -> str:
    """Choose a representative raw failure code."""

    codes = CATEGORY_TO_RAW_CODES.get(
        failure_category,
        ["UNKNOWN_ERROR"],
    )

    return str(rng.choice(codes))


def generate_payment_timestamp() -> pd.Timestamp:
    """Generate a payment timestamp from the last 90 days."""

    days_ago = int(
        rng.integers(0, 91)
    )

    minutes_ago = int(
        rng.integers(0, 24 * 60)
    )

    return (
        pd.Timestamp.now()
        - pd.Timedelta(days=days_ago)
        - pd.Timedelta(minutes=minutes_ago)
    )


# ---------------------------------------------------------
# Payment generation
# ---------------------------------------------------------

def generate_payments(
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate payment events.

    Payment events are generated per customer so that
    customer history can later be calculated directly
    from the actual payment records.
    """

    payments = []

    customer_records = customers.to_dict(
        orient="records"
    )

    # Track historical payment opportunities per customer.
    customer_attempt_counters = {
        customer["customer_id"]: 0
        for customer in customer_records
    }

    for i in range(NUM_PAYMENTS):

        customer = customer_records[
            rng.integers(
                0,
                len(customer_records),
            )
        ]

        customer_id = customer["customer_id"]

        customer_segment = customer[
            "customer_segment"
        ]

        # -------------------------------------------------
        # Decide whether this event belongs to an existing
        # payment opportunity or is a new payment.
        # -------------------------------------------------

        existing_attempt_probability = 0.12

        if (
            customer_attempt_counters[customer_id] > 0
            and rng.random()
            < existing_attempt_probability
        ):
            previous_attempt = (
                customer_attempt_counters[
                    customer_id
                ]
            )

            attempt_number = min(
                previous_attempt + 1,
                3,
            )

        else:

            attempt_number = 1

        customer_attempt_counters[
            customer_id
        ] = attempt_number

        # -------------------------------------------------
        # Payment method
        # -------------------------------------------------

        payment_method = str(
            rng.choice(
                PAYMENT_METHODS,
                p=[0.55, 0.30, 0.15],
            )
        )

        # -------------------------------------------------
        # Amount
        # -------------------------------------------------

        amount = generate_payment_amount(
            customer_segment
        )

        # -------------------------------------------------
        # Initial payment success probability
        # -------------------------------------------------

        if customer_segment == "high_value":

            success_probability = 0.88

        elif customer_segment == "regular":

            success_probability = 0.82

        else:

            success_probability = 0.70

        # Repeated attempts are slightly more likely
        # to fail in the historical dataset.
        if attempt_number > 1:
            success_probability -= 0.08

        is_successful = (
            rng.random()
            < success_probability
        )

        payment_id = (
            f"PAY_{i + 1:06d}"
        )

        timestamp = (
            generate_payment_timestamp()
        )

        # -------------------------------------------------
        # Successful payment
        # -------------------------------------------------

        if is_successful:

            payments.append(
                {
                    "payment_id": payment_id,
                    "customer_id": customer_id,
                    "amount": amount,
                    "payment_method": payment_method,
                    "status": "successful",
                    "raw_failure_code": None,
                    "failure_category": None,
                    "attempt_number": attempt_number,
                    "created_at": timestamp,
                }
            )

            continue

        # -------------------------------------------------
        # Failed payment
        # -------------------------------------------------

        failure_category = (
            choose_failure_category()
        )

        raw_failure_code = (
            choose_raw_failure_code(
                failure_category
            )
        )

        # -------------------------------------------------
        # Diagnostic noise
        # -------------------------------------------------

        noise_probability = rng.random()

        if noise_probability < 0.03:

            # Missing diagnostic code.
            raw_failure_code = None

        elif noise_probability < 0.08:

            # Ambiguous diagnostic code.
            raw_failure_code = (
                "UNSPECIFIED_DECLINE"
            )

        payments.append(
            {
                "payment_id": payment_id,
                "customer_id": customer_id,
                "amount": amount,
                "payment_method": payment_method,
                "status": "failed",
                "raw_failure_code": raw_failure_code,
                "failure_category": failure_category,
                "attempt_number": attempt_number,
                "created_at": timestamp,
            }
        )

    return pd.DataFrame(payments)


# ---------------------------------------------------------
# Recalculate customer payment history
# ---------------------------------------------------------

def update_customer_payment_history(
    customers: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate customer payment-history fields directly
    from the generated payment events.
    """

    payment_counts = (
        payments
        .assign(
            successful=(
                payments["status"]
                == "successful"
            ),
            failed=(
                payments["status"]
                == "failed"
            ),
        )
        .groupby("customer_id")
        .agg(
            successful_payment_count=(
                "successful",
                "sum",
            ),
            failed_payment_count=(
                "failed",
                "sum",
            ),
            total_payment_attempts=(
                "payment_id",
                "count",
            ),
        )
        .reset_index()
    )

    customers = customers.drop(
        columns=[
            "successful_payment_count",
            "failed_payment_count",
            "total_payment_attempts",
        ]
    )

    customers = customers.merge(
        payment_counts,
        on="customer_id",
        how="left",
    )

    customers[
        [
            "successful_payment_count",
            "failed_payment_count",
            "total_payment_attempts",
        ]
    ] = (
        customers[
            [
                "successful_payment_count",
                "failed_payment_count",
                "total_payment_attempts",
            ]
        ]
        .fillna(0)
        .astype(int)
    )

    return customers


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_customer_payment_consistency(
    customers: pd.DataFrame,
    payments: pd.DataFrame,
) -> None:
    """
    Confirm that customer-level payment counts exactly
    match the payment event table.
    """

    actual_counts = (
        payments
        .groupby("customer_id")
        .agg(
            successful_payment_count=(
                "status",
                lambda values: (
                    values == "successful"
                ).sum(),
            ),
            failed_payment_count=(
                "status",
                lambda values: (
                    values == "failed"
                ).sum(),
            ),
            total_payment_attempts=(
                "payment_id",
                "count",
            ),
        )
        .reset_index()
    )

    merged = customers.merge(
        actual_counts,
        on="customer_id",
        how="left",
        suffixes=(
            "_customer",
            "_actual",
        ),
    )

    for column in [
        "successful_payment_count",
        "failed_payment_count",
        "total_payment_attempts",
    ]:

        actual_column = (
            f"{column}_actual"
        )

        merged[actual_column] = (
            merged[actual_column]
            .fillna(0)
            .astype(int)
        )

        assert (
            merged[f"{column}_customer"]
            == merged[actual_column]
        ).all(), (
            f"Consistency check failed for {column}"
        )

    print(
        "✓ Customer/payment consistency check passed."
    )

def generate_subscription_amount(
    customer_segment: str,
) -> float:
    """Generate a subscription amount based on customer segment."""

    if customer_segment == "high_value":
        low = 5000
        high = MAX_SUBSCRIPTION_AMOUNT

    elif customer_segment == "regular":
        low = 1000
        high = 15000

    else:
        low = MIN_SUBSCRIPTION_AMOUNT
        high = 8000

    return round(
        float(rng.uniform(low, high)),
        2,
    )


def generate_subscriptions(
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate subscription events using existing customers.

    Some customers will also have payment events, creating
    cross-surface revenue-risk cases.
    """

    subscriptions = []

    customer_records = customers.to_dict(
        orient="records"
    )

    statuses = [
        "active",
        "failed",
        "cancelled",
    ]

    status_weights = [
        SUBSCRIPTION_ACTIVE_RATE,
        SUBSCRIPTION_FAILED_RATE,
        SUBSCRIPTION_CANCELLED_RATE,
    ]

    for i in range(len(customers)):

        customer = customer_records[i]

        customer_id = customer["customer_id"]

        customer_segment = customer[
            "customer_segment"
        ]

        subscription_id = (
            f"SUB_{i + 1:06d}"
        )

        amount = generate_subscription_amount(
            customer_segment
        )

        status = str(
            rng.choice(
                statuses,
                p=status_weights,
            )
        )

        created_at = (
            pd.Timestamp.now()
            - pd.Timedelta(
                days=int(rng.integers(0, 181))
            )
        )

        failure_code = None
        failure_category = None
        retry_count = 0
        next_retry_at = None

        if status == "failed":

            failure_category = (
                choose_failure_category()
            )

            failure_code = (
                choose_raw_failure_code(
                    failure_category
                )
            )

            # Introduce diagnostic noise.
            noise_probability = rng.random()

            if noise_probability < 0.03:
                failure_code = None

            elif noise_probability < 0.08:
                failure_code = (
                    "UNSPECIFIED_DECLINE"
                )

            retry_count = int(
                rng.integers(0, 3)
            )

            next_retry_at = (
                created_at
                + pd.Timedelta(
                    hours=int(
                        rng.integers(1, 48)
                    )
                )
            )

        subscriptions.append(
            {
                "subscription_id": subscription_id,
                "customer_id": customer_id,
                "amount": amount,
                "status": status,
                "failure_code": failure_code,
                "failure_category": failure_category,
                "retry_count": retry_count,
                "created_at": created_at,
                "next_retry_at": next_retry_at,
            }
        )

    return pd.DataFrame(subscriptions)

def generate_checkout_amount(
    customer_segment: str,
) -> float:
    """Generate checkout amount based on customer segment."""

    if customer_segment == "high_value":
        low = 5000
        high = MAX_CHECKOUT_AMOUNT

    elif customer_segment == "regular":
        low = 1000
        high = 50000

    else:
        low = MIN_CHECKOUT_AMOUNT
        high = 20000

    return round(
        float(rng.uniform(low, high)),
        2,
    )


def generate_checkouts(
    customers: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate checkout events using existing customers.

    A controlled portion of abandoned checkouts is linked
    to failed payment opportunities. The relationship is
    intentionally NOT stored in the checkout output.

    RecoverAI must discover the relationship using observable
    fields such as customer_id, amount, and timestamp.
    """

    checkouts = []

    customer_records = customers.to_dict(
        orient="records"
    )

    failed_payments = payments[
        payments["status"] == "failed"
    ].copy()

    failed_payments["created_at"] = pd.to_datetime(
        failed_payments["created_at"]
    )

    # Approximately 18% of abandoned checkouts will be
    # related to existing failed payments.
    LINKED_ABANDONMENT_RATE = 0.18

    for i in range(3500):

        # -------------------------------------------------
        # Decide whether this checkout is a linked event.
        # -------------------------------------------------

        create_linked_event = (
            len(failed_payments) > 0
            and rng.random()
            < LINKED_ABANDONMENT_RATE
        )

        if create_linked_event:

            # Select an existing failed payment.
            payment = failed_payments.iloc[
                rng.integers(
                    0,
                    len(failed_payments),
                )
            ]

            customer_id = payment[
                "customer_id"
            ]

            customer = customers[
                customers["customer_id"]
                == customer_id
            ].iloc[0]

            customer_segment = customer[
                "customer_segment"
            ]

            # Linked checkout amount stays close to
            # the failed payment amount, but isn't identical.
            payment_amount = float(
                payment["amount"]
            )

            amount_variation = rng.uniform(
                -0.06,
                0.06,
            )

            amount = round(
                payment_amount
                * (1 + amount_variation),
                2,
            )

            # Create the checkout near the failed payment.
            payment_timestamp = pd.Timestamp(
                payment["created_at"]
            )

            time_offset_minutes = int(
                rng.integers(
                    -180,
                    180,
                )
            )

            created_at = (
                payment_timestamp
                + pd.Timedelta(
                    minutes=time_offset_minutes
                )
            )

            # A linked event is intentionally abandoned.
            status = "abandoned"

            # Payment-related abandoned checkouts should
            # usually be at the payment stage.
            stage = str(
                rng.choice(
                    [
                        "payment_page",
                        "payment_attempt",
                    ]
                )
            )

        else:

            # -------------------------------------------------
            # Independent checkout event.
            # -------------------------------------------------

            customer = customer_records[
                rng.integers(
                    0,
                    len(customer_records),
                )
            ]

            customer_id = customer[
                "customer_id"
            ]

            customer_segment = customer[
                "customer_segment"
            ]

            statuses = [
                "completed",
                "abandoned",
                "expired",
            ]

            status_weights = [
                CHECKOUT_COMPLETED_RATE,
                CHECKOUT_ABANDONED_RATE,
                CHECKOUT_EXPIRED_RATE,
            ]

            status = str(
                rng.choice(
                    statuses,
                    p=status_weights,
                )
            )

            stage = str(
                rng.choice(
                    CHECKOUT_STAGES
                )
            )

            amount = generate_checkout_amount(
                customer_segment
            )

            created_at = (
                pd.Timestamp.now()
                - pd.Timedelta(
                    days=int(
                        rng.integers(
                            0,
                            91,
                        )
                    )
                )
                - pd.Timedelta(
                    minutes=int(
                        rng.integers(
                            0,
                            24 * 60,
                        )
                    )
                )
            )

        # -------------------------------------------------
        # Abandonment timestamp
        # -------------------------------------------------

        abandoned_at = None

        if status in {
            "abandoned",
            "expired",
        }:

            abandoned_at = (
                created_at
                + pd.Timedelta(
                    minutes=int(
                        rng.integers(
                            2,
                            120,
                        )
                    )
                )
            )

        checkout_id = (
            f"CHK_{i + 1:06d}"
        )

        checkouts.append(
            {
                "checkout_id": checkout_id,
                "customer_id": customer_id,
                "amount": amount,
                "status": status,
                "checkout_stage": stage,
                "created_at": created_at,
                "abandoned_at": abandoned_at,
            }
        )

    return pd.DataFrame(checkouts)
def generate_invoice_amount(
    customer_segment: str,
) -> float:
    """Generate an invoice amount based on customer segment."""

    if customer_segment == "high_value":
        low = 10000
        high = MAX_INVOICE_AMOUNT

    elif customer_segment == "regular":
        low = 5000
        high = 75000

    else:
        low = MIN_INVOICE_AMOUNT
        high = 25000

    return round(
        float(rng.uniform(low, high)),
        2,
    )


def generate_invoices(
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate invoice collection events using existing customers.
    """

    invoices = []

    customer_records = customers.to_dict(
        orient="records"
    )

    statuses = [
        "paid",
        "overdue",
        "cancelled",
    ]

    status_weights = [
        INVOICE_PAID_RATE,
        INVOICE_OVERDUE_RATE,
        INVOICE_CANCELLED_RATE,
    ]

    for i in range(len(customers)):

        customer = customer_records[i]

        customer_id = customer["customer_id"]

        customer_segment = customer[
            "customer_segment"
        ]

        invoice_id = (
            f"INV_{i + 1:06d}"
        )

        amount = generate_invoice_amount(
            customer_segment
        )

        status = str(
            rng.choice(
                statuses,
                p=status_weights,
            )
        )

        created_at = (
            pd.Timestamp.now()
            - pd.Timedelta(
                days=int(
                    rng.integers(0, 181)
                )
            )
        )

        due_date = (
            created_at
            + pd.Timedelta(
                days=int(
                    rng.integers(7, 31)
                )
            )
        )

        paid_at = None
        days_overdue = 0
        failure_code = None
        failure_category = None

        if status == "paid":

            paid_at = (
                due_date
                - pd.Timedelta(
                    days=int(
                        rng.integers(0, 7)
                    )
                )
            )

        elif status == "overdue":

            # Generate invoices that are genuinely past due.
            days_overdue = int(
                rng.integers(1, 91)
            )

            failure_category = (
                choose_failure_category()
            )

            failure_code = (
                choose_raw_failure_code(
                    failure_category
                )
            )

            # Some invoices have incomplete diagnostics.
            noise_probability = rng.random()

            if noise_probability < 0.03:

                failure_code = None

            elif noise_probability < 0.08:

                failure_code = (
                    "UNSPECIFIED_DECLINE"
                )

        invoices.append(
            {
                "invoice_id": invoice_id,
                "customer_id": customer_id,
                "amount": amount,
                "status": status,
                "due_date": due_date,
                "paid_at": paid_at,
                "days_overdue": days_overdue,
                "failure_code": failure_code,
                "failure_category": failure_category,
                "created_at": created_at,
            }
        )

    return pd.DataFrame(invoices)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    """Generate and validate the initial datasets."""

    output_directory = Path(
        "data/raw"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    customers = generate_customers()

    payments = generate_payments(
        customers
    )

    subscriptions = generate_subscriptions(
        customers
    )

    checkouts = generate_checkouts(
        customers,
        payments,
    )

    invoices = generate_invoices(
        customers
    )

    # Recalculate customer history from
    # those actual payment events.
    customers = (
        update_customer_payment_history(
            customers,
            payments,
        )
    )

    # Validate the relationship.
    validate_customer_payment_consistency(
        customers,
        payments,
    )

    # Save.
    customers_path = (
        output_directory
        / "customers.csv"
    )

    payments_path = (
        output_directory
        / "payments.csv"
    )
    subscriptions_path = (
        output_directory
        / "subscriptions.csv"
    )

    checkouts_path = (
        output_directory
        / "checkouts.csv"
    )

    invoices_path = (
        output_directory
        / "invoices.csv"
    )

    customers.to_csv(
        customers_path,
        index=False,
    )

    payments.to_csv(
        payments_path,
        index=False,
    )
    subscriptions.to_csv(
        subscriptions_path,
        index=False,
    )

    checkouts.to_csv(
        checkouts_path,
        index=False,
    )

    invoices.to_csv(
        invoices_path,
        index=False,
    )

    print()
    print("Invoice status:")
    print(
        invoices["status"]
        .value_counts()
    )

    print()
    print("Overdue invoice statistics:")
    print(
        invoices[
            invoices["status"] == "overdue"
        ][
            [
                "amount",
                "days_overdue",
            ]
        ].describe()
    )

    print()
    print("Saved:")
    print(customers_path)
    print(payments_path)
    print(subscriptions_path)
    print(checkouts_path)
    print(invoices_path)

    print()
    print(
        f"Generated {len(customers):,} customers."
    )

    print(
        f"Generated {len(payments):,} payments."
    )

    print()
    print("Payment status:")
    print(
        payments["status"]
        .value_counts()
    )

    print()
    print("Failure categories:")
    print(
        payments[
            payments["status"] == "failed"
        ]["failure_category"]
        .value_counts()
    )

    print()
    print("Saved:")
    print(customers_path)
    print(payments_path)

    print()
    print("Subscription status:")
    print(
        subscriptions["status"]
        .value_counts()
    )

    print()
    print("Subscription failure categories:")
    print(
        subscriptions[
            subscriptions["status"] == "failed"
        ]["failure_category"]
        .value_counts()
    )

    print()
    print("Saved:")
    print(customers_path)
    print(payments_path)
    print(subscriptions_path)

    print()
    print("Checkout status:")
    print(
        checkouts["status"]
        .value_counts()
    )

    print()
    print("Checkout stages:")
    print(
        checkouts["checkout_stage"]
        .value_counts()
    )

    print()
    print("Saved:")
    print(customers_path)
    print(payments_path)
    print(subscriptions_path)
    print(checkouts_path)


if __name__ == "__main__":
    main()