from pathlib import Path
import json

import numpy as np
import pandas as pd
from faker import Faker

from src.data import generator as base_generator


# ---------------------------------------------------------
# UNSEEN DATASET CONFIGURATION
# ---------------------------------------------------------

UNSEEN_SEED = 20260905

OUTPUT_DIRECTORY = Path(
    "data/unseen/raw"
)


# ---------------------------------------------------------
# ID ISOLATION
# ---------------------------------------------------------

def prefix_ids(
    customers: pd.DataFrame,
    payments: pd.DataFrame,
    subscriptions: pd.DataFrame,
    checkouts: pd.DataFrame,
    invoices: pd.DataFrame,
):
    """
    Give the unseen population its own identifiers.

    This makes it obvious that these are separate
    synthetic entities from the original dataset.
    """

    customer_map = {
        old_id: f"UNSEEN_{old_id}"
        for old_id in customers["customer_id"]
    }

    customers = customers.copy()
    payments = payments.copy()
    subscriptions = subscriptions.copy()
    checkouts = checkouts.copy()
    invoices = invoices.copy()

    # Customer IDs
    for df in [
        customers,
        payments,
        subscriptions,
        checkouts,
        invoices,
    ]:
        df["customer_id"] = (
            df["customer_id"]
            .map(customer_map)
        )

    # Event IDs
    payments["payment_id"] = (
        "UNSEEN_"
        + payments["payment_id"].astype(str)
    )

    subscriptions["subscription_id"] = (
        "UNSEEN_"
        + subscriptions["subscription_id"].astype(str)
    )

    checkouts["checkout_id"] = (
        "UNSEEN_"
        + checkouts["checkout_id"].astype(str)
    )

    invoices["invoice_id"] = (
        "UNSEEN_"
        + invoices["invoice_id"].astype(str)
    )

    return (
        customers,
        payments,
        subscriptions,
        checkouts,
        invoices,
    )


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_unseen_dataset(
    customers: pd.DataFrame,
    payments: pd.DataFrame,
    subscriptions: pd.DataFrame,
    checkouts: pd.DataFrame,
    invoices: pd.DataFrame,
) -> None:
    """Validate the fresh unseen population."""

    assert len(customers) > 0
    assert len(payments) > 0
    assert len(subscriptions) > 0
    assert len(checkouts) > 0
    assert len(invoices) > 0

    assert customers[
        "customer_id"
    ].str.startswith("UNSEEN_").all()

    assert payments[
        "payment_id"
    ].str.startswith("UNSEEN_").all()

    assert subscriptions[
        "subscription_id"
    ].str.startswith("UNSEEN_").all()

    assert checkouts[
        "checkout_id"
    ].str.startswith("UNSEEN_").all()

    assert invoices[
        "invoice_id"
    ].str.startswith("UNSEEN_").all()

    assert (
        customers["customer_id"]
        .is_unique
    )

    assert (
        payments["payment_id"]
        .is_unique
    )

    assert (
        subscriptions["subscription_id"]
        .is_unique
    )

    assert (
        checkouts["checkout_id"]
        .is_unique
    )

    assert (
        invoices["invoice_id"]
        .is_unique
    )

    base_generator.validate_customer_payment_consistency(
        customers,
        payments,
    )

    print(
        "✓ Unseen dataset validation passed."
    )


# ---------------------------------------------------------
# GENERATION
# ---------------------------------------------------------

def generate_unseen_dataset():
    """
    Generate a fresh synthetic population.

    Important:
    - Existing data/raw files are never modified.
    - Existing model is never retrained here.
    - A different random seed is used.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Use a fresh random state.
    #
    # The original generator uses module-level rng/fake.
    # We replace those only inside this process.
    # -----------------------------------------------------

    base_generator.rng = (
        np.random.default_rng(
            UNSEEN_SEED
        )
    )

    base_generator.fake = Faker(
        "en_IN"
    )

    base_generator.fake.seed_instance(
        UNSEEN_SEED
    )

    print(
        f"Generating unseen dataset "
        f"with seed {UNSEEN_SEED}..."
    )

    # -----------------------------------------------------
    # Generate the complete population using the same
    # production data-generation logic.
    # -----------------------------------------------------

    customers = (
        base_generator.generate_customers()
    )

    payments = (
        base_generator.generate_payments(
            customers
        )
    )

    subscriptions = (
        base_generator.generate_subscriptions(
            customers
        )
    )

    checkouts = (
        base_generator.generate_checkouts(
            customers,
            payments,
        )
    )

    invoices = (
        base_generator.generate_invoices(
            customers
        )
    )

    customers = (
        base_generator.update_customer_payment_history(
            customers,
            payments,
        )
    )

    # -----------------------------------------------------
    # Validate before saving.
    # -----------------------------------------------------

    base_generator.validate_customer_payment_consistency(
        customers,
        payments,
    )

    # -----------------------------------------------------
    # Isolate identifiers from the original population.
    # -----------------------------------------------------

    (
        customers,
        payments,
        subscriptions,
        checkouts,
        invoices,
    ) = prefix_ids(
        customers,
        payments,
        subscriptions,
        checkouts,
        invoices,
    )

    validate_unseen_dataset(
        customers,
        payments,
        subscriptions,
        checkouts,
        invoices,
    )

    # -----------------------------------------------------
    # Save fresh population.
    # -----------------------------------------------------

    paths = {
        "customers": (
            OUTPUT_DIRECTORY
            / "customers.csv"
        ),
        "payments": (
            OUTPUT_DIRECTORY
            / "payments.csv"
        ),
        "subscriptions": (
            OUTPUT_DIRECTORY
            / "subscriptions.csv"
        ),
        "checkouts": (
            OUTPUT_DIRECTORY
            / "checkouts.csv"
        ),
        "invoices": (
            OUTPUT_DIRECTORY
            / "invoices.csv"
        ),
    }

    customers.to_csv(
        paths["customers"],
        index=False,
    )

    payments.to_csv(
        paths["payments"],
        index=False,
    )

    subscriptions.to_csv(
        paths["subscriptions"],
        index=False,
    )

    checkouts.to_csv(
        paths["checkouts"],
        index=False,
    )

    invoices.to_csv(
        paths["invoices"],
        index=False,
    )

    # -----------------------------------------------------
    # Save metadata proving how the dataset was created.
    # -----------------------------------------------------

    manifest = {
        "dataset_type": "unseen_evaluation",
        "seed": UNSEEN_SEED,
        "model_retrained": False,
        "source_generator": (
            "src.data.generator"
        ),
        "customers": len(customers),
        "payments": len(payments),
        "subscriptions": len(
            subscriptions
        ),
        "checkouts": len(checkouts),
        "invoices": len(invoices),
        "generated_at": (
            base_generator.GENERATION_TIMESTAMP.isoformat()
        ),
    }

    with open(
        OUTPUT_DIRECTORY
        / "manifest.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )

    print()
    print(
        "================================"
    )
    print(
        "UNSEEN DATASET GENERATED"
    )
    print(
        "================================"
    )

    print(
        f"Seed: {UNSEEN_SEED}"
    )

    print(
        f"Customers: {len(customers):,}"
    )

    print(
        f"Payments: {len(payments):,}"
    )

    print(
        f"Subscriptions: "
        f"{len(subscriptions):,}"
    )

    print(
        f"Checkouts: {len(checkouts):,}"
    )

    print(
        f"Invoices: {len(invoices):,}"
    )

    print()

    print(
        "Model retrained: NO"
    )

    print()

    print(
        f"Saved to: "
        f"{OUTPUT_DIRECTORY}"
    )


def main():
    generate_unseen_dataset()


if __name__ == "__main__":
    main()