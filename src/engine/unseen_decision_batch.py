from pathlib import Path

import pandas as pd

from src.engine.decision_engine import evaluate_guardrails
from src.engine.recovery_policy import (
    get_initial_action,
    get_permitted_actions,
    get_recovery_sequence,
)


BASE_DIR = Path(__file__).resolve().parents[2]

ERV_PATH = (
    BASE_DIR
    / "data"
    / "unseen"
    / "processed"
    / "erv_scores.csv"
)

CASES_PATH = (
    BASE_DIR
    / "data"
    / "unseen"
    / "processed"
    / "recovery_cases.csv"
)

CUSTOMERS_PATH = (
    BASE_DIR
    / "data"
    / "unseen"
    / "raw"
    / "customers.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "unseen"
    / "processed"
    / "decisions.csv"
)


def normalize_failure_category(value):
    """Normalize failure category into policy vocabulary."""

    if pd.isna(value):
        return "unknown_failure"

    value = str(value).strip()

    if value.startswith("[") and value.endswith("]"):
        value = value.strip("[]").strip("'\" ")

    return value or "unknown_failure"


def main():

    print("Loading unseen recovery cases...")

    erv_df = pd.read_csv(ERV_PATH)
    cases_df = pd.read_csv(CASES_PATH)
    customers_df = pd.read_csv(CUSTOMERS_PATH)

    print(f"ERV rows loaded: {len(erv_df):,}")
    print(f"Cases loaded: {len(cases_df):,}")

    # ---------------------------------------------------------
    # Build complete case context
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
        column
        for column in customer_columns
        if column in customers_df.columns
    ]

    context_df = cases_df.merge(
        customers_df[customer_columns],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    if context_df["customer_segment"].isna().any():
        raise ValueError(
            "Unseen customer context is incomplete. "
            "Every unseen recovery case must match an unseen customer."
        )

    context_df["failure_category"] = (
        context_df["failure_categories"]
        .apply(normalize_failure_category)
    )

    context_df["communication_opt_in"] = (
        context_df["communication_opt_in"]
        .fillna(False)
        .astype(bool)
    )

    context_df["recovery_amount"] = pd.to_numeric(
        context_df["recovery_amount"],
        errors="coerce",
    ).fillna(0.0)

    # ---------------------------------------------------------
    # Policy-first decision
    # ---------------------------------------------------------

    decisions = []

    for _, case_row in context_df.iterrows():

        case_id = case_row["case_id"]

        # Get all action evaluations for this case.
        case_erv = erv_df[
            erv_df["case_id"].astype(str)
            == str(case_id)
        ].copy()

        if case_erv.empty:
            continue

        failure_category = (
            case_row["failure_category"]
        )

        # -----------------------------------------------------
        # 1. POLICY DEFINES THE SAFE ACTION SET
        # -----------------------------------------------------

        policy_action = get_initial_action(
            failure_category
        )

        permitted_actions = get_permitted_actions(
            failure_category
        )

        recovery_sequence = get_recovery_sequence(
            failure_category
        )

        # -----------------------------------------------------
        # 2. AI SELECTS THE BEST PERMITTED ACTION
        # -----------------------------------------------------

        permitted_rows = case_erv[
            case_erv["action"].astype(str).isin(permitted_actions)
        ].sort_values(
            ["erv", "recovery_probability"],
            ascending=[False, False],
        )

        ai_action = (
            str(permitted_rows.iloc[0]["action"])
            if not permitted_rows.empty
            else str(policy_action)
        )

        action_row = case_erv[
            case_erv["action"].astype(str)
            == ai_action
        ]

        if action_row.empty:

            action_row = case_erv.iloc[[0]]

        row = action_row.iloc[0]

        selected_erv = float(
            row.get("erv", 0.0)
        )

        # -----------------------------------------------------
        # 3. COMPLETE GUARDRAIL CONTEXT
        # -----------------------------------------------------

        case_context = {
            "case_id": case_id,

            "customer_id": case_row[
                "customer_id"
            ],

            "failure_category": (
                failure_category
            ),

            "communication_opt_in": (
                case_row[
                    "communication_opt_in"
                ]
            ),

            "attempt_number": 1,

            "recovery_amount": float(
                case_row[
                    "recovery_amount"
                ]
            ),

            "customer_segment": case_row.get(
                "customer_segment",
                "unknown",
            ),

            # IMPORTANT:
            # This was missing before.
            "erv": selected_erv,
        }

        # -----------------------------------------------------
        # 4. GUARDRAIL CHECK
        # -----------------------------------------------------

        guardrail = evaluate_guardrails(
            case_context,
            ai_action,
        )

        allowed = bool(
            guardrail.get(
                "allowed",
                False,
            )
        )

        guardrail_reason = guardrail.get(
            "reason",
            "Unknown guardrail result",
        )

        # -----------------------------------------------------
        # 5. FINAL ACTION
        # -----------------------------------------------------

        if allowed:

            final_action = ai_action

            decision_reason = (
                f"AI selected "
                f"'{ai_action}' within the policy-permitted actions "
                f"for failure "
                f"category "
                f"'{failure_category}'. "
                f"Guardrail allowed execution."
            )

        else:

            final_action = "stop"

            decision_reason = (
                f"Policy-selected action "
                f"'{policy_action}' was blocked "
                f"by a recovery guardrail: "
                f"{guardrail_reason}"
            )

        # -----------------------------------------------------
        # 6. SAVE DECISION + MODEL/ECONOMIC EVIDENCE
        # -----------------------------------------------------

        decisions.append(
            {
                "case_id": case_id,

                "customer_id": case_row[
                    "customer_id"
                ],

                "failure_category": (
                    failure_category
                ),

                "final_action": final_action,

                "policy_action": policy_action,
                "ai_recommendation": ai_action,

                "permitted_actions": ",".join(
                    permitted_actions
                ),

                "recovery_sequence": ",".join(
                    recovery_sequence
                ),

                "decision_reason": (
                    decision_reason
                ),

                "guardrail_status": (
                    "allowed"
                    if allowed
                    else "blocked"
                ),

                "guardrail_reason": (
                    guardrail_reason
                ),

                "recovery_probability": float(
                    row.get(
                        "recovery_probability",
                        0.0,
                    )
                ),

                "expected_recovery": float(
                    row.get(
                        "expected_recovery",
                        0.0,
                    )
                ),

                "action_cost": float(
                    row.get(
                        "action_cost",
                        0.0,
                    )
                ),

                "erv": selected_erv,
            }
        )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    decisions_df = pd.DataFrame(
        decisions
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    decisions_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print(
        "✓ Unseen policy decision engine completed."
    )

    print(
        f"Cases evaluated: "
        f"{decisions_df['case_id'].nunique():,}"
    )

    print()
    print("Final actions:")

    print(
        decisions_df[
            "final_action"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Guardrail status:")

    print(
        decisions_df[
            "guardrail_status"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Expected recovery ₹"
        f"{decisions_df['expected_recovery'].sum():,.2f}"
    )

    print(
        "Economic signal ₹"
        f"{decisions_df['erv'].sum():,.2f}"
    )

    print()
    print("Sample decisions:")

    print(
        decisions_df[
            [
                "case_id",
                "failure_category",
                "policy_action",
                "final_action",
                "guardrail_status",
                "recovery_probability",
                "decision_reason",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()