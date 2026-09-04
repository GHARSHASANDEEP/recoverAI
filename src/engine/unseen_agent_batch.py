import pandas as pd

from src.engine.recovery_agent import (
    run_recovery_case,
)


DECISIONS_PATH = (
    "data/unseen/processed/decisions.csv"
)

ERV_PATH = (
    "data/unseen/processed/erv_scores.csv"
)

CASES_PATH = (
    "data/unseen/processed/recovery_cases.csv"
)

CUSTOMERS_PATH = (
    "data/raw/customers.csv"
)

OUTPUT_PATH = (
    "data/unseen/processed/agent_results.csv"
)


def normalize_failure_category(value):
    """Normalize failure category into policy vocabulary."""

    if pd.isna(value):
        return "unknown_failure"

    value = str(value).strip()

    if value.startswith("[") and value.endswith("]"):
        value = value.strip("[]").strip("'\" ")

    return value or "unknown_failure"


def safe_float(value, default=0.0):
    """Convert a value to float safely."""

    if pd.isna(value):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    """Convert a value to integer safely."""

    if pd.isna(value):
        return default

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def extract_confidence(audit_events):
    """
    Extract the latest decision-confidence assessment
    recorded by the agent.
    """

    confidence_events = [
        event
        for event in audit_events
        if event.get("event")
        in {
            "initial_confidence_evaluation",
            "next_action_evaluation",
        }
    ]

    if not confidence_events:
        return {
            "confidence_score": None,
            "confidence_level": "not_evaluated",
            "probability_confidence": None,
            "action_margin": None,
            "margin_confidence": None,
        }

    latest = confidence_events[-1]

    return {
        "confidence_score": latest.get(
            "confidence_score"
        ),
        "confidence_level": latest.get(
            "confidence_level"
        ),
        "probability_confidence": latest.get(
            "probability_confidence"
        ),
        "action_margin": latest.get(
            "action_margin"
        ),
        "margin_confidence": latest.get(
            "margin_confidence"
        ),
    }


def main():

    print("Loading unseen agent inputs...")

    decisions = pd.read_csv(
        DECISIONS_PATH
    )

    erv = pd.read_csv(
        ERV_PATH
    )

    cases = pd.read_csv(
        CASES_PATH
    )

    customers = pd.read_csv(
        CUSTOMERS_PATH
    )

    print(
        f"Decisions loaded: {len(decisions):,}"
    )

    print(
        f"ERV rows loaded: {len(erv):,}"
    )

    print(
        f"Cases loaded: {len(cases):,}"
    )

    # ---------------------------------------------------------
    # Validate unseen dataset
    # ---------------------------------------------------------

    if not decisions[
        "customer_id"
    ].astype(str).str.startswith(
        "UNSEEN_"
    ).all():

        raise ValueError(
            "Non-unseen customer found in "
            "unseen agent pipeline."
        )

    # ---------------------------------------------------------
    # Build complete case/customer context
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
        if column in customers.columns
    ]

    full_cases = cases.merge(
        customers[customer_columns],
        on="customer_id",
        how="left",
    )

    full_cases["failure_category"] = (
        full_cases[
            "failure_categories"
        ].apply(
            normalize_failure_category
        )
    )

    full_cases["communication_opt_in"] = (
        full_cases[
            "communication_opt_in"
        ]
        .fillna(False)
        .astype(bool)
    )

    full_cases["recovery_amount"] = (
        pd.to_numeric(
            full_cases[
                "recovery_amount"
            ],
            errors="coerce",
        )
        .fillna(0.0)
    )

    # ---------------------------------------------------------
    # Fast case lookup
    # ---------------------------------------------------------

    case_lookup = {
        str(row["case_id"]): row
        for _, row in full_cases.iterrows()
    }

    results = []

    # ---------------------------------------------------------
    # CLOSED-LOOP RECOVERY AGENT
    # ---------------------------------------------------------

    for decision in decisions.to_dict(
        orient="records"
    ):

        case_id = str(
            decision["case_id"]
        )

        if case_id not in case_lookup:

            raise ValueError(
                f"Case context not found for "
                f"{case_id}"
            )

        case_row = case_lookup[
            case_id
        ]

        # -----------------------------------------------------
        # Complete case context
        # -----------------------------------------------------

        case = {
            "case_id": case_id,

            "customer_id": decision[
                "customer_id"
            ],

            "customer_segment": (
                case_row.get(
                    "customer_segment",
                    "unknown",
                )
                if not pd.isna(
                    case_row.get(
                        "customer_segment",
                        "unknown",
                    )
                )
                else "unknown"
            ),

            "communication_opt_in": bool(
                case_row.get(
                    "communication_opt_in",
                    False,
                )
            ),

            "customer_lifetime_value": safe_float(
                case_row.get(
                    "customer_lifetime_value",
                    0.0,
                )
            ),

            "successful_payment_count": safe_int(
                case_row.get(
                    "successful_payment_count",
                    0,
                )
            ),

            "failed_payment_count": safe_int(
                case_row.get(
                    "failed_payment_count",
                    0,
                )
            ),

            "total_payment_attempts": safe_int(
                case_row.get(
                    "total_payment_attempts",
                    0,
                )
            ),

            "failure_category": (
                decision[
                    "failure_category"
                ]
            ),

            "recovery_amount": safe_float(
                case_row.get(
                    "recovery_amount",
                    0.0,
                )
            ),

            # -------------------------------------------------
            # Policy decision
            # -------------------------------------------------

            "policy_action": decision[
                "policy_action"
            ],

            "final_action": decision[
                "final_action"
            ],

            "permitted_actions": decision[
                "permitted_actions"
            ],

            "recovery_sequence": decision[
                "recovery_sequence"
            ],

            "decision_reason": decision[
                "decision_reason"
            ],

            "guardrail_status": decision[
                "guardrail_status"
            ],

            "guardrail_reason": decision[
                "guardrail_reason"
            ],

            # -------------------------------------------------
            # Model/economic evidence
            # -------------------------------------------------

            "recovery_probability": safe_float(
                decision.get(
                    "recovery_probability",
                    0.0,
                )
            ),

            "expected_recovery": safe_float(
                decision.get(
                    "expected_recovery",
                    0.0,
                )
            ),

            "action_cost": safe_float(
                decision.get(
                    "action_cost",
                    0.0,
                )
            ),

            "erv": safe_float(
                decision.get(
                    "erv",
                    0.0,
                )
            ),

            # -------------------------------------------------
            # Agent state
            # -------------------------------------------------

            "attempt_number": 1,
        }

        # -----------------------------------------------------
        # All action evaluations for this case
        # -----------------------------------------------------

        action_scores = erv[
            erv["case_id"].astype(str)
            == case_id
        ].copy()

        if action_scores.empty:

            raise ValueError(
                f"No ERV scores found for "
                f"{case_id}"
            )

        # -----------------------------------------------------
        # Run closed-loop recovery agent
        # -----------------------------------------------------

        result = run_recovery_case(
            case,
            action_scores,
        )

        results.append(result)

    # ---------------------------------------------------------
    # Flatten agent results
    # ---------------------------------------------------------

    rows = []

    for result in results:

        audit_events = result[
            "audit_events"
        ]

        confidence = extract_confidence(
            audit_events
        )

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

                "confidence_score": confidence[
                    "confidence_score"
                ],

                "confidence_level": confidence[
                    "confidence_level"
                ],

                "probability_confidence": confidence[
                    "probability_confidence"
                ],

                "action_margin": confidence[
                    "action_margin"
                ],

                "margin_confidence": confidence[
                    "margin_confidence"
                ],

                "audit_event_count": len(
                    audit_events
                ),
            }
        )

    results_df = pd.DataFrame(rows)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    recovered = results_df[
        results_df["final_status"]
        == "recovered"
    ]

    escalated = results_df[
        results_df["final_status"]
        == "escalated"
    ]

    stopped = results_df[
        results_df["final_status"]
        == "stopped"
    ]

    total_cases = len(
        results_df
    )

    total_recovered = results_df[
        "total_recovered"
    ].sum()

    recovery_rate = (
        len(recovered)
        / total_cases
        if total_cases
        else 0.0
    )

    print()
    print(
        "✓ Unseen closed-loop agent completed."
    )

    print(
        f"Cases processed: "
        f"{total_cases:,}"
    )

    print(
        f"Recovered: "
        f"{len(recovered):,}"
    )

    print(
        f"Escalated: "
        f"{len(escalated):,}"
    )

    print(
        f"Stopped: "
        f"{len(stopped):,}"
    )

    print(
        f"Recovery rate: "
        f"{recovery_rate:.2%}"
    )

    print(
        f"Verified recovered revenue: "
        f"₹{total_recovered:,.2f}"
    )

    print(
        f"Total attempts: "
        f"{results_df['attempts'].sum():,}"
    )

    print()
    print("Confidence levels:")

    print(
        results_df[
            "confidence_level"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()