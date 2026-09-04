"""
RecoverAI V3 — case-level scoring for the recovery agent.

Run from the project root:
    python score_recoverability_v3.py

Produces action rows for compatibility with the existing recovery agent,
but the SAME case-level recoverability probability is attached to each
action. The agent/policy, not the ML model, chooses the action.
"""

from pathlib import Path
import joblib
import pandas as pd

from src.data.outcome_rules import ACTION_COSTS, get_amount_bucket

CASES_PATH = Path("data/processed/recovery_cases.csv")
CUSTOMERS_PATH = Path("data/raw/customers.csv")
MODEL_PATH = Path("models/recovery_probability_model_v3.joblib")
OUTPUT_PATH = Path("data/processed/erv_scores.csv")

ACTIONS = ["retry", "reminder", "escalate"]

CUSTOMER_COLUMNS = [
    "customer_id",
    "customer_segment",
    "customer_lifetime_value",
    "successful_payment_count",
    "failed_payment_count",
    "total_payment_attempts",
    "communication_opt_in",
]

def safe_ratio(a, b):
    return float(a) / max(float(b), 1.0)

def load_cases():
    cases = pd.read_csv(CASES_PATH)
    customers = pd.read_csv(CUSTOMERS_PATH)

    cases = cases.merge(
        customers[CUSTOMER_COLUMNS],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    cases["failure_category"] = (
        cases["failure_categories"]
        .fillna("unknown_failure")
        .astype(str)
        .str.split("|")
        .str[0]
    )

    cases["amount_bucket"] = cases["recovery_amount"].apply(get_amount_bucket)
    cases["attempt_number"] = 1

    return cases

def build_case_features(cases):
    rows = []

    for case in cases.to_dict(orient="records"):
        successful = max(int(case.get("successful_payment_count", 0)), 0)
        failed = max(int(case.get("failed_payment_count", 0)), 0)
        total = max(int(case.get("total_payment_attempts", 1)), 1)
        amount = float(case.get("recovery_amount", 0.0))
        lifetime = max(float(case.get("customer_lifetime_value", 1.0)), 1.0)

        rows.append({
            "failure_category": case.get("failure_category", "unknown_failure"),
            "customer_segment": case.get("customer_segment", "regular"),
            "amount_bucket": case.get("amount_bucket", get_amount_bucket(amount)),
            "attempt_number": int(case.get("attempt_number", 1)),
            "recovery_amount": amount,
            "customer_lifetime_value": lifetime,
            "successful_payment_count": successful,
            "failed_payment_count": failed,
            "total_payment_attempts": total,
            "historical_failure_rate": safe_ratio(failed, total),
            "communication_opt_in": int(bool(case.get("communication_opt_in", False))),
            "customer_success_rate": safe_ratio(successful, total),
            "failure_to_success_ratio": safe_ratio(failed, max(successful, 1)),
            "amount_to_lifetime_value_ratio": safe_ratio(amount, lifetime),
            "payment_history_volume": successful + failed,
        })

    return pd.DataFrame(rows)

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run train_recoverability_model_v3.py first."
        )

    cases = load_cases()
    features = build_case_features(cases)

    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        model = artifact["pipeline"]
        feature_columns = artifact["feature_columns"]
    else:
        model = artifact
        feature_columns = list(features.columns)

    # Add any missing expected columns as NA.
    for column in feature_columns:
        if column not in features.columns:
            features[column] = pd.NA

    X = features[feature_columns]
    probabilities = model.predict_proba(X)[:, 1]

    rows = []
    for case, probability in zip(cases.to_dict(orient="records"), probabilities):
        case_id = case["case_id"]
        amount = float(case["recovery_amount"])
        expected = float(probability) * amount

        for action in ACTIONS:
            cost = float(ACTION_COSTS[action])
            rows.append({
                "case_id": case_id,
                "customer_id": case["customer_id"],
                "action": action,
                "failure_category": case["failure_category"],
                "recovery_amount": amount,
                "action_cost": cost,
                "recovery_probability": float(probability),
                "expected_recovery": expected,
                # Compatibility only. It is NOT used by policy selection.
                "erv": expected - cost,
            })

    output = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"Cases scored: {len(cases):,}")
    print(f"Action rows: {len(output):,}")
    print(
        f"Probability range: "
        f"{output['recovery_probability'].min():.3f} → "
        f"{output['recovery_probability'].max():.3f}"
    )
    print(f"Saved: {OUTPUT_PATH}")
    print(
        "\nIMPORTANT: recovery_probability is case-level. "
        "Action selection remains policy/state/guardrail controlled."
    )

if __name__ == "__main__":
    main()
