"""Action-conditioned recommendation model for Judge Mode.

V3 estimates case-level recoverability. This companion model estimates how
the proposed action changes that probability, so the UI can show a genuine
next-best-action comparison. Policy and guardrails still have final authority.
"""

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "recovery_probability_model_v2.joblib"

CATEGORICAL_FEATURES = [
    "failure_category",
    "customer_segment",
    "amount_bucket",
    "action",
]

NUMERIC_FEATURES = [
    "attempt_number",
    "recovery_amount",
    "customer_lifetime_value",
    "successful_payment_count",
    "failed_payment_count",
    "total_payment_attempts",
    "historical_failure_rate",
    "communication_opt_in",
    "customer_success_rate",
    "failure_to_success_ratio",
    "amount_to_lifetime_value_ratio",
    "payment_history_volume",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
_MODEL = None


def _features(case: dict, action: str) -> pd.DataFrame:
    successful = max(int(case.get("successful_payment_count", 0)), 0)
    failed = max(int(case.get("failed_payment_count", 0)), 0)
    total = max(int(case.get("total_payment_attempts", 1)), 1)
    amount = float(case.get("recovery_amount", 0.0))
    lifetime = max(float(case.get("customer_lifetime_value", 1.0)), 1.0)

    return pd.DataFrame(
        [
            {
                "failure_category": case.get("failure_category", "unknown_failure"),
                "customer_segment": case.get("customer_segment", "regular"),
                "amount_bucket": case.get("amount_bucket", "0_5k"),
                "action": action,
                "attempt_number": int(case.get("attempt_number", 1)),
                "recovery_amount": amount,
                "customer_lifetime_value": lifetime,
                "successful_payment_count": successful,
                "failed_payment_count": failed,
                "total_payment_attempts": total,
                "historical_failure_rate": failed / total,
                "communication_opt_in": int(bool(case.get("communication_opt_in", False))),
                "customer_success_rate": successful / total,
                "failure_to_success_ratio": failed / max(successful, 1),
                "amount_to_lifetime_value_ratio": amount / lifetime,
                "payment_history_volume": successful + failed,
            }
        ]
    )[FEATURE_COLUMNS]


def predict_action_probability(case: dict, action: str) -> float:
    """Estimate recovery probability conditioned on one proposed action."""

    global _MODEL

    if _MODEL is None:
        _MODEL = joblib.load(MODEL_PATH)

    probability = float(_MODEL.predict_proba(_features(case, action))[0, 1])
    return max(0.0, min(1.0, probability))