import joblib
import pandas as pd


MODEL_PATH = (
    "models/recovery_probability_model_v3.joblib"
)


CATEGORICAL_FEATURES = [
    "failure_category",
    "customer_segment",
    "amount_bucket",
]


NUMERIC_FEATURES = [
    "attempt_number",
    "customer_lifetime_value",
    "successful_payment_count",
    "failed_payment_count",
    "total_payment_attempts",
    "historical_failure_rate",
    "communication_opt_in",
]


V3_FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# Load the model ONCE when this module is imported.
_MODEL = None


def load_model():
    """Load the trained model once and reuse it."""

    global _MODEL

    if _MODEL is None:
        _MODEL = joblib.load(
            MODEL_PATH
        )

    return _MODEL


def prepare_features(
    case: dict,
    action: str,
) -> pd.DataFrame:
    """Convert a recovery case into model features."""

    successful_payments = max(
        int(
            case[
                "successful_payment_count"
            ]
        ),
        0,
    )

    failed_payments = max(
        int(
            case[
                "failed_payment_count"
            ]
        ),
        0,
    )

    total_attempts = max(
        int(
            case[
                "total_payment_attempts"
            ]
        ),
        1,
    )

    recovery_amount = float(
        case[
            "recovery_amount"
        ]
    )

    lifetime_value = max(
        float(
            case[
                "customer_lifetime_value"
            ]
        ),
        1.0,
    )

    customer_success_rate = (
        successful_payments
        / total_attempts
    )

    failure_to_success_ratio = (
        failed_payments
        / max(
            successful_payments,
            1,
        )
    )

    amount_to_lifetime_value_ratio = (
        recovery_amount
        / lifetime_value
    )

    payment_history_volume = (
        successful_payments
        + failed_payments
    )

    row = {
        "failure_category": (
            case[
                "failure_category"
            ]
        ),
        "customer_segment": (
            case[
                "customer_segment"
            ]
        ),
        "amount_bucket": (
            case[
                "amount_bucket"
            ]
        ),
        "action": action,
        "attempt_number": int(
            case.get(
                "attempt_number",
                1,
            )
        ),
        "recovery_amount": recovery_amount,
        "customer_lifetime_value": lifetime_value,
        "successful_payment_count": (
            successful_payments
        ),
        "failed_payment_count": (
            failed_payments
        ),
        "total_payment_attempts": (
            total_attempts
        ),
        "historical_failure_rate": (
            failed_payments
            / total_attempts
        ),
        "communication_opt_in": int(
            bool(
                case[
                    "communication_opt_in"
                ]
            )
        ),
        "customer_success_rate": (
            customer_success_rate
        ),
        "failure_to_success_ratio": (
            failure_to_success_ratio
        ),
        "amount_to_lifetime_value_ratio": (
            amount_to_lifetime_value_ratio
        ),
        "payment_history_volume": (
            payment_history_volume
        ),
    }

    return pd.DataFrame([row])[V3_FEATURE_COLUMNS]


def predict_recovery_probability(
    case: dict,
    action: str,
) -> float:
    """Predict recovery probability for one case/action."""

    model = load_model()

    features = prepare_features(
        case,
        action,
    )

    probability = model.predict_proba(
        features
    )[0][1]

    return float(
        max(
            0.0,
            min(
                1.0,
                probability,
            ),
        )
    )