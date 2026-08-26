import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)


DATA_PATH = "data/processed/training_data.csv"

BEST_MODEL_PATH = (
    "models/recovery_probability_model_v2.joblib"
)

TARGET = "recovered"

RANDOM_STATE = 42


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

    # Engineered features
    "customer_success_rate",
    "failure_to_success_ratio",
    "amount_to_lifetime_value_ratio",
    "payment_history_volume",
]


def load_data():
    df = pd.read_csv(DATA_PATH)

    # ---------------------------------------------
    # Feature engineering
    # ---------------------------------------------

    df["customer_success_rate"] = (
        df["successful_payment_count"]
        /
        df["total_payment_attempts"].clip(
            lower=1
        )
    )

    df["failure_to_success_ratio"] = (
        df["failed_payment_count"]
        /
        df["successful_payment_count"].clip(
            lower=1
        )
    )

    df["amount_to_lifetime_value_ratio"] = (
        df["recovery_amount"]
        /
        df["customer_lifetime_value"].clip(
            lower=1
        )
    )

    df["payment_history_volume"] = (
        df["successful_payment_count"]
        +
        df["failed_payment_count"]
    )

    # Prevent numerical problems.
    df = df.replace(
        [float("inf"), -float("inf")],
        0,
    )

    df = df.fillna(0)

    return df


def build_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ]
    )


def build_logistic_model():

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest():

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):

    model.fit(
        X_train,
        y_train,
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_test,
            probabilities,
        ),
    }

    return model, metrics


def main():

    df = load_data()

    features = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    models = [
        (
            "logistic_regression_v2",
            build_logistic_model(),
        ),
        (
            "random_forest_v2",
            build_random_forest(),
        ),
    ]

    results = []
    trained_models = {}

    for name, model in models:

        print(
            f"Training {name}..."
        )

        trained_model, metrics = (
            evaluate_model(
                name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
            )
        )

        trained_models[name] = (
            trained_model
        )

        results.append(metrics)

    results_df = pd.DataFrame(
        results
    )

    print()
    print(
        "MODEL COMPARISON"
    )
    print(
        results_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # Select model primarily using ROC-AUC.
    # Lower Brier score breaks ties.
    # -------------------------------------------------

    ranked = results_df.sort_values(
        by=[
            "roc_auc",
            "brier_score",
        ],
        ascending=[
            False,
            True,
        ],
    )

    best_name = ranked.iloc[0][
        "model"
    ]

    best_model = trained_models[
        best_name
    ]

    os.makedirs(
        "models",
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        BEST_MODEL_PATH,
    )

    print()
    print(
        f"✓ Best model: {best_name}"
    )

    print(
        f"✓ Saved to: "
        f"{BEST_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()