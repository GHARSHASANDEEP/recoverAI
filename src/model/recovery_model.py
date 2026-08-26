import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)


DATA_PATH = "data/processed/training_data.csv"

MODEL_PATH = "models/recovery_probability_model.joblib"


TARGET = "recovered"


DROP_COLUMNS = [
    "case_id",
]


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
]


def load_training_data():
    """Load the generated training dataset."""

    df = pd.read_csv(
        DATA_PATH
    )

    return df


def build_pipeline():
    """Build the probability-model pipeline."""

    categorical_pipeline = (
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )

    numeric_pipeline = (
        StandardScaler()
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return pipeline


def train_model(
    df: pd.DataFrame,
):
    """Train the recovery probability model."""

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
            random_state=42,
            stratify=y,
        )
    )

    pipeline = build_pipeline()

    pipeline.fit(
        X_train,
        y_train,
    )

    probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    metrics = {
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

    return (
        pipeline,
        metrics,
    )


def save_model(
    pipeline,
):
    """Save the trained model."""

    os.makedirs(
        os.path.dirname(
            MODEL_PATH
        ),
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )


def main():

    df = load_training_data()

    pipeline, metrics = (
        train_model(df)
    )

    save_model(
        pipeline
    )

    print(
        "✓ Recovery probability model "
        "trained."
    )

    print()

    print(
        "Evaluation metrics:"
    )

    for name, value in metrics.items():

        print(
            f"{name}: "
            f"{value:.4f}"
        )

    print()

    print(
        f"Training rows: "
        f"{len(df):,}"
    )

    print(
        f"Model saved to: "
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()