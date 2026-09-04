from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "processed" / "training_data.csv"
MODEL_PATH = ROOT / "models" / "recovery_probability_model_v3.joblib"
REPORT_PATH = ROOT / "models" / "recovery_model_v3_report.json"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET DETECTION
# ============================================================

TARGET_CANDIDATES = [
    "recovered",
    "is_recovered",
    "recovery_success",
    "recovery_outcome",
    "recovery_status",
    "outcome",
    "label",
    "target",
]


def detect_target(df):
    for column in TARGET_CANDIDATES:
        if column in df.columns:
            return column

    # Fallback heuristic
    for column in df.columns:
        name = column.lower()

        if any(
            token in name
            for token in [
                "recover",
                "success",
                "outcome",
            ]
        ):
            unique = df[column].dropna().unique()

            if len(unique) == 2:
                return column

    raise ValueError(
        "Could not automatically detect a binary recovery target."
    )


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def normalize_target(series):
    if pd.api.types.is_numeric_dtype(series):
        values = series.dropna().unique()

        if set(values).issubset({0, 1}):
            return series.astype(int)

        if len(values) == 2:
            mapping = {
                values[0]: 0,
                values[1]: 1,
            }
            return series.map(mapping).astype(int)

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    positive = {
        "1",
        "true",
        "yes",
        "y",
        "success",
        "successful",
        "recovered",
        "paid",
        "captured",
        "completed",
    }

    negative = {
        "0",
        "false",
        "no",
        "n",
        "failure",
        "failed",
        "not_recovered",
        "unrecovered",
        "declined",
        "stopped",
    }

    result = pd.Series(index=series.index, dtype="float")

    result[text.isin(positive)] = 1
    result[text.isin(negative)] = 0

    unresolved = result.isna()

    if unresolved.any():
        unique = text[~result.isna()].unique()

        if len(unique) == 0:
            raise ValueError(
                "Unable to normalize target into binary values."
            )

    return result


# ============================================================
# CASE-LEVEL COLLAPSE
# ============================================================

def collapse_to_case_level(df):
    """
    Training data may contain multiple action rows per case.

    The recoverability model must predict ONE probability
    per payment-recovery case.

    Therefore action-level duplicates are collapsed.
    """

    if "case_id" not in df.columns:
        return df.copy()

    if not df["case_id"].duplicated().any():
        return df.copy()

    rows = []

    for case_id, group in df.groupby("case_id", sort=False):

        row = group.iloc[0].copy()

        # Recovery is a case-level outcome.
        if "recovered" in group.columns:
            row["recovered"] = int(
                group["recovered"]
                .fillna(0)
                .astype(int)
                .max()
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    return result.reset_index(drop=True)


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Training data not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Loaded training data: {len(df):,} rows")

target_column = detect_target(df)

print(f"Detected target: {target_column}")

df = collapse_to_case_level(df)

print(f"Case-level rows: {len(df):,}")

target = normalize_target(df[target_column])

valid = target.notna()

df = df.loc[valid].copy()
target = target.loc[valid].astype(int)

print(
    f"Positive rate: {target.mean():.2%}"
)


# ============================================================
# FEATURES
# ============================================================

# IMPORTANT:
# These columns can leak the answer or encode an action.
# They must NOT be used by the recoverability model.

EXCLUDED_COLUMNS = {
    target_column,
    "action",
    "final_action",
    "selected_action",
    "case_id",

    # Economic/action-selection outputs
    "erv",
    "expected_recovery",
    "expected_recovered",
    "action_cost",

    # Existing model outputs
    "recovery_probability",
    "predicted_probability",
    "score",

    # Outcome leakage
    "recovery_amount",
    "recovered_amount",
    "recovery_value",
    "recovery_outcome",
    "recovery_status",
}


feature_columns = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]


X = df[feature_columns].copy()
y = target


# Remove columns that are completely empty
empty_columns = [
    column
    for column in X.columns
    if X[column].isna().all()
]

if empty_columns:
    X = X.drop(columns=empty_columns)

feature_columns = list(X.columns)


print(
    f"Features used: {len(feature_columns)}"
)

print(
    "Excluded from ML:",
    sorted(EXCLUDED_COLUMNS.intersection(df.columns))
)


# ============================================================
# FEATURE TYPES
# ============================================================

numeric_features = list(
    X.select_dtypes(
        include=[
            "number",
            "bool",
        ]
    ).columns
)

categorical_features = [
    column
    for column in X.columns
    if column not in numeric_features
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ============================================================
# CANDIDATE MODELS
# ============================================================

models = {

    "logistic_regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        random_state=42,
    ),
}


# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []

trained_models = {}


for name, model in models.items():

    print()
    print(
        f"Training {name}..."
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    probabilities = pipeline.predict_proba(
        X_valid
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_valid,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_valid,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_valid,
        probabilities,
    )

    brier = brier_score_loss(
        y_valid,
        probabilities,
    )

    result = {
        "model": name,
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
    }

    results.append(result)

    trained_models[name] = pipeline

    print(
        f"  accuracy={accuracy:.3f} "
        f"ROC-AUC={roc_auc:.3f} "
        f"PR-AUC={pr_auc:.3f} "
        f"Brier={brier:.3f}"
    )


# ============================================================
# MODEL SELECTION
# ============================================================

# RecoverAI uses ML for probability estimation.
#
# Therefore:
#   1. Lower Brier = better probability quality
#   2. Higher ROC-AUC = better ranking
#   3. Higher PR-AUC = better positive-case ranking
#
# Accuracy is deliberately NOT the primary criterion.

results_sorted = sorted(
    results,
    key=lambda result: (
        result["brier_score"],
        -result["roc_auc"],
        -result["pr_auc"],
    ),
)


best_result = results_sorted[0]

best_name = best_result["model"]

best_model = trained_models[best_name]


print()
print("BEST MODEL")
print(
    json.dumps(
        best_result,
        indent=2,
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_PATH,
)


# ============================================================
# SAVE REPORT
# ============================================================

report = {
    "model_version": "v3_case_level",

    "target_column": target_column,

    "rows": int(len(X)),

    "positive_rate": float(
        y.mean()
    ),

    "feature_columns": feature_columns,

    "excluded_columns": sorted(
        EXCLUDED_COLUMNS.intersection(
            df.columns
        )
    ),

    "candidates": results,

    "selection_criterion": {
        "primary": "brier_score",
        "secondary": "roc_auc",
        "tertiary": "pr_auc",
        "accuracy_role": "diagnostic_only",
    },

    "selected_model": best_result,

    "note": (
        "The model predicts case recoverability. "
        "It does not select actions. Recovery policy, "
        "guardrails and state machine determine permitted actions."
    ),
}


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        report,
        file,
        indent=2,
    )


print()
print(
    f"Saved model: {MODEL_PATH}"
)

print(
    f"Saved report: {REPORT_PATH}"
)