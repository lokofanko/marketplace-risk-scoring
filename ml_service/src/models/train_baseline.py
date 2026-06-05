"""Train a simple Logistic Regression baseline for listing risk scoring."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "listing_events.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models" / "logreg_v1"

CATEGORICAL_FEATURES = ["category", "location"]
NUMERIC_FEATURES = [
    "price",
    "account_age_days",
    "num_ads_last_24h",
    "num_ads_last_7d",
    "is_verified_user",
    "previous_rejected_ads_count",
    "num_images",
    "has_telegram",
    "has_urgency_word",
    "has_external_contact",
    "price_to_category_median_ratio",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
THRESHOLDS = [0.3, 0.5, 0.75]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run src/data/simulate_listing_events.py first."
        )

    df = pd.read_csv(path)
    missing_columns = sorted(set(FEATURE_COLUMNS + ["label"]) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def confusion_at_thresholds(y_true: pd.Series, y_proba: pd.Series) -> dict[str, dict[str, int]]:
    matrices: dict[str, dict[str, int]] = {}
    for threshold in THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        matrices[str(threshold)] = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    return matrices


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_dataset_info(df: pd.DataFrame, train_rows: int, validation_rows: int) -> dict[str, object]:
    label_counts = df["label"].value_counts().sort_index()

    return {
        "data_path": str(DATA_PATH),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "target_column": "label",
        "label_counts": {str(label): int(count) for label, count in label_counts.items()},
        "label_rate": float(df["label"].mean()),
        "train_rows": int(train_rows),
        "validation_rows": int(validation_rows),
        "random_seed": RANDOM_SEED,
        "categorical_feature_values": {
            column: sorted(df[column].dropna().astype(str).unique().tolist())
            for column in CATEGORICAL_FEATURES
        },
        "numeric_feature_summary": {
            column: {
                "min": float(df[column].min()),
                "max": float(df[column].max()),
                "mean": float(df[column].mean()),
            }
            for column in NUMERIC_FEATURES
        },
    }


def main() -> None:
    df = load_dataset(DATA_PATH)
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_val, y_val_proba)),
        "pr_auc": float(average_precision_score(y_val, y_val_proba)),
        "log_loss": float(log_loss(y_val, y_val_proba)),
        "validation_label_rate": float(y_val.mean()),
        "confusion_matrices": confusion_at_thresholds(y_val, pd.Series(y_val_proba)),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "model.joblib"
    metrics_path = ARTIFACT_DIR / "metrics.json"
    model_info_path = ARTIFACT_DIR / "model_info.json"
    feature_columns_path = ARTIFACT_DIR / "feature_columns.json"
    dataset_info_path = ARTIFACT_DIR / "dataset_info.json"

    joblib.dump(pipeline, model_path)
    write_json(metrics_path, metrics)
    write_json(
        model_info_path,
        {
            "model_name": "logistic_regression_baseline",
            "model_version": "logreg_v1",
            "trained_at": datetime.now(UTC).isoformat(),
            "data_path": str(DATA_PATH),
            "target_column": "label",
            "rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "random_seed": RANDOM_SEED,
            "sklearn_version": sklearn.__version__,
        },
    )
    write_json(
        feature_columns_path,
        {
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "all_features": FEATURE_COLUMNS,
            "excluded_columns": ["listing_id", "user_id", "title", "description", "label"],
        },
    )
    write_json(dataset_info_path, build_dataset_info(df, len(X_train), len(X_val)))

    print(f"Dataset shape: {df.shape}")
    print(f"Label rate: {df['label'].mean():.4f}")
    print("Validation metrics:")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"  LogLoss: {metrics['log_loss']:.4f}")
    print("Artifact paths:")
    print(f"  model: {model_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  model_info: {model_info_path}")
    print(f"  feature_columns: {feature_columns_path}")
    print(f"  dataset_info: {dataset_info_path}")


if __name__ == "__main__":
    main()
