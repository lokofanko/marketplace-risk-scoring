"""Compare three small risk-scoring models on the same validation split."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "listing_events.csv"
COMPARISON_PATH = PROJECT_ROOT / "artifacts" / "model_comparison.json"
RANDOM_FOREST_DIR = PROJECT_ROOT / "artifacts" / "models" / "random_forest_v1"
BOOSTING_DIR = PROJECT_ROOT / "artifacts" / "models" / "hist_gradient_boosting_v1"

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


def build_pipeline(model: object) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def evaluate(y_true: pd.Series, probabilities: object) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "precision_at_0.5": float(precision_score(y_true, predictions)),
        "recall_at_0.5": float(recall_score(y_true, predictions)),
        "f1_at_0.5": float(f1_score(y_true, predictions)),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Generate the dataset before comparing models.")

    dataset = pd.read_csv(DATA_PATH)
    X = dataset[FEATURE_COLUMNS]
    y = dataset["label"].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_val)[:, 1]
        results[name] = evaluate(y_val, probabilities)
        trained_models[name] = pipeline

    winner = max(results, key=lambda name: results[name]["pr_auc"])
    report = {
        "dataset_rows": len(dataset),
        "validation_rows": len(X_val),
        "validation_label_rate": float(y_val.mean()),
        "selection_metric": "pr_auc",
        "best_model": winner,
        "models": results,
    }

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.write_text(json.dumps(report, indent=2) + "\n")
    RANDOM_FOREST_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models["random_forest"], RANDOM_FOREST_DIR / "model.joblib")
    (RANDOM_FOREST_DIR / "metrics.json").write_text(
        json.dumps(results["random_forest"], indent=2) + "\n"
    )
    BOOSTING_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models["hist_gradient_boosting"], BOOSTING_DIR / "model.joblib")
    (BOOSTING_DIR / "metrics.json").write_text(
        json.dumps(results["hist_gradient_boosting"], indent=2) + "\n"
    )

    print(f"Dataset shape: {dataset.shape}")
    print(f"Label rate: {dataset['label'].mean():.4f}")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    print(f"\nBest model by PR-AUC: {winner}")
    print(f"Comparison report: {COMPARISON_PATH}")
    print(f"Random Forest artifact: {RANDOM_FOREST_DIR / 'model.joblib'}")
    print(f"Boosting artifact: {BOOSTING_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
