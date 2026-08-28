"""Loading and inference helpers for the sklearn model artifact."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import MODEL_NAME, MODEL_PATH, MODEL_VERSION
from app.schemas import ScoreRequest

FEATURE_COLUMNS = [
    "category",
    "location",
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


class ModelUnavailableError(RuntimeError):
    """Raised when the model artifact cannot be used."""


_model: Any | None = None


def load_model(model_path: Path | None = None) -> Any:
    global _model

    path = model_path or MODEL_PATH
    try:
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        candidate = joblib.load(path)
        if not hasattr(candidate, "predict_proba"):
            raise TypeError("Model artifact does not implement predict_proba")
    except Exception as exc:
        _model = None
        raise ModelUnavailableError(str(exc)) from exc

    _model = candidate
    return _model


def get_model() -> Any:
    if _model is None:
        return load_model()
    return _model


def model_status() -> str:
    try:
        get_model()
    except ModelUnavailableError:
        return "unavailable"
    return "loaded"


def predict_probability(request: ScoreRequest) -> float:
    features = pd.DataFrame(
        [{column: getattr(request, column) for column in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS,
    )
    probability = float(get_model().predict_proba(features)[0, 1])
    return min(max(probability, 0.0), 1.0)


def read_model_info() -> dict[str, object]:
    get_model()
    info_path = MODEL_PATH.with_name("model_info.json")
    metadata: dict[str, object] = {}
    if info_path.is_file():
        try:
            metadata = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}

    return {
        "model_name": str(metadata.get("model_name", MODEL_NAME)),
        "model_version": str(metadata.get("model_version", MODEL_VERSION)),
        "status": "loaded",
        "model_type": "sklearn Pipeline with LogisticRegression",
        "feature_count": len(FEATURE_COLUMNS),
    }
