from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "logreg_v1" / "model.joblib"


def test_model_artifact_exists():
    assert MODEL_PATH.exists()


def test_model_artifact_predicts_probability():
    model = joblib.load(MODEL_PATH)

    features = pd.DataFrame([
        {
            "category": "electronics",
            "location": "Moscow",
            "price": 200.0,
            "account_age_days": 2,
            "num_ads_last_24h": 12,
            "num_ads_last_7d": 24,
            "is_verified_user": False,
            "previous_rejected_ads_count": 2,
            "num_images": 1,
            "has_telegram": True,
            "has_urgency_word": True,
            "has_external_contact": True,
            "price_to_category_median_ratio": 0.22,
        }
    ])

    probability = model.predict_proba(features)[0, 1]

    assert 0.0 <= probability <= 1.0