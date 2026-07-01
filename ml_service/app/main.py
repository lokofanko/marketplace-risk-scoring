from pathlib import Path
from datetime import datetime, timezone
import json

import os
import psycopg

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "logreg_v1" / "model.joblib"
PREDICTION_LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://risk_user:risk_password@localhost:5433/marketplace_risk",
)

model = joblib.load(MODEL_PATH)

def write_prediction_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")  

def write_prediction_log_to_db(record: dict) -> None:
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prediction_logs (
                    listing_id,
                    risk_score,
                    risk_level,
                    recommended_action,
                    model_version
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    record["listing_id"],
                    record["risk_score"],
                    record["risk_level"],
                    record["recommended_action"],
                    record["model_version"],
                ),
            )

def read_recent_prediction_logs(limit: int = 10) -> list[dict]:
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    listing_id,
                    risk_score,
                    risk_level,
                    recommended_action,
                    model_version,
                    created_at
                FROM prediction_logs
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            )

            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "listing_id": row[1],
            "risk_score": row[2],
            "risk_level": row[3],
            "recommended_action": row[4],
            "model_version": row[5],
            "created_at": row[6].isoformat(),
        }
        for row in rows
    ]


class ScoreRequest(BaseModel):
    listing_id: str
    title: str
    description: str
    price: float
    category: str
    location: str
    account_age_days: int
    num_ads_last_24h: int
    num_ads_last_7d: int
    is_verified_user: bool
    previous_rejected_ads_count: int
    num_images: int
    has_telegram: bool
    has_urgency_word: bool
    has_external_contact: bool
    price_to_category_median_ratio: float


class ScoreResponse(BaseModel):
    listing_id: str
    risk_score: float
    message: str
    risk_level: str
    recommended_action: str
    model_version: str
    risk_factors: list[str]


@app.get("/")
def root():
    return {"message": "Marketplace Risk Scoring ML Service"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_name": "logistic_regression_baseline",
        "model_version": "logreg_v1",
        "status": "loaded",
        "model_path": str(MODEL_PATH),
    }

@app.get("/logs")
def get_prediction_logs(limit: int = 10):
    return {
        "logs": read_recent_prediction_logs(limit=limit)
    }

@app.get("/listings/{listing_id}")
def get_listing(listing_id: str):
    return {
        "listing_id": listing_id,
        "message": "Listing placeholder",
    }


@app.post("/score", response_model=ScoreResponse)
def score_listing(request: ScoreRequest):
    features = pd.DataFrame([{
        "category": request.category,
        "location": request.location,
        "price": request.price,
        "account_age_days": request.account_age_days,
        "num_ads_last_24h": request.num_ads_last_24h,
        "num_ads_last_7d": request.num_ads_last_7d,
        "is_verified_user": request.is_verified_user,
        "previous_rejected_ads_count": request.previous_rejected_ads_count,
        "num_images": request.num_images,
        "has_telegram": request.has_telegram,
        "has_urgency_word": request.has_urgency_word,
        "has_external_contact": request.has_external_contact,
        "price_to_category_median_ratio": request.price_to_category_median_ratio,
    }])

    risk_score = float(model.predict_proba(features)[0, 1])

    if risk_score < 0.3:
        risk_level = "low"
        recommended_action = "approve"
    elif risk_score < 0.75:
        risk_level = "medium"
        recommended_action = "manual_review"
    else:
        risk_level = "high"
        recommended_action = "block"
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "listing_id": request.listing_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "model_version": "logreg_v1",
    }

    write_prediction_log(PREDICTION_LOG_PATH, log_record)
    write_prediction_log_to_db(log_record)

    return {
        "listing_id": request.listing_id,
        "risk_score": risk_score,
        "message": "Risk score calculated by logreg_v1",
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "model_version": "logreg_v1",
        "risk_factors": [],
    }
