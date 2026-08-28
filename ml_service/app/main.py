"""FastAPI application for marketplace listing risk scoring."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, status

from app import db, model
from app.config import (
    APPROVE_THRESHOLD,
    BLOCK_THRESHOLD,
    DATABASE_URL,
    MODEL_VERSION,
)
from app.policy import collect_risk_factors, decide_risk
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionLogsResponse,
    ScoreRequest,
    ScoreResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        model.load_model()
    except model.ModelUnavailableError as exc:
        logger.error("Model initialization failed: %s", exc)

    if DATABASE_URL:
        try:
            db.initialize_database(DATABASE_URL)
        except db.DatabaseUnavailableError as exc:
            logger.warning("Database initialization failed: %s", exc)

    yield


app = FastAPI(
    title="Marketplace Risk Scoring",
    description="Portfolio prototype for real-time listing risk scoring.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Marketplace Risk Scoring ML Service"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_state = model.model_status()
    if DATABASE_URL:
        database_is_available = db.check_database(DATABASE_URL)
        database_state = "available" if database_is_available else "unavailable"
    else:
        database_state = "disabled"

    service_state = (
        "ok"
        if model_state == "loaded" and database_state != "unavailable"
        else "degraded"
    )
    return HealthResponse(
        status=service_state,
        model_status=model_state,
        database_status=database_state,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        return ModelInfoResponse.model_validate(model.read_model_info())
    except model.ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is unavailable",
        ) from exc


@app.get("/logs", response_model=PredictionLogsResponse)
def get_prediction_logs(
    limit: int = Query(default=10, ge=1, le=100),
) -> PredictionLogsResponse:
    try:
        logs = db.read_recent_prediction_logs(DATABASE_URL, limit=limit)
    except db.DatabaseUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return PredictionLogsResponse(logs=logs)


@app.post("/score", response_model=ScoreResponse)
def score_listing(request: ScoreRequest) -> ScoreResponse:
    try:
        risk_score = model.predict_probability(request)
    except model.ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is unavailable",
        ) from exc
    except Exception as exc:
        logger.exception("Model inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed",
        ) from exc

    risk_level, recommended_action = decide_risk(
        risk_score,
        approve_threshold=APPROVE_THRESHOLD,
        block_threshold=BLOCK_THRESHOLD,
    )
    response = ScoreResponse(
        listing_id=request.listing_id,
        risk_score=risk_score,
        risk_level=risk_level,
        recommended_action=recommended_action,
        model_version=MODEL_VERSION,
        risk_factors=collect_risk_factors(request),
    )

    if DATABASE_URL:
        try:
            db.insert_prediction_log(DATABASE_URL, response.model_dump())
        except db.DatabaseUnavailableError as exc:
            logger.warning("Prediction was not written to PostgreSQL: %s", exc)

    return response
