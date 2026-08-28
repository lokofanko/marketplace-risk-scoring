"""Request and response schemas for the scoring API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskLevel = Literal["low", "medium", "high"]
RecommendedAction = Literal["approve", "manual_review", "block"]


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: Literal["listing_create"] = "listing_create"
    listing_id: str = Field(min_length=1, max_length=100)
    user_id: str | None = Field(default=None, max_length=100)
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=5_000)
    price: float = Field(gt=0)
    category: str = Field(min_length=1, max_length=100)
    location: str = Field(min_length=1, max_length=100)
    account_age_days: int = Field(ge=0)
    num_ads_last_24h: int = Field(ge=0)
    num_ads_last_7d: int = Field(ge=0)
    is_verified_user: bool
    previous_rejected_ads_count: int = Field(ge=0)
    num_images: int = Field(ge=0)
    has_telegram: bool
    has_urgency_word: bool
    has_external_contact: bool
    price_to_category_median_ratio: float = Field(gt=0)


class ScoreResponse(BaseModel):
    listing_id: str
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    recommended_action: RecommendedAction
    model_version: str
    risk_factors: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_status: Literal["loaded", "unavailable"]
    database_status: Literal["available", "unavailable", "disabled"]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    status: Literal["loaded"]
    model_type: str
    feature_count: int


class PredictionLog(BaseModel):
    id: int
    listing_id: str
    risk_score: float
    risk_level: RiskLevel
    recommended_action: RecommendedAction
    model_version: str
    created_at: datetime


class PredictionLogsResponse(BaseModel):
    logs: list[PredictionLog]
