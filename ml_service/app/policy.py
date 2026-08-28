"""Threshold policy applied after model inference."""

from app.schemas import RecommendedAction, RiskLevel, ScoreRequest


def decide_risk(
    risk_score: float,
    approve_threshold: float = 0.3,
    block_threshold: float = 0.75,
) -> tuple[RiskLevel, RecommendedAction]:
    if not 0 <= risk_score <= 1:
        raise ValueError("risk_score must be between 0 and 1")
    if not 0 < approve_threshold < block_threshold < 1:
        raise ValueError("thresholds must satisfy 0 < approve < block < 1")

    if risk_score < approve_threshold:
        return "low", "approve"
    if risk_score < block_threshold:
        return "medium", "manual_review"
    return "high", "block"


def collect_risk_factors(request: ScoreRequest) -> list[str]:
    factors: list[str] = []

    if request.account_age_days < 7:
        factors.append("new_account")
    if request.num_ads_last_24h >= 5 or request.num_ads_last_7d >= 15:
        factors.append("high_listing_velocity")
    if not request.is_verified_user:
        factors.append("unverified_user")
    if request.previous_rejected_ads_count > 0:
        factors.append("previous_rejections")
    if request.price_to_category_median_ratio < 0.55:
        factors.append("low_price")
    if request.has_telegram or request.has_external_contact:
        factors.append("off_platform_contact")
    if request.has_urgency_word:
        factors.append("urgency_language")
    if request.num_images <= 1:
        factors.append("few_images")

    return factors
