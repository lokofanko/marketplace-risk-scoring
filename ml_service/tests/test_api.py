from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_score_valid_payload_returns_prediction():
    payload = {
        "listing_id": "listing_test_001",
        "title": "iPhone 15 Pro very cheap urgent",
        "description": "Prepayment only, contact me in Telegram",
        "price": 200.0,
        "category": "electronics",
        "location": "Moscow",
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

    response = client.post("/score", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["listing_id"] == "listing_test_001"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_level"] in ["low", "medium", "high"]
    assert data["recommended_action"] in ["approve", "manual_review", "block"]
    assert data["model_version"] == "logreg_v1"

def test_score_invalid_price_returns_422():
    payload = {
        "listing_id": "listing_test_001",
        "title": "iPhone 15 Pro very cheap urgent",
        "description": "Prepayment only, contact me in Telegram",
        "price": "дешево",
        "category": "electronics",
        "location": "Moscow",
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

    response = client.post("/score", json=payload)

    assert response.status_code == 422