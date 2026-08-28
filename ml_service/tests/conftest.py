import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def score_payload() -> dict[str, object]:
    return {
        "event_type": "listing_create",
        "listing_id": "listing_test_001",
        "user_id": "user_test_001",
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


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "DATABASE_URL", None)
    with TestClient(main.app) as test_client:
        yield test_client
