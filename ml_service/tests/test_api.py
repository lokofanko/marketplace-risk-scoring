from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app import main
from app.schemas import ScoreResponse


def test_health(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_status": "loaded",
        "database_status": "disabled",
    }


def test_model_info(client: TestClient):
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "logistic_regression_baseline",
        "model_version": "logreg_v1",
        "status": "loaded",
        "model_type": "sklearn Pipeline with LogisticRegression",
        "feature_count": 13,
    }


def test_score_valid_request(client: TestClient, score_payload: dict[str, object]):
    response = client.post("/score", json=score_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["listing_id"] == "listing_test_001"
    assert 0 <= data["risk_score"] <= 1
    assert data["risk_level"] in {"low", "medium", "high"}
    assert data["recommended_action"] in {
        "approve",
        "manual_review",
        "block",
    }
    assert data["model_version"] == "logreg_v1"
    assert "new_account" in data["risk_factors"]


def test_score_response_schema(
    client: TestClient,
    score_payload: dict[str, object],
):
    response = client.post("/score", json=score_payload)

    assert response.status_code == 200
    parsed = ScoreResponse.model_validate(response.json())
    assert parsed.listing_id == score_payload["listing_id"]


def test_score_invalid_price_returns_422(
    client: TestClient,
    score_payload: dict[str, object],
):
    score_payload["price"] = "cheap"

    response = client.post("/score", json=score_payload)

    assert response.status_code == 422


def test_logs_endpoint_mocked(client: TestClient, monkeypatch):
    created_at = datetime(2026, 6, 6, tzinfo=UTC)
    monkeypatch.setattr(
        main.db,
        "read_recent_prediction_logs",
        lambda database_url, limit: [
            {
                "id": 1,
                "listing_id": "listing_test_001",
                "risk_score": 0.91,
                "risk_level": "high",
                "recommended_action": "block",
                "model_version": "logreg_v1",
                "created_at": created_at,
            }
        ],
    )

    response = client.get("/logs?limit=1")

    assert response.status_code == 200
    assert response.json()["logs"][0]["listing_id"] == "listing_test_001"


def test_logs_limit_validation(client: TestClient):
    assert client.get("/logs?limit=0").status_code == 422
    assert client.get("/logs?limit=101").status_code == 422
