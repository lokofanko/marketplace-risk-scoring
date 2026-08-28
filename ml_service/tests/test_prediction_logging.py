from fastapi.testclient import TestClient

from app import main
from app.db import DatabaseUnavailableError


def test_score_still_works_when_database_logging_fails(
    client: TestClient,
    monkeypatch,
    score_payload: dict[str, object],
):
    monkeypatch.setattr(main, "DATABASE_URL", "postgresql://unavailable/test")

    def fail_to_insert(*args, **kwargs):
        raise DatabaseUnavailableError("database unavailable in test")

    monkeypatch.setattr(main.db, "insert_prediction_log", fail_to_insert)

    response = client.post("/score", json=score_payload)

    assert response.status_code == 200
    assert response.json()["model_version"] == "logreg_v1"
