import json

from app.main import write_prediction_log


def test_write_prediction_log_creates_jsonl_record(tmp_path):
    log_path = tmp_path / "predictions.jsonl"

    record = {
        "timestamp": "2026-06-09T00:00:00+00:00",
        "listing_id": "listing_test_001",
        "risk_score": 0.91,
        "risk_level": "high",
        "recommended_action": "block",
        "model_version": "logreg_v1",
    }

    write_prediction_log(log_path, record)

    lines = log_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert json.loads(lines[0]) == record