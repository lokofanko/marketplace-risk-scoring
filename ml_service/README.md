# Marketplace Risk Scoring ML Service

A small portfolio ML-service prototype that scores classified marketplace listings
before publication. It demonstrates the path from a trained sklearn artifact to an
HTTP API, a threshold policy, and PostgreSQL prediction logging.

This is intentionally a compact learning project. It is not presented as a
production anti-fraud platform or a high-load system.

## What it demonstrates

- reproducible synthetic listing data generation;
- a Logistic Regression baseline wrapped in a full sklearn `Pipeline`;
- FastAPI request validation and `predict_proba` inference;
- an explicit `approve` / `manual_review` / `block` policy layer;
- prediction audit logs stored in PostgreSQL;
- Docker Compose for the API and database;
- API, policy, and model artifact tests that do not require a test database.

## Architecture

```text
Marketplace backend / API client
              |
              | POST /score
              v
        FastAPI + Pydantic
              |
              v
  sklearn preprocessing + LogisticRegression
              |
              v
     threshold policy and risk factors
          /                 \
         v                   v
  JSON response      PostgreSQL prediction_logs

Synthetic generator -> CSV -> training script -> model.joblib + JSON artifacts
```

The model artifact contains preprocessing and classification in one object, so
serving only needs a single `predict_proba` call.

## Service layout

```text
app/
  config.py       Environment configuration
  schemas.py      Pydantic request and response models
  model.py        Artifact loading and inference
  policy.py       Decision thresholds and risk factors
  db.py           PostgreSQL initialization and prediction logs
  main.py         FastAPI application and endpoints
artifacts/models/logreg_v1/
tests/
Dockerfile
pyproject.toml
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Model and database readiness summary |
| `GET` | `/model-info` | Loaded model name, version, and feature count |
| `POST` | `/score` | Score a listing and apply the decision policy |
| `GET` | `/logs?limit=10` | Read recent prediction audit logs |
| `GET` | `/docs` | Interactive OpenAPI documentation |

`/logs` accepts limits from 1 to 100. It returns `503` when PostgreSQL is not
configured or cannot be reached. Scoring remains available if database logging
temporarily fails.

## Score example

```bash
curl -X POST http://localhost:8010/score \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "listing_create",
    "listing_id": "listing_456",
    "user_id": "user_123",
    "title": "iPhone 15 Pro very cheap urgent",
    "description": "Prepayment only, contact me in Telegram",
    "price": 200.0,
    "category": "electronics",
    "location": "Moscow",
    "account_age_days": 2,
    "num_ads_last_24h": 12,
    "num_ads_last_7d": 24,
    "is_verified_user": false,
    "previous_rejected_ads_count": 2,
    "num_images": 1,
    "has_telegram": true,
    "has_urgency_word": true,
    "has_external_contact": true,
    "price_to_category_median_ratio": 0.22
  }'
```

Example response:

```json
{
  "listing_id": "listing_456",
  "risk_score": 0.9999,
  "risk_level": "high",
  "recommended_action": "block",
  "model_version": "logreg_v1",
  "risk_factors": [
    "new_account",
    "high_listing_velocity",
    "unverified_user",
    "previous_rejections",
    "low_price",
    "off_platform_contact",
    "urgency_language",
    "few_images"
  ]
}
```

Other demo commands:

```bash
curl http://localhost:8010/health
curl http://localhost:8010/model-info
curl "http://localhost:8010/logs?limit=10"
```

## Run with Docker Compose

From the repository root:

```bash
cp .env.example .env
make docker-up
docker compose ps
```

The API is available at `http://localhost:8010`, Swagger UI at
`http://localhost:8010/docs`, and PostgreSQL at `localhost:5433`.

Stop the stack with:

```bash
make docker-down
```

## Run locally

Install dependencies and start the service:

```bash
cd ml_service
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

Without `DATABASE_URL`, inference endpoints work and database status is reported as
`disabled`; `/logs` returns `503`. To use PostgreSQL while running the API locally,
start the database and export the host connection URL from `.env.example`.

## Tests

From the repository root:

```bash
make test
```

The tests cover health, model metadata, valid and invalid scoring requests,
response schema validation, exact policy thresholds, model artifact inference,
mocked logs, and graceful database logging failure.

## Baseline training

From `ml_service/`:

```bash
uv sync --group training
uv run python src/data/simulate_listing_events.py
uv run python src/models/train_baseline.py
uv run python src/models/compare_models.py
```

The generator writes `data/processed/listing_events.csv`. Training writes the full
pipeline and metadata to `artifacts/models/logreg_v1/`:

- `model.joblib`
- `metrics.json`
- `model_info.json`
- `feature_columns.json`
- `dataset_info.json`

The synthetic labels are sampled from a transparent risk probability. Feature
distributions overlap, and the generator adds interaction effects and unobserved
noise, so the task is intentionally imperfect rather than directly reproducing a
fixed rule.

`compare_models.py` evaluates Logistic Regression, Random Forest, and Histogram
Gradient Boosting on the same split. It writes `artifacts/model_comparison.json`
and saves the candidate models under `artifacts/models/`. The API continues to
use `logreg_v1` until a model is deliberately promoted.

## Limitations

- Training data is synthetic and simplified.
- Thresholds are hand-selected and not optimized against business costs.
- Risk factors are rule-based explanations, not model attribution values.
- The service loads one local model artifact and has no registry or rollback flow.
- Database logging is synchronous and intended for a small prototype workload.
- Authentication, rate limiting, drift monitoring, and production security controls
  are outside this project scope.

## Possible next steps

- validate the request contract with the marketplace backend;
- add calibration and cost-based threshold evaluation;
- separate offline and online feature definitions more explicitly;
- add model version promotion and rollback rules;
- add a small integration test in CI using a temporary PostgreSQL service.
