# Marketplace Risk Scoring

A portfolio monorepo for a mini classified marketplace with a real-time ML listing
risk scoring prototype.

## Goal

Build an end-to-end system where a marketplace backend calls an ML risk scoring service before publishing user listings.

## Main Flow

1. User creates a listing.
2. Backend extracts listing and user features.
3. Backend calls the ML scoring service.
4. ML service returns `risk_score`, `risk_level`, `recommended_action`, `model_version`, and `risk_factors`.
5. Backend publishes, reviews, or blocks the listing.
6. Moderator feedback is stored as labels for future retraining.

## Repository Layout

- `backend/` - Django/DRF marketplace backend
- `ml_service/` - FastAPI ML scoring service and ML pipeline
- `shared/` - API contracts and example payloads
- `infra/` - infrastructure placeholders
- `docs/` - architecture, API contract, model card, interview story
- `sql/` - future SQL schema and analytics queries

## Current Status

The ML service includes a synthetic data generator, Logistic Regression baseline,
FastAPI scoring endpoint, threshold policy, PostgreSQL prediction logs, tests, and
Docker Compose setup. See [ml_service/README.md](ml_service/README.md) for API examples,
local setup, limitations, and model details.

The Django marketplace backend is an imported scaffold and is not yet integrated
with the scoring API.
