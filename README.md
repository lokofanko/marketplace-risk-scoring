# Marketplace Risk Scoring

A production-style ML engineering project for real-time trust and safety scoring in a mini classified marketplace.

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

Initial monorepo skeleton and uv environments. The backend now includes an imported
Django classified marketplace scaffold that will be adapted step by step.

No ML logic, risk scoring integration, training scripts, or Docker logic has been implemented yet.
