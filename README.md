# Marketplace Risk Scoring

A production-style ML engineering project for real-time trust and safety scoring in a mini classified marketplace.

## Goal

Build an end-to-end ML risk scoring service that can be integrated into a marketplace backend. The service will score listing creation events and return a risk-based decision: `approve`, `manual_review`, or `block`.

## Planned Flow

1. User creates a listing in the marketplace.
2. Backend sends listing and user features to the ML service.
3. ML service returns `risk_score`, `risk_level`, `recommended_action`, `model_version`, and `risk_factors`.
4. Backend publishes, reviews, or blocks the listing.
5. Moderator feedback is stored for future model retraining.

## Planned Architecture

The project is structured as a standalone ML service that will later integrate with a Django/DRF marketplace backend.

- Marketplace backend owns users, listings, moderation workflows, and persistence.
- ML service owns feature extraction, model inference, risk policy, and prediction metadata.
- Offline ML modules own data simulation, training, evaluation, reporting, and future retraining workflows.
- Artifacts and reports are stored locally during early development and can later be moved to MLflow or object storage.

## Planned Components

- Synthetic event/data simulator
- Feature engineering pipeline
- Logistic Regression baseline
- Tree/boosting baseline
- Model evaluation with ROC-AUC, PR-AUC, and LogLoss
- Threshold policy and business cost function
- FastAPI model serving
- Prediction logging
- MLflow experiment tracking
- Docker Compose setup
- Monitoring and drift checks
- SQL analytics
- Documentation and interview-ready project report

## Repository Layout

```text
app/          Future service application package
src/          Future offline ML modules
configs/      Configuration files
data/         Local raw, interim, and processed data
artifacts/    Local model artifacts, reports, and metrics
notebooks/    Exploratory notebooks
sql/          Analytical SQL drafts
tests/        Test suite
docs/         Architecture, API contract, and model documentation
```

## Current Status

Repository skeleton and uv environment initialized. Implementation will be added step by step.

No training scripts, simulator scripts, FastAPI endpoints, model code, Docker logic, or business logic have been implemented yet.

## Development

Install and synchronize the local environment with uv:

```bash
uv sync
```

Run future project commands through the Makefile as they are added:

```bash
make help
```
