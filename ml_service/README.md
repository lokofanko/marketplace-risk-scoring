# ML Service

This package will own the real-time ML risk scoring service and offline ML workflow.

## Planned Responsibilities

- Synthetic data simulation
- Feature extraction and feature engineering
- Baseline model training
- Model evaluation and reporting
- Threshold policy for `approve`, `manual_review`, and `block`
- FastAPI model serving
- Local model artifacts and metrics
- MLflow experiment tracking later
- Monitoring and drift checks later

## Baseline ML pipeline

The first baseline pipeline uses synthetic listing creation events and a simple
Logistic Regression model.

Generate synthetic data:

```bash
uv run python src/data/simulate_listing_events.py
```

Train the baseline model:

```bash
uv run python src/models/train_baseline.py
```

The generator writes `data/processed/listing_events.csv`. The training script saves
the full sklearn preprocessing and model pipeline to `artifacts/models/logreg_v1/`
along with metrics, model metadata, and feature column definitions.

MLflow, Docker, monitoring, and production model serving integration will be added later.
