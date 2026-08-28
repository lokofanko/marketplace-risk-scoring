"""Configuration read from environment variables."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        PROJECT_ROOT / "artifacts" / "models" / "logreg_v1" / "model.joblib",
    )
)
MODEL_NAME = os.getenv("MODEL_NAME", "logistic_regression_baseline")
MODEL_VERSION = os.getenv("MODEL_VERSION", "logreg_v1")

DATABASE_URL = os.getenv("DATABASE_URL")

APPROVE_THRESHOLD = float(os.getenv("APPROVE_THRESHOLD", "0.3"))
BLOCK_THRESHOLD = float(os.getenv("BLOCK_THRESHOLD", "0.75"))
