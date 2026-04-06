"""Shared paths and ML settings for the fraud-mlops package."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Data

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


# Artifacts (single source of truth)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

#  ONE MODEL PATH ONLY
MODEL_OUTPUT_PATH = ARTIFACTS_DIR / "model.joblib"
MODEL_PATH = MODEL_OUTPUT_PATH  #  same path for training + inference


# MLflow

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "fraud-detection-v2"


# Inference

THRESHOLD = 0.3