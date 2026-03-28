"""Shared paths and ML settings for the fraud-mlops package."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" /"raw"/ "creditcard.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_OUTPUT_PATH = ARTIFACTS_DIR / "model.joblib"
PIPELINE_OUTPUT_PATH = ARTIFACTS_DIR / "pipeline.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
PIPELINE_PATH = PROJECT_ROOT / "models" / "pipeline.pkl"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "fraud-detection"
THRESHOLD = 0.7