import joblib
import pandas as pd

from ..config import MODEL_PATH, THRESHOLD

#  Load full pipeline (model + scaler)
pipeline = joblib.load(MODEL_PATH)


def predict(data: pd.DataFrame):
    probs = pipeline.predict_proba(data)[:, 1]
    preds = (probs > THRESHOLD).astype(int)

    return {
        "fraud_probability": float(probs[0]),
        "fraud_prediction": bool(preds[0])
    }