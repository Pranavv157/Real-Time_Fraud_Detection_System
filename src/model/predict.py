import joblib
import pandas as pd
from ..config import MODEL_PATH, PIPELINE_PATH, THRESHOLD

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

def predict(data: pd.DataFrame):
    data_transformed = pipeline.transform(data)
    probs = model.predict_proba(data_transformed)[:, 1]
    preds = (probs > THRESHOLD).astype(int)
    return {
    "fraud_probability": float(probs),
    "fraud_prediction": bool(preds)
}