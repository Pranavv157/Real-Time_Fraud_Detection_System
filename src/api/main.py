from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

from src.api.schema import Transaction
from src.config import MODEL_PATH, THRESHOLD

app = FastAPI(title="Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = joblib.load(MODEL_PATH)


@app.post("/predict")
def predict(data: Transaction):
    df = pd.DataFrame([data.model_dump()])

    prob = pipeline.predict_proba(df)[0][1]
    pred = int(prob > THRESHOLD)

    return {
        "fraud_probability": float(prob),
        "fraud_prediction": bool(pred)
    }