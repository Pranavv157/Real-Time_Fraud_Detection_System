from fastapi import FastAPI
import joblib
import pandas as pd

from src.api.schema import Transaction
from src.config import MODEL_PATH, PIPELINE_PATH, THRESHOLD

app = FastAPI()

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)


@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}


@app.post("/predict")
def predict(data: Transaction):
    df = pd.DataFrame([data.model_dump()])

    # APPLY SAME PIPELINE
    df_transformed = pipeline.transform(df)

    probs = model.predict_proba(df_transformed)[:, 1]
    prediction = (probs > THRESHOLD).astype(int)[0]

    return {
        "fraud": bool(prediction)
    }