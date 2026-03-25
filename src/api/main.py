from fastapi import FastAPI
import joblib
import pandas as pd
from src.api.schema import Transaction

app = FastAPI()

model = joblib.load("models/model.pkl")
pipeline = joblib.load("models/pipeline.pkl")


@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}


@app.post("/predict")
def predict(data: Transaction):
    df = pd.DataFrame([data.model_dump()])

    # APPLY SAME PIPELINE
    df_transformed = pipeline.transform(df)

    probs = model.predict_proba(df_transformed)[:, 1]
    threshold = 0.7
    prediction = (probs > threshold).astype(int)[0]

    return {
        "fraud": bool(prediction)
    }