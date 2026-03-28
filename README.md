# 🚀 Fraud Detection System (End-to-End ML)

An end-to-end machine learning system for detecting fraudulent transactions using XGBoost, with experiment tracking (MLflow), evaluation metrics, and a FastAPI-based real-time inference API.

---

## 📌 Features

* ⚡ XGBoost model with class imbalance handling
* 🧠 Feature preprocessing pipeline
* 📊 MLflow experiment tracking (precision, recall, F1, ROC-AUC)
* 📉 Evaluation with Confusion Matrix & ROC Curve
* 🔁 Threshold tuning for optimal fraud detection
* 🚀 FastAPI for real-time predictions
* 📦 Dockerized for portable deployment

---

## 🏗️ Architecture

```text
        ┌──────────────┐
        │ Raw Dataset  │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ Preprocessing│
        │  Pipeline    │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │  XGBoost     │
        │   Model      │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ MLflow       │
        │ Tracking     │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ Saved Model  │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ FastAPI      │
        │ Inference    │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ API Response │
        └──────────────┘
```

---

## 📊 Model Performance

| Threshold | Precision | Recall   | F1 Score   |
| --------- | --------- | -------- | ---------- |
| 0.3       | 0.40      | 0.87     | 0.55       |
| 0.5       | 0.63      | 0.84     | 0.72       |
| 0.7       | **0.79**  | **0.84** | **0.82** ✅ |

✔ Selected threshold: **0.7**

---

## 📈 Evaluation

* ROC-AUC ≈ **0.98**
* High recall ensures fraud detection
* Balanced precision reduces false positives

---

## 🚀 Running Locally

```bash
git clone <repo>
cd fraud-mlops
pip install -r requirements.txt

python -m src.model.train
uvicorn src.api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🐳 Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

---

## 🧪 Example API Request

```json
{
  "Time": 10000,
  "V1": -3,
  "V2": 2,
  "V3": -1,
  "V4": 2,
  "Amount": 1000
}
```

---

## 🎥 Demo

(Add your video link here)will be adding soon

---

## 🛠️ Tech Stack

* Python
* XGBoost
* Scikit-learn
* MLflow
* FastAPI
* Docker

---

## 📌 Future Improvements

* Model explainability (SHAP)
* Cloud deployment
* Streaming inference
