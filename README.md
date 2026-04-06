# 🚀 Fraud Detection System (End-to-End ML + Deployment)

An end-to-end machine learning system for detecting fraudulent transactions using **XGBoost**, with a **production-style FastAPI API**, **Streamlit UI**, **MLflow tracking**, and **Dockerized deployment**.

---

# 📌 Overview

This project demonstrates how to build a **real-world ML system**, focusing not just on model performance but on:

* Feature consistency
* API deployment
* Experiment tracking
* End-to-end integration

---

# 🧠 Key Design Decision

> The model uses only **Time** and **Amount** as input features.

### Why?

* Simulates **real-time inference constraints**
* Avoids dependency on unavailable engineered features
* Ensures **consistent inputs between training and production**

⚠️ Trade-off: Reduced model accuracy (explained below)

---

# 🏗️ Architecture

```id="arch123"
        ┌──────────────┐
        │ Raw Dataset  │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ Feature      │
        │ Selection    │
        │ (Time,Amount)
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ ML Pipeline  │
        │ (Scaler +    │
        │  XGBoost)    │
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
        │ (Dockerized) │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ Streamlit UI │
        └──────────────┘
```

---

# 📊 Model Performance

| Threshold | Precision | Recall | F1 Score |
| --------- | --------- | ------ | -------- |
| 0.3       | 0.0044    | 0.7347 | 0.0087   |
| 0.5       | 0.0085    | 0.5408 | 0.0167   |
| 0.7       | 0.0193    | 0.4184 | 0.0368   |

✔ Selected threshold: **0.7**

---

# 📈 Evaluation Insights

* ROC-AUC ≈ **0.78**
* Model can **rank fraud vs legit reasonably well**
* However, **prediction confidence is low**

---

# ⚠️ Why is Precision Low?

This is intentional and an important design trade-off.

### 1️⃣ Severe Class Imbalance

* Fraud cases ≈ **0.17%**
* Majority of transactions are legitimate

👉 Leads to naturally low precision

---

### 2️⃣ Limited Feature Set

The original dataset contains engineered features (`V1–V28`) derived from PCA.

This project uses only:

```text
Time + Amount
```

👉 These features:

* Do NOT capture behavioral patterns
* Provide weak fraud signals

---

### 3️⃣ Model Confidence Distribution

The model outputs probabilities mostly in:

```text
0.01 – 0.20 range
```

👉 Rarely produces high-confidence predictions

---

# 🧠 Key Insight

> **This project prioritizes system design over raw model performance**

---

# 🖥️ System Workflow

```id="flow123"
Streamlit UI → FastAPI (Docker) → ML Pipeline → Prediction
```

---

# 🚀 Running Locally

```bash
git clone <your-repo>
cd fraud-mlops
pip install -r requirements.txt
```

### Train Model

```bash
python -m src.model.train
```

### Run API

```bash
uvicorn src.api.main:app --reload --port 8001
```

### Run UI

```bash
streamlit run ui/app.py
```

---

# 🧪 Example API Request

```json
{
  "Time": 10000,
  "Amount": 500
}
```

---

# 🐳 Docker

### Build Image

```bash
docker build -t fraud-api .
```

### Run Container

```bash
docker run -p 8001:8000 fraud-api
```

👉 API available at:

```
http://127.0.0.1:8001/docs
```

---

# 🛠️ Tech Stack

* Python
* XGBoost
* Scikit-learn
* MLflow
* FastAPI
* Streamlit
* Docker

---

# 📌 Limitations

* Uses only **Time + Amount**
* Low precision due to missing behavioral features
* Not suitable for real-world fraud detection without enhancement

---

# 🚀 Future Improvements

* Feature engineering (transaction history, user behavior)
* Model explainability (SHAP)
* MLflow model registry integration
* Cloud deployment (AWS / Render)
* Real-time streaming inference

---

# 💡 Key Learnings

* Importance of **feature consistency in ML systems**
* Handling **data distribution mismatch**
* Designing **production-ready ML pipelines**
* Trade-offs between **accuracy vs deployability**

---

# 👨‍💻 Author

Pranav Shinde

---
