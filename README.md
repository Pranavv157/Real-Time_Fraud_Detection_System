# 🚀 Fraud Detection System (End-to-End ML + Cloud Deployment)

An end-to-end machine learning system for detecting fraudulent transactions using **XGBoost**, deployed as a **Dockerized FastAPI service on AWS EC2**, with a **Streamlit frontend**, **MLflow tracking**, and **production-aware design choices**.

---

# 📌 Overview

This project demonstrates how to build a **real-world ML system**, focusing not just on model performance but on:

* Feature consistency between training & inference
* API-first deployment (FastAPI)
* Containerization using Docker
* Cloud deployment on AWS EC2
* End-to-end integration with a frontend UI

---

# 🧠 Key Design Decision

> The model uses only **Time** and **Amount** as input features.

### Why?

* Simulates **real-time inference constraints**
* Avoids dependency on unavailable engineered features (V1–V28)
* Ensures **training–serving consistency**

⚠️ Trade-off: Reduced model accuracy (explained below)

---

# 🏗️ Architecture

```
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
        │ (artifact)   │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ FastAPI      │
        │ (Dockerized) │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ AWS EC2      │
        │ Deployment   │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ Streamlit UI │
        └──────────────┘
```

---

# 🌐 Live System Flow

```
User → Streamlit UI → FastAPI (AWS EC2) → ML Model → Prediction
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
* Model can **rank fraud vs legitimate transactions**
* However, **prediction confidence is low due to limited features**

---

# ⚠️ Why is Precision Low?

### 1️⃣ Severe Class Imbalance

* Fraud ≈ **0.17%**
* Leads to very low precision by default

---

### 2️⃣ Limited Feature Set

Only:

```
Time + Amount
```

Missing:

* Behavioral signals
* Transaction history
* PCA-derived features (V1–V28)

---

### 3️⃣ Model Confidence

* Most predictions fall in **0.01–0.20 probability range**
* Rarely produces high-confidence fraud predictions

---

# 🧠 Key Insight

> This project prioritizes **system design, deployment, and real-world constraints** over raw model accuracy.

---

# 🚀 Deployment (AWS EC2 + Docker)
# Frontend deployment(Streamlit)


---

# 🔐 Security Considerations

* Environment variables used for configuration
* CORS configured for controlled access
* API key mechanism can be added for protection
* EC2 security group restricts exposed ports

---

# 🐳 Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
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
* AWS EC2

---

# 📌 Limitations

* Uses only **Time + Amount**
* Low precision due to missing features
* No HTTPS (HTTP only via EC2 public IP)
* Public IP changes when instance restarts

---

# 🚀 Future Improvements

* Feature engineering (behavioral + temporal features)
* Model explainability (SHAP)
* HTTPS (Nginx + domain)
* CI/CD pipeline (GitHub Actions → AWS)
* Deploy via ECS / serverless instead of EC2


---

# 💡 Key Learnings

* Feature consistency is critical in ML systems
* Docker simplifies reproducibility
* Cloud infra requires cost awareness
* ML ≠ just model — it’s a system

---



---
