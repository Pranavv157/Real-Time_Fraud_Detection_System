import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier #type:ignore
import mlflow#type:ignore
import mlflow.sklearn#type:ignore
import matplotlib.pyplot as plt
import joblib

from src.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_OUTPUT_PATH,
)
from pathlib import Path

Path(MODEL_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# Load data
df = pd.read_csv(DATA_PATH)

#  ONLY TWO FEATURES
X = df[["Time", "Amount"]]
y = df["Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle imbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

#  Full pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss"
    ))
])

with mlflow.start_run(run_name="xgb_time_amount"):

    pipeline.fit(X_train, y_train)

    y_probs = pipeline.predict_proba(X_test)[:, 1]

    for threshold in [0.3, 0.5, 0.7]:
        y_pred = (y_probs > threshold).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric(f"precision_{threshold}", precision)
        mlflow.log_metric(f"recall_{threshold}", recall)
        mlflow.log_metric(f"f1_{threshold}", f1)

        print(f"\nThreshold: {threshold}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_artifact("roc_curve.png")

    mlflow.sklearn.log_model(pipeline, "model")

# Save
joblib.dump(pipeline, MODEL_OUTPUT_PATH)

print(" Training complete")