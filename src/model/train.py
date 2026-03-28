import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
from pathlib import Path

from src.features.pipeline import create_pipeline
from src.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_OUTPUT_PATH,
    PIPELINE_OUTPUT_PATH,
    THRESHOLD,
)
from xgboost import XGBClassifier  # type: ignore[reportMissingImports]
import mlflow  # type: ignore[reportMissingImports]
import mlflow.sklearn # type: ignore[reportMissingImports]
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Set local tracking URI and experiment
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# Load data
df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline
pipeline = create_pipeline()

# Fit pipeline on training data
X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Train model
# Calculate imbalance ratio
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss"
)

for threshold in [0.3, 0.5, 0.7]:
    with mlflow.start_run(run_name=f"xgb_threshold_{threshold}"):

        model.fit(X_train_transformed, y_train)

        y_probs = model.predict_proba(X_test_transformed)[:, 1]
        y_pred = (y_probs > threshold).astype(int)

        # Log parameters
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", threshold)

        # Metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Threshold: {threshold}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
           

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.colorbar()
        plt.savefig("confusion_matrix.png")
        plt.close()

        mlflow.log_artifact("confusion_matrix.png")

        fpr, tpr, _ = roc_curve(y_test, y_probs)

        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig("roc_curve.png")
        plt.close()

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_artifact("roc_curve.png")

        mlflow.sklearn.log_model(model, "model")