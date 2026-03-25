import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

from src.features.pipeline import create_pipeline
from xgboost import XGBClassifier  # type: ignore[reportMissingImports]

# Load data
df = pd.read_csv("data/raw/creditcard.csv")

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

model.fit(X_train_transformed, y_train)
# Evaluate
y_pred = model.predict(X_test_transformed)
print(classification_report(y_test, y_pred))

y_probs = model.predict_proba(X_test_transformed)[:, 1]
threshold = 0.7  # try 0.3, 0.5, 0.7

y_pred = (y_probs > threshold).astype(int)

# Save model and pipeline
joblib.dump(model, "models/model.pkl")
joblib.dump(pipeline, "models/pipeline.pkl")

print("Model and pipeline saved!")