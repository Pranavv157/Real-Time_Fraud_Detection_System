import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

#  Load data
df = pd.read_csv("data/raw/creditcard.csv")

#  Split features and target
X = df.drop("Class", axis=1)
y = df["Class"]

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Train model (with imbalance handling)
model = LogisticRegression(max_iter=5000, class_weight="balanced")
model.fit(X_train, y_train)

#  Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#  Save model
joblib.dump(model, "models/model.pkl")

print("Model trained and saved!")