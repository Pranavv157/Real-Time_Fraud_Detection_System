from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    return pipeline