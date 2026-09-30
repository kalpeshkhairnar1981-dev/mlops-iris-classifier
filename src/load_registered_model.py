"""Load the staged Iris model from MLflow and run a sample prediction."""

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "iris_features.csv"
MODEL_URI = "models:/iris-classifier-prod/Staging"
FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]


def main() -> None:
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    model = mlflow.sklearn.load_model(MODEL_URI)
    sample = pd.read_csv(DATA_PATH)[FEATURE_COLS].head(1)
    prediction = model.predict(sample)
    print(f"Loaded model: {type(model).__name__}")
    print(f"Model URI: {MODEL_URI}")
    print(f"Predicted encoded species: {prediction[0]}")


if __name__ == "__main__":
    main()