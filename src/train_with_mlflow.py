"""Train and track three Iris classifiers with MLflow."""

import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mlflow.models import infer_signature


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "iris_features.csv"
EXPERIMENT_NAME = "iris-classification-baseline"
FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]
TARGET_COL = "species"


def main() -> None:
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = pd.read_csv(DATA_PATH)
    missing_columns = set(FEATURE_COLS + [TARGET_COL]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Feature CSV is missing columns: {sorted(missing_columns)}")

    features = data[FEATURE_COLS].copy()
    features = features.fillna(features.median())
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(data[TARGET_COL])
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=200, C=1.0),
        "random_forest_shallow": RandomForestClassifier(
            n_estimators=50, max_depth=3, random_state=42
        ),
        "random_forest_deep": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42
        ),
    }
    results = []

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision_macro": precision_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
            }
            mlflow.log_param("model_type", model_name)
            mlflow.log_params(
                {
                    key: value
                    for key, value in model.get_params().items()
                    if key in {"C", "max_iter", "n_estimators", "max_depth", "random_state"}
                    and value is not None
                }
            )
            if isinstance(model, RandomForestClassifier) and model.max_depth is None:
                mlflow.log_param("max_depth", "None")
            mlflow.log_metrics(metrics)

            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = Path(temp_dir) / f"confusion_matrix_{model_name}.png"
                display = ConfusionMatrixDisplay(
                    confusion_matrix=confusion_matrix(y_test, predictions),
                    display_labels=label_encoder.classes_,
                )
                display.plot()
                plt.tight_layout()
                plt.savefig(image_path)
                plt.close()
                mlflow.log_artifact(str(image_path))

            signature = infer_signature(X_test, model.predict(X_test))
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=X_test.head(5),
            )
            results.append({"model": model_name, **metrics})

    results_frame = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(results_frame.to_string(index=False))
    print(f"\nBest model by f1_macro: {results_frame.iloc[0]['model']}")


if __name__ == "__main__":
    main()