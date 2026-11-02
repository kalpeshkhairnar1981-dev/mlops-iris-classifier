"""Train and save the Iris classifier for container deployment."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "iris_features.csv"
MODEL_DIR = ROOT / "models"
FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]


def train_final_model(
    data_path: str | Path = DATA_PATH,
    model_dir: str | Path = MODEL_DIR,
) -> tuple[RandomForestClassifier, LabelEncoder, int]:
    data = pd.read_csv(data_path)
    required_columns = [*FEATURE_COLUMNS, "species"]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if data.empty:
        raise ValueError("Dataset must contain at least one row")

    features = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    medians = features.median()
    if medians.isna().any():
        raise ValueError("Feature columns must contain at least one numeric value")
    features = features.fillna(medians)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["species"])
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )
    model.fit(features, labels)

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "final_model.joblib")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")
    return model, label_encoder, len(data)


def main() -> None:
    _, label_encoder, row_count = train_final_model()
    print("Training completed.")
    print(f"Rows used: {row_count}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Saved model and label encoder to {MODEL_DIR}")


if __name__ == "__main__":
    main()
