"""End-to-end model quality gate using the Iris data pipeline."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.pipeline.collect import collect_data
from src.pipeline.features import engineer_features
from src.pipeline.preprocess import preprocess
from src.pipeline.validate import validate

MIN_ACCEPTABLE_ACCURACY = 0.90
FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]


def test_model_meets_minimum_accuracy_threshold(tmp_path):
    raw_path = tmp_path / "raw" / "iris.csv"
    preprocessed_path = tmp_path / "processed" / "iris.csv"
    features_path = tmp_path / "features" / "iris.csv"

    collect_data(raw_path)
    preprocess(raw_path, preprocessed_path)
    engineer_features(preprocessed_path, features_path)
    data = validate(features_path)

    labels = LabelEncoder().fit_transform(data["species"])
    features = data[FEATURE_COLUMNS].fillna(data[FEATURE_COLUMNS].median())
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))

    assert accuracy >= MIN_ACCEPTABLE_ACCURACY
