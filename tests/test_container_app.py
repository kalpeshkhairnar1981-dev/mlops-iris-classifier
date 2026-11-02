"""Tests for final model training and the container prediction API."""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

from src.serve import FEATURE_ORDER, create_handler
from src.train_final_model import train_final_model


class StubModel:
    def predict(self, features):
        assert features.shape == (1, len(FEATURE_ORDER))
        return np.array([0])

    def predict_proba(self, features):
        return np.array([[1.0, 0.0, 0.0]])


@pytest.fixture
def api_url():
    label_encoder = LabelEncoder().fit(["setosa", "versicolor", "virginica"])
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        create_handler(StubModel(), label_encoder),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def _request(api_url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{api_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def _valid_payload():
    return {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2,
        "sepal_area": 17.85,
        "petal_area": 0.28,
        "sepal_to_petal_length_ratio": 3.642857,
    }


def test_health_endpoint_returns_healthy_status(api_url):
    status, payload = _request(api_url, "/health")

    assert status == 200
    assert payload == {"status": "healthy"}


def test_prediction_endpoint_returns_species_and_probabilities(api_url):
    status, payload = _request(api_url, "/predict", _valid_payload())

    assert status == 200
    assert payload["prediction"] == "setosa"
    assert len(payload["probabilities"]) == 3
    assert sum(payload["probabilities"]) == pytest.approx(1.0)


def test_prediction_endpoint_rejects_missing_features(api_url):
    status, payload = _request(api_url, "/predict", {"sepal length (cm)": 5.1})

    assert status == 400
    assert payload["error"] == "Missing features"
    assert payload["missing"] == FEATURE_ORDER[1:]


def test_prediction_endpoint_rejects_non_numeric_features(api_url):
    payload = _valid_payload()
    payload["petal width (cm)"] = "small"

    status, response = _request(api_url, "/predict", payload)

    assert status == 400
    assert response == {"error": "Features must be finite numbers"}


def test_final_model_training_saves_reloadable_artifacts(tmp_path):
    iris = load_iris(as_frame=True)
    frame = iris.frame.rename(columns={"target": "target_index"})
    frame["species"] = frame["target_index"].map(dict(enumerate(iris.target_names)))
    frame["sepal_area"] = (
        frame["sepal length (cm)"] * frame["sepal width (cm)"]
    )
    frame["petal_area"] = (
        frame["petal length (cm)"] * frame["petal width (cm)"]
    )
    frame["sepal_to_petal_length_ratio"] = (
        frame["sepal length (cm)"] / frame["petal length (cm)"]
    )
    data_path = tmp_path / "iris_features.csv"
    model_dir = tmp_path / "models"
    frame.to_csv(data_path, index=False)

    model, label_encoder, row_count = train_final_model(data_path, model_dir)

    saved_model = joblib.load(model_dir / "final_model.joblib")
    saved_encoder = joblib.load(model_dir / "label_encoder.joblib")
    prediction = saved_encoder.inverse_transform(
        [saved_model.predict(frame[FEATURE_ORDER])[0]]
    )[0]

    assert (model_dir / "final_model.joblib").exists()
    assert (model_dir / "label_encoder.joblib").exists()
    assert row_count == len(frame)
    assert set(label_encoder.classes_) == {"setosa", "versicolor", "virginica"}
    assert prediction in label_encoder.classes_
