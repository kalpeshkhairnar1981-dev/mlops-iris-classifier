"""Smoke tests for fitting a model and validating its prediction outputs."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.train import evaluate_model, load_data, train_model


@pytest.fixture
def fitted_model_and_test_data():
    features, labels = make_classification(
        n_samples=90,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = train_model(x_train, y_train, n_estimators=20, max_depth=5)
    return model, x_test, y_test


def test_training_returns_predictive_model(fitted_model_and_test_data):
    model, _, _ = fitted_model_and_test_data

    assert callable(model.predict)


def test_load_data_returns_expected_train_test_split():
    x_train, x_test, y_train, y_test = load_data()

    assert len(x_train) == len(y_train) == 120
    assert len(x_test) == len(y_test) == 30


def test_predictions_have_expected_shape_and_labels(fitted_model_and_test_data):
    model, x_test, y_test = fitted_model_and_test_data

    predictions = model.predict(x_test)

    assert predictions.shape == y_test.shape
    assert set(predictions).issubset({0, 1, 2})


def test_evaluate_model_returns_accuracy_and_report(
    fitted_model_and_test_data, capsys
):
    model, x_test, y_test = fitted_model_and_test_data

    accuracy, report = evaluate_model(model, x_test, y_test)

    assert 0.0 <= accuracy <= 1.0
    assert "precision" in report
    assert "precision" in capsys.readouterr().out


def test_predicted_probabilities_are_valid(fitted_model_and_test_data):
    model, x_test, _ = fitted_model_and_test_data

    probabilities = model.predict_proba(x_test)

    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
