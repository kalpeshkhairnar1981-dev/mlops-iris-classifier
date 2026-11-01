"""Tests for feature calculations and edge cases."""

import pandas as pd
import pytest

from src.pipeline.features import engineer_features


def _write_input(tmp_path, petal_lengths):
    data = pd.DataFrame(
        {
            "sepal length (cm)": [5.0] * len(petal_lengths),
            "sepal width (cm)": [4.0] * len(petal_lengths),
            "petal length (cm)": petal_lengths,
            "petal width (cm)": [1.0] * len(petal_lengths),
            "species": ["setosa"] * len(petal_lengths),
        }
    )
    input_path = tmp_path / "input.csv"
    data.to_csv(input_path, index=False)
    return input_path


def test_areas_and_length_ratio_are_computed(tmp_path):
    input_path = _write_input(tmp_path, [2.0])
    output_path = tmp_path / "nested" / "features.csv"

    result = engineer_features(input_path, output_path)

    assert result.loc[0, "sepal_area"] == pytest.approx(20.0)
    assert result.loc[0, "petal_area"] == pytest.approx(2.0)
    assert result.loc[0, "sepal_to_petal_length_ratio"] == pytest.approx(2.5)
    assert output_path.exists()


def test_petal_length_bin_assigns_expected_categories(tmp_path):
    input_path = _write_input(tmp_path, [1.0, 3.0, 6.0])

    result = engineer_features(input_path, tmp_path / "features.csv")

    assert result["petal_length_bin"].astype(str).tolist() == [
        "short",
        "medium",
        "long",
    ]


def test_zero_petal_length_produces_missing_ratio(tmp_path):
    input_path = _write_input(tmp_path, [0.0])

    result = engineer_features(input_path, tmp_path / "features.csv")

    assert pd.isna(result.loc[0, "sepal_to_petal_length_ratio"])
