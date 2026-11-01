"""Tests for successful and failing data validation."""

import pandas as pd
import pytest

from src.pipeline.validate import DataValidationError, validate

VALID_ROW = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
    "species": "setosa",
    "sepal_area": 17.85,
    "petal_area": 0.28,
    "sepal_to_petal_length_ratio": 3.64,
    "petal_length_bin": "short",
}


def _write_csv(tmp_path, data):
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)
    return path


def test_validate_accepts_clean_data(tmp_path):
    path = _write_csv(tmp_path, pd.DataFrame([VALID_ROW, VALID_ROW]))

    result = validate(path)

    assert len(result) == 2


def test_validate_reports_missing_columns_without_key_error(tmp_path):
    data = pd.DataFrame([VALID_ROW]).drop(columns=["species"])
    path = _write_csv(tmp_path, data)

    with pytest.raises(DataValidationError, match="Missing expected columns"):
        validate(path)


def test_validate_rejects_unrecognized_species(tmp_path):
    row = {**VALID_ROW, "species": "unicorn"}
    path = _write_csv(tmp_path, pd.DataFrame([row]))

    with pytest.raises(DataValidationError, match="Unexpected species values"):
        validate(path)


def test_validate_rejects_out_of_range_measurements(tmp_path):
    row = {**VALID_ROW, "sepal length (cm)": 12.0}
    path = _write_csv(tmp_path, pd.DataFrame([row]))

    with pytest.raises(DataValidationError, match="out of expected range"):
        validate(path)


def test_validate_rejects_null_values(tmp_path):
    row = {**VALID_ROW, "petal_area": None}
    path = _write_csv(tmp_path, pd.DataFrame([row]))

    with pytest.raises(DataValidationError, match="Unexpected null values"):
        validate(path)
