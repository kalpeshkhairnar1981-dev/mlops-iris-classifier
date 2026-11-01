"""Tests for preprocessing behavior, including common data edge cases."""

import pandas as pd
import pytest

from src.pipeline.preprocess import preprocess

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


@pytest.fixture
def raw_csv(tmp_path):
    data = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 4.9, 4.7],
            "sepal width (cm)": [3.5, 3.0, 3.2],
            "petal length (cm)": [1.4, 1.4, 1.3],
            "petal width (cm)": [0.2, 0.2, 0.2],
            "species": ["setosa", "setosa", "setosa"],
            "collected_at": ["2026-01-01"] * 3,
        }
    )
    path = tmp_path / "raw.csv"
    data.to_csv(path, index=False)
    return path

def test_preprocess_converts_invalid_numeric_text(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": ["invalid", "5.0", "6.0"],
        "sepal width (cm)": [3.0, 3.1, 3.2],
        "petal length (cm)": [1.0, 1.1, 1.2],
        "petal width (cm)": [0.2, 0.3, 0.4],
        "species": ["setosa", "setosa", "setosa"],
    })

    input_path = tmp_path / "invalid.csv"
    output_path = tmp_path / "output.csv"
    df.to_csv(input_path, index=False)

    result = preprocess(input_path, output_path)

    assert result["sepal length (cm)"].notna().all()
    assert result.loc[0, "sepal length (cm)"] == 5.5


def test_preprocess_creates_output_directory(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [3.0],
        "petal length (cm)": [1.0],
        "petal width (cm)": [0.2],
        "species": ["setosa"],
        "collected_at": ["2026-01-01"],
    })

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "nested" / "folder" / "output.csv"
    df.to_csv(input_path, index=False)

    result = preprocess(input_path, output_path)

    assert output_path.exists()
    assert "collected_at" not in result.columns


def test_preprocess_handles_all_missing_numeric_column(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [None, None],
        "sepal width (cm)": [3.0, 3.2],
        "petal length (cm)": [1.0, 1.2],
        "petal width (cm)": [0.2, 0.3],
        "species": ["setosa", "versicolor"],
    })

    input_path = tmp_path / "missing.csv"
    output_path = tmp_path / "output.csv"
    df.to_csv(input_path, index=False)

    result = preprocess(input_path, output_path)

    assert len(result) == 2
    assert result["sepal length (cm)"].isna().all()
    
def test_preprocess_removes_duplicate_rows(tmp_path):
    import pandas as pd
    from src.pipeline.preprocess import preprocess

    df = pd.DataFrame({
        "sepal length (cm)": [5.0, 5.0, 6.0],
        "sepal width (cm)": [3.0, 3.0, 3.2],
        "petal length (cm)": [1.0, 1.0, 1.2],
        "petal width (cm)": [0.2, 0.2, 0.3],
        "species": ["setosa", "setosa", "versicolor"],
    })

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    df.to_csv(input_path, index=False)

    result = preprocess(input_path, output_path)

    assert len(result) == 2
    assert not result.duplicated().any()


def test_preprocess_removes_rows_with_missing_species(tmp_path):
    import pandas as pd
    from src.pipeline.preprocess import preprocess

    df = pd.DataFrame({
        "sepal length (cm)": [5.0, 6.0, 7.0],
        "sepal width (cm)": [3.0, 3.2, 3.4],
        "petal length (cm)": [1.0, 1.2, 1.4],
        "petal width (cm)": [0.2, 0.3, 0.4],
        "species": ["setosa", None, "versicolor"],
    })

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    df.to_csv(input_path, index=False)

    result = preprocess(input_path, output_path)

    assert len(result) == 2
    assert result["species"].notna().all()

def test_preprocess_accepts_relative_paths(tmp_path, monkeypatch):
    import pandas as pd
    from src.pipeline.preprocess import preprocess

    df = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [3.0],
        "petal length (cm)": [1.0],
        "petal width (cm)": [0.2],
        "species": ["setosa"],
    })

    monkeypatch.setattr("src.pipeline.preprocess.ROOT", tmp_path)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    df.to_csv(input_dir / "iris.csv", index=False)

    result = preprocess(
        "input/iris.csv",
        "output/processed.csv"
    )

    assert len(result) == 1
    assert (tmp_path / "output" / "processed.csv").exists()
