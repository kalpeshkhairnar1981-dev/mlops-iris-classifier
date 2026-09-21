# Data Pipeline Workflow

## Overview

This project uses a four-stage DVC pipeline for the Iris dataset:

collect -> preprocess -> features -> validate

## Stage 1: collect

- Purpose: Ingest raw Iris data from sklearn and save it in the raw data zone.
- Input: none (generated from source library)
- Output: `data/raw/iris_raw.csv`
- Notes: Adds a `collected_at` timestamp for provenance.

## Stage 2: preprocess

- Purpose: Clean the raw data before feature engineering.
- Input: `data/raw/iris_raw.csv`
- Output: `data/processed/iris_preprocessed.csv`
- Checks:
  - drop duplicate rows
  - coerce numeric columns to numeric dtype
  - impute missing values using median
  - drop rows with missing species values
  - drop the `collected_at` column

## Stage 3: features

- Purpose: Create model-ready derived features.
- Input: `data/processed/iris_preprocessed.csv`
- Output: `data/processed/iris_features.csv`
- Features created:
  - `sepal_area`
  - `petal_area`
  - `sepal_to_petal_length_ratio`
  - `petal_length_bin`

## Stage 4: validate

- Purpose: Enforce schema and range checks before data enters training.
- Input: `data/processed/iris_features.csv`
- Output: none; it raises an error if invalid data is detected.
- Validation rules:
  - required columns must exist
  - no null values allowed
  - species must be one of setosa, versicolor, virginica
  - sepal length should be between 3.0 and 9.0
  - sepal width should be between 1.5 and 5.5
  - petal length should be between 0.5 and 8.0
  - petal width should be between 0.05 and 3.0

## Dependency graph

```text
collect
  -> preprocess
      -> features
          -> validate
```

This is defined in `dvc.yaml`, and DVC uses the declared dependencies to decide whether each stage should rerun.
