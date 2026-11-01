# Testing Strategy

Run the complete suite from the repository root:

```bash
python -m pytest --cov=src --cov-report=term-missing --junitxml=test-results.xml
```

The suite is independent of the MLflow server and uses temporary directories
for generated CSVs and model input data.

| Test file                         | Layer                              | Coverage                                                                                                |
| --------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `tests/test_preprocess.py`        | Data processing                    | Duplicate removal, median imputation, missing targets, empty input, and output creation.                |
| `tests/test_features.py`          | Data processing                    | Derived areas, length ratios, length bins, zero-length handling, and output creation.                   |
| `tests/test_validate.py`          | Data validation                    | Valid rows, missing schema, invalid labels, out-of-range values, and nulls.                             |
| `tests/test_training_pipeline.py` | Training pipeline and model output | Train/test split, model fitting, prediction shape and labels, probability validity, evaluation outputs. |
| `tests/test_model_quality.py`     | End-to-end model quality           | Collection through validation, then a stratified holdout accuracy gate of at least 0.90.                |

The current run passes **18 tests** with **45%** full-source coverage
(`--cov=src`), with the
pipeline modules covered at 76% for feature engineering, 82% for preprocessing,
80% for validation, 81% for data collection, and 75% for baseline training.
The total includes unrelated MLflow scripts and command-line entry points that
the suite intentionally does not execute.

# Experiment 8: Testing Strategy

## Objective

To test data processing functions, the training pipeline, and model outputs using pytest and pytest-cov.

## Test Files

| Test File                 | Testing Layer | Purpose                                                                                   |
| ------------------------- | ------------- | ----------------------------------------------------------------------------------------- |
| test_preprocess.py        | Data          | Tests duplicate removal, missing-value imputation, missing targets, and empty data.       |
| test_features.py          | Data          | Tests feature calculations, petal length categories, and zero-division handling.          |
| test_validate.py          | Data          | Tests valid data, missing columns, invalid species, out-of-range values, and null values. |
| test_training_pipeline.py | Pipeline      | Tests model training, predictions, output shapes, evaluation, and probabilities.          |
| test_model_quality.py     | Model Quality | Checks prediction labels and minimum acceptable accuracy.                                 |

## Test Results

- Total tests executed: 18
- Tests passed: 18
- Tests failed: 0
- Execution time: 25.44 seconds
- Overall source-code coverage: 45%
- JUnit XML report: test-results.xml

## Module Coverage

- preprocess.py: 82%
- features.py: 76%
- validate.py: 80%

## Conclusion

All 18 collected automated tests passed successfully. Pytest generated the JUnit XML report and coverage report. Further tests are required to meet the specified coverage target of above 85% for the preprocessing, feature engineering, and validation modules.
