# Continuous Integration Pipeline

## Overview

The GitHub Actions workflow in `.github/workflows/ci.yml` runs on pushes to
`main` and `develop`, and on pull requests targeting either branch. It compiles
and lints the Python source, runs the test suite, produces test and coverage
reports, and checks the model's minimum accuracy.

## Jobs and dependencies

The jobs run sequentially through their `needs` relationships:

```text
build-and-lint
      |
      v
  test-suite
      |
      v
model-quality-gate
```

### `build-and-lint`

Runs on Python 3.11. It installs the project requirements and Flake8, compiles
`src` and `tests`, then lints `src` with a 100-character line limit. The job
must pass before the test suite starts.

### `test-suite`

Runs after `build-and-lint`. It collects the Iris data and runs the
preprocessing and feature-engineering stages before executing all tests under
`tests/`. Pytest writes `test-results.xml` and `coverage.xml`; the workflow
uploads these as the `test-and-coverage-reports` artifact even when the test
step fails, if the files were created.

### `model-quality-gate`

Runs after `test-suite`. It executes `tests/test_model_quality.py`, which
exercises the data pipeline and trains a Random Forest classifier. The test
fails if holdout accuracy is below `0.90`.

## Enforcing checks on `main`

Branch protection is configured in the GitHub repository settings, not in the
workflow file. To prevent unverified changes from being merged into `main`,
create an active branch ruleset targeting `main`, require pull requests, and
require these status checks:

- `build-and-lint`
- `test-suite`
- `model-quality-gate`

GitHub may display a workflow-qualified check name; select the names shown for
the workflow's jobs. Once enabled, verify the checks appear on a pull request
targeting `main`. The ruleset must be configured in GitHub by a repository
administrator.

## Demonstrating the model-quality gate

Perform this demonstration only on a temporary feature branch with a pull
request targeting `main`; do not commit the test change to `main`.

1. Change `MIN_ACCEPTABLE_ACCURACY` in `tests/test_model_quality.py` to a value
   greater than `1.0` (for example, `1.01`).
2. Push the branch and open or update its pull request.
3. Confirm that `model-quality-gate` fails at the accuracy assertion and that
   the required check prevents the pull request from being merged.
4. Restore the threshold to `0.90`, push the correction, and confirm the jobs
   pass before merging.

This is a verification procedure, not a record that the demonstration has
already been run. Record the pull request and Actions run in the project
history when the demonstration is performed.
