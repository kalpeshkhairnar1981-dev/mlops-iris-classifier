# mlops-iris-classifier — Version A + B (resolved)

A sample ML project used to demonstrate Git-based version control
workflows in an MLOps context.

## Practical 6: MLflow Tracking and Model Registry

Install the project requirements in the Python environment you plan to use:

```bash
python -m pip install -r requirements.txt
```

In one terminal, start the MLflow tracking server from the repository root:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

In a second terminal, from the same repository root, point the scripts at the
server and run the tracked experiments:

```bash
# Git Bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/train_with_mlflow.py
python src/register_best_model.py
python src/load_registered_model.py
```

On Windows Command Prompt, set the tracking URI with
`set MLFLOW_TRACKING_URI=http://127.0.0.1:5000`; in PowerShell, use
`$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"`.

The training script logs three runs to `iris-classification-baseline`, including
metrics, confusion-matrix images, and signature-bearing scikit-learn model
artifacts. The registration script selects the highest `f1_macro` run and
registers it as `iris-classifier-prod` in the `Staging` stage. The final script
loads that staged model and runs inference against a row from the feature CSV.
Open `http://127.0.0.1:5000` to compare runs and inspect the registry. The
server must remain running while the training and registry scripts execute.

## Setup

```bash
pip install -r requirements.txt
python src/train.py
```
