"""Register the highest-F1 run from the baseline MLflow experiment."""

import os

import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "iris-classification-baseline"
REGISTERED_MODEL_NAME = "iris-classifier-prod"


def main() -> None:
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment {EXPERIMENT_NAME!r} was not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.f1_macro DESC"],
        max_results=1000,
    )
    runs = [run for run in runs if "f1_macro" in run.data.metrics]
    if not runs:
        raise RuntimeError(f"No completed runs with f1_macro in {EXPERIMENT_NAME!r}.")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        stage="Staging",
    )

    print(f"Best run: {best_run.info.run_id}")
    print(f"Model type: {best_run.data.params.get('model_type', 'unknown')}")
    print(f"f1_macro: {best_run.data.metrics['f1_macro']:.4f}")
    print(f"Registered: {REGISTERED_MODEL_NAME} version {model_version.version}")
    print(f"Model URI: models:/{REGISTERED_MODEL_NAME}/Staging")


if __name__ == "__main__":
    main()