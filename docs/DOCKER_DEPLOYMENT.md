# Practical 10: Docker Containerization

## Overview

The Docker image contains a trained Iris classifier and a small HTTP server.
The multi-stage build runs the project's collect, preprocess, and feature
engineering stages, then trains the model in a builder stage. The runtime
stage contains only the pinned inference dependencies, the model artifacts,
and `src/serve.py`; training-only pandas, source data, and build tools are not
copied into it.

The build starts from the scikit-learn Iris dataset, so it does not depend on
an ignored local CSV, a DVC remote, or a model artifact already in `models/`.
The builder and runtime use the same pinned scikit-learn, NumPy, SciPy, and
joblib versions to keep serialized model loading consistent.

## Prerequisites

- Docker Desktop with the Linux container engine running (or Docker Engine on
  Linux).
- Python 3.11 is used inside the image.
- The repository root as the Docker build context.

Verify Docker is available:

```powershell
docker --version
docker info
```

## Build the image

Run these commands from the repository root. The optional builder-stage image
provides a size comparison: it retains training dependencies and intermediate
data, unlike the default runtime image.

```powershell
docker build --target builder -t iris-classifier:builder-reference .
docker build -t iris-classifier:v1 .
docker image inspect --format='Builder reference: {{.Size}} bytes' iris-classifier:builder-reference
docker image inspect --format='Runtime image: {{.Size}} bytes' iris-classifier:v1
docker history iris-classifier:v1
```

The GitHub Actions `container-build-test` job runs both builds and reports
their sizes in the workflow logs. Image size depends on the platform and
binary dependency wheels; use the measured output rather than assuming a
specific size. The first GitHub Actions build reported 473,896,230 bytes for
the builder reference and 396,409,906 bytes for the runtime image, a reduction
of about 77.5 MB (16%). These images are hundreds of megabytes rather than
tens of megabytes because scikit-learn and its NumPy/SciPy dependencies are
large binary packages. See [the verified Actions run](https://github.com/kalpeshkhairnar1981-dev/mlops-iris-classifier/actions/runs/41428272061)
for the build and endpoint checks.

## Run and inspect the container

```powershell
docker run --detach --name iris-classifier-container --publish 8000:8000 iris-classifier:v1
docker ps
docker inspect --format='{{.State.Health.Status}}' iris-classifier-container
```

The image runs as a non-root user and has a Docker health check that calls
`/health`. Wait for the health status to become `healthy` before sending
prediction requests.

## Test the HTTP API

Health check:

```powershell
curl.exe --fail http://localhost:8000/health
```

Expected response:

```json
{"status": "healthy"}
```

Prediction request:

```powershell
curl.exe --fail --request POST http://localhost:8000/predict `
  --header "Content-Type: application/json" `
  --data '{"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2,"sepal_area":17.85,"petal_area":0.28,"sepal_to_petal_length_ratio":3.642857}'
```

The response contains a species prediction and probabilities for all three
classes. Exact probabilities can vary with the installed library build.

Check input validation (this deliberately returns HTTP 400):

```powershell
curl.exe --include --request POST http://localhost:8000/predict `
  --header "Content-Type: application/json" `
  --data '{"sepal length (cm)":5.1}'
```

The JSON response lists the missing feature names. Unknown routes return 404;
malformed JSON, missing/non-numeric features, and non-finite values are
rejected.

## Automated verification

`tests/test_container_app.py` checks model artifact creation and exercises the
health and prediction endpoints, including invalid input. The GitHub Actions
`container-build-test` job additionally builds both image stages, runs the
runtime image, waits for its health check, and verifies valid and invalid
requests. It prints the two image sizes to the Actions log.

## Logs and cleanup

```powershell
docker logs iris-classifier-container
docker stop iris-classifier-container
docker rm iris-classifier-container
```

The image remains available locally after the container is removed. Remove it
only when no longer needed:

```powershell
docker image rm iris-classifier:v1 iris-classifier:builder-reference
```
