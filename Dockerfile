FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-serving.txt .
RUN python -m pip install --no-cache-dir -r requirements-serving.txt \
    && python -m pip install --no-cache-dir pandas==2.3.3

COPY src/pipeline/ ./src/pipeline/
COPY src/train_final_model.py src/serve.py ./src/
RUN python src/pipeline/collect.py --output data/raw/iris_raw.csv \
    && python src/pipeline/preprocess.py \
        --input data/raw/iris_raw.csv \
        --output data/processed/iris_preprocessed.csv \
    && python src/pipeline/features.py \
        --input data/processed/iris_preprocessed.csv \
        --output data/processed/iris_features.csv \
    && python src/train_final_model.py

CMD ["python", "src/serve.py"]

FROM python:3.11-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-serving.txt .
RUN python -m pip install --no-cache-dir -r requirements-serving.txt

COPY --from=builder /app/models/ ./models/
COPY src/serve.py ./src/serve.py

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["python", "src/serve.py"]
