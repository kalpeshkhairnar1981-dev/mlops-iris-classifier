"""HTTP health and prediction endpoints for the containerized Iris classifier."""

import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_model.joblib"
ENCODER_PATH = ROOT / "models" / "label_encoder.joblib"
FEATURE_ORDER = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]
MAX_CONTENT_LENGTH = 16_384


def create_handler(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
) -> type[BaseHTTPRequestHandler]:
    class PredictionHandler(BaseHTTPRequestHandler):
        def _respond(self, status: int, payload: dict[str, Any]) -> None:
            response = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(response)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._respond(200, {"status": "healthy"})
            else:
                self._respond(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._respond(404, {"error": "not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._respond(400, {"error": "Invalid Content-Length"})
                return
            if content_length <= 0:
                self._respond(400, {"error": "Request body is required"})
                return
            if content_length > MAX_CONTENT_LENGTH:
                self._respond(413, {"error": "Request body is too large"})
                return

            try:
                body = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._respond(400, {"error": "Invalid JSON"})
                return
            if not isinstance(body, dict):
                self._respond(400, {"error": "JSON body must be an object"})
                return

            missing = [name for name in FEATURE_ORDER if name not in body]
            if missing:
                self._respond(400, {"error": "Missing features", "missing": missing})
                return

            raw_values = [body[name] for name in FEATURE_ORDER]
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in raw_values
            ):
                self._respond(400, {"error": "Features must be finite numbers"})
                return

            try:
                values = [float(value) for value in raw_values]
            except OverflowError:
                self._respond(400, {"error": "Features must be finite numbers"})
                return
            if any(not math.isfinite(value) for value in values):
                self._respond(400, {"error": "Features must be finite numbers"})
                return

            features = np.asarray([values])
            prediction_index = int(model.predict(features)[0])
            prediction = str(label_encoder.inverse_transform([prediction_index])[0])
            probabilities = model.predict_proba(features)[0].tolist()
            self._respond(
                200,
                {"prediction": prediction, "probabilities": probabilities},
            )

    return PredictionHandler


def main() -> None:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    server = HTTPServer(("0.0.0.0", 8000), create_handler(model, label_encoder))
    print("Iris server listening on port 8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
