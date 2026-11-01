"""
Stage 3: Feature Engineering.
Derives new, model-useful features from the raw measurements.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("features")

ROOT = Path(__file__).resolve().parents[2]


def engineer_features(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    input_file = Path(input_path)
    output_file = Path(output_path)
    if not input_file.is_absolute():
        input_file = ROOT / input_file
    if not output_file.is_absolute():
        output_file = ROOT / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)
    df["sepal_area"] = df["sepal length (cm)"] * df["sepal width (cm)"]
    df["petal_area"] = df["petal length (cm)"] * df["petal width (cm)"]
    df["sepal_to_petal_length_ratio"] = (
        df["sepal length (cm)"] / df["petal length (cm)"].replace(0, pd.NA)
    )
    df["petal_length_bin"] = pd.cut(
        df["petal length (cm)"],
        bins=[0, 2, 4.5, 7],
        labels=["short", "medium", "long"],
    )
    df.to_csv(output_file, index=False)
    logger.info("Engineered %d features -> %s", df.shape[1], output_file)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/iris_preprocessed.csv")
    parser.add_argument("--output", default="data/processed/iris_features.csv")
    args = parser.parse_args()
    engineer_features(args.input, args.output)
