
"""
Stage 2: Data Preprocessing.
Handles missing values, duplicate removal, and type correction.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess")

ROOT = Path(__file__).resolve().parents[2]
NUMERIC_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def preprocess(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    input_file = Path(input_path)
    output_file = Path(output_path)
    if not input_file.is_absolute():
        input_file = ROOT / input_file
    if not output_file.is_absolute():
        output_file = ROOT / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)
    initial_rows = len(df)

    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", initial_rows - len(df))

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Imputed %d missing values in '%s' with median=%.3f",
                        n_missing, col, median_val)

    df = df.dropna(subset=["species"])
    df.drop(columns=["collected_at"], inplace=True, errors="ignore")
    df.to_csv(output_file, index=False)
    logger.info("Preprocessed %d rows -> %s", len(df), output_file)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/iris_raw.csv")
    parser.add_argument("--output", default="data/processed/iris_preprocessed.csv")
    args = parser.parse_args()
    preprocess(args.input, args.output)
