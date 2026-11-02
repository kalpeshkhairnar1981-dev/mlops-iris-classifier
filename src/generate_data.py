from pathlib import Path

from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "iris_v1.csv"
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

iris = load_iris(as_frame=True)
df = iris.frame
df.to_csv(DATA_PATH, index=False)
print(f"Saved {len(df)} rows to {DATA_PATH}")
# Remove the collection timestamp because it isn't a model feature.