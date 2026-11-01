from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "iris_v1.csv"
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
rng = np.random.default_rng(42)
noise = rng.normal(0, 0.05, size=(20, 4))
sample = df.sample(20, random_state=42).reset_index(drop=True)
sample.iloc[:, :4] = sample.iloc[:, :4].values + noise
augmented = pd.concat([df, sample], ignore_index=True)
augmented.to_csv(DATA_PATH, index=False)
print(f"Dataset now has {len(augmented)} rows")
