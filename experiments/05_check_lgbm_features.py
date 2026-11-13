from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/grouped"

features_to_check = [
    "Fwd IAT Std",
    "Flow IAT Std",
    "Bwd Header Len",
    "Fwd IAT Min",
    "Dst Port",
    "TotLen Bwd Pkts",
    "Flow Duration",
]

for split in ["train", "validation"]:
    data = pd.read_csv(DATA_DIR / f"{split}_lgbm_grouped.csv")

    print(f"\n========== {split.upper()} ==========")

    for feature in features_to_check:
        values = pd.to_numeric(data[feature], errors="coerce")

        print(f"\n{feature}")
        print(values.describe(percentiles=[0.50, 0.90, 0.99, 0.999]).to_string())
        print("NaN:", values.isna().sum())
        print("Inf:", np.isinf(values).sum())
        print("Above float32 max:", (values.abs() > np.finfo(np.float32).max).sum())