from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

model = joblib.load(
    ROOT / "models/risk_assessment/lightgbm_early_stopping.joblib"
)

features = joblib.load(
    ROOT / "models/risk_assessment/risk_model_feature_names.joblib"
)

data_dir = ROOT / "data/processed/grouped"

for split in ["train", "validation"]:
    file_path = data_dir / f"{split}_lgbm_grouped.csv"
    data = pd.read_csv(file_path)

    X = data[features]
    scores = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "Label": data["Label"],
        "Score": scores
    })

    print(f"\n===== {split.upper()} =====")
    print("Rows:", len(data))
    print("Actual trees:", model.booster_.num_trees())
    print("Unique scores:", result["Score"].nunique())

    print("\nScores by actual label:")
    print(result.groupby("Label")["Score"].describe().to_string())

    print("\nMost common scores:")
    print(result["Score"].value_counts().head(10).to_string())
