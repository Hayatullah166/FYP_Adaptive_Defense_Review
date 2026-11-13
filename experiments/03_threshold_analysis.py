from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models/risk_assessment"
DATA_FILE = ROOT / "data/processed/grouped/validation_lgbm_grouped.csv"
OUTPUT_FILE = ROOT / "data/reports/lightgbm_threshold_analysis.csv"

def main():
    model = joblib.load(MODEL_DIR / "lightgbm_early_stopping.joblib")
    feature_names = joblib.load(
        MODEL_DIR / "risk_model_feature_names.joblib"
    )

    data = pd.read_csv(DATA_FILE)
    X = data[feature_names]
    y = data["Label"].astype(int)

    scores = model.predict_proba(X)[:, 1]

    results = []

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        predictions = (scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "Threshold": threshold,
            "Precision": precision_score(
                y, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y, predictions, zero_division=0
            ),
            "F1": f1_score(
                y, predictions, zero_division=0
            ),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

    report = pd.DataFrame(results)
    report.to_csv(OUTPUT_FILE, index=False)

    print(report.to_string(index=False))
    print("\nReport saved:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
