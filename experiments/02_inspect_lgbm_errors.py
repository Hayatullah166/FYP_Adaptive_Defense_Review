from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models/risk_assessment"
DATA_FILE = ROOT / "data/processed/grouped/validation_lgbm_grouped.csv"
SCORE_FILE = ROOT / "data/reports/risk_model_validation_predictions.csv"
OUTPUT_FILE = ROOT / "data/reports/lightgbm_prediction_diagnostics.csv"


def main():
    model = joblib.load(MODEL_DIR / "lightgbm_risk_model.joblib")
    saved_features = joblib.load(
        MODEL_DIR / "risk_model_feature_names.joblib"
    )

    data = pd.read_csv(DATA_FILE)
    scores = pd.read_csv(SCORE_FILE)

    X = data[saved_features].copy()

    if len(data) != len(scores):
        raise ValueError("Validation and prediction rows do not match.")

    if not np.array_equal(
        data["Label"].astype(int).to_numpy(),
        scores["Actual_Label"].astype(int).to_numpy()
    ):
        raise ValueError("Labels do not match.")

    fresh_scores = model.predict_proba(X)[:, 1]
    saved_scores = scores["LightGBM_Risk_Score"].to_numpy()

    print("Model classes:", model.classes_)
    print("Model feature count:", len(saved_features))
    print("Validation feature count:", X.shape[1])

    print(
        "Feature names match:",
        list(X.columns) == list(saved_features)
    )

    print(
        "Saved and fresh predictions match:",
        np.allclose(saved_scores, fresh_scores)
    )

    print(
        "Maximum prediction difference:",
        np.max(np.abs(saved_scores - fresh_scores))
    )

    result = data.copy()
    result["Saved_Score"] = saved_scores
    result["Fresh_Score"] = fresh_scores

    missed = result[
        (result["Label"] == 1) &
        (result["Fresh_Score"] < 0.5)
    ]

    false_positives = result[
        (result["Label"] == 0) &
        (result["Fresh_Score"] >= 0.5)
    ]

    print("\nFresh prediction results:")
    print("Missed attacks:", len(missed))
    print("High-risk benign:", len(false_positives))

    print("\nFresh score distribution:")
    print(
        result.groupby("Label")["Fresh_Score"]
        .describe().to_string()
    )

    print("\nMissed attack score counts:")
    print(missed["Fresh_Score"].value_counts().head(15).to_string())

    result.to_csv(OUTPUT_FILE, index=False)
    print("\nDiagnostic report saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()