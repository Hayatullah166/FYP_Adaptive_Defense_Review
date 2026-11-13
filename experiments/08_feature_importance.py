"""
FYP Feature Importance Analysis
Compares feature importance for:
1. Random Forest
2. XGBoost
3. LightGBM with Early Stopping

Uses saved models only. Does not retrain models.
"""

from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "risk_assessment"
REPORT_DIR = ROOT / "data" / "reports"

MODEL_FILES = {
    "RandomForest": MODEL_DIR / "randomforest_risk_model.joblib",
    "XGBoost": MODEL_DIR / "xgboost_risk_model.joblib",
    "LightGBM": MODEL_DIR / "lightgbm_early_stopping.joblib",
}


def get_importance(model, model_name):

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            f"{model_name} does not provide feature_importances_."
        )

    return model.feature_importances_


def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    feature_names_path = (
        MODEL_DIR / "risk_model_feature_names.joblib"
    )

    if not feature_names_path.exists():
        raise FileNotFoundError(feature_names_path)

    feature_names = joblib.load(feature_names_path)

    importance_data = {
        "Feature": feature_names
    }

    for model_name, model_path in MODEL_FILES.items():

        print(f"\nLoading {model_name}...")

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        model = joblib.load(model_path)

        importance = get_importance(model, model_name)

        if len(importance) != len(feature_names):
            raise ValueError(
                f"Feature count mismatch in {model_name}."
            )

        importance_data[model_name] = importance

    importance_df = pd.DataFrame(importance_data)

    for model_name in MODEL_FILES:
        importance_df[model_name + "_Rank"] = (
            importance_df[model_name]
            .rank(ascending=False, method="min")
            .astype(int)
        )

    importance_df.to_csv(
        REPORT_DIR / "feature_importance_all_models.csv",
        index=False,
    )

    print("\n========== TOP 15 FEATURES ==========")

    for model_name in MODEL_FILES:

        print(f"\n{model_name}")

        top_features = (
            importance_df[
                ["Feature", model_name, model_name + "_Rank"]
            ]
            .sort_values(model_name, ascending=False)
            .head(15)
        )

        print(top_features.to_string(index=False))

    print(
        "\nFull report saved:",
        REPORT_DIR / "feature_importance_all_models.csv"
    )


if __name__ == "__main__":
    main()