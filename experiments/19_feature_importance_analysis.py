import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

# =========================================================
# 1. CONFIGURATION
# =========================================================

SEED = 42
N_REPEATS = 3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "grouped")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

RAW_VAL_PATH = os.path.join(
    DATA_DIR, "validation_lgbm_grouped.csv"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR, "risk_assessment",
    "risk_model_feature_names.joblib"
)

MODEL_PATH = os.path.join(
    MODEL_DIR, "risk_assessment",
    "xgboost_risk_model.joblib"
)

RESULT_PATH = os.path.join(
    REPORT_DIR, "risk_engine_feature_importance.csv"
)

SUMMARY_PATH = os.path.join(
    REPORT_DIR, "risk_engine_feature_importance_summary.json"
)

print("=" * 70)
print("SCRIPT 19 - XGBOOST FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

# =========================================================
# 2. LOAD DATA AND MODEL
# =========================================================

print("\n[1/6] Loading data and model...")

for path in [RAW_VAL_PATH, FEATURE_PATH, MODEL_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

df = pd.read_csv(RAW_VAL_PATH)

feature_cols = list(joblib.load(FEATURE_PATH))
model = joblib.load(MODEL_PATH)

X = df[feature_cols].copy()
y = df["Label"].astype(int).to_numpy()

if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
    raise ValueError("Validation features contain invalid values.")

if set(np.unique(y)) != {0, 1}:
    raise ValueError("Expected binary labels 0 and 1.")

if len(feature_cols) != 68:
    raise ValueError("Expected exactly 68 features.")

print("Validation records:", len(X))
print("Features:", len(feature_cols))
print("Benign records:", int((y == 0).sum()))
print("Malicious records:", int((y == 1).sum()))

# =========================================================
# 3. BASELINE PERFORMANCE
# =========================================================

print("\n[2/6] Calculating baseline performance...")

baseline_probabilities = model.predict_proba(X)[:, 1]

baseline_ap = average_precision_score(
    y,
    baseline_probabilities
)

print("Baseline PR-AUC:", round(baseline_ap, 6))

# =========================================================
# 4. NATIVE XGBOOST GAIN IMPORTANCE
# =========================================================

print("\n[3/6] Extracting native XGBoost gain importance...")

booster = model.get_booster()

gain_scores = booster.get_score(
    importance_type="gain"
)

gain_importance = {}

for key, value in gain_scores.items():

    if key in feature_cols:
        feature_name = key

    elif key.startswith("f") and key[1:].isdigit():
        index = int(key[1:])

        if index >= len(feature_cols):
            continue

        feature_name = feature_cols[index]

    else:
        continue

    gain_importance[feature_name] = float(value)

# Include unused features with zero gain.
gain_values = np.array(
    [gain_importance.get(col, 0.0) for col in feature_cols],
    dtype=np.float64
)

gain_total = gain_values.sum()

if gain_total > 0:
    gain_percent = gain_values / gain_total * 100
else:
    gain_percent = np.zeros(len(feature_cols))

# =========================================================
# 5. PERMUTATION IMPORTANCE
# =========================================================

print("\n[4/6] Running permutation importance...")
print("Scoring metric: Average Precision (PR-AUC)")
print("Repeats per feature:", N_REPEATS)
print("Records used:", len(X))

permutation = permutation_importance(
    estimator=model,
    X=X,
    y=y,
    scoring="average_precision",
    n_repeats=N_REPEATS,
    random_state=SEED,
    n_jobs=1
)

permutation_mean = permutation.importances_mean
permutation_std = permutation.importances_std

# =========================================================
# 6. COMBINE AND SAVE RESULTS
# =========================================================

print("\n[5/6] Combining importance results...")

results = pd.DataFrame({
    "feature": feature_cols,
    "xgboost_gain": gain_values,
    "xgboost_gain_percent": gain_percent,
    "permutation_ap_mean": permutation_mean,
    "permutation_ap_std": permutation_std
})

results["gain_rank"] = (
    results["xgboost_gain"]
    .rank(method="min", ascending=False)
    .astype(int)
)

results["permutation_rank"] = (
    results["permutation_ap_mean"]
    .rank(method="min", ascending=False)
    .astype(int)
)

results["importance_rank_average"] = (
    results[["gain_rank", "permutation_rank"]]
    .mean(axis=1)
)

results = results.sort_values(
    "importance_rank_average",
    ascending=True
).reset_index(drop=True)

results.to_csv(
    RESULT_PATH,
    index=False
)

summary = {
    "experiment": "XGBoost Risk Engine Feature Importance",
    "validation_only": True,
    "records": int(len(X)),
    "features": int(len(feature_cols)),
    "benign_records": int((y == 0).sum()),
    "malicious_records": int((y == 1).sum()),
    "baseline_validation_pr_auc": float(baseline_ap),
    "permutation_metric": "average_precision",
    "permutation_repeats": N_REPEATS,
    "native_importance_type": "gain",
    "top_10_combined": results.head(10)["feature"].tolist(),
    "interpretation_note": (
        "Feature importance is descriptive, not causal. "
        "Correlated features can share or distort importance. "
        "Validation-set permutation importance is not independent "
        "test-set evidence."
    )
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("\n[6/6] TOP 15 FEATURES")
print("=" * 70)

print(
    results[
        [
            "feature",
            "xgboost_gain_percent",
            "permutation_ap_mean",
            "permutation_ap_std",
            "gain_rank",
            "permutation_rank"
        ]
    ].head(15).to_string(index=False)
)

print("\nSaved reports:")
print(RESULT_PATH)
print(SUMMARY_PATH)

print("\nSCRIPT 19 COMPLETED SUCCESSFULLY")
print("=" * 70)

