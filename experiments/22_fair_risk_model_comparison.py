from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

ROOT = Path(".")
MODEL_DIR = ROOT / "models/risk_assessment"
DATA_DIR = ROOT / "data/processed/grouped"
REPORT_DIR = ROOT / "data/reports"

FEATURE_FILE = MODEL_DIR / "risk_model_feature_names.joblib"

features = joblib.load(FEATURE_FILE)

train = pd.read_csv(DATA_DIR / "train_lgbm_grouped.csv")
valid = pd.read_csv(DATA_DIR / "validation_lgbm_grouped.csv")
test = pd.read_csv(DATA_DIR / "test_lgbm_grouped.csv")

X_valid = valid[features]
y_valid = valid["Label"].astype(int)

print("=" * 75)
print("DATASET SPLIT AUDIT")
print("=" * 75)

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", valid),
    ("TEST", test),
]:
    print(
        f"{name}: rows={len(df)}, "
        f"benign={(df['Label'] == 0).sum()}, "
        f"attacks={(df['Label'] == 1).sum()}"
    )

print("\nTest set is NOT used for model selection or predictions.")

candidates = {
    "LightGBM Early Stopping": "lightgbm_early_stopping.joblib",
    "LightGBM 126 Trees": "lightgbm_risk_model.joblib",
    "Random Forest": "randomforest_risk_model.joblib",
    "XGBoost": "xgboost_risk_model.joblib",
}

results = []

print("\n" + "=" * 75)
print("SAME-VALIDATION-SET MODEL COMPARISON")
print("=" * 75)

for name, filename in candidates.items():

    path = MODEL_DIR / filename

    if not path.exists():
        print(f"\n{name}: MODEL NOT FOUND")
        continue

    model = joblib.load(path)

    print(f"\nEvaluating: {name}")
    print("Model type:", type(model).__name__)
    print("Expected features:", len(features))
    print("Model features:", getattr(model, "n_features_in_", "Unknown"))

    if getattr(model, "n_features_in_", len(features)) != len(features):
        print("SKIPPED: Feature count mismatch")
        continue

    scores = model.predict_proba(X_valid)[:, 1]
    predictions = (scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_valid, predictions, labels=[0, 1]
    ).ravel()

    timings = []

    for _ in range(5):
        start = time.perf_counter()
        model.predict_proba(X_valid)
        timings.append(
            (time.perf_counter() - start) * 1000
        )

    result = {
        "Model": name,
        "PR_AUC": average_precision_score(y_valid, scores),
        "ROC_AUC": roc_auc_score(y_valid, scores),
        "Precision": precision_score(
            y_valid, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_valid, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_valid, predictions, zero_division=0
        ),
        "Brier_Score": brier_score_loss(y_valid, scores),
        "Log_Loss": log_loss(y_valid, scores),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Inference_ms_per_batch": np.mean(timings),
        "Unique_Scores": len(np.unique(scores)),
    }

    results.append(result)

    print("PR-AUC:", result["PR_AUC"])
    print("ROC-AUC:", result["ROC_AUC"])
    print("F1:", result["F1"])
    print("Brier:", result["Brier_Score"])
    print("Log loss:", result["Log_Loss"])
    print("Confusion matrix:", tn, fp, fn, tp)

if results:
    report = pd.DataFrame(results)

    output = REPORT_DIR / "risk_model_fair_validation_comparison.csv"
    report.to_csv(output, index=False)

    print("\n" + "=" * 75)
    print("FINAL VALIDATION COMPARISON")
    print("=" * 75)
    print(report.to_string(index=False))
    print("\nSaved:", output)

print("\nCOMPARISON COMPLETED")
