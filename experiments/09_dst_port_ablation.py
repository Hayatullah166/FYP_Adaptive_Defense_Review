from pathlib import Path
import time

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "grouped"
REPORT_DIR = ROOT / "data" / "reports"

train = pd.read_csv(DATA_DIR / "train_lgbm_grouped.csv")
valid = pd.read_csv(DATA_DIR / "validation_lgbm_grouped.csv")

excluded = ["Label", "AttackType"]

X_train = train.drop(columns=excluded)
y_train = train["Label"].astype(int)

X_valid = valid.drop(columns=excluded)
y_valid = valid["Label"].astype(int)

positive = int((y_train == 1).sum())
negative = int((y_train == 0).sum())
ratio = negative / positive

results = []

for experiment, features in [
    ("Original_68_Features", list(X_train.columns)),
    ("Without_Dst_Port_67_Features",
     [c for c in X_train.columns if c != "Dst Port"]),
]:

    print(f"\nRunning: {experiment}")
    print(f"Features: {len(features)}")

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=ratio,
        objective="binary",
        verbosity=-1,
        n_jobs=-1,
        random_state=42,
    )

    start = time.perf_counter()

    model.fit(
        X_train[features],
        y_train,
        eval_X=X_valid[features],
        eval_y=y_valid,
        eval_metric="binary_logloss",
        callbacks=[early_stopping(20, verbose=True)],
    )

    train_time = time.perf_counter() - start

    scores = model.predict_proba(X_valid[features])[:, 1]
    predicted = (scores >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_valid, predicted, labels=[0, 1]
    ).ravel()

    results.append({
        "Experiment": experiment,
        "Feature_Count": len(features),
        "Best_Iteration": model.best_iteration_,
        "Precision": precision_score(y_valid, predicted, zero_division=0),
        "Recall": recall_score(y_valid, predicted, zero_division=0),
        "F1": f1_score(y_valid, predicted, zero_division=0),
        "PR_AUC": average_precision_score(y_valid, scores),
        "ROC_AUC": roc_auc_score(y_valid, scores),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Training_Seconds": train_time,
    })

REPORT_DIR.mkdir(parents=True, exist_ok=True)

report = pd.DataFrame(results)
output = REPORT_DIR / "dst_port_ablation_comparison.csv"
report.to_csv(output, index=False)

print("\n========== ABLATION COMPARISON ==========")
print(report.to_string(index=False))
print(f"\nReport saved: {output}")
print("\nBoth experiments used validation data only.")
print("No existing model or test data was changed.")