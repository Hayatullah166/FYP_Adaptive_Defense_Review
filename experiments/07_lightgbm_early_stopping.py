
from pathlib import Path
import time
import joblib
import pandas as pd

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data/processed/grouped"
MODEL_DIR = ROOT / "models/risk_assessment"
REPORT_DIR = ROOT / "data/reports"

TRAIN_FILE = DATA_DIR / "train_lgbm_grouped.csv"
VALID_FILE = DATA_DIR / "validation_lgbm_grouped.csv"
FEATURE_FILE = MODEL_DIR / "risk_model_feature_names.joblib"

OUTPUT_MODEL = MODEL_DIR / "lightgbm_early_stopping.joblib"
OUTPUT_REPORT = REPORT_DIR / "lightgbm_early_stopping.csv"


def main():
    train = pd.read_csv(TRAIN_FILE)
    valid = pd.read_csv(VALID_FILE)
    features = joblib.load(FEATURE_FILE)

    X_train = train[features]
    y_train = train["Label"].astype(int)

    X_valid = valid[features]
    y_valid = valid["Label"].astype(int)

    ratio = (y_train == 0).sum() / (y_train == 1).sum()

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

    print("Starting early-stopping experiment...")
    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        eval_X=X_valid,
        eval_y=y_valid,
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(stopping_rounds=20, verbose=True),
            log_evaluation(period=10),
        ],
    )

    training_time = time.perf_counter() - start

    best_iteration = model.best_iteration_
    scores = model.predict_proba(
        X_valid, num_iteration=best_iteration
    )[:, 1]

    predictions = (scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_valid, predictions, labels=[0, 1]
    ).ravel()

    report = pd.DataFrame([{
        "Requested_Iterations": 300,
        "Actual_Iterations": model.n_iter_,
        "Best_Iteration": best_iteration,
        "Precision": precision_score(
            y_valid, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_valid, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_valid, predictions, zero_division=0
        ),
        "PR_AUC": average_precision_score(y_valid, scores),
        "ROC_AUC": roc_auc_score(y_valid, scores),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Training_Seconds": training_time,
    }])

    joblib.dump(model, OUTPUT_MODEL)
    report.to_csv(OUTPUT_REPORT, index=False)

    print("\nBest iteration:", best_iteration)
    print("Actual iterations:", model.n_iter_)
    print("Training time:", round(training_time, 3), "seconds")
    print("\nValidation results:")
    print(report.to_string(index=False))
    print("\nModel saved:", OUTPUT_MODEL)
    print("Report saved:", OUTPUT_REPORT)


if __name__ == "__main__":
    main()
