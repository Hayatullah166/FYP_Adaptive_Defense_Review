"""
FYP Risk Assessment Model Comparison
Random Forest vs XGBoost vs LightGBM (Early Stopping)

Uses grouped training and validation data only.
Test dataset remains untouched.
"""

from pathlib import Path
import time
import warnings

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "grouped"
MODEL_DIR = ROOT / "models" / "risk_assessment"
REPORT_DIR = ROOT / "data" / "reports"

TRAIN_FILE = DATA_DIR / "train_lgbm_grouped.csv"
VALID_FILE = DATA_DIR / "validation_lgbm_grouped.csv"

RANDOM_STATE = 42
THRESHOLD = 0.50


def load_data():

    for path in (TRAIN_FILE, VALID_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    train = pd.read_csv(TRAIN_FILE)
    valid = pd.read_csv(VALID_FILE)

    features = [
        c for c in train.columns
        if c not in ("Label", "AttackType")
    ]

    valid_features = [
        c for c in valid.columns
        if c not in ("Label", "AttackType")
    ]

    if features != valid_features:
        raise ValueError("Train and validation features do not match.")

    X_train = train[features]
    y_train = train["Label"].astype(int)

    X_valid = valid[features]
    y_valid = valid["Label"].astype(int)

    if X_train.isna().any().any() or X_valid.isna().any().any():
        raise ValueError("Missing values found in features.")

    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Unexpected training labels.")

    if set(y_valid.unique()) != {0, 1}:
        raise ValueError("Unexpected validation labels.")

    return X_train, y_train, X_valid, y_valid, features


def build_models(y_train):

    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())

    ratio = negative / positive

    models = {

        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=ratio,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

    }

    lightgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=ratio,
        objective="binary",
        verbosity=-1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return models, lightgbm


def evaluate_model(
    name,
    model,
    X_valid,
    y_valid,
    train_seconds,
    actual_iterations=None,
):

    start = time.perf_counter()

    scores = model.predict_proba(X_valid)[:, 1]

    inference_seconds = time.perf_counter() - start

    predicted = (scores >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_valid,
        predicted,
        labels=[0, 1],
    ).ravel()

    result = {
        "Model": name,
        "Threshold": THRESHOLD,
        "Precision": precision_score(
            y_valid, predicted, zero_division=0
        ),
        "Recall": recall_score(
            y_valid, predicted, zero_division=0
        ),
        "F1": f1_score(
            y_valid, predicted, zero_division=0
        ),
        "PR_AUC": average_precision_score(y_valid, scores),
        "ROC_AUC": roc_auc_score(y_valid, scores),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Actual_Iterations": actual_iterations,
        "Train_Time_Seconds": train_seconds,
        "Inference_Milliseconds_Per_Row": (
            inference_seconds * 1000 / len(X_valid)
        ),
    }

    print(
        f"{name}: "
        f"PR-AUC={result['PR_AUC']:.4f} | "
        f"Recall={result['Recall']:.4f} | "
        f"F1={result['F1']:.4f}"
    )

    return result, scores


def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading grouped datasets...")

    X_train, y_train, X_valid, y_valid, features = load_data()

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_valid):,}")
    print(f"Features: {len(features)}")
    print("Test dataset will NOT be used.")

    models, lightgbm = build_models(y_train)

    results = []
    score_table = pd.DataFrame({
        "Actual_Label": y_valid.to_numpy()
    })

    # Random Forest and XGBoost
    for name, model in models.items():

        print(f"\nTraining {name}...")

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        result, scores = evaluate_model(
            name,
            model,
            X_valid,
            y_valid,
            train_seconds,
            getattr(model, "n_estimators", None),
        )

        results.append(result)

        joblib.dump(
            model,
            MODEL_DIR / f"{name.lower()}_risk_model.joblib"
        )

        score_table[f"{name}_Risk_Score"] = scores

    # LightGBM with early stopping
    print("\nTraining LightGBM with Early Stopping...")

    start = time.perf_counter()

    lightgbm.fit(
        X_train,
        y_train,
        eval_X=X_valid,
        eval_y=y_valid,
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(
                stopping_rounds=20,
                verbose=True,
            )
        ],
    )

    train_seconds = time.perf_counter() - start

    best_iteration = lightgbm.best_iteration_

    print(f"\nLightGBM best iteration: {best_iteration}")

    result, scores = evaluate_model(
        "LightGBM_EarlyStopping",
        lightgbm,
        X_valid,
        y_valid,
        train_seconds,
        best_iteration,
    )

    results.append(result)

    joblib.dump(
        lightgbm,
        MODEL_DIR / "lightgbm_early_stopping.joblib"
    )

    score_table["LightGBM_EarlyStopping_Risk_Score"] = scores

    # Save comparison
    metrics = pd.DataFrame(results)

    metrics.to_csv(
        REPORT_DIR / "risk_model_comparison.csv",
        index=False,
    )

    score_table.to_csv(
        REPORT_DIR / "risk_model_validation_predictions.csv",
        index=False,
    )

    joblib.dump(
        features,
        MODEL_DIR / "risk_model_feature_names.joblib",
    )

    print("\n========== FINAL VALIDATION COMPARISON ==========")

    print(
        metrics.sort_values(
            "PR_AUC",
            ascending=False,
        ).to_string(index=False)
    )

    print("\nModels saved in:", MODEL_DIR)

    print(
        "\nComparison report:",
        REPORT_DIR / "risk_model_comparison.csv"
    )

    print("\nValidation predictions saved.")

    print("\nNo model has been selected automatically.")
    print("All results are validation results, not final test results.")


if __name__ == "__main__":
    main()