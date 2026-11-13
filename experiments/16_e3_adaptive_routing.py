"""
E3: Adaptive Risk-Aware Fast/Deep Routing
Validation implementation.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, roc_auc_score,
    confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPED_DIR = os.path.join(BASE_DIR, "data", "processed", "grouped")
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RISK_MODEL_PATH = os.path.join(
    MODEL_DIR, "risk_assessment", "xgboost_risk_model.joblib"
)
FEATURE_NAMES_PATH = os.path.join(
    MODEL_DIR, "risk_assessment", "risk_model_feature_names.joblib"
)
MLP_CHECKPOINT_PATH = os.path.join(
    MODEL_DIR, "mlp", "mlp_adversarial_best.pth"
)

RISK_DATA_PATH = os.path.join(
    GROUPED_DIR, "validation_lgbm_grouped.csv"
)
MLP_DATA_PATH = os.path.join(
    GROUPED_DIR, "validation_mlp_scaled_grouped.csv"
)

OUTPUT_CSV = os.path.join(
    REPORT_DIR, "e3_adaptive_validation_predictions.csv"
)
OUTPUT_JSON = os.path.join(
    REPORT_DIR, "e3_adaptive_validation_summary.json"
)

RISK_THRESHOLD = 0.01
MLP_THRESHOLD = 0.9073078036308289
SQUEEZE_DECIMALS = 2
DEVICE = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


def calculate_metrics(y_true, y_pred, probabilities):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, y_pred, zero_division=0
        )),
        "pr_auc": float(average_precision_score(
            y_true, probabilities
        )),
        "roc_auc": float(roc_auc_score(
            y_true, probabilities
        )),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 65)
    print("E3 ADAPTIVE FAST/DEEP ROUTING - VALIDATION")
    print("=" * 65)

    # --------------------------------------------------
    # 1. Load validation datasets
    # --------------------------------------------------
    risk_df = pd.read_csv(RISK_DATA_PATH)
    mlp_df = pd.read_csv(MLP_DATA_PATH)

    if len(risk_df) != len(mlp_df):
        raise ValueError("Risk and MLP validation row counts differ.")

    if not np.array_equal(
        risk_df["Label"].to_numpy(),
        mlp_df["Label"].to_numpy()
    ):
        raise ValueError("Validation labels are not row-aligned.")

    if not np.array_equal(
        risk_df["AttackType"].astype(str).to_numpy(),
        mlp_df["AttackType"].astype(str).to_numpy()
    ):
        raise ValueError("Validation AttackType rows are not aligned.")

    y_true = risk_df["Label"].to_numpy(dtype=np.int32)

    # --------------------------------------------------
    # 2. Load XGBoost risk engine
    # --------------------------------------------------
    risk_model = joblib.load(RISK_MODEL_PATH)
    risk_features = joblib.load(FEATURE_NAMES_PATH)
    risk_features = list(risk_features)

    X_risk = risk_df[risk_features].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(X_risk).all():
        raise ValueError("Non-finite values in risk input.")

    start = time.perf_counter()
    risk_probabilities = risk_model.predict_proba(
        pd.DataFrame(X_risk, columns=risk_features)
    )[:, 1]
    risk_model_seconds = time.perf_counter() - start

    predicted_class = (risk_probabilities >= 0.5).astype(np.int32)
    uncertainty = np.minimum(
        risk_probabilities, 1.0 - risk_probabilities
    )

    # --------------------------------------------------
    # 3. Adaptive routing
    # --------------------------------------------------
    fast_mask = (
        (predicted_class == 0) &
        (uncertainty < RISK_THRESHOLD)
    )
    deep_mask = ~fast_mask

    # Fast Path predictions: XGBoost predicted benign.
    e3_predictions = predicted_class.copy()
    e3_probabilities = risk_probabilities.copy()

    # --------------------------------------------------
    # 4. Load adversarially trained MLP
    # --------------------------------------------------
    try:
        checkpoint = torch.load(
            MLP_CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            MLP_CHECKPOINT_PATH,
            map_location=DEVICE
        )

    mlp_features = list(checkpoint["feature_names"])

    if mlp_features != risk_features:
        raise ValueError(
            "MLP and Risk Engine feature names/order do not match."
        )

    mlp_model = MLP(len(mlp_features)).to(DEVICE)
    mlp_model.load_state_dict(checkpoint["model_state_dict"])
    mlp_model.eval()

    # --------------------------------------------------
    # 5. Deep Path: feature squeezing -> adversarial MLP
    # --------------------------------------------------
    X_mlp = mlp_df[mlp_features].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(X_mlp).all():
        raise ValueError("Non-finite values in MLP input.")

    deep_indices = np.flatnonzero(deep_mask)

    if len(deep_indices) > 0:
        X_deep = X_mlp[deep_indices]

        X_deep_squeezed = np.round(
            X_deep, decimals=SQUEEZE_DECIMALS
        ).astype(np.float32)

        deep_tensor = torch.tensor(
            X_deep_squeezed, dtype=torch.float32
        )

        start = time.perf_counter()

        with torch.no_grad():
            deep_logits = mlp_model(deep_tensor)
            deep_probs = torch.sigmoid(
                deep_logits
            ).cpu().numpy().reshape(-1)

        deep_model_seconds = time.perf_counter() - start

        deep_predictions = (
            deep_probs >= MLP_THRESHOLD
        ).astype(np.int32)

        e3_predictions[deep_indices] = deep_predictions
        e3_probabilities[deep_indices] = deep_probs

    else:
        deep_probs = np.array([], dtype=np.float32)
        deep_model_seconds = 0.0

    # --------------------------------------------------
    # 6. Metrics and path distribution
    # --------------------------------------------------
    metrics = calculate_metrics(
        y_true, e3_predictions, e3_probabilities
    )

    benign_mask = y_true == 0
    malicious_mask = y_true == 1

    total = len(y_true)
    fast_count = int(fast_mask.sum())
    deep_count = int(deep_mask.sum())

    summary = {
        "experiment": "E3 Adaptive Fast/Deep Routing",
        "dataset": "CICIDS-2018 grouped validation",
        "records": int(total),
        "malicious_records": int(malicious_mask.sum()),
        "benign_records": int(benign_mask.sum()),
        "risk_threshold": RISK_THRESHOLD,
        "risk_rule": "min(p_attack, 1-p_attack)",
        "routing_rule": (
            "Fast if predicted benign and risk < threshold; "
            "otherwise Deep"
        ),
        "deep_path": (
            "Feature squeezing followed by adversarially trained MLP"
        ),
        "squeeze_decimals": SQUEEZE_DECIMALS,
        "mlp_threshold": MLP_THRESHOLD,
        "fast_path_records": fast_count,
        "deep_path_records": deep_count,
        "fast_path_share_percent": (
            fast_count / total * 100 if total else 0.0
        ),
        "deep_path_share_percent": (
            deep_count / total * 100 if total else 0.0
        ),
        "fast_benign_records": int((fast_mask & benign_mask).sum()),
        "fast_malicious_records": int((fast_mask & malicious_mask).sum()),
        "deep_benign_records": int((deep_mask & benign_mask).sum()),
        "deep_malicious_records": int((deep_mask & malicious_mask).sum()),
        "risk_engine_inference_seconds": risk_model_seconds,
        "deep_mlp_inference_seconds": deep_model_seconds,
        "e3_metrics": metrics,
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_validation_pr_auc": float(
            checkpoint["best_validation_pr_auc"]
        ),
        "note": (
            "Validation-only result. Threshold is provisional. "
            "No independent test evaluation is performed here."
        )
    }

    # --------------------------------------------------
    # 7. Save row-level predictions and summary
    # --------------------------------------------------
    output_df = pd.DataFrame({
        "Actual_Label": y_true,
        "AttackType": risk_df["AttackType"].astype(str),
        "XGBoost_Attack_Probability": risk_probabilities,
        "Uncertainty_Risk": uncertainty,
        "Route": np.where(fast_mask, "Fast", "Deep"),
        "E3_Prediction": e3_predictions,
        "E3_Attack_Probability": e3_probabilities
    })

    output_df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nPATH DISTRIBUTION")
    print("Fast:", fast_count)
    print("Deep:", deep_count)
    print("Fast path share: {:.4f}%".format(
        summary["fast_path_share_percent"]
    ))
    print("Fast benign:", summary["fast_benign_records"])
    print("Fast malicious:", summary["fast_malicious_records"])
    print("Deep benign:", summary["deep_benign_records"])
    print("Deep malicious:", summary["deep_malicious_records"])

    print("\nE3 VALIDATION METRICS")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved predictions:", OUTPUT_CSV)
    print("Saved summary:", OUTPUT_JSON)
    print("\nValidation run complete. This is not test-set evaluation.")


if __name__ == "__main__":
    main()
