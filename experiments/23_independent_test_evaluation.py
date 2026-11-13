from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models" / "risk_assessment"
MLP_PATH = ROOT / "models" / "mlp" / "mlp_adversarial_best.pth"
SCALER_PATH = (
    ROOT / "data" / "processed" / "grouped" /
    "models" / "standard_scaler_grouped.joblib"
)
TEST_PATH = (
    ROOT / "data" / "processed" / "grouped" /
    "test_lgbm_grouped.csv"
)
REPORT_DIR = ROOT / "data" / "reports"

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


def main():

    print("=" * 75)
    print("SCRIPT 23 - INDEPENDENT TEST EVALUATION")
    print("=" * 75)

    required = [
        TEST_PATH,
        MODEL_DIR / "xgboost_risk_model.joblib",
        MODEL_DIR / "risk_model_feature_names.joblib",
        MLP_PATH,
        SCALER_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")

    # --------------------------------------------------
    # 1. Load untouched test dataset and frozen artifacts
    # --------------------------------------------------

    test = pd.read_csv(TEST_PATH)

    if "Label" not in test.columns:
        raise ValueError("Test dataset has no Label column.")

    labels = pd.to_numeric(test["Label"], errors="coerce")

    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Test labels must contain only 0 and 1.")

    y_true = labels.to_numpy(dtype=np.int32)

    features = list(
        joblib.load(MODEL_DIR / "risk_model_feature_names.joblib")
    )

    risk_model = joblib.load(
        MODEL_DIR / "xgboost_risk_model.joblib"
    )

    scaler = joblib.load(SCALER_PATH)

    try:
        checkpoint = torch.load(
            MLP_PATH,
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            MLP_PATH,
            map_location=DEVICE
        )

    mlp_features = list(checkpoint["feature_names"])

    if features != mlp_features:
        raise ValueError("XGBoost and MLP feature orders do not match.")

    if len(features) != 68:
        raise ValueError(f"Expected 68 features, found {len(features)}.")

    mlp_model = MLP(len(features)).to(DEVICE)
    mlp_model.load_state_dict(checkpoint["model_state_dict"])
    mlp_model.eval()

    print("\nTEST DATA AUDIT")
    print("Records:", len(test))
    print("Features:", len(features))
    print("Benign:", int((y_true == 0).sum()))
    print("Attacks:", int((y_true == 1).sum()))
    print("Test data used for training/tuning: NO")

    # --------------------------------------------------
    # 2. Validate and prepare raw features
    # --------------------------------------------------

    missing = [f for f in features if f not in test.columns]

    if missing:
        raise ValueError(f"Missing features: {missing[:10]}")

    X_df = test[features].apply(pd.to_numeric, errors="coerce")

    if X_df.isna().any().any():
        raise ValueError("Test features contain missing/non-numeric values.")

    X_raw = X_df.to_numpy(dtype=np.float32)

    if not np.isfinite(X_raw).all():
        raise ValueError("Test features contain infinity or invalid values.")

    # --------------------------------------------------
    # 3. XGBoost risk assessment - raw features
    # --------------------------------------------------

    print("\nRunning frozen XGBoost risk engine...")

    risk_probabilities = risk_model.predict_proba(
        pd.DataFrame(X_raw, columns=features)
    )[:, 1]

    predicted_class = (risk_probabilities >= 0.5).astype(np.int32)

    uncertainty = np.minimum(
        risk_probabilities,
        1.0 - risk_probabilities
    )

    # --------------------------------------------------
    # 4. Adaptive routing - exact dashboard condition
    # --------------------------------------------------

    fast_mask = (
        (predicted_class == 0) &
        (uncertainty < RISK_THRESHOLD)
    )

    deep_mask = ~fast_mask

    fast_indices = np.flatnonzero(fast_mask)
    deep_indices = np.flatnonzero(deep_mask)

    final_predictions = predicted_class.copy()
    final_probabilities = risk_probabilities.copy()

    # --------------------------------------------------
    # 5. Deep path - scaler, squeezing, adversarial MLP
    # --------------------------------------------------

    print("Running saved scaler and feature squeezing...")

    X_scaled = scaler.transform(X_raw).astype(np.float32)

    if not np.isfinite(X_scaled).all():
        raise ValueError("Scaled test features contain invalid values.")

    deep_probabilities = np.array([], dtype=np.float32)

    if len(deep_indices) > 0:

        X_deep = X_scaled[deep_indices]

        X_squeezed = np.round(
            X_deep,
            decimals=SQUEEZE_DECIMALS
        ).astype(np.float32)

        deep_tensor = torch.tensor(
            X_squeezed,
            dtype=torch.float32,
            device=DEVICE
        )

        with torch.no_grad():
            logits = mlp_model(deep_tensor)
            deep_probabilities = torch.sigmoid(
                logits
            ).cpu().numpy().reshape(-1)

        deep_predictions = (
            deep_probabilities >= MLP_THRESHOLD
        ).astype(np.int32)

        final_predictions[deep_indices] = deep_predictions
        final_probabilities[deep_indices] = deep_probabilities

    # --------------------------------------------------
    # 6. Final evaluation
    # --------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        final_predictions,
        labels=[0, 1]
    ).ravel()

    final_metrics = {
        "Accuracy": float(accuracy_score(y_true, final_predictions)),
        "Precision": float(precision_score(
            y_true, final_predictions, zero_division=0
        )),
        "Recall": float(recall_score(
            y_true, final_predictions, zero_division=0
        )),
        "F1": float(f1_score(
            y_true, final_predictions, zero_division=0
        )),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    # Standalone XGBoost probability metrics
    xgb_predictions = (risk_probabilities >= 0.5).astype(np.int32)

    xgb_tn, xgb_fp, xgb_fn, xgb_tp = confusion_matrix(
        y_true, xgb_predictions, labels=[0, 1]
    ).ravel()

    xgb_metrics = {
        "PR_AUC": float(average_precision_score(
            y_true, risk_probabilities
        )),
        "ROC_AUC": float(roc_auc_score(
            y_true, risk_probabilities
        )),
        "Brier_Score": float(brier_score_loss(
            y_true, risk_probabilities
        )),
        "Log_Loss": float(log_loss(
            y_true, risk_probabilities, labels=[0, 1]
        )),
        "TN": int(xgb_tn),
        "FP": int(xgb_fp),
        "FN": int(xgb_fn),
        "TP": int(xgb_tp),
    }

    # Deep-only conditional performance
    deep_metrics = None

    if len(deep_indices) > 0:
        deep_y = y_true[deep_indices]
        deep_pred = final_predictions[deep_indices]

        deep_metrics = {
            "Records": int(len(deep_indices)),
            "Actual_Attacks": int((deep_y == 1).sum()),
            "Actual_Benign": int((deep_y == 0).sum()),
            "Accuracy": float(accuracy_score(deep_y, deep_pred)),
            "Precision": float(precision_score(
                deep_y, deep_pred, zero_division=0
            )),
            "Recall": float(recall_score(
                deep_y, deep_pred, zero_division=0
            )),
            "F1": float(f1_score(
                deep_y, deep_pred, zero_division=0
            )),
        }

    # --------------------------------------------------
    # 7. Routing audit
    # --------------------------------------------------

    fast_count = int(fast_mask.sum())
    deep_count = int(deep_mask.sum())

    fast_attacks = int((y_true[fast_mask] == 1).sum())
    deep_attacks = int((y_true[deep_mask] == 1).sum())

    routing = {
        "Fast_Records": fast_count,
        "Deep_Records": deep_count,
        "Fast_Percent": float(100 * fast_count / len(test)),
        "Deep_Percent": float(100 * deep_count / len(test)),
        "Attacks_Routed_Fast": fast_attacks,
        "Attacks_Routed_Deep": deep_attacks,
        "Benign_Routed_Fast": int(
            ((y_true == 0) & fast_mask).sum()
        ),
        "Benign_Routed_Deep": int(
            ((y_true == 0) & deep_mask).sum()
        ),
    }

    # --------------------------------------------------
    # 8. Save per-record predictions and reports
    # --------------------------------------------------

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = test.copy()
    results["Actual_Label"] = y_true
    results["XGBoost_Attack_Probability"] = risk_probabilities
    results["Uncertainty_Risk"] = uncertainty
    results["Route"] = np.where(fast_mask, "Fast", "Deep")
    results["Deep_MLP_Attack_Probability"] = np.nan
    results.loc[
        deep_indices, "Deep_MLP_Attack_Probability"
    ] = deep_probabilities
    results["Final_Prediction"] = final_predictions
    results["Final_Probability"] = final_probabilities
    results["Correct"] = (
        final_predictions == y_true
    )

    csv_path = REPORT_DIR / "e3_independent_test_predictions.csv"
    json_path = REPORT_DIR / "e3_independent_test_summary.json"

    results.to_csv(csv_path, index=False)

    summary = {
        "evaluation": "Independent test-set evaluation",
        "test_records": int(len(test)),
        "features": int(len(features)),
        "test_used_for_training_or_tuning": False,
        "risk_model": "XGBoost",
        "deep_classifier": "Adversarial MLP",
        "risk_threshold": RISK_THRESHOLD,
        "mlp_threshold": MLP_THRESHOLD,
        "feature_squeezing_decimals": SQUEEZE_DECIMALS,
        "device": str(DEVICE),
        "final_pipeline_metrics": final_metrics,
        "xgboost_standalone_metrics": xgb_metrics,
        "deep_path_conditional_metrics": deep_metrics,
        "routing": routing,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # --------------------------------------------------
    # 9. Print results
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL ADAPTIVE PIPELINE - INDEPENDENT TEST RESULTS")
    print("=" * 75)

    for key, value in final_metrics.items():
        print(f"{key}: {value}")

    print("\nXGBOOST STANDALONE METRICS")
    for key, value in xgb_metrics.items():
        print(f"{key}: {value}")

    print("\nADAPTIVE ROUTING")
    for key, value in routing.items():
        print(f"{key}: {value}")

    print("\nDEEP PATH CONDITIONAL METRICS")
    print(deep_metrics)

    print("\nSAVED REPORTS")
    print(csv_path)
    print(json_path)

    print("\nSCRIPT 23 COMPLETED SUCCESSFULLY")
    print("No training, threshold tuning, or model selection was performed.")


if __name__ == "__main__":
    main()
