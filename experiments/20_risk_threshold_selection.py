import os
import json
import time
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
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
GROUPED_DIR = os.path.join(DATA_DIR, "processed", "grouped")

RAW_VAL_PATH = os.path.join(
    GROUPED_DIR, "validation_lgbm_grouped.csv"
)

SCALED_VAL_PATH = os.path.join(
    GROUPED_DIR, "validation_mlp_scaled_grouped.csv"
)

FEATURE_NAMES_PATH = os.path.join(
    GROUPED_DIR, "models", "feature_names_grouped.joblib"
)

RISK_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "risk_assessment",
    "xgboost_risk_model.joblib"
)

RISK_FEATURES_PATH = os.path.join(
    BASE_DIR, "models", "risk_assessment",
    "risk_model_feature_names.joblib"
)

MLP_CHECKPOINT_PATH = os.path.join(
    BASE_DIR, "models", "mlp",
    "mlp_adversarial_best.pth"
)

OUTPUT_CSV = os.path.join(
    REPORT_DIR, "risk_threshold_selection.csv"
)

OUTPUT_JSON = os.path.join(
    REPORT_DIR, "risk_threshold_selection_summary.json"
)

DEVICE = torch.device("cpu")

CURRENT_THRESHOLD = 0.01
MLP_THRESHOLD = 0.9073078036308289
SQUEEZE_DECIMALS = 2

THRESHOLDS = [
    0.0,
    0.00001,
    0.000025,
    0.00005,
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01
]

os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 75)
print("SCRIPT 20 - E3 RISK THRESHOLD SELECTION")
print("=" * 75)


# =========================================================
# 2. MLP ARCHITECTURE
# =========================================================

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


# =========================================================
# 3. LOAD AND VERIFY DATA
# =========================================================

print("\n[1/7] Loading validation data and models...")

required_paths = [
    RAW_VAL_PATH,
    SCALED_VAL_PATH,
    FEATURE_NAMES_PATH,
    RISK_MODEL_PATH,
    RISK_FEATURES_PATH,
    MLP_CHECKPOINT_PATH
]

for path in required_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file missing: {path}"
        )

raw_df = pd.read_csv(RAW_VAL_PATH)
scaled_df = pd.read_csv(SCALED_VAL_PATH)

feature_cols = list(joblib.load(FEATURE_NAMES_PATH))
risk_feature_cols = list(joblib.load(RISK_FEATURES_PATH))

if feature_cols != risk_feature_cols:
    raise ValueError(
        "Grouped feature names and XGBoost feature names do not match."
    )

if len(feature_cols) != 68:
    raise ValueError(
        f"Expected 68 features, found {len(feature_cols)}."
    )

if len(raw_df) != len(scaled_df):
    raise ValueError(
        "Raw and scaled validation row counts do not match."
    )

if "Label" not in raw_df.columns:
    raise ValueError("Label column missing from raw validation data.")

if "Label" in scaled_df.columns:
    if not np.array_equal(
        raw_df["Label"].to_numpy(),
        scaled_df["Label"].to_numpy()
    ):
        raise ValueError("Raw and scaled labels are not aligned.")

if "AttackType" in raw_df.columns and "AttackType" in scaled_df.columns:
    if not raw_df["AttackType"].equals(scaled_df["AttackType"]):
        raise ValueError("Raw and scaled AttackType rows are not aligned.")

X_raw = raw_df[feature_cols].to_numpy(dtype=np.float32)
X_scaled = scaled_df[feature_cols].to_numpy(dtype=np.float32)
y = raw_df["Label"].to_numpy(dtype=np.int64)

if not np.isfinite(X_raw).all():
    raise ValueError("Raw validation features contain NaN or infinity.")

if not np.isfinite(X_scaled).all():
    raise ValueError("Scaled validation features contain NaN or infinity.")

if not np.isin(np.unique(y), [0, 1]).all():
    raise ValueError("Labels must be binary 0/1.")

print("Validation records:", len(y))
print("Features:", len(feature_cols))
print("Benign:", int((y == 0).sum()))
print("Malicious:", int((y == 1).sum()))


# =========================================================
# 4. LOAD MODELS
# =========================================================

print("\n[2/7] Loading XGBoost Risk Engine and MLP...")

risk_model = joblib.load(RISK_MODEL_PATH)

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

if mlp_features != feature_cols:
    raise ValueError(
        "MLP checkpoint feature names/order do not match."
    )

mlp_model = MLP(len(mlp_features)).to(DEVICE)
mlp_model.load_state_dict(checkpoint["model_state_dict"])
mlp_model.eval()

print("XGBoost loaded successfully.")
print("MLP checkpoint loaded successfully.")
print("MLP best epoch:", checkpoint.get("best_epoch", "Not stored"))


# =========================================================
# 5. XGBOOST RISK PREDICTIONS
# =========================================================

print("\n[3/7] Calculating XGBoost risk probabilities...")

raw_frame = pd.DataFrame(
    X_raw,
    columns=feature_cols
)

xgb_prob = risk_model.predict_proba(raw_frame)[:, 1]

xgb_class = (xgb_prob >= 0.5).astype(np.int64)

uncertainty = np.minimum(
    xgb_prob,
    1.0 - xgb_prob
)

print("XGBoost predictions ready.")
print("XGBoost benign predictions:", int((xgb_class == 0).sum()))
print("XGBoost malicious predictions:", int((xgb_class == 1).sum()))


# =========================================================
# 6. DEEP PATH PREDICTIONS - CALCULATE ONCE
# =========================================================

print("\n[4/7] Running feature squeezing and adversarially trained MLP...")

deep_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32,
    device=DEVICE
)

squeezed_tensor = torch.round(
    deep_tensor * (10 ** SQUEEZE_DECIMALS)
) / (10 ** SQUEEZE_DECIMALS)

start_time = time.perf_counter()

with torch.no_grad():
    deep_logits = mlp_model(squeezed_tensor)

    deep_prob = torch.sigmoid(
        deep_logits
    ).cpu().numpy().reshape(-1)

deep_inference_seconds = time.perf_counter() - start_time

deep_class = (
    deep_prob >= MLP_THRESHOLD
).astype(np.int64)

print("Deep-path predictions ready.")
print("MLP inference time:", round(deep_inference_seconds, 6), "seconds")


# =========================================================
# 7. EVALUATE EACH THRESHOLD
# =========================================================

print("\n[5/7] Evaluating candidate thresholds...")

results = []

for threshold in THRESHOLDS:

    # Fast only when XGBoost predicts benign
    # and uncertainty is below the selected threshold.
    fast_mask = (
        (xgb_class == 0) &
        (uncertainty < threshold)
    )

    deep_mask = ~fast_mask

    final_predictions = np.where(
        fast_mask,
        xgb_class,
        deep_class
    ).astype(np.int64)

    final_probabilities = np.where(
        fast_mask,
        xgb_prob,
        deep_prob
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        final_predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y, final_predictions)

    precision = precision_score(
        y, final_predictions, zero_division=0
    )

    recall = recall_score(
        y, final_predictions, zero_division=0
    )

    f1 = f1_score(
        y, final_predictions, zero_division=0
    )

    pr_auc = average_precision_score(
        y, final_probabilities
    )

    roc_auc = roc_auc_score(
        y, final_probabilities
    )

    total = len(y)
    fast_count = int(fast_mask.sum())
    deep_count = int(deep_mask.sum())

    fast_malicious = int(
        ((y == 1) & fast_mask).sum()
    )

    deep_malicious = int(
        ((y == 1) & deep_mask).sum()
    )

    fast_benign = int(
        ((y == 0) & fast_mask).sum()
    )

    deep_benign = int(
        ((y == 0) & deep_mask).sum()
    )

    results.append({
        "risk_threshold": threshold,
        "fast_records": fast_count,
        "deep_records": deep_count,
        "fast_share_percent": 100 * fast_count / total,
        "deep_share_percent": 100 * deep_count / total,
        "fast_benign_records": fast_benign,
        "fast_malicious_records": fast_malicious,
        "deep_benign_records": deep_benign,
        "deep_malicious_records": deep_malicious,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n[6/7] THRESHOLD COMPARISON")
print("=" * 105)

display_cols = [
    "risk_threshold",
    "fast_share_percent",
    "deep_share_percent",
    "fast_malicious_records",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "FN"
]

print(
    results_df[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =========================================================
# 8. BASELINE AND SUMMARY
# =========================================================

baseline_rows = results_df[
    np.isclose(
        results_df["risk_threshold"],
        CURRENT_THRESHOLD
    )
]

if len(baseline_rows) != 1:
    raise ValueError(
        "Current threshold 0.0001 is missing from candidates."
    )

baseline = baseline_rows.iloc[0].to_dict()

summary = {
    "experiment": "E3 Risk Threshold Selection",
    "dataset": "CICIDS-2018 grouped validation",
    "validation_only": True,
    "independent_test_evaluation": False,
    "total_records": int(len(y)),
    "malicious_records": int((y == 1).sum()),
    "benign_records": int((y == 0).sum()),
    "feature_count": int(len(feature_cols)),
    "risk_rule": "min(p_attack, 1-p_attack)",
    "routing_rule": (
        "Fast if XGBoost predicts benign and uncertainty < threshold; "
        "otherwise Deep"
    ),
    "deep_path": (
        "Feature squeezing followed by adversarially trained MLP"
    ),
    "mlp_threshold": MLP_THRESHOLD,
    "squeeze_decimals": SQUEEZE_DECIMALS,
    "thresholds_evaluated": THRESHOLDS,
    "current_threshold": CURRENT_THRESHOLD,
    "current_threshold_results": baseline,
    "deep_mlp_full_validation_inference_seconds": (
        float(deep_inference_seconds)
    ),
    "results_csv": OUTPUT_CSV,
    "important_note": (
        "Threshold comparison reuses the same validation records and "
        "fixed model predictions. Results are validation-specific. "
        "Threshold selection is not independent test evidence."
    )
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("\n[7/7] SAVED REPORTS")
print("=" * 75)
print(OUTPUT_CSV)
print(OUTPUT_JSON)

print("\nCURRENT THRESHOLD: 0.0001")
print("Fast share:", round(baseline["fast_share_percent"], 4), "%")
print("Deep share:", round(baseline["deep_share_percent"], 4), "%")
print("Recall:", round(baseline["recall"], 6))
print("F1:", round(baseline["f1"], 6))
print("PR-AUC:", round(baseline["pr_auc"], 6))
print("False negatives:", baseline["FN"])
print("Malicious records sent to Fast:", baseline["fast_malicious_records"])

print("\nSCRIPT 20 COMPLETED SUCCESSFULLY")
print("=" * 75)
