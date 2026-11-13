"""
E3 Adaptive Routing - Adversarial Evaluation
Targeted FGSM and PGD attacks against the adversarially trained MLP.
Adversarial inputs are generated in standardized feature space.
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. CONFIGURATION
# =========================================================

SEED = 42
DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "grouped")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

RAW_VAL_PATH = os.path.join(
    DATA_DIR, "validation_lgbm_grouped.csv"
)

SCALED_VAL_PATH = os.path.join(
    DATA_DIR, "validation_mlp_scaled_grouped.csv"
)

SCALED_TRAIN_PATH = os.path.join(
    DATA_DIR, "train_mlp_scaled_grouped.csv"
)

SCALER_PATH = os.path.join(
    DATA_DIR, "models",
    "standard_scaler_grouped.joblib"
)

RISK_MODEL_PATH = os.path.join(
    MODEL_DIR, "risk_assessment",
    "xgboost_risk_model.joblib"
)

FEATURE_NAMES_PATH = os.path.join(
    MODEL_DIR, "risk_assessment",
    "risk_model_feature_names.joblib"
)

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR, "mlp", "mlp_adversarial_best.pth"
)

RESULT_PATH = os.path.join(
    REPORT_DIR, "e3_adversarial_routing_results.csv"
)

EXAMPLES_PATH = os.path.join(
    REPORT_DIR, "e3_adversarial_routing_examples.csv"
)

SUMMARY_PATH = os.path.join(
    REPORT_DIR, "e3_adversarial_routing_summary.json"
)

RISK_THRESHOLD = 0.0001
MLP_THRESHOLD = 0.9073078036308289
SQUEEZE_DECIMALS = 2

EPSILONS = [0.01, 0.02, 0.05]
PGD_STEPS = 10


# =========================================================
# 2. REPRODUCIBILITY
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 65)
print("E3 ADVERSARIAL ROUTING EXPERIMENT")
print("=" * 65)


# =========================================================
# 3. MLP ARCHITECTURE
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
# 4. LOAD DATA AND MODELS
# =========================================================

print("\n[1/8] Loading datasets and models...")

raw_df = pd.read_csv(RAW_VAL_PATH)
scaled_val_df = pd.read_csv(SCALED_VAL_PATH)
scaled_train_df = pd.read_csv(SCALED_TRAIN_PATH)

scaler = joblib.load(SCALER_PATH)
risk_model = joblib.load(RISK_MODEL_PATH)
risk_feature_cols = list(joblib.load(FEATURE_NAMES_PATH))

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

mlp_feature_cols = list(checkpoint["feature_names"])

if not isinstance(scaler, StandardScaler):
    raise TypeError("Loaded scaler is not a StandardScaler.")

if scaler.n_features_in_ != len(risk_feature_cols):
    raise ValueError("Scaler and risk model feature counts differ.")

if len(risk_feature_cols) != 68:
    raise ValueError(
        f"Expected 68 features, found {len(risk_feature_cols)}."
    )

if risk_feature_cols != mlp_feature_cols:
    raise ValueError(
        "Risk model and MLP feature orders do not match."
    )

feature_cols = risk_feature_cols

for col in feature_cols:
    if col not in raw_df.columns:
        raise ValueError(f"Missing raw feature: {col}")

    if col not in scaled_val_df.columns:
        raise ValueError(f"Missing scaled validation feature: {col}")

    if col not in scaled_train_df.columns:
        raise ValueError(f"Missing scaled training feature: {col}")

if len(raw_df) != len(scaled_val_df):
    raise ValueError("Raw and scaled validation row counts differ.")

for col in ["Label", "AttackType"]:
    if col in raw_df.columns and col in scaled_val_df.columns:
        if not raw_df[col].equals(scaled_val_df[col]):
            raise ValueError(
                f"Raw and scaled validation rows are misaligned: {col}"
            )

if not raw_df["Label"].equals(scaled_val_df["Label"]):
    raise ValueError("Validation labels are not aligned.")

if "AttackType" in raw_df.columns:
    if not raw_df["AttackType"].equals(scaled_val_df["AttackType"]):
        raise ValueError("Validation attack types are not aligned.")

X_raw = raw_df[feature_cols].to_numpy(dtype=np.float32)
X_scaled = scaled_val_df[feature_cols].to_numpy(dtype=np.float32)
X_train_scaled = scaled_train_df[feature_cols].to_numpy(
    dtype=np.float32
)

y_val = raw_df["Label"].to_numpy(dtype=np.int64)

if not np.isfinite(X_raw).all():
    raise ValueError("Raw validation data contains NaN or infinity.")

if not np.isfinite(X_scaled).all():
    raise ValueError("Scaled validation data contains NaN or infinity.")

if not np.isfinite(X_train_scaled).all():
    raise ValueError("Scaled training data contains NaN or infinity.")

model = MLP(len(feature_cols)).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Raw validation rows:", len(X_raw))
print("Scaled validation rows:", len(X_scaled))
print("Training rows:", len(X_train_scaled))
print("Features:", len(feature_cols))
print("Scaler:", type(scaler).__name__)
print("MLP checkpoint loaded successfully.")


# =========================================================
# 5. ATTACK DATA AND BOUNDS
# =========================================================

print("\n[2/8] Preparing malicious validation records...")

malicious_mask = y_val == 1

X_malicious_scaled = X_scaled[malicious_mask]
X_malicious_raw = X_raw[malicious_mask]

malicious_indices = np.where(malicious_mask)[0]

if len(X_malicious_scaled) == 0:
    raise ValueError("No malicious validation records found.")

train_min = X_train_scaled.min(axis=0)
train_max = X_train_scaled.max(axis=0)

malicious_raw_min = X_malicious_raw.min(axis=0)
malicious_raw_max = X_malicious_raw.max(axis=0)

print("Malicious validation records:", len(X_malicious_scaled))


# =========================================================
# 6. E3 ROUTING FUNCTION
# =========================================================

def run_e3(X_raw_input, X_scaled_input):

    raw_frame = pd.DataFrame(
        X_raw_input,
        columns=feature_cols
    )

    xgb_probabilities = risk_model.predict_proba(
        raw_frame
    )[:, 1]

    xgb_predictions = (
        xgb_probabilities >= 0.5
    ).astype(np.int64)

    uncertainty = np.minimum(
        xgb_probabilities,
        1.0 - xgb_probabilities
    )

    fast_mask = (
        (xgb_predictions == 0)
        & (uncertainty < RISK_THRESHOLD)
    )

    final_predictions = np.zeros(
        len(X_raw_input),
        dtype=np.int64
    )

    final_predictions[fast_mask] = (
        xgb_predictions[fast_mask]
    )

    deep_mask = ~fast_mask

    if deep_mask.any():

        deep_inputs = torch.tensor(
            X_scaled_input[deep_mask],
            dtype=torch.float32,
            device=DEVICE
        )

        squeezed_inputs = torch.round(
            deep_inputs * (10 ** SQUEEZE_DECIMALS)
        ) / (10 ** SQUEEZE_DECIMALS)

        with torch.no_grad():

            logits = model(squeezed_inputs).squeeze(1)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        final_predictions[deep_mask] = (
            probabilities >= MLP_THRESHOLD
        ).astype(np.int64)

    routes = np.where(fast_mask, "Fast", "Deep")

    return {
        "predictions": final_predictions,
        "routes": routes,
        "xgb_probabilities": xgb_probabilities,
        "fast_mask": fast_mask,
        "deep_mask": deep_mask
    }


# =========================================================
# 7. TARGETED FGSM
# =========================================================

def targeted_fgsm(X, epsilon):

    inputs = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )

    inputs.requires_grad_(True)

    target_labels = torch.zeros(
        len(inputs),
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.BCEWithLogitsLoss()

    model.zero_grad(set_to_none=True)

    logits = model(inputs).squeeze(1)
    loss = criterion(logits, target_labels)

    loss.backward()

    gradient = inputs.grad.detach()

    adversarial = (
        inputs.detach()
        - epsilon * gradient.sign()
    )

    lower = torch.tensor(
        train_min,
        dtype=torch.float32,
        device=DEVICE
    )

    upper = torch.tensor(
        train_max,
        dtype=torch.float32,
        device=DEVICE
    )

    adversarial = torch.maximum(
        torch.minimum(adversarial, upper),
        lower
    )

    return adversarial.detach().cpu().numpy()


# =========================================================
# 8. TARGETED PGD
# =========================================================

def targeted_pgd(X, epsilon, steps=PGD_STEPS):

    original = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )

    lower = torch.tensor(
        train_min,
        dtype=torch.float32,
        device=DEVICE
    )

    upper = torch.tensor(
        train_max,
        dtype=torch.float32,
        device=DEVICE
    )

    target_labels = torch.zeros(
        len(original),
        dtype=torch.float32,
        device=DEVICE
    )

    adversarial = original.detach().clone()

    random_noise = torch.empty_like(adversarial).uniform_(
        -epsilon, epsilon
    )

    adversarial = adversarial + random_noise

    adversarial = torch.maximum(
        torch.minimum(adversarial, original + epsilon),
        original - epsilon
    )

    adversarial = torch.maximum(
        torch.minimum(adversarial, upper),
        lower
    )

    alpha = epsilon / steps
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(steps):

        adversarial = adversarial.detach()
        adversarial.requires_grad_(True)

        model.zero_grad(set_to_none=True)

        logits = model(adversarial).squeeze(1)
        loss = criterion(logits, target_labels)

        loss.backward()

        gradient = adversarial.grad.detach()

        adversarial = (
            adversarial.detach()
            - alpha * gradient.sign()
        )

        perturbation = torch.clamp(
            adversarial - original,
            min=-epsilon,
            max=epsilon
        )

        adversarial = original + perturbation

        adversarial = torch.maximum(
            torch.minimum(adversarial, upper),
            lower
        )

    return adversarial.detach().cpu().numpy()


# =========================================================
# 9. CLEAN E3 BASELINE
# =========================================================

print("\n[3/8] Evaluating clean E3 routing...")

clean_result = run_e3(
    X_malicious_raw,
    X_malicious_scaled
)

clean_predictions = clean_result["predictions"]

clean_detected = int(
    np.sum(clean_predictions == 1)
)

clean_fast = int(
    np.sum(clean_result["routes"] == "Fast")
)

clean_deep = int(
    np.sum(clean_result["routes"] == "Deep")
)

clean_detection_rate = (
    clean_detected / len(y_val[malicious_mask]) * 100
)

print("Clean malicious records:", len(X_malicious_scaled))
print("Clean detected:", clean_detected)
print("Clean detection rate:", round(clean_detection_rate, 4), "%")
print("Clean Fast:", clean_fast)
print("Clean Deep:", clean_deep)


# =========================================================
# 10. ATTACK EVALUATION
# =========================================================

print("\n[4/8] Running FGSM and PGD experiments...")

attack_functions = {
    "Targeted FGSM": targeted_fgsm,
    "Targeted PGD": targeted_pgd
}

all_results = []
all_examples = []

for attack_name, attack_function in attack_functions.items():

    for epsilon in EPSILONS:

        print(
            f"\nRunning {attack_name} | epsilon={epsilon}"
        )

        start_time = time.perf_counter()

        if attack_name == "Targeted PGD":

            X_adv_scaled = attack_function(
                X_malicious_scaled,
                epsilon,
                PGD_STEPS
            )

        else:

            X_adv_scaled = attack_function(
                X_malicious_scaled,
                epsilon
            )

        # Convert adversarial standardized inputs back to raw scale.
        X_adv_raw = scaler.inverse_transform(
            X_adv_scaled
        ).astype(np.float32)

        # =====================================================
        # DIAGNOSTIC: ADVERSARIAL FEATURE VALIDITY
        # =====================================================

        print("\nINTEGER FEATURE VALIDITY CHECK")
        print("=" * 65)

        integer_features = [
            col for col in feature_cols
            if np.isclose(
                scaler.inverse_transform(scaled_train_df[feature_cols].to_numpy(dtype=np.float64))[:, feature_cols.index(col)],
                np.round(
                    scaler.inverse_transform(scaled_train_df[feature_cols].to_numpy(dtype=np.float64))[:, feature_cols.index(col)]
                )
            ).all()
        ]

        for col in integer_features:
            j = feature_cols.index(col)
            values = X_adv_raw[:, j]

            fractional = int(
                (~np.isclose(values, np.round(values))).sum()
            )

            train_min_value = scaler.inverse_transform(scaled_train_df[feature_cols].to_numpy(dtype=np.float64))[:, feature_cols.index(col)].min()
            train_max_value = scaler.inverse_transform(scaled_train_df[feature_cols].to_numpy(dtype=np.float64))[:, feature_cols.index(col)].max()

            outside = int(
                ((values < train_min_value) |
                 (values > train_max_value)).sum()
            )

            if fractional or outside:
                print(
                    f"{col}: fractional={fractional}/{len(values)}, "
                    f"outside_train_range={outside}/{len(values)}"
                )

        print("Integer features checked:", len(integer_features))
        print("Feature validity diagnostic complete.")

        if not np.isfinite(X_adv_raw).all():
            raise ValueError(
                f"{attack_name} generated invalid raw values."
            )

        # Evaluate adaptive routing on adversarial records.
        adv_result = run_e3(
            X_adv_raw,
            X_adv_scaled
        )

        adv_predictions = adv_result["predictions"]

        # ASR denominator: records detected by clean E3.
        originally_detected_mask = (
            clean_predictions == 1
        )

        denominator = int(
            originally_detected_mask.sum()
        )

        successful_evasions = (
            originally_detected_mask
            & (adv_predictions == 0)
        )

        evasions = int(
            successful_evasions.sum()
        )

        asr = (
            evasions / denominator * 100
            if denominator > 0 else 0.0
        )

        detected = int(
            np.sum(adv_predictions == 1)
        )

        detection_rate = (
            detected / len(adv_predictions) * 100
        )

        fast_count = int(
            np.sum(adv_result["routes"] == "Fast")
        )

        deep_count = int(
            np.sum(adv_result["routes"] == "Deep")
        )

        total = len(adv_predictions)

        fast_malicious = int(
            np.sum(
                (adv_result["routes"] == "Fast")
                & (adv_predictions == 1)
            )
        )

        fast_evasions = int(
            np.sum(
                (adv_result["routes"] == "Fast")
                & (adv_predictions == 0)
            )
        )

        # Count rows with at least one feature outside the
        # observed clean malicious validation feature ranges.
        outside_validation_range = np.any(
            (X_adv_raw < malicious_raw_min)
            | (X_adv_raw > malicious_raw_max),
            axis=1
        )

        outside_range_count = int(
            outside_validation_range.sum()
        )

        elapsed_seconds = (
            time.perf_counter() - start_time
        )

        result_row = {
            "attack": attack_name,
            "epsilon": epsilon,
            "pgd_steps": (
                PGD_STEPS
                if attack_name == "Targeted PGD"
                else 0
            ),
            "total_malicious_records": total,
            "clean_detected_denominator": denominator,
            "successful_evasions": evasions,
            "ASR_percent": asr,
            "detected_attacks": detected,
            "detection_rate_percent": detection_rate,
            "fast_path_records": fast_count,
            "deep_path_records": deep_count,
            "fast_path_share_percent": (
                fast_count / total * 100 if total else 0.0
            ),
            "deep_path_share_percent": (
                deep_count / total * 100 if total else 0.0
            ),
            "fast_path_malicious_detections": fast_malicious,
            "fast_path_evasions": fast_evasions,
            "outside_malicious_validation_range_rows": (
                outside_range_count
            ),
            "runtime_seconds": elapsed_seconds
        }

        all_results.append(result_row)

        # Save a small record-level sample for inspection.
        for i in range(total):

            all_examples.append({
                "attack": attack_name,
                "epsilon": epsilon,
                "validation_row_index": int(
                    malicious_indices[i]
                ),
                "clean_prediction": int(
                    clean_predictions[i]
                ),
                "adversarial_prediction": int(
                    adv_predictions[i]
                ),
                "clean_route": str(
                    clean_result["routes"][i]
                ),
                "adversarial_route": str(
                    adv_result["routes"][i]
                ),
                "clean_xgb_attack_probability": float(
                    clean_result["xgb_probabilities"][i]
                ),
                "adversarial_xgb_attack_probability": float(
                    adv_result["xgb_probabilities"][i]
                ),
                "successful_evasion": bool(
                    successful_evasions[i]
                ),
                "outside_malicious_validation_range": bool(
                    outside_validation_range[i]
                )
            })

        print("ASR:", round(asr, 4), "%")
        print("Detection rate:", round(detection_rate, 4), "%")
        print("Fast:", fast_count, "| Deep:", deep_count)
        print(
            "Outside validation feature range:",
            outside_range_count
        )


# =========================================================
# 11. SAVE RESULTS
# =========================================================

print("\n[5/8] Saving experiment reports...")

results_df = pd.DataFrame(all_results)
examples_df = pd.DataFrame(all_examples)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

examples_df.to_csv(
    EXAMPLES_PATH,
    index=False
)

summary = {
    "experiment": "E3 Adaptive Adversarial Routing",
    "evaluation_split": "Validation",
    "validation_only": True,
    "total_validation_records": int(len(raw_df)),
    "malicious_validation_records": int(len(X_malicious_scaled)),
    "feature_count": int(len(feature_cols)),
    "risk_threshold": RISK_THRESHOLD,
    "mlp_threshold": MLP_THRESHOLD,
    "feature_squeezing_decimals": SQUEEZE_DECIMALS,
    "epsilon_values": EPSILONS,
    "pgd_steps": PGD_STEPS,
    "attack_space": "Standardized feature space",
    "adversarial_raw_values": (
        "Generated by inverse_transform using StandardScaler"
    ),
    "clean_baseline": {
        "malicious_records": int(len(X_malicious_scaled)),
        "detected_attacks": clean_detected,
        "detection_rate_percent": clean_detection_rate,
        "fast_path_records": clean_fast,
        "deep_path_records": clean_deep
    },
    "attack_results": all_results
}

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# =========================================================
# 12. FINAL SUMMARY
# =========================================================

print("\n[6/8] EXPERIMENT COMPLETED")
print("=" * 65)

print("\nClean E3 baseline:")
print("Detection rate:", round(clean_detection_rate, 4), "%")
print("Fast:", clean_fast)
print("Deep:", clean_deep)

print("\nAdversarial results:")
print(
    results_df[
        [
            "attack",
            "epsilon",
            "ASR_percent",
            "detection_rate_percent",
            "fast_path_share_percent",
            "deep_path_share_percent"
        ]
    ].to_string(index=False)
)

print("\nSaved files:")
print(RESULT_PATH)
print(EXAMPLES_PATH)
print(SUMMARY_PATH)

print("\nImportant:")
print(
    "Results are validation-set digital evasion simulations, "
    "not real network packet attacks."
)

print("=" * 65)



