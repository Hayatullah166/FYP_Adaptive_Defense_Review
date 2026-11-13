
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================================================
# 1. Paths and configuration
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR, "data", "processed", "grouped",
    "train_mlp_scaled_grouped.csv"
)

VAL_PATH = os.path.join(
    BASE_DIR, "data", "processed", "grouped",
    "validation_mlp_scaled_grouped.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "mlp", "mlp_baseline_best.pth"
)

SUMMARY_PATH = os.path.join(
    BASE_DIR, "data", "reports", "mlp_baseline_summary.json"
)

REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

# Epsilon values in standardized feature space
EPSILONS = [0.01, 0.02, 0.05]

# =========================================================
# 2. Recreate the exact trained MLP architecture
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
# 3. Load trained model and threshold
# =========================================================

print("\nLoading trained MLP checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

feature_cols = checkpoint["feature_names"]

model = MLP(len(feature_cols)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with open(SUMMARY_PATH, "r") as f:
    summary = json.load(f)

tuned_threshold = float(
    summary["validation_tuned_threshold"]
)

thresholds = {
    "0.50": 0.50,
    "validation_tuned": tuned_threshold
}

print("Model loaded successfully.")
print("Best epoch:", checkpoint["best_epoch"])
print("Number of features:", len(feature_cols))
print("Validation-tuned threshold:", tuned_threshold)


# =========================================================
# 4. Load validation data
# =========================================================

print("\nLoading validation dataset...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

X_train = train_df[feature_cols].astype(np.float32)
X_val = val_df[feature_cols].astype(np.float32)

y_val = val_df["Label"].astype(int).to_numpy()

if not np.isfinite(X_val.to_numpy()).all():
    raise ValueError("Validation features contain NaN or infinity.")

if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Training features contain NaN or infinity.")

# Only malicious validation records are attacked.
attack_indices = np.where(y_val == 1)[0]

if len(attack_indices) == 0:
    raise ValueError("No malicious records found in validation data.")

X_attack = X_val.iloc[attack_indices].to_numpy()

print("Validation records:", len(val_df))
print("Malicious records:", len(attack_indices))


# =========================================================
# 5. Establish training feature bounds
# =========================================================

# Keep adversarial values within the observed training
# range in standardized feature space.

lower_bounds = X_train.min().to_numpy(dtype=np.float32)
upper_bounds = X_train.max().to_numpy(dtype=np.float32)

lower_tensor = torch.tensor(
    lower_bounds, dtype=torch.float32, device=DEVICE
)

upper_tensor = torch.tensor(
    upper_bounds, dtype=torch.float32, device=DEVICE
)


# =========================================================
# 6. Get clean predictions
# =========================================================

X_clean_tensor = torch.tensor(
    X_attack, dtype=torch.float32, device=DEVICE
)

with torch.no_grad():
    clean_logits = model(X_clean_tensor).squeeze(1)
    clean_probabilities = torch.sigmoid(clean_logits).numpy()

print("\nClean attack-record predictions calculated.")


# =========================================================
# 7. Generate targeted FGSM adversarial examples
# =========================================================

# Target class = 0 (Benign).
#
# We minimize BCE loss toward the benign target.
# Therefore, subtract epsilon * sign(gradient).

def generate_targeted_fgsm(model, inputs, epsilon):

    x = inputs.clone().detach().requires_grad_(True)

    model.zero_grad()

    logits = model(x).squeeze(1)

    target_benign = torch.zeros_like(logits)

    loss = nn.BCEWithLogitsLoss()(
        logits,
        target_benign
    )

    loss.backward()

    gradient_sign = x.grad.sign()

    adversarial = (
        x - epsilon * gradient_sign
    )

    # Keep values inside observed training feature bounds.
    adversarial = torch.maximum(
        torch.minimum(adversarial, upper_tensor),
        lower_tensor
    )

    return adversarial.detach()


# =========================================================
# 8. Evaluate clean and adversarial predictions
# =========================================================

results = []
example_results = []

# Clean baseline predictions
for threshold_name, threshold in thresholds.items():

    clean_predictions = (
        clean_probabilities >= threshold
    ).astype(int)

    clean_detected = int(clean_predictions.sum())

    results.append({
        "attack": "Clean",
        "epsilon": 0.0,
        "threshold": threshold_name,
        "threshold_value": threshold,
        "total_malicious_records": len(attack_indices),
        "clean_detected_attacks": clean_detected,
        "evasion_successes": 0,
        "ASR_percent": 0.0,
        "attack_record_detection_rate_percent":
            clean_detected / len(attack_indices) * 100
    })


print("\nGenerating FGSM attacks...")

for epsilon in EPSILONS:

    start_time = time.time()

    adversarial_tensor = generate_targeted_fgsm(
        model,
        X_clean_tensor,
        epsilon
    )

    with torch.no_grad():

        adversarial_logits = model(
            adversarial_tensor
        ).squeeze(1)

        adversarial_probabilities = (
            torch.sigmoid(adversarial_logits).numpy()
        )

    for threshold_name, threshold in thresholds.items():

        clean_predictions = (
            clean_probabilities >= threshold
        ).astype(int)

        adversarial_predictions = (
            adversarial_probabilities >= threshold
        ).astype(int)

        # Count only attacks correctly detected before
        # perturbation. This is the ASR denominator.
        originally_detected = (
            clean_predictions == 1
        )

        successful_evasions = (
            originally_detected &
            (adversarial_predictions == 0)
        )

        denominator = int(originally_detected.sum())
        successes = int(successful_evasions.sum())

        asr = (
            successes / denominator * 100
            if denominator > 0 else 0.0
        )

        results.append({
            "attack": "Targeted FGSM",
            "epsilon": epsilon,
            "threshold": threshold_name,
            "threshold_value": threshold,
            "total_malicious_records": len(attack_indices),
            "clean_detected_attacks": denominator,
            "evasion_successes": successes,
            "ASR_percent": asr,
            "attack_record_detection_rate_percent":
                int(adversarial_predictions.sum())
                / len(attack_indices) * 100
        })

    # Save individual record scores for analysis.
    for i, original_index in enumerate(attack_indices):

        example_results.append({
            "validation_row_index": int(original_index),
            "epsilon": epsilon,
            "clean_probability": float(clean_probabilities[i]),
            "adversarial_probability":
                float(adversarial_probabilities[i]),
            "clean_prediction_050":
                int(clean_probabilities[i] >= 0.50),
            "adversarial_prediction_050":
                int(adversarial_probabilities[i] >= 0.50)
        })

    print(
        f"Epsilon {epsilon:.3f} completed in "
        f"{time.time() - start_time:.2f} seconds"
    )


# =========================================================
# 9. Save reports
# =========================================================

results_df = pd.DataFrame(results)

report_path = os.path.join(
    REPORT_DIR, "mlp_fgsm_attack_results.csv"
)

results_df.to_csv(report_path, index=False)

examples_df = pd.DataFrame(example_results)

examples_path = os.path.join(
    REPORT_DIR, "mlp_fgsm_validation_examples.csv"
)

examples_df.to_csv(examples_path, index=False)


# =========================================================
# 10. Final summary
# =========================================================

print("\n========================================")
print("FGSM EXPERIMENT COMPLETED")
print("========================================")

print("\nResults:")
print(results_df.to_string(index=False))

print("\nSaved files:")
print(report_path)
print(examples_path)

print("\nTest dataset was not used.")
print("MLP checkpoint was not modified.")
print("========================================")
