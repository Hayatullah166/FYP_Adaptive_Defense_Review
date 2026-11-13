
import os
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =====================================================
# 1. REPRODUCIBILITY AND CONFIGURATION
# =====================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/processed/grouped"
MODEL_DIR = "models/mlp"
REPORT_DIR = "data/reports"

CHECKPOINT_PATH = f"{MODEL_DIR}/mlp_baseline_best.pth"

EPSILONS = [0.01, 0.02, 0.05]
PGD_STEPS = 10
SQUEEZE_DECIMALS = [1, 2]

THRESHOLDS = {
    "0.50": 0.50,
    "validation_tuned": 0.9297593832015991
}

os.makedirs(REPORT_DIR, exist_ok=True)

print("Device:", DEVICE)


# =====================================================
# 2. MLP ARCHITECTURE
# =====================================================

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
        return self.network(x).squeeze(1)


# =====================================================
# 3. LOAD CHECKPOINT
# =====================================================

print("\nLoading MLP checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

feature_cols = checkpoint["feature_names"]

model = MLP(len(feature_cols)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded.")
print("Best epoch:", checkpoint["best_epoch"])
print("Features:", len(feature_cols))


# =====================================================
# 4. LOAD TRAINING AND VALIDATION DATA
# =====================================================

print("\nLoading datasets...")

train_df = pd.read_csv(
    f"{DATA_DIR}/train_mlp_scaled_grouped.csv"
)

val_df = pd.read_csv(
    f"{DATA_DIR}/validation_mlp_scaled_grouped.csv"
)

train_features = [
    c for c in train_df.columns
    if c not in ["Label", "AttackType"]
]

val_features = [
    c for c in val_df.columns
    if c not in ["Label", "AttackType"]
]

if train_features != feature_cols or val_features != feature_cols:
    raise ValueError("Feature order does not match MLP checkpoint.")

X_train = train_df[feature_cols].to_numpy(dtype=np.float32)

X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
y_val = val_df["Label"].to_numpy(dtype=np.int64)

# Attack only malicious validation records.
malicious_mask = y_val == 1

X_malicious = X_val[malicious_mask]

print("Validation records:", len(X_val))
print("Malicious records:", len(X_malicious))

# Use observed training feature bounds.
train_min = X_train.min(axis=0)
train_max = X_train.max(axis=0)


# =====================================================
# 5. MODEL PREDICTIONS
# =====================================================

def predict_probabilities(X):

    model.eval()

    with torch.no_grad():

        tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=DEVICE
        )

        logits = model(tensor)
        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy()


# =====================================================
# 6. TARGETED FGSM
# =====================================================

def generate_fgsm(X, epsilon):

    model.eval()

    original = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )

    adversarial = original.clone().detach()
    adversarial.requires_grad_(True)

    # Target class 0 = Benign.
    target = torch.zeros(
        len(original),
        dtype=torch.float32,
        device=DEVICE
    )

    loss_fn = nn.BCEWithLogitsLoss()

    logits = model(adversarial)
    loss = loss_fn(logits, target)

    model.zero_grad()
    loss.backward()

    with torch.no_grad():

        attacked = (
            adversarial - epsilon * adversarial.grad.sign()
        )

        attacked = torch.maximum(
            torch.minimum(attacked, torch.tensor(
                train_max, dtype=torch.float32, device=DEVICE
            )),
            torch.tensor(
                train_min, dtype=torch.float32, device=DEVICE
            )
        )

    return attacked.cpu().numpy()


# =====================================================
# 7. TARGETED PGD
# =====================================================

def generate_pgd(X, epsilon, steps=10):

    model.eval()

    original = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )

    # Random initialization inside epsilon ball.
    adversarial = original + torch.empty_like(original).uniform_(
        -epsilon, epsilon
    )

    adversarial = torch.maximum(
        torch.minimum(adversarial, original + epsilon),
        original - epsilon
    )

    alpha = epsilon / steps

    target = torch.zeros(
        len(original),
        dtype=torch.float32,
        device=DEVICE
    )

    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(steps):

        adversarial = adversarial.detach()
        adversarial.requires_grad_(True)

        logits = model(adversarial)
        loss = loss_fn(logits, target)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():

            adversarial = (
                adversarial - alpha * adversarial.grad.sign()
            )

            # Project into original epsilon ball.
            adversarial = torch.maximum(
                torch.minimum(
                    adversarial,
                    original + epsilon
                ),
                original - epsilon
            )

    return adversarial.detach().cpu().numpy()


# =====================================================
# 8. FEATURE SQUEEZING
# =====================================================

def squeeze_features(X, decimals):

    # Quantize standardized tabular features.
    return np.round(X, decimals=decimals).astype(np.float32)


# =====================================================
# 9. EVALUATION
# =====================================================

results = []


def evaluate(
    attack_name,
    epsilon,
    steps,
    condition,
    X,
    baseline_clean_detected
):

    probabilities = predict_probabilities(X)

    for threshold_name, threshold_value in THRESHOLDS.items():

        predictions = (probabilities >= threshold_value).astype(int)

        detected = int(predictions.sum())
        evasions = int(len(predictions) - detected)

        asr = (
            evasions / baseline_clean_detected * 100
            if baseline_clean_detected > 0 else 0
        )

        detection_rate = (
            detected / len(predictions) * 100
            if len(predictions) > 0 else 0
        )

        results.append({
            "attack": attack_name,
            "epsilon": epsilon,
            "steps": steps,
            "condition": condition,
            "threshold": threshold_name,
            "threshold_value": threshold_value,
            "total_malicious_records": len(predictions),
            "detected_attacks": detected,
            "evasion_successes": evasions,
            "ASR_percent": asr,
            "detection_rate_percent": detection_rate
        })


# =====================================================
# 10. CLEAN INPUT EXPERIMENT
# =====================================================

print("\nEvaluating clean inputs...")

clean_probabilities = predict_probabilities(X_malicious)

clean_detected = int(
    (clean_probabilities >= THRESHOLDS["validation_tuned"]).sum()
)

if clean_detected == 0:
    raise ValueError("No clean malicious records detected at tuned threshold.")

evaluate(
    "Clean",
    0.0,
    0,
    "No squeezing",
    X_malicious,
    clean_detected
)

for decimals in SQUEEZE_DECIMALS:

    squeezed = squeeze_features(X_malicious, decimals)

    evaluate(
        "Clean",
        0.0,
        0,
        f"{decimals} decimal places",
        squeezed,
        clean_detected
    )


# =====================================================
# 11. FGSM AND PGD EXPERIMENTS
# =====================================================

for epsilon in EPSILONS:

    print(f"\nProcessing epsilon: {epsilon}")

    # -------------------------
    # FGSM
    # -------------------------

    fgsm_examples = generate_fgsm(
        X_malicious,
        epsilon
    )

    evaluate(
        "FGSM",
        epsilon,
        1,
        "No squeezing",
        fgsm_examples,
        clean_detected
    )

    for decimals in SQUEEZE_DECIMALS:

        squeezed = squeeze_features(
            fgsm_examples,
            decimals
        )

        evaluate(
            "FGSM",
            epsilon,
            1,
            f"{decimals} decimal places",
            squeezed,
            clean_detected
        )

    # -------------------------
    # PGD
    # -------------------------

    # Reset seed for reproducible PGD initialization.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    pgd_examples = generate_pgd(
        X_malicious,
        epsilon,
        PGD_STEPS
    )

    evaluate(
        "PGD",
        epsilon,
        PGD_STEPS,
        "No squeezing",
        pgd_examples,
        clean_detected
    )

    for decimals in SQUEEZE_DECIMALS:

        squeezed = squeeze_features(
            pgd_examples,
            decimals
        )

        evaluate(
            "PGD",
            epsilon,
            PGD_STEPS,
            f"{decimals} decimal places",
            squeezed,
            clean_detected
        )


# =====================================================
# 12. SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)

output_path = (
    f"{REPORT_DIR}/mlp_feature_squeezing_results.csv"
)

results_df.to_csv(output_path, index=False)

print("\n" + "=" * 65)
print("FEATURE SQUEEZING EXPERIMENT COMPLETED")
print("=" * 65)

print(
    results_df.to_string(index=False)
)

print("\nSaved:")
print(os.path.abspath(output_path))

print("\nValidation dataset was used.")
print("Test dataset was NOT used.")
print("Original MLP checkpoint was NOT modified.")
