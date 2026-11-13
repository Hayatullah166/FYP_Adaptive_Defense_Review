
import os
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score
)


# =========================================================
# 1. Configuration
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

BASELINE_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "mlp",
    "mlp_baseline_best.pth"
)

ADV_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "mlp",
    "mlp_adversarial_best.pth"
)

REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

RESULT_PATH = os.path.join(
    REPORT_DIR, "mlp_adversarial_comparison_results.csv"
)

EXAMPLES_PATH = os.path.join(
    REPORT_DIR, "mlp_adversarial_comparison_examples.csv"
)

SUMMARY_PATH = os.path.join(
    REPORT_DIR, "mlp_adversarial_comparison_summary.json"
)

DEVICE = torch.device("cpu")

SEED = 42
EPSILONS = [0.01, 0.02, 0.05]
PGD_STEPS = 10


# =========================================================
# 2. Reproducibility
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# 3. Load data
# =========================================================

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

feature_cols = [
    col for col in train_df.columns
    if col not in ["Label", "AttackType"]
]

val_feature_cols = [
    col for col in val_df.columns
    if col not in ["Label", "AttackType"]
]

if feature_cols != val_feature_cols:
    raise ValueError("Feature columns do not match.")

X_train = train_df[feature_cols].to_numpy(
    dtype=np.float32
)

X_val = val_df[feature_cols].to_numpy(
    dtype=np.float32
)

y_val = val_df["Label"].to_numpy(
    dtype=np.int64
)

if not np.isfinite(X_train).all():
    raise ValueError("Non-finite training values found.")

if not np.isfinite(X_val).all():
    raise ValueError("Non-finite validation values found.")

# Attack only malicious validation records.
attack_indices = np.where(y_val == 1)[0]

X_attack = X_val[attack_indices]

print("Validation records:", len(X_val))
print("Validation malicious records:", len(attack_indices))
print("Number of features:", len(feature_cols))


# =========================================================
# 4. Tensor conversion and feature bounds
# =========================================================

X_val_tensor = torch.tensor(
    X_val, dtype=torch.float32, device=DEVICE
)

X_attack_tensor = torch.tensor(
    X_attack, dtype=torch.float32, device=DEVICE
)

lower_bound = torch.tensor(
    X_train.min(axis=0),
    dtype=torch.float32,
    device=DEVICE
)

upper_bound = torch.tensor(
    X_train.max(axis=0),
    dtype=torch.float32,
    device=DEVICE
)


# =========================================================
# 5. MLP architecture
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
# 6. Load both trained models
# =========================================================

def load_model(model_path):

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    if checkpoint["feature_names"] != feature_cols:
        raise ValueError(
            f"Feature mismatch in {model_path}"
        )

    model = MLP(len(feature_cols)).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


models = {
    "Baseline MLP": load_model(BASELINE_MODEL_PATH),
    "Adversarial MLP": load_model(ADV_MODEL_PATH)
}


# =========================================================
# 7. Prediction function
# =========================================================

def predict_probabilities(model, inputs):

    model.eval()

    with torch.no_grad():

        logits = model(inputs).squeeze(1)

        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy()


# =========================================================
# 8. Select validation F1 threshold
# =========================================================

def find_best_f1_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1] /
        (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))

    threshold = float(thresholds[best_index])

    return threshold, float(f1_values[best_index])


# =========================================================
# 9. Targeted FGSM attack
# =========================================================

def generate_targeted_fgsm(model, inputs, epsilon):

    model.eval()

    x = inputs.detach().clone().requires_grad_(True)

    model.zero_grad()

    logits = model(x).squeeze(1)

    # Target label: Benign (0)
    target_benign = torch.zeros_like(logits)

    loss = nn.BCEWithLogitsLoss()(
        logits,
        target_benign
    )

    gradients = torch.autograd.grad(
        loss,
        x,
        only_inputs=True
    )[0]

    # Move against the gradient to reduce attack probability.
    adversarial = (
        x.detach() -
        epsilon * gradients.detach().sign()
    )

    return adversarial.detach()


# =========================================================
# 10. Targeted PGD attack
# =========================================================

def generate_targeted_pgd(
    model,
    inputs,
    epsilon,
    steps=PGD_STEPS
):

    model.eval()

    original = inputs.detach()

    # Random initialization within epsilon ball.
    random_noise = torch.empty_like(original).uniform_(
        -epsilon,
        epsilon
    )

    adversarial = original + random_noise


    alpha = epsilon / steps

    for _ in range(steps):

        x = adversarial.detach().clone().requires_grad_(True)

        model.zero_grad()

        logits = model(x).squeeze(1)

        target_benign = torch.zeros_like(logits)

        loss = nn.BCEWithLogitsLoss()(
            logits,
            target_benign
        )

        gradients = torch.autograd.grad(
            loss,
            x,
            only_inputs=True
        )[0]

        # Targeted update.
        adversarial = (
            x.detach() -
            alpha * gradients.detach().sign()
        )

        # Project perturbation into epsilon ball.
        perturbation = torch.clamp(
            adversarial - original,
            min=-epsilon,
            max=epsilon
        )

        adversarial = original + perturbation


    return adversarial.detach()


# =========================================================
# 11. Evaluate one attack condition
# =========================================================

def evaluate_condition(
    model_name,
    attack_name,
    epsilon,
    probabilities,
    clean_predictions,
    thresholds,
    clean_pr_auc,
    examples
):

    output_rows = []

    for threshold_name, threshold in thresholds.items():

        adversarial_predictions = (
            probabilities >= threshold
        ).astype(int)

        # Only attacks correctly detected before perturbation
        # are included in the ASR denominator.
        originally_detected = (
            clean_predictions[threshold_name] == 1
        )

        successful_evasions = (
            originally_detected &
            (adversarial_predictions == 0)
        )

        conditional_denominator = int(originally_detected.sum())
        successes = int(successful_evasions.sum())

        # Proposal-defined ASR: successful evasions / all malicious records.
        proposal_asr = (
            successes / len(attack_indices) * 100
            if len(attack_indices) > 0 else 0.0
        )

        # Conditional ASR: successful evasions / initially detected attacks.
        conditional_asr = (
            successes / conditional_denominator * 100
            if conditional_denominator > 0 else 0.0
        )

        detection_rate = (
            adversarial_predictions.sum()
            / len(adversarial_predictions) * 100
        )

        output_rows.append({
            "model": model_name,
            "attack": attack_name,
            "epsilon": epsilon,
            "threshold_type": threshold_name,
            "threshold": threshold,
            "clean_validation_PR_AUC": clean_pr_auc,
            "malicious_records": len(attack_indices),
            "clean_detected_attacks": conditional_denominator,
            "evasion_successes": successes,
            "ASR_percent": proposal_asr,
            "ASR_conditional_percent": conditional_asr,
            "adversarial_detection_rate_percent": detection_rate
        })

    return output_rows


# =========================================================
# 12. Run comparison
# =========================================================

all_results = []
all_examples = []
threshold_summary = {}

for model_name, model in models.items():

    print("\n========================================")
    print("Evaluating:", model_name)
    print("========================================")

    clean_probabilities = predict_probabilities(
        model,
        X_val_tensor
    )

    clean_attack_probabilities = clean_probabilities[
        attack_indices
    ]

    # Each model gets its own validation-tuned threshold.
    tuned_threshold, tuned_f1 = find_best_f1_threshold(
        y_val,
        clean_probabilities
    )

    thresholds = {
        "0.50": 0.50,
        "Validation_F1_tuned": tuned_threshold
    }

    clean_predictions = {}

    for threshold_name, threshold in thresholds.items():

        clean_predictions[threshold_name] = (
            clean_attack_probabilities >= threshold
        ).astype(int)

    clean_pr_auc = average_precision_score(
        y_val,
        clean_probabilities
    )

    threshold_summary[model_name] = {
        "clean_validation_PR_AUC": float(clean_pr_auc),
        "validation_F1_tuned_threshold": tuned_threshold,
        "validation_best_F1": tuned_f1
    }

    print("Clean validation PR-AUC:", clean_pr_auc)
    print("F1-tuned threshold:", tuned_threshold)
    print("Best validation F1:", tuned_f1)

    # Clean condition.
    for threshold_name, threshold in thresholds.items():

        detected = int(
            clean_predictions[threshold_name].sum()
        )

        all_results.append({
            "model": model_name,
            "attack": "Clean",
            "epsilon": 0.0,
            "threshold_type": threshold_name,
            "threshold": threshold,
            "clean_validation_PR_AUC": clean_pr_auc,
            "malicious_records": len(attack_indices),
            "clean_detected_attacks": detected,
            "evasion_successes": 0,
            "ASR_percent": 0.0,
            "adversarial_detection_rate_percent":
                detected / len(attack_indices) * 100
        })

    # FGSM and PGD.
    for epsilon in EPSILONS:

        print(f"\nEpsilon: {epsilon}")

        # Targeted FGSM.
        fgsm_tensor = generate_targeted_fgsm(
            model,
            X_attack_tensor,
            epsilon
        )

        fgsm_probabilities = predict_probabilities(
            model,
            fgsm_tensor
        )

        all_results.extend(
            evaluate_condition(
                model_name,
                "Targeted FGSM",
                epsilon,
                fgsm_probabilities,
                clean_predictions,
                thresholds,
                clean_pr_auc,
                all_examples
            )
        )

        for i, original_index in enumerate(attack_indices):

            all_examples.append({
                "model": model_name,
                "attack": "Targeted FGSM",
                "epsilon": epsilon,
                "validation_row_index": int(original_index),
                "clean_probability":
                    float(clean_attack_probabilities[i]),
                "adversarial_probability":
                    float(fgsm_probabilities[i])
            })

        # Reset seed so both models receive the same
        # random-start PGD initialization.
        torch.manual_seed(SEED)

        # Targeted PGD.
        pgd_tensor = generate_targeted_pgd(
            model,
            X_attack_tensor,
            epsilon
        )

        pgd_probabilities = predict_probabilities(
            model,
            pgd_tensor
        )

        all_results.extend(
            evaluate_condition(
                model_name,
                "Targeted PGD",
                epsilon,
                pgd_probabilities,
                clean_predictions,
                thresholds,
                clean_pr_auc,
                all_examples
            )
        )

        for i, original_index in enumerate(attack_indices):

            all_examples.append({
                "model": model_name,
                "attack": "Targeted PGD",
                "epsilon": epsilon,
                "validation_row_index": int(original_index),
                "clean_probability":
                    float(clean_attack_probabilities[i]),
                "adversarial_probability":
                    float(pgd_probabilities[i])
            })


# =========================================================
# 13. Save reports
# =========================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

examples_df = pd.DataFrame(all_examples)

examples_df.to_csv(
    EXAMPLES_PATH,
    index=False
)

summary = {
    "evaluation": "Baseline vs FGSM adversarially trained MLP",
    "dataset": "CICIDS-2018 grouped validation split",
    "malicious_validation_records": int(len(attack_indices)),
    "attacks": ["Targeted FGSM", "Targeted PGD"],
    "epsilons": EPSILONS,
    "pgd_steps": PGD_STEPS,
    "threshold_selection": "Maximum F1 on clean validation data",
    "model_thresholds": threshold_summary,
    "test_dataset_used": False
}

with open(SUMMARY_PATH, "w") as file:
    json.dump(summary, file, indent=4)


# =========================================================
# 14. Final output
# =========================================================

print("\n========================================")
print("MODEL COMPARISON COMPLETED")
print("========================================")

print("\nThreshold summary:")
print(json.dumps(threshold_summary, indent=4))

print("\nResults:")
print(
    results_df[
        [
            "model",
            "attack",
            "epsilon",
            "threshold_type",
            "ASR_percent",
            "adversarial_detection_rate_percent"
        ]
    ].to_string(index=False)
)

print("\nSaved files:")
print(RESULT_PATH)
print(EXAMPLES_PATH)
print(SUMMARY_PATH)

print("\nTest dataset was not used.")
print("Neither model checkpoint was modified.")
print("========================================")
