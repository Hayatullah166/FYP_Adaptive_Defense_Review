
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "grouped",
    "train_mlp_scaled_grouped.csv"
)

VAL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "grouped",
    "validation_mlp_scaled_grouped.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mlp",
    "mlp_baseline_best.pth"
)

SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "reports",
    "mlp_baseline_summary.json"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "reports"
)

os.makedirs(REPORT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

EPSILONS = [0.01, 0.02, 0.05]

STEPS = 10

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)


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
# 3. LOAD TRAINED MODEL
# =========================================================

print("\nLoading trained MLP checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

feature_cols = checkpoint["feature_names"]

model = MLP(len(feature_cols)).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")

print("Best epoch:", checkpoint["best_epoch"])

print("Number of features:", len(feature_cols))


# =========================================================
# 4. LOAD VALIDATION THRESHOLD
# =========================================================

with open(SUMMARY_PATH, "r") as file:

    summary = json.load(file)

tuned_threshold = float(
    summary["validation_tuned_threshold"]
)

thresholds = {

    "0.50": 0.50,

    "validation_tuned": tuned_threshold

}

print("\nThresholds:")

print(thresholds)


# =========================================================
# 5. LOAD DATASETS
# =========================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)

val_df = pd.read_csv(VAL_PATH)

X_train = train_df[feature_cols].astype(
    np.float32
)

X_val = val_df[feature_cols].astype(
    np.float32
)

y_val = val_df["Label"].astype(
    int
).to_numpy()


# Validate data

if not np.isfinite(X_train.to_numpy()).all():

    raise ValueError(
        "Training data contains NaN or infinity."
    )

if not np.isfinite(X_val.to_numpy()).all():

    raise ValueError(
        "Validation data contains NaN or infinity."
    )


# Select malicious validation records only

attack_indices = np.where(y_val == 1)[0]

if len(attack_indices) == 0:

    raise ValueError(
        "No malicious validation records found."
    )


X_original = torch.tensor(

    X_val.iloc[attack_indices].to_numpy(),

    dtype=torch.float32,

    device=DEVICE

)

print("Validation records:", len(val_df))

print("Malicious records:", len(attack_indices))


# =========================================================
# 6. TRAINING FEATURE BOUNDS
# =========================================================

# Bounds are calculated from training data only.

lower = torch.tensor(

    X_train.min().to_numpy(dtype=np.float32),

    dtype=torch.float32,

    device=DEVICE

)

upper = torch.tensor(

    X_train.max().to_numpy(dtype=np.float32),

    dtype=torch.float32,

    device=DEVICE

)


# =========================================================
# 7. CLEAN PREDICTIONS
# =========================================================

print("\nCalculating clean predictions...")

with torch.no_grad():

    clean_logits = model(
        X_original
    ).squeeze(1)

    clean_probs = torch.sigmoid(
        clean_logits
    ).numpy()


# =========================================================
# 8. TARGETED PGD ATTACK
# =========================================================

def generate_targeted_pgd(
    model,
    original,
    epsilon,
    steps
):

    # -----------------------------------------------------
    # Random initialization
    # -----------------------------------------------------

    random_noise = torch.empty_like(
        original
    ).uniform_(
        -epsilon,
        epsilon
    )

    adversarial = original + random_noise

    # Target class = 0 (Benign)

    target = torch.zeros(

        adversarial.shape[0],

        dtype=torch.float32,

        device=DEVICE

    )


    # PGD step size

    alpha = epsilon / steps


    # -----------------------------------------------------
    # Iterative optimization
    # -----------------------------------------------------

    for step in range(steps):

        adversarial = (

            adversarial.detach().requires_grad_(True)

        )

        model.zero_grad()

        logits = model(
            adversarial
        ).squeeze(1)


        # Targeted loss:
        # Minimize probability of Attack class.

        loss = nn.BCEWithLogitsLoss()(

            logits,

            target

        )

        loss.backward()


        gradient_sign = adversarial.grad.sign()


        # Move toward Benign target

        updated = (

            adversarial - alpha * gradient_sign

        ).detach()


        # -------------------------------------------------
        # Project perturbation into epsilon ball
        # -------------------------------------------------

        perturbation = updated - original

        perturbation = torch.clamp(

            perturbation,

            min=-epsilon,

            max=epsilon

        )

        adversarial = original + perturbation


    return adversarial


# =========================================================
# 9. EVALUATE CLEAN BASELINE
# =========================================================

results = []

examples = []


for threshold_name, threshold in thresholds.items():

    clean_predictions = (

        clean_probs >= threshold

    ).astype(int)


    results.append({

        "attack": "Clean",

        "epsilon": 0.0,

        "steps": 0,

        "threshold": threshold_name,

        "threshold_value": threshold,

        "total_malicious_records": len(attack_indices),

        "clean_detected_attacks": int(
            clean_predictions.sum()
        ),

        "evasion_successes": 0,

        "ASR_percent": 0.0,

        "attack_record_detection_rate_percent":

            float(
                clean_predictions.mean() * 100
            )

    })


# =========================================================
# 10. GENERATE AND EVALUATE PGD
# =========================================================

print("\nGenerating PGD attacks...")

for epsilon in EPSILONS:

    start_time = time.time()


    # Generate adversarial records

    adversarial = generate_targeted_pgd(

        model,

        X_original,

        epsilon,

        STEPS

    )


    # Get adversarial probabilities

    with torch.no_grad():

        adversarial_logits = model(
            adversarial
        ).squeeze(1)

        adversarial_probs = torch.sigmoid(
            adversarial_logits
        ).numpy()


    # -----------------------------------------------------
    # Evaluate both thresholds
    # -----------------------------------------------------

    for threshold_name, threshold in thresholds.items():

        clean_predictions = (

            clean_probs >= threshold

        )

        adversarial_predictions = (

            adversarial_probs >= threshold

        )


        # ASR denominator:
        # Records correctly detected before attack.

        originally_detected = clean_predictions


        successful_evasions = (

            originally_detected &

            (~adversarial_predictions)

        )


        denominator = int(
            originally_detected.sum()
        )

        successes = int(
            successful_evasions.sum()
        )


        asr = (

            successes / denominator * 100

            if denominator > 0

            else 0.0

        )


        detection_rate = (

            adversarial_predictions.mean() * 100

        )


        results.append({

            "attack": "Targeted PGD",

            "epsilon": epsilon,

            "steps": STEPS,

            "threshold": threshold_name,

            "threshold_value": threshold,

            "total_malicious_records": len(attack_indices),

            "clean_detected_attacks": denominator,

            "evasion_successes": successes,

            "ASR_percent": asr,

            "attack_record_detection_rate_percent":

                float(detection_rate)

        })


    # -----------------------------------------------------
    # Save individual probabilities
    # -----------------------------------------------------

    for i, original_index in enumerate(attack_indices):

        examples.append({

            "validation_row_index": int(
                original_index
            ),

            "epsilon": epsilon,

            "steps": STEPS,

            "clean_probability": float(
                clean_probs[i]
            ),

            "adversarial_probability": float(
                adversarial_probs[i]
            ),

            "clean_prediction_050": int(
                clean_probs[i] >= 0.50
            ),

            "adversarial_prediction_050": int(
                adversarial_probs[i] >= 0.50
            )

        })


    elapsed = time.time() - start_time

    print(

        f"PGD epsilon {epsilon:.3f} completed "

        f"in {elapsed:.2f} seconds"

    )


# =========================================================
# 11. SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(results)

examples_df = pd.DataFrame(examples)


results_path = os.path.join(

    REPORT_DIR,

    "mlp_pgd_attack_results.csv"

)


examples_path = os.path.join(

    REPORT_DIR,

    "mlp_pgd_validation_examples.csv"

)


results_df.to_csv(

    results_path,

    index=False

)


examples_df.to_csv(

    examples_path,

    index=False

)


# =========================================================
# 12. FINAL REPORT
# =========================================================

print("\n========================================")

print("PGD EXPERIMENT COMPLETED")

print("========================================")


print("\nResults:")

print(

    results_df.to_string(
        index=False
    )

)


print("\nSaved files:")

print(results_path)

print(examples_path)


print("\nTest dataset was not used.")

print("Original MLP checkpoint was not modified.")

print("Random initialization seed: 42")

print("PGD iterations:", STEPS)


print("========================================")
