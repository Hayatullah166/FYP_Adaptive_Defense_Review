
import os
import json
import time
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score


# =========================================================
# 1. Configuration
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data", "processed", "grouped",
    "train_mlp_scaled_grouped.csv"
)

VAL_PATH = os.path.join(
    BASE_DIR,
    "data", "processed", "grouped",
    "validation_mlp_scaled_grouped.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models", "mlp")
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

OUTPUT_MODEL = os.path.join(
    MODEL_DIR, "mlp_adversarial_best.pth"
)

HISTORY_PATH = os.path.join(
    REPORT_DIR, "mlp_adversarial_training_history.csv"
)

SUMMARY_PATH = os.path.join(
    REPORT_DIR, "mlp_adversarial_summary.json"
)

DEVICE = torch.device("cpu")

SEED = 42
EPSILON = 0.02

BATCH_SIZE = 2048
EPOCHS = 20
PATIENCE = 5

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


# =========================================================
# 2. Reproducibility
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE)
print("Adversarial training epsilon:", EPSILON)


# =========================================================
# 3. Load training and validation datasets
# =========================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

feature_cols = [
    col for col in train_df.columns
    if col not in ["Label", "AttackType"]
]

if feature_cols != [
    col for col in val_df.columns
    if col not in ["Label", "AttackType"]
]:
    raise ValueError("Training and validation feature columns differ.")

X_train = train_df[feature_cols].to_numpy(
    dtype=np.float32
)

y_train = train_df["Label"].to_numpy(
    dtype=np.float32
)

X_val = val_df[feature_cols].to_numpy(
    dtype=np.float32
)

y_val = val_df["Label"].to_numpy(
    dtype=np.float32
)

if not np.isfinite(X_train).all():
    raise ValueError("Non-finite values found in training data.")

if not np.isfinite(X_val).all():
    raise ValueError("Non-finite values found in validation data.")

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Number of features:", len(feature_cols))
print("Training attacks:", int(y_train.sum()))
print("Training benign:", int(len(y_train) - y_train.sum()))


# =========================================================
# 4. Convert data to PyTorch tensors
# =========================================================

X_train_tensor = torch.tensor(
    X_train, dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.reshape(-1, 1), dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val, dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val.reshape(-1, 1), dtype=torch.float32
)

dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

generator = torch.Generator()
generator.manual_seed(SEED)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator
)


# =========================================================
# 5. Training feature boundaries
# =========================================================

lower_bound = torch.tensor(
    X_train.min(axis=0),
    dtype=torch.float32
)

upper_bound = torch.tensor(
    X_train.max(axis=0),
    dtype=torch.float32
)


# =========================================================
# 6. Define MLP architecture
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


model = MLP(
    input_features=len(feature_cols)
).to(DEVICE)


# =========================================================
# 7. Loss function and optimizer
# =========================================================

positive = float(y_train.sum())
negative = float(len(y_train) - positive)

pos_weight = torch.tensor(
    [negative / positive],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("\nPositive class weight:", pos_weight.item())


# =========================================================
# 8. Generate untargeted FGSM training examples
# =========================================================

def generate_training_fgsm(inputs, labels):

    # Create a separate input tensor for gradient calculation.
    x = inputs.detach().clone().requires_grad_(True)

    # Disable dropout while calculating the attack gradient.
    model.eval()

    logits = model(x)

    # Use original labels: this is label-preserving FGSM.
    attack_loss = criterion(logits, labels)

    gradients = torch.autograd.grad(
        attack_loss,
        x,
        only_inputs=True
    )[0]

    # Untargeted FGSM: move in the direction that increases loss.
    adversarial = (
        x.detach() +
        EPSILON * gradients.detach().sign()
    )

    # Keep features within observed training bounds.
    adversarial = torch.maximum(
        torch.minimum(adversarial, upper_bound),
        lower_bound
    )

    return adversarial.detach()


# =========================================================
# 9. Validation function
# =========================================================

def evaluate_validation():

    model.eval()

    with torch.no_grad():

        logits = model(X_val_tensor).squeeze(1)

        probabilities = torch.sigmoid(logits).numpy()

    val_ap = average_precision_score(
        y_val,
        probabilities
    )

    return val_ap


# =========================================================
# 10. Adversarial training
# =========================================================

best_ap = -1.0
best_epoch = 0
epochs_without_improvement = 0

history = []

start_time = time.time()

print("\nStarting adversarial training...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_records = 0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        # Generate adversarial examples using true labels.
        batch_adv = generate_training_fgsm(
            batch_x,
            batch_y
        )

        # Train on both clean and adversarial examples.
        combined_x = torch.cat(
            [batch_x, batch_adv],
            dim=0
        )

        combined_y = torch.cat(
            [batch_y, batch_y],
            dim=0
        )

        model.train()

        optimizer.zero_grad()

        logits = model(combined_x)

        loss = criterion(
            logits,
            combined_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(combined_x)
        total_records += len(combined_x)

    avg_loss = total_loss / total_records

    # Select model using clean validation PR-AUC.
    val_ap = evaluate_validation()

    history.append({
        "epoch": epoch + 1,
        "training_loss": avg_loss,
        "validation_pr_auc": val_ap
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {avg_loss:.6f} | "
        f"Validation PR-AUC: {val_ap:.6f}"
    )

    if val_ap > best_ap:

        best_ap = val_ap
        best_epoch = epoch + 1

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_names": feature_cols,
                "best_epoch": best_epoch,
                "best_validation_pr_auc": best_ap,
                "training_epsilon": EPSILON,
                "attack_method": "Untargeted FGSM",
                "training_method": "Clean + adversarial examples"
            },
            OUTPUT_MODEL
        )

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("Early stopping activated.")
        break


# =========================================================
# 11. Save training history
# =========================================================

training_time = time.time() - start_time

history_df = pd.DataFrame(history)
history_df.to_csv(HISTORY_PATH, index=False)


# =========================================================
# 12. Save summary
# =========================================================

summary = {
    "training_method": "FGSM adversarial training",
    "attack_method": "Untargeted FGSM",
    "training_epsilon": EPSILON,
    "training_examples": "Clean + adversarial",
    "best_epoch": best_epoch,
    "best_validation_pr_auc": best_ap,
    "training_time_seconds": training_time,
    "number_of_features": len(feature_cols),
    "batch_size": BATCH_SIZE,
    "max_epochs": EPOCHS,
    "early_stopping_patience": PATIENCE,
    "positive_class_weight": pos_weight.item(),
    "device": str(DEVICE)
}

with open(SUMMARY_PATH, "w") as file:
    json.dump(summary, file, indent=4)


# =========================================================
# 13. Final output
# =========================================================

print("\n========================================")
print("ADVERSARIAL TRAINING COMPLETED")
print("========================================")

print("Best epoch:", best_epoch)
print("Best validation PR-AUC:", best_ap)
print("Training time:", round(training_time, 2), "seconds")

print("\nSaved model:")
print(OUTPUT_MODEL)

print("\nSaved reports:")
print(HISTORY_PATH)
print(SUMMARY_PATH)

print("\nBaseline checkpoint was not modified.")
print("Test dataset was not used.")
print("========================================")
