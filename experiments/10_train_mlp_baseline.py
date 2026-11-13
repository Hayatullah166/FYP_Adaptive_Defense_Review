
import os
import json
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)

# -----------------------------
# 1. Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/processed/grouped"
OUTPUT_DIR = "models/mlp"
REPORT_DIR = "data/reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("Device:", DEVICE)

# -----------------------------
# 2. Load datasets
# -----------------------------
train_df = pd.read_csv(
    f"{DATA_DIR}/train_mlp_scaled_grouped.csv"
)

val_df = pd.read_csv(
    f"{DATA_DIR}/validation_mlp_scaled_grouped.csv"
)

feature_cols = [
    c for c in train_df.columns
    if c not in ["Label", "AttackType"]
]

if feature_cols != [
    c for c in val_df.columns
    if c not in ["Label", "AttackType"]
]:
    raise ValueError("Train and validation features do not match.")

X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
y_train = train_df["Label"].to_numpy(dtype=np.float32)

X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
y_val = val_df["Label"].to_numpy(dtype=np.int64)

if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
    raise ValueError("Dataset contains NaN or infinite values.")

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Training labels:", np.unique(y_train, return_counts=True))

# -----------------------------
# 3. Convert to PyTorch tensors
# -----------------------------
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

X_val_tensor = torch.tensor(X_val)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2048,
    shuffle=True
)

# -----------------------------
# 4. Define MLP architecture
# -----------------------------
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


model = MLP(len(feature_cols)).to(DEVICE)

# -----------------------------
# 5. Class imbalance handling
# -----------------------------
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
    lr=0.001,
    weight_decay=0.0001
)

print("Positive class weight:", pos_weight.item())

# -----------------------------
# 6. Validation function
# -----------------------------
def get_validation_probabilities():

    model.eval()

    with torch.no_grad():
        logits = model(X_val_tensor.to(DEVICE))
        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy()


# -----------------------------
# 7. Model training
# -----------------------------
EPOCHS = 20
PATIENCE = 5

best_ap = -1
best_epoch = 0
epochs_without_improvement = 0

history = []

start_time = time.time()

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    val_prob = get_validation_probabilities()

    val_ap = average_precision_score(
        y_val,
        val_prob
    )

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
                "best_validation_pr_auc": best_ap
            },
            f"{OUTPUT_DIR}/mlp_baseline_best.pth"
        )

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping activated.")
        break

training_time = time.time() - start_time

# -----------------------------
# 8. Load best model
# -----------------------------
checkpoint = torch.load(
    f"{OUTPUT_DIR}/mlp_baseline_best.pth",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

print("Best epoch:", best_epoch)
print("Best validation PR-AUC:", best_ap)

# -----------------------------
# 9. Validation evaluation
# -----------------------------
val_prob = get_validation_probabilities()

# Standard threshold
pred_050 = (val_prob >= 0.50).astype(int)

# Find best F1 threshold on validation only
precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_prob
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_index = int(np.argmax(f1_values))
best_threshold = float(thresholds[best_index])

pred_tuned = (val_prob >= best_threshold).astype(int)


def calculate_metrics(y_true, y_pred, probabilities):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "PR_AUC": average_precision_score(
            y_true, probabilities
        ),
        "ROC_AUC": roc_auc_score(
            y_true, probabilities
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


metrics_050 = calculate_metrics(
    y_val, pred_050, val_prob
)

metrics_tuned = calculate_metrics(
    y_val, pred_tuned, val_prob
)

results = {
    "best_epoch": best_epoch,
    "best_validation_pr_auc": best_ap,
    "training_time_seconds": training_time,
    "threshold_050": metrics_050,
    "validation_tuned_threshold": best_threshold,
    "tuned_threshold_metrics": metrics_tuned
}

# -----------------------------
# 10. Save reports
# -----------------------------
pd.DataFrame(history).to_csv(
    f"{REPORT_DIR}/mlp_training_history.csv",
    index=False
)

pd.DataFrame([
    {"threshold": 0.50, **metrics_050},
    {
        "threshold": best_threshold,
        **metrics_tuned
    }
]).to_csv(
    f"{REPORT_DIR}/mlp_baseline_metrics.csv",
    index=False
)

pd.DataFrame({
    "y_true": y_val,
    "probability": val_prob,
    "prediction_050": pred_050,
    "prediction_tuned": pred_tuned
}).to_csv(
    f"{REPORT_DIR}/mlp_validation_predictions.csv",
    index=False
)

with open(
    f"{REPORT_DIR}/mlp_baseline_summary.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

print("\n========== MLP BASELINE RESULTS ==========")
print(json.dumps(results, indent=4))
print("\nReports saved successfully.")
print("Training time:", round(training_time, 2), "seconds")
