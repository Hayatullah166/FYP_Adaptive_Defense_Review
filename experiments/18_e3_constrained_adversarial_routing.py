
import os
import json
import time
import random
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

# =========================================================
# 1. CONFIGURATION
# =========================================================

SEED = 42
DEVICE = torch.device("cpu")

EPSILONS = [0.01, 0.02, 0.05]
PGD_STEPS = 10

RISK_THRESHOLD = 0.01
MLP_THRESHOLD = 0.9073078036308289
SQUEEZE_DECIMALS = 2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "grouped")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

RAW_VAL_PATH = os.path.join(DATA_DIR, "validation_lgbm_grouped.csv")
RAW_TRAIN_PATH = os.path.join(DATA_DIR, "train_lgbm_grouped.csv")
SCALED_VAL_PATH = os.path.join(DATA_DIR, "validation_mlp_scaled_grouped.csv")
SCALED_TRAIN_PATH = os.path.join(DATA_DIR, "train_mlp_scaled_grouped.csv")

SCALER_PATH = os.path.join(
    DATA_DIR, "models", "standard_scaler_grouped.joblib"
)
FEATURE_PATH = os.path.join(
    MODEL_DIR, "risk_assessment", "risk_model_feature_names.joblib"
)
RISK_MODEL_PATH = os.path.join(
    MODEL_DIR, "risk_assessment", "xgboost_risk_model.joblib"
)
CHECKPOINT_PATH = os.path.join(
    MODEL_DIR, "mlp", "mlp_adversarial_best.pth"
)

RESULT_PATH = os.path.join(
    REPORT_DIR, "e3_constrained_adversarial_routing_results.csv"
)
EXAMPLE_PATH = os.path.join(
    REPORT_DIR, "e3_constrained_adversarial_routing_examples.csv"
)
SUMMARY_PATH = os.path.join(
    REPORT_DIR, "e3_constrained_adversarial_routing_summary.json"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


print("=" * 70)
print("SCRIPT 18 - CONSTRAINED ADVERSARIAL ROUTING")
print("EPSILON-PRESERVING DOMAIN PROJECTION")
print("=" * 70)


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
# 3. LOAD DATA AND MODELS
# =========================================================

print("\n[1/8] Loading data and models...")

required_files = [
    RAW_VAL_PATH,
    RAW_TRAIN_PATH,
    SCALED_VAL_PATH,
    SCALED_TRAIN_PATH,
    SCALER_PATH,
    FEATURE_PATH,
    RISK_MODEL_PATH,
    CHECKPOINT_PATH
]

missing_files = [
    path for path in required_files
    if not os.path.exists(path)
]

if missing_files:
    raise FileNotFoundError(
        "Missing required files:\n" + "\n".join(missing_files)
    )

raw_df = pd.read_csv(RAW_VAL_PATH)
raw_train_df = pd.read_csv(RAW_TRAIN_PATH)

scaled_val_df = pd.read_csv(SCALED_VAL_PATH)
scaled_train_df = pd.read_csv(SCALED_TRAIN_PATH)

scaler = joblib.load(SCALER_PATH)
feature_cols = list(joblib.load(FEATURE_PATH))
risk_model = joblib.load(RISK_MODEL_PATH)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

mlp_feature_cols = list(checkpoint["feature_names"])

if feature_cols != mlp_feature_cols:
    raise ValueError("XGBoost and MLP feature orders differ.")

if len(feature_cols) != 68:
    raise ValueError(
        f"Expected 68 features, found {len(feature_cols)}."
    )

if scaler.n_features_in_ != len(feature_cols):
    raise ValueError("Scaler feature count mismatch.")

for col in feature_cols:

    for frame_name, frame in [
        ("Raw validation", raw_df),
        ("Raw training", raw_train_df),
        ("Scaled validation", scaled_val_df),
        ("Scaled training", scaled_train_df)
    ]:

        if col not in frame.columns:
            raise ValueError(
                f"Missing feature {col} in {frame_name}."
            )

if "Label" not in raw_df.columns:
    raise ValueError("Label column missing from validation data.")

if "Label" not in raw_train_df.columns:
    raise ValueError("Label column missing from training data.")

if not raw_df["Label"].equals(scaled_val_df["Label"]):
    raise ValueError("Validation labels are not aligned.")

if len(raw_df) != len(scaled_val_df):
    raise ValueError("Raw and scaled validation row counts differ.")

X_raw = raw_df[feature_cols].to_numpy(dtype=np.float64)

X_train_raw = raw_train_df[
    feature_cols
].to_numpy(dtype=np.float64)

X_scaled_csv = scaled_val_df[
    feature_cols
].to_numpy(dtype=np.float64)

X_train_scaled = scaled_train_df[
    feature_cols
].to_numpy(dtype=np.float64)

y_val = raw_df["Label"].to_numpy(dtype=np.int64)

if not np.isfinite(X_raw).all():
    raise ValueError("Raw validation has invalid values.")

if not np.isfinite(X_train_raw).all():
    raise ValueError("Raw training has invalid values.")

if not np.isfinite(X_scaled_csv).all():
    raise ValueError("Scaled validation has invalid values.")

if not np.isfinite(X_train_scaled).all():
    raise ValueError("Scaled training has invalid values.")

# Verify raw/scaled alignment.
X_scaled_from_raw = scaler.transform(X_raw)

alignment_difference = np.abs(
    X_scaled_from_raw - X_scaled_csv
)

alignment_max_difference = float(
    alignment_difference.max()
)

print(
    "Raw/scaled alignment maximum difference:",
    alignment_max_difference
)

if alignment_max_difference > 1e-4:
    raise ValueError(
        "Raw and scaled validation data do not align. "
        "Check scaler and feature order."
    )

model = MLP(len(feature_cols)).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

scale = np.asarray(
    scaler.scale_,
    dtype=np.float64
)

if np.any(scale <= 0):
    raise ValueError("Scaler contains invalid scale values.")

print("Validation rows:", len(X_raw))
print("Training rows:", len(X_train_raw))
print("Features:", len(feature_cols))
print("Scaler:", type(scaler).__name__)
print("Models loaded successfully.")


# =========================================================
# 4. IDENTIFY FEATURE DOMAINS
# =========================================================

print("\n[2/8] Identifying feature domains...")

binary_features = {
    "Fwd PSH Flags",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "ECE Flag Cnt",
    "CWE Flag Count"
}

protocol_features = {"Protocol"}

integer_keywords = [
    "port",
    "duration",
    "tot",
    "count",
    "length",
    "len",
    "subflow",
    "window",
    "packet",
    "header",
    "active max",
    "active min",
    "idle max",
    "idle min"
]

domain_types = {}

for j, col in enumerate(feature_cols):

    lower = col.lower()
    training_values = X_train_raw[:, j]

    if col in binary_features:

        unique_values = np.unique(training_values)

        if np.all(np.isin(unique_values, [0.0, 1.0])):
            domain_types[col] = "binary"
        else:
            domain_types[col] = "continuous"

    elif col in protocol_features:

        domain_types[col] = "categorical"

    elif any(word in lower for word in integer_keywords):

        integer_like = np.all(
            np.isclose(
                training_values,
                np.round(training_values),
                atol=1e-6
            )
        )

        if integer_like:
            domain_types[col] = "integer"
        else:
            domain_types[col] = "continuous"

    else:

        domain_types[col] = "continuous"


category_values = {}

for j, col in enumerate(feature_cols):

    if domain_types[col] == "categorical":

        category_values[col] = np.sort(
            np.unique(X_train_raw[:, j])
        )


domain_counts = {
    kind: sum(
        value == kind
        for value in domain_types.values()
    )
    for kind in [
        "binary",
        "categorical",
        "integer",
        "continuous"
    ]
}

print("Binary features:", domain_counts["binary"])
print("Categorical features:", domain_counts["categorical"])
print("Integer features:", domain_counts["integer"])
print("Continuous features:", domain_counts["continuous"])


# =========================================================
# 5. CONSTRAINED RAW FEATURE PROJECTION
# =========================================================

def project_raw(candidate_raw, original_raw, epsilon):

    candidate = np.asarray(
        candidate_raw,
        dtype=np.float64
    ).copy()

    original = np.asarray(
        original_raw,
        dtype=np.float64
    )

    if candidate.shape != original.shape:
        raise ValueError("Candidate/original shape mismatch.")

    if candidate.shape[1] != len(feature_cols):
        raise ValueError("Feature dimension mismatch.")

    # Raw-space equivalent of the standardized epsilon budget.
    raw_budget = epsilon * scale

    for j, col in enumerate(feature_cols):

        # IMPORTANT:
        # Do not clip to training min/max.
        # Validation values may legitimately be outside
        # the observed training range.
        #
        # The epsilon interval is the primary constraint.

        lower = (
            original[:, j] - raw_budget[j]
        )

        upper = (
            original[:, j] + raw_budget[j]
        )

        values = np.clip(
            candidate[:, j],
            lower,
            upper
        )

        kind = domain_types[col]

        # ---------------------------------------------
        # Binary features
        # ---------------------------------------------

        if kind == "binary":

            for i in range(len(values)):

                feasible = [
                    v for v in [0.0, 1.0]
                    if lower[i] - 1e-9 <= v <= upper[i] + 1e-9
                ]

                if feasible:

                    values[i] = min(
                        feasible,
                        key=lambda v: abs(v - values[i])
                    )

                else:

                    values[i] = original[i, j]

        # ---------------------------------------------
        # Categorical features
        # ---------------------------------------------

        elif kind == "categorical":

            allowed = category_values[col]

            for i in range(len(values)):

                feasible = allowed[
                    (allowed >= lower[i] - 1e-9)
                    & (allowed <= upper[i] + 1e-9)
                ]

                if len(feasible):

                    values[i] = feasible[
                        np.argmin(
                            np.abs(feasible - values[i])
                        )
                    ]

                else:

                    values[i] = original[i, j]

        # ---------------------------------------------
        # Integer features
        # ---------------------------------------------

        elif kind == "integer":

            for i in range(len(values)):

                smallest = int(
                    np.ceil(lower[i] - 1e-9)
                )

                largest = int(
                    np.floor(upper[i] + 1e-9)
                )

                if smallest <= largest:

                    rounded = int(np.rint(values[i]))

                    values[i] = np.clip(
                        rounded,
                        smallest,
                        largest
                    )

                else:

                    values[i] = original[i, j]

        # ---------------------------------------------
        # Continuous features
        # ---------------------------------------------

        # Continuous values remain clipped only to the
        # per-record epsilon interval.

        candidate[:, j] = values

    return candidate


# =========================================================
# 6. VERIFY CONSTRAINTS
# =========================================================

def verify_constraints(
    adversarial_raw,
    original_raw,
    original_scaled,
    epsilon
):

    adversarial_scaled = scaler.transform(
        adversarial_raw
    )

    scaled_delta = (
        adversarial_scaled - original_scaled
    )

    absolute_delta = np.abs(scaled_delta)

    violation_mask = (
        absolute_delta > epsilon + 1e-5
    )

    violation_rows, violation_cols = np.where(
        violation_mask
    )

    binary_invalid = 0
    categorical_invalid = 0
    integer_fractional = 0

    domain_errors = []

    for j, col in enumerate(feature_cols):

        values = adversarial_raw[:, j]
        kind = domain_types[col]

        if kind == "binary":

            bad = ~np.isin(values, [0.0, 1.0])
            count = int(bad.sum())
            binary_invalid += count

        elif kind == "categorical":

            bad = ~np.isin(
                values,
                category_values[col]
            )

            count = int(bad.sum())
            categorical_invalid += count

        elif kind == "integer":

            bad = ~np.isclose(
                values,
                np.round(values),
                atol=1e-6
            )

            count = int(bad.sum())
            integer_fractional += count

        else:

            count = 0

        if count:

            domain_errors.append({
                "feature": col,
                "domain": kind,
                "invalid_values": count
            })

    max_linf = (
        float(absolute_delta.max())
        if absolute_delta.size
        else 0.0
    )

    return {
        "scaled_delta": scaled_delta,
        "max_linf": max_linf,
        "budget_violations": int(
            violation_mask.any(axis=1).sum()
        ),
        "violation_rows": violation_rows,
        "violation_cols": violation_cols,
        "binary_invalid": binary_invalid,
        "categorical_invalid": categorical_invalid,
        "integer_fractional": integer_fractional,
        "domain_errors": domain_errors
    }


# =========================================================
# 7. E3 ROUTING
# =========================================================

def run_e3(raw_input, scaled_input):

    raw_frame = pd.DataFrame(
        raw_input,
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

    deep_mask = ~fast_mask

    predictions = np.zeros(
        len(raw_input),
        dtype=np.int64
    )

    if deep_mask.any():

        deep_inputs = torch.tensor(
            scaled_input[deep_mask],
            dtype=torch.float32,
            device=DEVICE
        )

        squeezed_inputs = torch.round(
            deep_inputs * (10 ** SQUEEZE_DECIMALS)
        ) / (10 ** SQUEEZE_DECIMALS)

        with torch.no_grad():

            logits = model(
                squeezed_inputs
            ).squeeze(1)

            probabilities = torch.sigmoid(
                logits
            ).cpu().numpy()

        predictions[deep_mask] = (
            probabilities >= MLP_THRESHOLD
        ).astype(np.int64)

    return {
        "predictions": predictions,
        "routes": np.where(fast_mask, "Fast", "Deep"),
        "xgb_probabilities": xgb_probabilities,
        "fast_mask": fast_mask,
        "deep_mask": deep_mask
    }


# =========================================================
# 8. CONSTRAINED FGSM / PGD
# =========================================================

def generate_constrained_attack(
    original_raw,
    epsilon,
    method
):

    original_scaled = scaler.transform(
        original_raw
    )

    current_raw = original_raw.copy()

    target_labels = torch.zeros(
        len(original_raw),
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.BCEWithLogitsLoss()

    steps = 1 if method == "FGSM" else PGD_STEPS

    alpha = (
        epsilon
        if method == "FGSM"
        else epsilon / PGD_STEPS
    )

    # PGD random initialization.
    if method == "PGD":

        random_noise = np.random.uniform(
            -epsilon,
            epsilon,
            size=original_scaled.shape
        )

        starting_scaled = (
            original_scaled + random_noise
        )

        starting_raw = scaler.inverse_transform(
            starting_scaled
        )

        current_raw = project_raw(
            starting_raw,
            original_raw,
            epsilon
        )

    for step in range(steps):

        current_scaled = scaler.transform(
            current_raw
        ).astype(np.float32)

        inputs = torch.tensor(
            current_scaled,
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True
        )

        model.zero_grad(set_to_none=True)

        logits = model(inputs).squeeze(1)

        loss = criterion(
            logits,
            target_labels
        )

        loss.backward()

        gradient = (
            inputs.grad.detach().cpu().numpy()
        )

        if method == "FGSM":

            candidate_scaled = (
                current_scaled
                - epsilon * np.sign(gradient)
            )

        else:

            candidate_scaled = (
                current_scaled
                - alpha * np.sign(gradient)
            )

        candidate_raw = scaler.inverse_transform(
            candidate_scaled
        )

        current_raw = project_raw(
            candidate_raw,
            original_raw,
            epsilon
        )

    final_scaled = scaler.transform(
        current_raw
    )

    return current_raw, final_scaled


# =========================================================
# 9. PREPARE MALICIOUS VALIDATION DATA
# =========================================================

print("\n[3/8] Preparing malicious validation records...")

malicious_mask = y_val == 1

malicious_indices = np.where(
    malicious_mask
)[0]

X_mal_raw = X_raw[malicious_mask]

X_mal_scaled = X_scaled_csv[
    malicious_mask
]

if len(X_mal_raw) == 0:
    raise ValueError("No malicious validation records found.")

print("Malicious records:", len(X_mal_raw))


# =========================================================
# 10. CLEAN BASELINE
# =========================================================

print("\n[4/8] Evaluating clean baseline...")

clean_result = run_e3(
    X_mal_raw,
    X_mal_scaled
)

clean_predictions = clean_result["predictions"]

clean_detected = int(
    (clean_predictions == 1).sum()
)

clean_fast = int(
    (clean_result["routes"] == "Fast").sum()
)

clean_deep = int(
    (clean_result["routes"] == "Deep").sum()
)

clean_denominator = clean_detected

print("Clean detected:", clean_detected)
print("Clean Fast:", clean_fast)
print("Clean Deep:", clean_deep)

if clean_denominator == 0:
    raise ValueError(
        "Clean baseline detected zero malicious records. "
        "ASR denominator would be undefined."
    )


# =========================================================
# 11. ATTACK EXPERIMENTS
# =========================================================

print("\n[5/8] Running constrained attack experiments...")

all_results = []
all_examples = []

for method in ["FGSM", "PGD"]:

    for epsilon in EPSILONS:

        print("\n" + "-" * 65)
        print(f"Running {method} | epsilon={epsilon}")
        print("-" * 65)

        start_time = time.perf_counter()

        adv_raw, adv_scaled = generate_constrained_attack(
            X_mal_raw,
            epsilon,
            method
        )

        # Verify using the actual transformed adversarial data.
        verification = verify_constraints(
            adv_raw,
            X_mal_raw,
            X_mal_scaled,
            epsilon
        )

        actual_scaled_delta = verification["scaled_delta"]

        max_linf = verification["max_linf"]

        budget_violations = verification[
            "budget_violations"
        ]

        binary_invalid = verification[
            "binary_invalid"
        ]

        categorical_invalid = verification[
            "categorical_invalid"
        ]

        integer_fractional = verification[
            "integer_fractional"
        ]

        # ---------------------------------------------
        # Print epsilon violations
        # ---------------------------------------------

        if budget_violations:

            print("\nEPSILON VIOLATION DETAILS:")

            for r, c in zip(
                verification["violation_rows"],
                verification["violation_cols"]
            ):

                print({
                    "record": int(r),
                    "feature": feature_cols[c],
                    "original_raw": float(X_mal_raw[r, c]),
                    "adversarial_raw": float(adv_raw[r, c]),
                    "actual_scaled_delta": float(
                        actual_scaled_delta[r, c]
                    ),
                    "epsilon": epsilon
                })

        if integer_fractional:

            print("\nFRACTIONAL INTEGER DETAILS:")

            for j, col in enumerate(feature_cols):

                if domain_types[col] != "integer":
                    continue

                bad = np.where(
                    ~np.isclose(
                        adv_raw[:, j],
                        np.round(adv_raw[:, j]),
                        atol=1e-6
                    )
                )[0]

                for r in bad:

                    print({
                        "record": int(r),
                        "feature": col,
                        "original": float(X_mal_raw[r, j]),
                        "adversarial": float(adv_raw[r, j])
                    })

        if verification["domain_errors"]:

            print("\nDOMAIN ERRORS:")
            print(verification["domain_errors"])

        # Fail safely. Do not save invalid results.
        if budget_violations > 0:
            raise ValueError(
                f"Epsilon budget violation detected for "
                f"{method}, epsilon={epsilon}."
            )

        if binary_invalid > 0:
            raise ValueError("Invalid binary values detected.")

        if categorical_invalid > 0:
            raise ValueError("Invalid categorical values detected.")

        if integer_fractional > 0:
            raise ValueError("Fractional integer values detected.")

        # Confirm generated scaled input matches fresh transform.
        transformed_adv = scaler.transform(adv_raw)

        scaled_recheck_difference = float(
            np.max(
                np.abs(transformed_adv - adv_scaled)
            )
        )

        if scaled_recheck_difference > 1e-5:
            raise ValueError(
                "Adversarial raw/scaled data mismatch."
            )

        # ---------------------------------------------
        # Evaluate adaptive routing
        # ---------------------------------------------

        adv_result = run_e3(
            adv_raw,
            transformed_adv
        )

        adv_predictions = adv_result["predictions"]

        successful_evasions = (
            (clean_predictions == 1)
            & (adv_predictions == 0)
        )

        evasions = int(
            successful_evasions.sum()
        )

        asr = (
            evasions / clean_denominator * 100
        )

        detected = int(
            (adv_predictions == 1).sum()
        )

        detection_rate = (
            detected / len(adv_predictions) * 100
        )

        fast_count = int(
            (adv_result["routes"] == "Fast").sum()
        )

        deep_count = int(
            (adv_result["routes"] == "Deep").sum()
        )

        changed_features = (
            np.abs(adv_raw - X_mal_raw) > 1e-9
        )

        changed_rows = int(
            changed_features.any(axis=1).sum()
        )

        changed_cells = int(
            changed_features.sum()
        )

        runtime = (
            time.perf_counter() - start_time
        )

        # ---------------------------------------------
        # Result row
        # ---------------------------------------------

        result_row = {

            "attack": method,

            "epsilon": epsilon,

            "pgd_steps": (
                PGD_STEPS
                if method == "PGD"
                else 0
            ),

            "malicious_records": len(adv_predictions),

            "clean_detected_denominator": clean_denominator,

            "successful_evasions": evasions,

            "ASR_percent": asr,

            "detected_attacks": detected,

            "detection_rate_percent": detection_rate,

            "fast_records": fast_count,

            "deep_records": deep_count,

            "fast_share_percent": (
                fast_count / len(adv_predictions) * 100
            ),

            "deep_share_percent": (
                deep_count / len(adv_predictions) * 100
            ),

            "rows_with_changed_features": changed_rows,

            "changed_feature_cells": changed_cells,

            "max_actual_scaled_linf": max_linf,

            "epsilon_budget_violations": budget_violations,

            "invalid_binary_values": binary_invalid,

            "invalid_protocol_values": categorical_invalid,

            "fractional_integer_values": integer_fractional,

            "runtime_seconds": runtime
        }

        all_results.append(result_row)

        # ---------------------------------------------
        # Record-level examples
        # ---------------------------------------------

        for i in range(len(adv_predictions)):

            all_examples.append({

                "attack": method,

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

                "clean_xgb_probability": float(
                    clean_result["xgb_probabilities"][i]
                ),

                "adversarial_xgb_probability": float(
                    adv_result["xgb_probabilities"][i]
                ),

                "successful_evasion": bool(
                    successful_evasions[i]
                ),

                "changed_feature_count": int(
                    changed_features[i].sum()
                ),

                "actual_scaled_linf": float(
                    np.max(
                        np.abs(actual_scaled_delta[i])
                    )
                )
            })

        print("\nRESULT:")
        print("ASR:", round(asr, 4), "%")
        print("Detection rate:", round(detection_rate, 4), "%")
        print("Fast:", fast_count, "| Deep:", deep_count)
        print("Changed rows:", changed_rows)
        print("Maximum actual scaled L-infinity:", max_linf)
        print("Epsilon violations:", budget_violations)
        print("Invalid binary values:", binary_invalid)
        print("Invalid protocol values:", categorical_invalid)
        print("Fractional integer values:", integer_fractional)
        print("Runtime:", round(runtime, 4), "seconds")


# =========================================================
# 12. SAVE REPORTS
# =========================================================

print("\n[6/8] Saving experiment reports...")

results_df = pd.DataFrame(all_results)

examples_df = pd.DataFrame(all_examples)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

examples_df.to_csv(
    EXAMPLE_PATH,
    index=False
)

summary = {

    "experiment": "E3 Constrained Adversarial Routing",

    "validation_only": True,

    "attack_space": "Constrained tabular feature-space simulation",

    "not_packet_level": True,

    "feature_count": len(feature_cols),

    "malicious_validation_records": len(X_mal_raw),

    "epsilon_values": EPSILONS,

    "pgd_steps": PGD_STEPS,

    "risk_threshold": RISK_THRESHOLD,

    "mlp_threshold": MLP_THRESHOLD,

    "feature_squeezing_decimals": SQUEEZE_DECIMALS,

    "raw_scaled_alignment_max_difference": alignment_max_difference,

    "projection": {

        "training_raw_min_max_clipping": False,

        "per_record_standardized_epsilon_budget": True,

        "binary_domain_projection": True,

        "observed_protocol_categories_only": True,

        "integer_domain_projection": True,

        "continuous_features_preserved": True,

        "relational_packet_constraints": False
    },

    "domain_feature_counts": domain_counts,

    "clean_baseline": {

        "malicious_records": len(X_mal_raw),

        "detected": clean_detected,

        "fast": clean_fast,

        "deep": clean_deep
    },

    "all_constraint_checks_passed": True,

    "attack_results": all_results
}

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# =========================================================
# 13. FINAL SUMMARY
# =========================================================

print("\n[7/8] FINAL RESULTS")

print("=" * 70)

print(
    results_df[
        [
            "attack",
            "epsilon",
            "ASR_percent",
            "detection_rate_percent",
            "fast_share_percent",
            "deep_share_percent",
            "max_actual_scaled_linf",
            "epsilon_budget_violations",
            "fractional_integer_values"
        ]
    ].to_string(index=False)
)

print("\nSaved reports:")

print(RESULT_PATH)
print(EXAMPLE_PATH)
print(SUMMARY_PATH)

print("\nIMPORTANT INTERPRETATION:")

print(
    "These results represent constrained tabular feature-space "
    "simulations, not real packet-level attacks."
)

print(
    "Binary, categorical, integer and epsilon constraints are checked. "
    "Relational consistency between network-flow features is not guaranteed."
)

print(
    "Validation values are not forcibly clipped to training min/max."
)

print("\n[8/8] SCRIPT 18 COMPLETED SUCCESSFULLY")

print("=" * 70)
