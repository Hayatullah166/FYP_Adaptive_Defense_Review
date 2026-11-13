
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# -----------------------------------
# PATHS
# -----------------------------------
RAW_PATH = "../data/raw/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"
OUTPUT_DIR = "../data/processed"
MODEL_DIR = "../models"
REPORT_DIR = "../data/reports"

for folder in [OUTPUT_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(folder, exist_ok=True)

# -----------------------------------
# LOAD ORIGINAL DATA
# -----------------------------------
print("Loading original raw dataset...")
df = pd.read_csv(RAW_PATH, low_memory=False)

print("Original shape:", df.shape)

# -----------------------------------
# BASIC CLEANING
# -----------------------------------

# Remove rows without a target label
df = df.dropna(subset=["Label"]).copy()

# Preserve original attack category
df["AttackType"] = df["Label"].astype(str).str.strip()

# Remove exact duplicate records
df = df.drop_duplicates().copy()

# Replace positive and negative infinity with missing values
df = df.replace([np.inf, -np.inf], np.nan)

# Remove the 10 previously audited constant/all-zero columns
columns_to_remove = [
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg"
]

existing_columns = [
    col for col in columns_to_remove if col in df.columns
]
df = df.drop(columns=existing_columns)

# Timestamp is retained nowhere in model inputs
df = df.drop(columns=["Timestamp"])

# Binary target: Benign = 0, all attacks = 1
df["Label"] = (
    df["AttackType"].str.lower() != "benign"
).astype(int)

# Separate features and targets
X = df.drop(columns=["Label", "AttackType"])
y = df["Label"]
attack_type = df["AttackType"]

# Ensure all model features are numeric
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()

if non_numeric:
    raise ValueError(
        f"Non-numeric model features found: {non_numeric}"
    )

print("\nRecords after basic cleaning:", len(df))
print("Feature count:", X.shape[1])
print("Binary labels:")
print(y.value_counts())

# -----------------------------------
# TRAIN / VALIDATION / TEST SPLIT
# -----------------------------------
X_train, X_temp, y_train, y_temp, type_train, type_temp = (
    train_test_split(
        X, y, attack_type,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
)

X_val, X_test, y_val, y_test, type_val, type_test = (
    train_test_split(
        X_temp, y_temp, type_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )
)

# -----------------------------------
# IMPUTATION: FIT ON TRAIN ONLY
# -----------------------------------
print("\nFitting median imputer on training data only...")

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)

# -----------------------------------
# SCALING: FIT ON TRAIN ONLY
# -----------------------------------
print("Fitting scaler on training data only...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

# -----------------------------------
# SAVE OUTPUT FILES
# -----------------------------------
def save_split(name, features, labels, attack_types):
    output = pd.DataFrame(
        features,
        columns=X.columns
    )

    output["Label"] = labels.to_numpy()
    output["AttackType"] = attack_types.to_numpy()

    output.to_csv(
        f"{OUTPUT_DIR}/{name}.csv",
        index=False
    )

# Imputed but unscaled data for LightGBM
save_split("train_lgbm", X_train_imp, y_train, type_train)
save_split("validation_lgbm", X_val_imp, y_val, type_val)
save_split("test_lgbm", X_test_imp, y_test, type_test)

# Imputed and scaled data for MLP
save_split("train_mlp_scaled", X_train_scaled, y_train, type_train)
save_split("validation_mlp_scaled", X_val_scaled, y_val, type_val)
save_split("test_mlp_scaled", X_test_scaled, y_test, type_test)

# -----------------------------------
# SAVE TRAINED PREPROCESSING OBJECTS
# -----------------------------------
joblib.dump(
    imputer,
    f"{MODEL_DIR}/median_imputer.joblib"
)

joblib.dump(
    scaler,
    f"{MODEL_DIR}/standard_scaler.joblib"
)

joblib.dump(
    list(X.columns),
    f"{MODEL_DIR}/feature_names.joblib"
)

# -----------------------------------
# SUMMARY
# -----------------------------------
print("\n========== FINAL SPLIT SUMMARY ==========")

for name, labels in [
    ("TRAIN", y_train),
    ("VALIDATION", y_val),
    ("TEST", y_test)
]:
    print(f"\n{name}")
    print("Records:", len(labels))
    print("Benign:", int((labels == 0).sum()))
    print("Attack:", int((labels == 1).sum()))

print("\n========== PREPROCESSING CHECK ==========")
print("Train missing after imputation:", np.isnan(X_train_imp).sum())
print("Validation missing after imputation:", np.isnan(X_val_imp).sum())
print("Test missing after imputation:", np.isnan(X_test_imp).sum())

print("\n========== SAVED FILES ==========")
print("LightGBM: train_lgbm.csv, validation_lgbm.csv, test_lgbm.csv")
print("MLP: train_mlp_scaled.csv, validation_mlp_scaled.csv, test_mlp_scaled.csv")
print("Imputer and scaler saved in models folder.")

print("\nLeakage-safe preprocessing completed.")
