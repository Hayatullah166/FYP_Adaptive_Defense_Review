
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ==========================================
# PATHS
# ==========================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(
    ROOT,
    "data", "raw",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT, "data", "processed", "grouped"
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR, "models"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR, "reports"
)

for folder in [OUTPUT_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(folder, exist_ok=True)

# ==========================================
# LOAD AND CLEAN DATA
# ==========================================

print("Loading original dataset...")
df = pd.read_csv(RAW_PATH, low_memory=False)

print("Original shape:", df.shape)

df = df.dropna(subset=["Label"]).copy()

df["AttackType"] = df["Label"].astype(str).str.strip()

df = df.drop_duplicates().copy()

df = df.replace([np.inf, -np.inf], np.nan)

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

df = df.drop(
    columns=[c for c in columns_to_remove if c in df.columns]
)

df = df.drop(columns=["Timestamp"])

df["Label"] = (
    df["AttackType"].str.lower() != "benign"
).astype(int)

X = df.drop(columns=["Label", "AttackType"])
y = df["Label"].reset_index(drop=True)
attack_type = df["AttackType"].reset_index(drop=True)

if X.select_dtypes(exclude=[np.number]).shape[1] > 0:
    raise ValueError("Non-numeric features found.")

X = X.reset_index(drop=True)

print("\nCleaned records:", len(X))
print("Feature count:", X.shape[1])
print("Class distribution:")
print(y.value_counts())

# ==========================================
# CREATE GROUP IDs
# Identical feature vectors = same group
# ==========================================

print("\nCreating identical-feature groups...")

hashes = pd.util.hash_pandas_object(
    X,
    index=False
)

groups, unique_hashes = pd.factorize(hashes)

print("Unique feature groups:", len(unique_hashes))

# Verify that identical groups never have
# conflicting binary labels.

group_check = pd.DataFrame({
    "Group": groups,
    "Label": y
})

conflicts = (
    group_check.groupby("Group")["Label"].nunique() > 1
).sum()

if conflicts:
    raise ValueError(
        f"{conflicts} groups have conflicting binary labels. "
        "Stop and investigate before splitting."
    )

# ==========================================
# GROUPED STRATIFIED SPLIT
# 20 folds: approximately 14/3/3
# ==========================================

print("\nCreating grouped stratified splits...")

splitter = StratifiedGroupKFold(
    n_splits=20,
    shuffle=True,
    random_state=42
)

fold_ids = np.full(len(X), -1, dtype=int)

for fold_number, (_, test_indices) in enumerate(
    splitter.split(X, y, groups)
):
    fold_ids[test_indices] = fold_number

if (fold_ids == -1).any():
    raise ValueError("Some records were not assigned to a fold.")

train_mask = np.isin(fold_ids, range(0, 14))
val_mask = np.isin(fold_ids, range(14, 17))
test_mask = np.isin(fold_ids, range(17, 20))

X_train = X.loc[train_mask].copy()
X_val = X.loc[val_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_val = y.loc[val_mask].copy()
y_test = y.loc[test_mask].copy()

type_train = attack_type.loc[train_mask].copy()
type_val = attack_type.loc[val_mask].copy()
type_test = attack_type.loc[test_mask].copy()

# ==========================================
# VERIFY GROUP SEPARATION
# ==========================================

train_groups = set(groups[train_mask])
val_groups = set(groups[val_mask])
test_groups = set(groups[test_mask])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

print("\nGroup separation verified: no shared groups.")

# ==========================================
# IMPUTATION: TRAIN ONLY
# ==========================================

print("\nFitting median imputer on training data...")

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)

# ==========================================
# SCALING: TRAIN ONLY
# ==========================================

print("Fitting scaler on training data...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

# ==========================================
# SAVE DATASETS
# ==========================================

def save_split(name, features, labels, types):
    output = pd.DataFrame(
        features,
        columns=X.columns
    )

    output["Label"] = labels.to_numpy()
    output["AttackType"] = types.to_numpy()

    output.to_csv(
        os.path.join(OUTPUT_DIR, name + ".csv"),
        index=False
    )

save_split("train_lgbm_grouped", X_train_imp, y_train, type_train)
save_split("validation_lgbm_grouped", X_val_imp, y_val, type_val)
save_split("test_lgbm_grouped", X_test_imp, y_test, type_test)

save_split("train_mlp_scaled_grouped", X_train_scaled, y_train, type_train)
save_split("validation_mlp_scaled_grouped", X_val_scaled, y_val, type_val)
save_split("test_mlp_scaled_grouped", X_test_scaled, y_test, type_test)

# ==========================================
# SAVE PREPROCESSING OBJECTS
# ==========================================

joblib.dump(
    imputer,
    os.path.join(MODEL_DIR, "median_imputer_grouped.joblib")
)

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "standard_scaler_grouped.joblib")
)

joblib.dump(
    list(X.columns),
    os.path.join(MODEL_DIR, "feature_names_grouped.joblib")
)

# ==========================================
# SAVE SPLIT SUMMARY
# ==========================================

summary = []

for name, labels, group_ids in [
    ("train", y_train, train_groups),
    ("validation", y_val, val_groups),
    ("test", y_test, test_groups)
]:
    summary.append({
        "Dataset": name,
        "Rows": len(labels),
        "Unique_Groups": len(group_ids),
        "Benign": int((labels == 0).sum()),
        "Attack": int((labels == 1).sum()),
        "Attack_Percentage": round(
            100 * (labels == 1).mean(), 5
        )
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    os.path.join(REPORT_DIR, "grouped_split_summary.csv"),
    index=False
)

print("\n========== GROUPED SPLIT SUMMARY ==========")
print(summary_df.to_string(index=False))

print("\n========== OUTPUT LOCATION ==========")
print(OUTPUT_DIR)

print("\nGrouped preprocessing completed.")
