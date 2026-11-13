
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Project root
ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "LightGBM Train": ROOT / "data/processed/train_lgbm.csv",
    "LightGBM Validation": ROOT / "data/processed/validation_lgbm.csv",
    "LightGBM Test": ROOT / "data/processed/test_lgbm.csv",
    "MLP Train": ROOT / "data/processed/train_mlp_scaled.csv",
    "MLP Validation": ROOT / "data/processed/validation_mlp_scaled.csv",
    "MLP Test": ROOT / "data/processed/test_mlp_scaled.csv",
    "Imputer": ROOT / "models/median_imputer.joblib",
    "Scaler": ROOT / "models/standard_scaler.joblib",
    "Feature Names": ROOT / "models/feature_names.joblib",
}

print("=" * 65)
print("FYP MODEL FILE VERIFICATION")
print("=" * 65)

# 1. Check file existence and size
missing_files = []

for name, path in FILES.items():
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"[OK] {name}: {size_mb:.2f} MB")
    else:
        print(f"[MISSING] {name}: {path}")
        missing_files.append(name)

if missing_files:
    raise FileNotFoundError(
        f"Missing files: {', '.join(missing_files)}"
    )

# 2. Load preprocessing artifacts
imputer = joblib.load(FILES["Imputer"])
scaler = joblib.load(FILES["Scaler"])
feature_names = joblib.load(FILES["Feature Names"])

print("\n--- PREPROCESSING ARTIFACTS ---")
print("Imputer features:", imputer.n_features_in_)
print("Scaler features:", scaler.n_features_in_)
print("Saved feature names:", len(feature_names))

# 3. Validate all six datasets
dataset_results = {}

for name in list(FILES.keys())[:6]:
    path = FILES[name]
    df = pd.read_csv(path)

    feature_columns = [
        col for col in df.columns
        if col not in ["Label", "AttackType"]
    ]

    missing = int(df.isna().sum().sum())

    numeric = df.select_dtypes(include=[np.number])
    infinite = int(np.isinf(numeric.to_numpy()).sum())

    duplicates = int(df.duplicated().sum())

    label_counts = (
        df["Label"].value_counts().to_dict()
        if "Label" in df.columns else {}
    )

    dataset_results[name] = {
        "rows": len(df),
        "columns": len(df.columns),
        "features": len(feature_columns),
        "missing": missing,
        "infinite": infinite,
        "duplicates": duplicates,
        "labels": label_counts,
        "feature_columns": feature_columns,
    }

    print(f"\n--- {name} ---")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Features:", len(feature_columns))
    print("Missing values:", missing)
    print("Infinite values:", infinite)
    print("Duplicate rows:", duplicates)
    print("Labels:", label_counts)

    if len(feature_columns) != 68:
        print("[WARNING] Expected 68 features.")

    if missing != 0 or infinite != 0 or duplicates != 0:
        print("[WARNING] Dataset needs investigation.")

# 4. Check feature alignment
print("\n--- FEATURE ALIGNMENT ---")

reference = dataset_results["LightGBM Train"]["feature_columns"]
alignment_ok = True

for name, result in dataset_results.items():
    if result["feature_columns"] != reference:
        print(f"[FAIL] Feature order mismatch: {name}")
        alignment_ok = False
    else:
        print(f"[OK] Feature order: {name}")

if len(feature_names) != len(reference):
    print("[FAIL] Saved feature-name count does not match datasets.")
    alignment_ok = False

elif list(feature_names) != reference:
    print("[FAIL] Saved feature names/order do not match datasets.")
    alignment_ok = False
else:
    print("[OK] Saved feature names match datasets.")

# 5. Check artifact dimensions
artifacts_ok = (
    imputer.n_features_in_ == len(reference)
    and scaler.n_features_in_ == len(reference)
)

print("\n--- FINAL RESULT ---")

if alignment_ok and artifacts_ok:
    print("[PASS] Feature alignment and artifact dimensions verified.")
else:
    print("[FAIL] Please investigate the warnings above.")

print("\nVerification finished.")
