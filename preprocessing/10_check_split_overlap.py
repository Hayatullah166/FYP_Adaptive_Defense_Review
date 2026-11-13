
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "data" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

files = {
    "train": DATA / "train_lgbm.csv",
    "validation": DATA / "validation_lgbm.csv",
    "test": DATA / "test_lgbm.csv",
}

frames = {
    name: pd.read_csv(path)
    for name, path in files.items()
}

feature_cols = [
    c for c in frames["train"].columns
    if c not in ["Label", "AttackType"]
]

# Check AttackType differences within identical feature groups
print("=" * 65)
print("1. IDENTICAL FEATURES WITH DIFFERENT ATTACK TYPES")
print("=" * 65)

attack_type_report = []

for name, df in frames.items():
    grouped = df.groupby(feature_cols, dropna=False)

    type_counts = grouped["AttackType"].nunique(dropna=False)
    label_counts = grouped["Label"].nunique(dropna=False)

    different_types = type_counts[type_counts > 1]
    different_labels = label_counts[label_counts > 1]

    print(f"\n{name.upper()}")
    print("Feature groups with multiple AttackTypes:", len(different_types))
    print("Feature groups with multiple binary Labels:", len(different_labels))

    attack_type_report.append({
        "Dataset": name,
        "Groups_With_Multiple_AttackTypes": len(different_types),
        "Groups_With_Multiple_Binary_Labels": len(different_labels),
    })

# Check overlap between splits using exact feature tuples
print("\n" + "=" * 65)
print("2. CROSS-SPLIT FEATURE OVERLAP")
print("=" * 65)

feature_sets = {}

for name, df in frames.items():
    feature_sets[name] = set(
        map(tuple, df[feature_cols].itertuples(index=False, name=None))
    )

pairs = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]

overlap_report = []

for first, second in pairs:
    common = feature_sets[first].intersection(feature_sets[second])

    print(f"\n{first} vs {second}")
    print("Unique feature vectors in common:", len(common))

    overlap_report.append({
        "Split_1": first,
        "Split_2": second,
        "Common_Unique_Feature_Vectors": len(common),
    })

# Save reports
pd.DataFrame(attack_type_report).to_csv(
    REPORTS / "attacktype_duplicate_check.csv", index=False
)

pd.DataFrame(overlap_report).to_csv(
    REPORTS / "cross_split_overlap.csv", index=False
)

print("\nReports saved:")
print(REPORTS / "attacktype_duplicate_check.csv")
print(REPORTS / "cross_split_overlap.csv")
print("\nInvestigation complete. No data was modified.")
