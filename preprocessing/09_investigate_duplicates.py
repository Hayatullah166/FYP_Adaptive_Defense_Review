
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "data" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

datasets = {
    "train_lgbm": PROCESSED / "train_lgbm.csv",
    "validation_lgbm": PROCESSED / "validation_lgbm.csv",
    "test_lgbm": PROCESSED / "test_lgbm.csv",
}

summary = []

for name, path in datasets.items():
    print(f"\n{'=' * 60}")
    print("Investigating:", name)

    df = pd.read_csv(path)

    feature_cols = [
        c for c in df.columns
        if c not in ["Label", "AttackType"]
    ]

    # Duplicate complete rows
    full_duplicate_count = int(df.duplicated().sum())

    # Hash only the 68 feature values
    feature_hash = pd.util.hash_pandas_object(
        df[feature_cols],
        index=False
    )

    hash_df = pd.DataFrame({
        "feature_hash": feature_hash,
        "Label": df["Label"],
        "AttackType": df["AttackType"]
    })

    feature_duplicate_count = int(
        hash_df.duplicated(
            subset=["feature_hash"]
        ).sum()
    )

    # Check whether identical feature vectors have conflicting labels
    label_counts = (
        hash_df.groupby("feature_hash")["Label"].nunique()
    )

    conflicting_hashes = label_counts[label_counts > 1].index
    conflicting_rows = hash_df[
        hash_df["feature_hash"].isin(conflicting_hashes)
    ]

    conflicting_row_count = len(conflicting_rows)

    print("Total rows:", len(df))
    print("Complete-row duplicates:", full_duplicate_count)
    print("Feature-only duplicate rows:", feature_duplicate_count)
    print("Rows with conflicting labels:", conflicting_row_count)

    summary.append({
        "Dataset": name,
        "Rows": len(df),
        "Complete_Row_Duplicates": full_duplicate_count,
        "Feature_Only_Duplicate_Rows": feature_duplicate_count,
        "Conflicting_Label_Rows": conflicting_row_count
    })

    if conflicting_row_count:
        print("\nConflicting label examples:")
        print(
            conflicting_rows.head(10).to_string(index=False)
        )

    del df, hash_df, feature_hash, label_counts

report = pd.DataFrame(summary)
output = REPORTS / "duplicate_investigation.csv"
report.to_csv(output, index=False)

print("\n" + "=" * 60)
print("INVESTIGATION SUMMARY")
print(report.to_string(index=False))
print("\nReport saved to:", output)
