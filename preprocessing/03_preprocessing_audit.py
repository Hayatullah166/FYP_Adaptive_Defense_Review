import os
import numpy as np
import pandas as pd

# Dataset path
DATA_PATH = "../data/raw/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"

# Reports folder
REPORT_DIR = "../data/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

print("\n========== DATASET OVERVIEW ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# 1. Missing and infinite values
numeric_df = df.select_dtypes(include=np.number)

missing = df.isna().sum()
infinite = pd.Series(
    np.isinf(numeric_df).sum(),
    index=numeric_df.columns
)

# 2. Duplicate investigation
duplicate_mask = df.duplicated(keep=False)
duplicate_rows = df[duplicate_mask]

print("\n========== DUPLICATE INVESTIGATION ==========")
print("Duplicate occurrences:", df.duplicated().sum())
print("Rows involved in duplicate groups:", duplicate_mask.sum())

if "Label" in df.columns:
    print("\nLabels among duplicate-group rows:")
    print(duplicate_rows["Label"].value_counts(dropna=False))

# 3. Missing and infinite values report
quality_report = pd.DataFrame({
    "missing_values": missing,
    "infinite_values": infinite.reindex(df.columns, fill_value=0)
})

quality_report.to_csv(
    os.path.join(REPORT_DIR, "preprocessing_audit_quality.csv")
)

# 4. Constant and all-zero columns
constant_columns = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

all_zero_columns = [
    col for col in numeric_df.columns
    if (numeric_df[col].fillna(0) == 0).all()
]

print("\n========== CONSTANT COLUMNS ==========")
print(constant_columns)

print("\n========== ALL-ZERO NUMERIC COLUMNS ==========")
print(all_zero_columns)

# 5. Special values in numeric columns
print("\n========== NEGATIVE VALUE COUNTS ==========")
negative_counts = (numeric_df < 0).sum()
print(negative_counts[negative_counts > 0])

# 6. Label analysis
print("\n========== LABEL ANALYSIS ==========")
if "Label" in df.columns:
    print(df["Label"].value_counts(dropna=False))

# 7. Save audit summary
summary = pd.DataFrame({
    "metric": [
        "total_rows",
        "total_columns",
        "missing_cells",
        "infinite_cells",
        "duplicate_occurrences",
        "rows_in_duplicate_groups",
        "constant_columns_count",
        "all_zero_columns_count",
        "missing_labels"
    ],
    "value": [
        len(df),
        len(df.columns),
        int(df.isna().sum().sum()),
        int(np.isinf(numeric_df).sum().sum()),
        int(df.duplicated().sum()),
        int(duplicate_mask.sum()),
        len(constant_columns),
        len(all_zero_columns),
        int(df["Label"].isna().sum()) if "Label" in df.columns else 0
    ]
})

summary.to_csv(
    os.path.join(REPORT_DIR, "preprocessing_audit_summary.csv"),
    index=False
)

print("\nAudit reports saved in:", REPORT_DIR)
print("\nAudit completed. Original dataset was not modified.")