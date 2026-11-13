
import os
import numpy as np
import pandas as pd

# ==========================================
# 1. DATASET PATH
# ==========================================

FILE_PATH = (
    "../data/raw/"
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"
)

REPORT_DIR = "../data/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 60)
print("DATA QUALITY ANALYSIS")
print("=" * 60)

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(FILE_PATH, low_memory=False)

print("Dataset loaded successfully!")

# ==========================================
# 3. BASIC INFORMATION
# ==========================================

print("\n--- BASIC INFORMATION ---")

print("Total Rows:", len(df))
print("Total Columns:", len(df.columns))
print("Total Cells:", df.size)
print("Memory Usage (MB):", round(
    df.memory_usage(deep=True).sum() / (1024 ** 2), 2
))

# ==========================================
# 4. MISSING VALUES
# ==========================================

print("\n--- MISSING VALUES ---")

missing = df.isna().sum()
missing_report = missing[missing > 0].sort_values(ascending=False)

print("Total Missing Values:", int(missing.sum()))
print(missing_report.to_string() if len(missing_report) else "None")

# ==========================================
# 5. INFINITE VALUES
# ==========================================

print("\n--- INFINITE VALUES ---")

numeric_df = df.select_dtypes(include=np.number)

infinite = pd.Series(
    np.isinf(numeric_df).sum(),
    name="Infinite Values"
)

infinite_report = infinite[infinite > 0].sort_values(ascending=False)

print("Total Infinite Values:", int(infinite.sum()))
print(infinite_report.to_string() if len(infinite_report) else "None")

# ==========================================
# 6. DUPLICATE ROWS
# ==========================================

print("\n--- DUPLICATE ROWS ---")

duplicate_count = int(df.duplicated().sum())

print("Duplicate Rows:", duplicate_count)

# ==========================================
# 7. LABEL DISTRIBUTION
# ==========================================

print("\n--- LABEL DISTRIBUTION ---")

if "Label" in df.columns:
    label_counts = df["Label"].value_counts(dropna=False)

    print(label_counts.to_string())

    label_report = label_counts.rename_axis("Label").reset_index(
        name="Record Count"
    )

    label_report["Percentage"] = (
        label_report["Record Count"] / len(df) * 100
    ).round(4)
else:
    label_report = pd.DataFrame()
    print("Label column not found!")

# ==========================================
# 8. NUMERIC STATISTICS
# ==========================================

print("\n--- NUMERIC FEATURE STATISTICS ---")

statistics = numeric_df.describe().T

print(statistics.to_string())

# ==========================================
# 9. SAVE REPORTS
# ==========================================

quality_report = pd.DataFrame({
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isna().sum(),
    "Infinite Values": pd.Series({
        col: int(np.isinf(numeric_df[col]).sum())
        if col in numeric_df.columns else 0
        for col in df.columns
    }),
    "Unique Values": df.nunique(dropna=False)
})

quality_report.to_csv(
    os.path.join(REPORT_DIR, "data_quality_report.csv")
)

statistics.to_csv(
    os.path.join(REPORT_DIR, "numeric_statistics.csv")
)

if not label_report.empty:
    label_report.to_csv(
        os.path.join(REPORT_DIR, "label_distribution.csv"),
        index=False
    )

# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing Values:", int(df.isna().sum().sum()))
print("Infinite Values:", int(infinite.sum()))
print("Duplicate Rows:", duplicate_count)

print("\nReports saved in:", REPORT_DIR)
print("Original dataset was NOT modified.")
