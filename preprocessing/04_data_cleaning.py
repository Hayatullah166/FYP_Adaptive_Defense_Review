import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATA_PATH = "../data/raw/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"
OUTPUT_PATH = "../data/processed/cicids2018_cleaned.csv"
REPORT_DIR = "../data/reports"

os.makedirs("../data/processed", exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD ORIGINAL DATA
# --------------------------------------------------
print("Loading original dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

initial_rows = len(df)
initial_columns = len(df.columns)

print("Original rows:", initial_rows)
print("Original columns:", initial_columns)

# --------------------------------------------------
# STEP 1: REMOVE ROWS WITH MISSING LABELS
# --------------------------------------------------
before = len(df)
df = df.dropna(subset=["Label"]).copy()
missing_label_removed = before - len(df)

print("\nMissing-label rows removed:", missing_label_removed)

# --------------------------------------------------
# STEP 2: REMOVE EXACT DUPLICATES
# --------------------------------------------------
before = len(df)
df = df.drop_duplicates(keep="first").copy()
duplicates_removed = before - len(df)

print("Duplicate rows removed:", duplicates_removed)

# --------------------------------------------------
# STEP 3: REPLACE INFINITE VALUES WITH NaN
# --------------------------------------------------
numeric_columns = df.select_dtypes(include=np.number).columns

infinite_count = int(
    np.isinf(df[numeric_columns]).sum().sum()
)

df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf], np.nan
)

print("Infinite values replaced:", infinite_count)

# --------------------------------------------------
# STEP 4: IDENTIFY AND REMOVE CONSTANT FEATURES
# --------------------------------------------------
constant_columns = [
    col for col in df.columns
    if col != "Label" and df[col].nunique(dropna=False) <= 1
]

df = df.drop(columns=constant_columns)

print("Constant columns removed:", constant_columns)

# --------------------------------------------------
# STEP 5: IDENTIFY AND REMOVE ALL-ZERO FEATURES
# --------------------------------------------------
numeric_columns = df.select_dtypes(include=np.number).columns

all_zero_columns = [
    col for col in numeric_columns
    if df[col].notna().any()
    and df[col].dropna().eq(0).all()
]

df = df.drop(columns=all_zero_columns)

print("All-zero columns removed:", all_zero_columns)

# --------------------------------------------------
# STEP 6: HANDLE REMAINING MISSING VALUES
# --------------------------------------------------
numeric_columns = df.select_dtypes(include=np.number).columns
categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.drop("Label", errors="ignore")

imputation_report = []

for col in numeric_columns:
    missing_before = int(df[col].isna().sum())

    if missing_before > 0:
        median_value = df[col].median()

        if pd.notna(median_value):
            df[col] = df[col].fillna(median_value)

        imputation_report.append({
            "column": col,
            "missing_values_filled": missing_before,
            "method": "Median"
        })

for col in categorical_columns:
    missing_before = int(df[col].isna().sum())

    if missing_before > 0:
        mode_values = df[col].mode()

        if not mode_values.empty:
            df[col] = df[col].fillna(mode_values.iloc[0])

        imputation_report.append({
            "column": col,
            "missing_values_filled": missing_before,
            "method": "Mode"
        })

# --------------------------------------------------
# STEP 7: FINAL VALIDATION
# --------------------------------------------------
remaining_missing = int(df.isna().sum().sum())

remaining_infinite = int(
    np.isinf(
        df.select_dtypes(include=np.number)
    ).sum().sum()
)

final_rows = len(df)
final_columns = len(df.columns)

print("\n========== CLEANING SUMMARY ==========")
print("Initial rows:", initial_rows)
print("Final rows:", final_rows)
print("Rows removed:", initial_rows - final_rows)
print("Initial columns:", initial_columns)
print("Final columns:", final_columns)
print("Remaining missing values:", remaining_missing)
print("Remaining infinite values:", remaining_infinite)
print("Remaining duplicates:", int(df.duplicated().sum()))

# --------------------------------------------------
# SAVE CLEANED DATASET AND REPORTS
# --------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)

pd.DataFrame(imputation_report).to_csv(
    os.path.join(REPORT_DIR, "imputation_report.csv"),
    index=False
)

cleaning_summary = pd.DataFrame({
    "metric": [
        "initial_rows",
        "final_rows",
        "initial_columns",
        "final_columns",
        "missing_label_rows_removed",
        "duplicate_rows_removed",
        "infinite_values_replaced",
        "constant_columns_removed",
        "all_zero_columns_removed",
        "remaining_missing_values",
        "remaining_infinite_values"
    ],
    "value": [
        initial_rows,
        final_rows,
        initial_columns,
        final_columns,
        missing_label_removed,
        duplicates_removed,
        infinite_count,
        len(constant_columns),
        len(all_zero_columns),
        remaining_missing,
        remaining_infinite
    ]
})

cleaning_summary.to_csv(
    os.path.join(REPORT_DIR, "cleaning_summary.csv"),
    index=False
)

print("\nCleaned dataset saved at:")
print(OUTPUT_PATH)

print("\nCleaning reports saved in:")
print(REPORT_DIR)

print("\nOriginal dataset was NOT modified.")