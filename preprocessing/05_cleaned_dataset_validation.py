import pandas as pd
import numpy as np

path = "../data/processed/cicids2018_cleaned.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(path, low_memory=False)

print("\n========== DATASET ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n========== LABEL DISTRIBUTION ==========")
print(df["Label"].value_counts())

print("\n========== LABEL PERCENTAGES ==========")
print(
    (df["Label"].value_counts(normalize=True) * 100)
    .round(4)
)

print("\n========== DATA TYPES ==========")
print(df.dtypes.value_counts())

print("\n========== NON-NUMERIC FEATURES ==========")
print(df.select_dtypes(exclude=np.number).columns.tolist())

print("\n========== FINAL QUALITY CHECK ==========")
print("Missing:", df.isna().sum().sum())

numeric = df.select_dtypes(include=np.number)
print("Infinite:", np.isinf(numeric).sum().sum())
print("Duplicates:", df.duplicated().sum())

print("\nValidation completed.")