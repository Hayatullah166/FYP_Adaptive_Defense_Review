
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(r"D:\FYP_Risk_Engine")
DATA_FILE = PROJECT_DIR / "data" / "raw" / "cic.csv"

print("=" * 55)
print("       CICIDS-2018 DATASET INSPECTION")
print("=" * 55)

# 1. Check whether file exists
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

print(f"\nFile: {DATA_FILE.name}")
print(f"Size: {DATA_FILE.stat().st_size / (1024**2):.2f} MB")

# 2. Load dataset
print("\nLoading dataset...")
df = pd.read_csv(DATA_FILE, low_memory=False)

# 3. Dataset dimensions
print("\n1. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 4. Column names
print("\n2. COLUMN NAMES")
for number, column in enumerate(df.columns, start=1):
    print(f"{number}. {column}")

# 5. First five records
print("\n3. FIRST FIVE RECORDS")
print(df.head().to_string())

# 6. Data types
print("\n4. DATA TYPES")
print(df.dtypes.to_string())

# 7. Missing values
print("\n5. MISSING VALUES")
missing = df.isnull().sum()
print(missing[missing > 0].to_string())

# 8. Duplicate records (only count, no deletion)
print("\n6. DUPLICATE RECORDS")
print("Duplicate rows:", df.duplicated().sum())

# 9. Label distribution
print("\n7. LABEL DISTRIBUTION")
if "Label" in df.columns:
    print(df["Label"].value_counts(dropna=False).to_string())
else:
    print("Label column not found.")

print("\nInspection completed. Original CSV was not modified.")
