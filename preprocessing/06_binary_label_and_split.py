
import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = "../data/processed/cicids2018_cleaned.csv"
OUTPUT_DIR = "../data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading cleaned dataset...")
df = pd.read_csv(INPUT_PATH, low_memory=False)

# Preserve original attack categories
df["AttackType"] = df["Label"].str.strip()

# Remove Timestamp from model inputs
df = df.drop(columns=["Timestamp"])

# Binary encoding: Benign = 0, Attack = 1
df["Label"] = (
    df["AttackType"].str.lower() != "benign"
).astype(int)

print("\n========== BINARY LABEL DISTRIBUTION ==========")
print(df["Label"].value_counts())
print((df["Label"].value_counts(normalize=True) * 100).round(4))

# Model features must exclude both labels
X = df.drop(columns=["Label", "AttackType"])
y = df["Label"]

# 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Split temporary data equally: 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Save each split with binary label and original attack type
for name, features, labels in [
    ("train", X_train, y_train),
    ("validation", X_val, y_val),
    ("test", X_test, y_test)
]:
    output = features.copy()
    output["Label"] = labels
    output["AttackType"] = df.loc[features.index, "AttackType"]
    output.to_csv(f"{OUTPUT_DIR}/{name}.csv", index=False)

    print(f"\n========== {name.upper()} ==========")
    print("Records:", len(output))
    print("Benign:", int((labels == 0).sum()))
    print("Attack:", int((labels == 1).sum()))
    print("Attack percentage:", round(labels.mean() * 100, 4))
    print("\nOriginal attack categories:")
    print(output["AttackType"].value_counts())

print("\nFeature count:", X.shape[1])
print("\nAll three split files saved successfully.")
