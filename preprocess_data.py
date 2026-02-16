import pandas as pd
import numpy as np
import os

# Check files in current folder
print("Files in this folder:")
print(os.listdir())

# ✅ Your CSV file name
file_name = "Ecommerce Dataset.csv"

# Load dataset
df = pd.read_csv(file_name)

print("\nOriginal Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# -------------------------------------------------
# 1. Remove unwanted column if exists
# -------------------------------------------------
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# -------------------------------------------------
# 2. Replace invalid values with NaN
# -------------------------------------------------
df.replace(["", " ", "NA", "N/A", "null", "None"], np.nan, inplace=True)

# -------------------------------------------------
# 3. Convert ID and ProdID to numeric if they exist
# -------------------------------------------------
if "ID" in df.columns:
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df = df[df["ID"].notna()]
    df = df[df["ID"] != 0]

if "ProdID" in df.columns:
    df["ProdID"] = pd.to_numeric(df["ProdID"], errors="coerce")
    df = df[df["ProdID"].notna()]
    df = df[df["ProdID"] != 0]

# -------------------------------------------------
# 4. Clean text columns
# -------------------------------------------------
text_columns = df.select_dtypes(include="object").columns

for col in text_columns:
    df[col] = df[col].fillna("")
    df[col] = df[col].str.strip()

# -------------------------------------------------
# 5. Remove "|" from image links if exists
# -------------------------------------------------
if "ImageURL" in df.columns:
    df["ImageURL"] = df["ImageURL"].str.replace("|", "", regex=False)

# -------------------------------------------------
# 6. Reset index
# -------------------------------------------------
df.reset_index(drop=True, inplace=True)

print("\nCleaned Dataset Shape:", df.shape)

# -------------------------------------------------
# Save cleaned file
# -------------------------------------------------
df.to_csv("cleaned_data.csv", index=False)

print("\n✅ Data cleaned successfully!")
print("📁 New file created: cleaned_data.csv")
