"""
00_explore_raw.py
-----------------
Run this FIRST before anything else.
It inspects the raw CPCB CSV so you can see exact column names,
date format, and PM2.5 ranges before writing the pipeline.

Usage:
    python code/01_data_prep/00_explore_raw.py
"""

import pandas as pd
import os
import glob

# ── Find the raw CSV file ─────────────────────────────────────────────────────
raw_folder = "data/raw"
csv_files = glob.glob(os.path.join(raw_folder, "*.csv"))

if not csv_files:
    print("ERROR: No CSV files found in data/raw/")
    print("Make sure you have downloaded and unzipped the Kaggle dataset there.")
    exit(1)

print(f"Found {len(csv_files)} CSV file(s):")
for f in csv_files:
    size_mb = os.path.getsize(f) / 1e6
    print(f"  {f}  ({size_mb:.1f} MB)")

# Load the first (or largest) CSV
target = max(csv_files, key=os.path.getsize)
print(f"\nLoading: {target}")

df = pd.read_csv(target, nrows=5000)  # load sample first

print(f"\n── Shape ─────────────────────────────────────")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

print(f"\n── Columns ───────────────────────────────────")
for col in df.columns:
    print(f"  '{col}'")

print(f"\n── First 3 rows ──────────────────────────────")
print(df.head(3).to_string())

print(f"\n── Data types ────────────────────────────────")
print(df.dtypes)

print(f"\n── Null counts ───────────────────────────────")
print(df.isnull().sum())

# Try to find PM2.5 column
pm25_cols = [c for c in df.columns if 'pm2' in c.lower() or 'pm_2' in c.lower()]
pm10_cols = [c for c in df.columns if 'pm10' in c.lower() or 'pm_10' in c.lower()]
date_cols  = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
city_cols  = [c for c in df.columns if 'city' in c.lower() or 'station' in c.lower()]

print(f"\n── Detected columns ──────────────────────────")
print(f"  PM2.5 candidates : {pm25_cols}")
print(f"  PM10  candidates : {pm10_cols}")
print(f"  Date  candidates : {date_cols}")
print(f"  City  candidates : {city_cols}")

if pm25_cols:
    col = pm25_cols[0]
    print(f"\n── PM2.5 stats ('{col}') ─────────────────────")
    print(df[col].describe())

print("\n── Done. Update column names in 01_download_openmeteo.py accordingly ──")
