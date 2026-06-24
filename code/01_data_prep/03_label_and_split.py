"""
03_label_and_split.py
---------------------
Takes the merged dataset and:
  1. Caps PM2.5 at 500 µg/m³ (removes sensor ceiling artefact at 1000.0)
  2. Imputes PM10 nulls using per-city mean
  3. Labels each row with a 6-class AQI category (Nepal DoE standard)
  4. Splits 70/15/15 TIME-ORDERED WITHIN EACH CITY (so every city appears in
     train/val/test; earliest hours -> train, latest -> test)
  5. Applies SMOTE only to the training split
  6. Saves train / val / test CSVs

NOTE ON THE SPLIT
-----------------
The split is done per-city and then concatenated. Slicing the globally
city-sorted frame (sort by city, then time) would NOT be time-ordered — it
would put whole cities into different splits (e.g. test = only Patna), leaking
location instead of holding out time. Cross-location experiments are handled
separately in code/05_cross_location/.

Usage:
    pip install imbalanced-learn --break-system-packages   # already satisfied if imblearn imports
    python code/01_data_prep/03_label_and_split.py
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

os.makedirs("data/processed", exist_ok=True)

# ── Nepal DoE AQI breakpoints (PM2.5 µg/m³) ──────────────────────────────────
AQI_CLASSES = [
    (0.0,   12.0,  "Good"),
    (12.1,  35.4,  "Moderate"),
    (35.5,  55.4,  "Unhealthy for Sensitive Groups"),
    (55.5,  150.4, "Unhealthy"),
    (150.5, 250.4, "Very Unhealthy"),
    (250.5, 500.0, "Hazardous"),
]

def pm25_to_class(pm25):
    for cls, (lo, hi, _) in enumerate(AQI_CLASSES):
        if lo <= pm25 <= hi:
            return cls
    return -1  # out of range — will be dropped

# ── Load merged data ──────────────────────────────────────────────────────────
in_path = "data/processed/merged_india_aq_weather.csv"
print(f"Loading: {in_path}")
df = pd.read_csv(in_path)
df["datetime"] = pd.to_datetime(df["datetime"])
print(f"  Loaded : {len(df):,} rows")

# ── Fix 1: Cap PM2.5 at 500 µg/m³ ────────────────────────────────────────────
print("\n── Fix 1: PM2.5 ceiling ──────────────────────────")
clipped = (df["PM2.5"] > 500).sum()
print(f"  Rows with PM2.5 > 500 µg/m³ : {clipped:,}  → capped at 500")
df["PM2.5"] = df["PM2.5"].clip(upper=500.0)
print(f"  PM2.5 range after cap        : {df['PM2.5'].min():.1f} – {df['PM2.5'].max():.1f} µg/m³")

# ── Fix 2: Impute PM10 nulls using per-city mean ──────────────────────────────
print("\n── Fix 2: PM10 imputation ────────────────────────")
pm10_nulls = df["PM10"].isnull().sum()
print(f"  PM10 nulls before : {pm10_nulls:,}  ({pm10_nulls/len(df)*100:.1f}%)")
city_pm10_mean = df.groupby("city")["PM10"].transform("mean")
df["PM10"] = df["PM10"].fillna(city_pm10_mean)
print(f"  PM10 nulls after  : {df['PM10'].isnull().sum():,}")

# ── Step 1: Label AQI classes ─────────────────────────────────────────────────
print("\n── Step 1: Labeling AQI classes ──────────────────")
df["aqi_class"] = df["PM2.5"].apply(pm25_to_class)
invalid = (df["aqi_class"] == -1).sum()
if invalid > 0:
    print(f"  Dropping {invalid} out-of-range rows")
df = df[df["aqi_class"] >= 0].reset_index(drop=True)

print(f"\n  Class distribution (full labeled set):")
total = len(df)
for cls, (lo, hi, label) in enumerate(AQI_CLASSES):
    count = (df["aqi_class"] == cls).sum()
    bar   = "█" * int(count / total * 40)
    print(f"  {cls} {label:<35} {count:6,}  ({count/total*100:4.1f}%)  {bar}")

df.to_csv("data/processed/dataset_labeled.csv", index=False)
print(f"\n  Saved: data/processed/dataset_labeled.csv  ({len(df):,} rows)")

# ── Step 2: Time-aware split, PER CITY then concatenated ─────────────────────
print("\n── Step 2: Time-aware split (per city, 70/15/15) ─")
FEATURES = ["PM2.5", "PM10", "temperature", "humidity"]

train_parts, val_parts, test_parts = [], [], []
for city, grp in df.groupby("city"):
    grp = grp.sort_values("datetime").reset_index(drop=True)
    n   = len(grp)
    tr  = int(n * 0.70)
    va  = int(n * 0.85)
    train_parts.append(grp.iloc[:tr])
    val_parts.append(grp.iloc[tr:va])
    test_parts.append(grp.iloc[va:])

train_raw = pd.concat(train_parts).sort_values(["city", "datetime"]).reset_index(drop=True)
val_raw   = pd.concat(val_parts).sort_values(["city", "datetime"]).reset_index(drop=True)
test_raw  = pd.concat(test_parts).sort_values(["city", "datetime"]).reset_index(drop=True)

print(f"  Train (raw) : {len(train_raw):,} rows  cities={train_raw['city'].nunique()}")
print(f"  Val         : {len(val_raw):,} rows  cities={val_raw['city'].nunique()}")
print(f"  Test        : {len(test_raw):,} rows  cities={test_raw['city'].nunique()}")
print(f"\n  Temporal coverage (per split, all cities present):")
print(f"    Train : {train_raw['datetime'].min().date()} → {train_raw['datetime'].max().date()}")
print(f"    Val   : {val_raw['datetime'].min().date()} → {val_raw['datetime'].max().date()}")
print(f"    Test  : {test_raw['datetime'].min().date()} → {test_raw['datetime'].max().date()}")

# ── Step 3: SMOTE on training set only ───────────────────────────────────────
print("\n── Step 3: SMOTE on training set only ────────────")
X_tr = train_raw[FEATURES].values
y_tr = train_raw["aqi_class"].values.astype(int)

print(f"  Before SMOTE: {dict(sorted(Counter(y_tr).items()))}")
# k_neighbors must be < the smallest minority class size
min_class = min(Counter(y_tr).values())
k = min(5, max(1, min_class - 1))
if k < 5:
    print(f"  (smallest class has {min_class} samples → using k_neighbors={k})")
smote = SMOTE(random_state=42, k_neighbors=k)
X_bal, y_bal = smote.fit_resample(X_tr, y_tr)
print(f"  After SMOTE : {dict(sorted(Counter(y_bal).items()))}")

train_final = pd.DataFrame(X_bal, columns=FEATURES)
train_final["aqi_class"] = y_bal.astype(int)

# Val and test keep city + datetime for cross-location experiments later
val_final  = val_raw[["city", "datetime"] + FEATURES + ["aqi_class"]].reset_index(drop=True)
test_final = test_raw[["city", "datetime"] + FEATURES + ["aqi_class"]].reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────
train_final.to_csv("data/processed/train.csv", index=False)
val_final.to_csv("data/processed/val.csv",     index=False)
test_final.to_csv("data/processed/test.csv",   index=False)

print(f"\n── Final datasets ────────────────────────────────")
print(f"  train.csv : {len(train_final):,} rows  (SMOTE applied)")
print(f"  val.csv   : {len(val_final):,} rows")
print(f"  test.csv  : {len(test_final):,} rows")

print(f"\n── Test set class distribution ───────────────────")
for cls, (lo, hi, label) in enumerate(AQI_CLASSES):
    count = (test_final["aqi_class"] == cls).sum()
    print(f"  {cls} {label:<35} {count:5,}")

print(f"\n── Feature stats (train post-SMOTE) ─────────────")
print(train_final[FEATURES].describe().round(2))

print(f"\n✓  Dataset preparation complete.")
print(f"   Next: python code/03_baseline_models/train_random_forest.py")
