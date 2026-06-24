"""
02_merge_datasets.py
--------------------
Merges CPCB pollution data (PM2.5 + PM10) with
Open-Meteo weather data (temperature + humidity)
by matching on datetime and city.

DATA LAYOUT NOTE
----------------
The CPCB Kaggle dataset is NOT one big CSV. It is ~470 per-station files
(AP001.csv, DL002.csv, ...) with NO city column inside them. The city for
each station lives in `stations_info.csv`, keyed by `file_name`. Inside each
station file the relevant columns are:
    'From Date'        -> hourly timestamp
    'PM2.5 (ug/m3)'    -> PM2.5
    'PM10 (ug/m3)'     -> PM10
Temperature/humidity inside the station files is sparse and inconsistent
(only ~35/81 target stations have a Temp column, mostly <25% populated),
so weather is taken from Open-Meteo instead — see 01_download_openmeteo.py.

This script:
  1. reads stations_info.csv and keeps stations in the target cities
  2. loads each target station file, tags it with its city
  3. averages PM2.5/PM10 across stations per city-hour (one clean row per
     city-hour); set AGGREGATE_PER_CITY_HOUR = False to keep station-level rows
  4. merges with Open-Meteo weather on datetime + city

Usage:
    python code/01_data_prep/02_merge_datasets.py
"""

import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_FOLDER     = "data/raw"
STATIONS_INFO  = os.path.join(RAW_FOLDER, "stations_info.csv")
WEATHER_PATH   = os.path.join(RAW_FOLDER, "openmeteo_weather_india.csv")
OUT_PATH       = "data/processed/merged_india_aq_weather.csv"

TARGET_CITIES  = ["Delhi", "Mumbai", "Kolkata", "Lucknow", "Patna"]
YEAR_MIN, YEAR_MAX = 2020, 2023

# Real column names inside each per-station CPCB file
CPCB_DATE_COL  = "From Date"
CPCB_PM25_COL  = "PM2.5 (ug/m3)"
CPCB_PM10_COL  = "PM10 (ug/m3)"

# Average pollutant readings across all stations in a city for each hour.
# False -> keep one row per station per hour (station-level, larger dataset).
AGGREGATE_PER_CITY_HOUR = True

# ── Load station registry and select target stations ──────────────────────────
if not os.path.exists(STATIONS_INFO):
    print(f"ERROR: {STATIONS_INFO} not found.")
    print("Make sure the CPCB dataset (incl. stations_info.csv) is unzipped in data/raw/")
    exit(1)

stations = pd.read_csv(STATIONS_INFO)
stations["city"] = stations["city"].astype(str).str.strip().str.title()
targets = stations[stations["city"].isin(TARGET_CITIES)].copy()

if targets.empty:
    print(f"ERROR: No stations found for target cities {TARGET_CITIES}")
    exit(1)

print(f"Target stations: {len(targets)} across {targets['city'].nunique()} cities")
print(targets["city"].value_counts().to_string())

# ── Load each target station file ─────────────────────────────────────────────
frames = []
missing = []
for _, row in targets.iterrows():
    fname = row["file_name"]
    city  = row["city"]
    path  = os.path.join(RAW_FOLDER, f"{fname}.csv")
    if not os.path.exists(path):
        missing.append(fname)
        continue

    # Only read the columns we need; some station files omit one of them.
    usecols = lambda c: c in (CPCB_DATE_COL, CPCB_PM25_COL, CPCB_PM10_COL)
    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    if CPCB_DATE_COL not in df.columns or CPCB_PM25_COL not in df.columns:
        continue  # unusable station

    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(df[CPCB_DATE_COL], errors="coerce")
    out["city"]     = city
    out["PM2.5"]    = pd.to_numeric(df[CPCB_PM25_COL], errors="coerce")
    out["PM10"]     = pd.to_numeric(df[CPCB_PM10_COL], errors="coerce") \
                        if CPCB_PM10_COL in df.columns else pd.NA
    out["station"]  = fname
    frames.append(out)

if missing:
    print(f"\nWARNING: {len(missing)} station file(s) missing on disk: {missing[:10]}"
          + (" ..." if len(missing) > 10 else ""))

if not frames:
    print("\nERROR: No usable station files were loaded.")
    exit(1)

pollution = pd.concat(frames, ignore_index=True)
print(f"\nLoaded {len(frames)} station files -> {len(pollution):,} raw rows")

# ── Clean pollution data ──────────────────────────────────────────────────────
pollution = pollution[pollution["datetime"].notna()]
pollution = pollution.dropna(subset=["PM2.5"])
pollution = pollution[pollution["PM2.5"] >= 0]
pollution = pollution[pollution["datetime"].dt.year.between(YEAR_MIN, YEAR_MAX)]
pollution["datetime"] = pollution["datetime"].dt.floor("h")

print(f"After cleaning ({YEAR_MIN}-{YEAR_MAX}): {len(pollution):,} rows")
print(f"  Cities      : {sorted(pollution['city'].unique().tolist())}")
print(f"  PM2.5 range : {pollution['PM2.5'].min():.1f} - {pollution['PM2.5'].max():.1f} ug/m3")

# ── Optionally average across stations per city-hour ──────────────────────────
if AGGREGATE_PER_CITY_HOUR:
    pollution = (
        pollution.groupby(["city", "datetime"], as_index=False)
        .agg(**{"PM2.5": ("PM2.5", "mean"), "PM10": ("PM10", "mean")})
    )
    print(f"\nAggregated to one row per city-hour: {len(pollution):,} rows")

# ── Load weather data ─────────────────────────────────────────────────────────
if not os.path.exists(WEATHER_PATH):
    print(f"\nERROR: Weather file not found: {WEATHER_PATH}")
    print("Run 01_download_openmeteo.py first.")
    exit(1)

print(f"\nLoading weather data from: {WEATHER_PATH}")
weather = pd.read_csv(WEATHER_PATH)
weather["datetime"] = pd.to_datetime(weather["datetime"]).dt.floor("h")
weather["city"]     = weather["city"].astype(str).str.strip().str.title()
print(f"  Weather rows: {len(weather):,}")

# ── Merge ─────────────────────────────────────────────────────────────────────
print("\nMerging on datetime + city...")
merged = pd.merge(pollution, weather, on=["datetime", "city"], how="inner")
merged = merged.sort_values(["city", "datetime"]).reset_index(drop=True)

print(f"  Merged rows : {len(merged):,}")
print(f"  Columns     : {list(merged.columns)}")
print(f"  Nulls       : {merged.isnull().sum().to_dict()}")

# ── Save ──────────────────────────────────────────────────────────────────────
merged.to_csv(OUT_PATH, index=False)
print(f"\nSaved to: {OUT_PATH}")
print("\n-- City breakdown (PM2.5) --")
print(merged.groupby("city")["PM2.5"].agg(["count", "mean", "max"]).round(1).to_string())
