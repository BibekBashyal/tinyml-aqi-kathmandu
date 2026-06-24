"""
01_download_openmeteo.py
------------------------
Downloads hourly temperature and humidity from Open-Meteo
for 5 South Asian cities matching the CPCB pollution dataset.

No API key needed — Open-Meteo is free for non-commercial use.
License: CC BY 4.0

Usage:
    python code/01_data_prep/01_download_openmeteo.py
"""

import requests
import pandas as pd
import time
import os

# ── Configuration ─────────────────────────────────────────────────────────────
START_DATE = "2020-01-01"
END_DATE   = "2023-12-31"

CITIES = {
    "Delhi":   {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":  {"lat": 19.0760, "lon": 72.8777},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Patna":   {"lat": 25.5941, "lon": 85.1376},
}

API_URL = "https://archive-api.open-meteo.com/v1/archive"
OUT_FILE = "data/raw/openmeteo_weather_india.csv"

os.makedirs("data/raw", exist_ok=True)

# ── Fetch data for each city ──────────────────────────────────────────────────
all_records = []

for city, coords in CITIES.items():
    print(f"Fetching weather for {city}...", end=" ", flush=True)

    params = {
        "latitude":    coords["lat"],
        "longitude":   coords["lon"],
        "start_date":  START_DATE,
        "end_date":    END_DATE,
        "hourly":      "temperature_2m,relative_humidity_2m",
        "timezone":    "Asia/Kolkata",
        "format":      "json"
    }

    try:
        r = requests.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        hourly = data.get("hourly", {})
        times  = hourly.get("time", [])
        temps  = hourly.get("temperature_2m", [])
        humids = hourly.get("relative_humidity_2m", [])

        df = pd.DataFrame({
            "datetime":    times,
            "city":        city,
            "temperature": temps,
            "humidity":    humids,
        })

        all_records.append(df)
        print(f"✓  {len(df)} rows")

    except Exception as e:
        print(f"✗  ERROR: {e}")

    time.sleep(1)  # be polite to the API

# ── Combine and save ──────────────────────────────────────────────────────────
if not all_records:
    print("ERROR: No data fetched. Check your internet connection.")
    exit(1)

weather_df = pd.concat(all_records, ignore_index=True)
weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

weather_df.to_csv(OUT_FILE, index=False)

print(f"\n── Summary ───────────────────────────────────────")
print(f"Total rows  : {len(weather_df):,}")
print(f"Cities      : {weather_df['city'].unique().tolist()}")
print(f"Date range  : {weather_df['datetime'].min()} → {weather_df['datetime'].max()}")
print(f"Temperature : {weather_df['temperature'].min():.1f}°C – {weather_df['temperature'].max():.1f}°C")
print(f"Humidity    : {weather_df['humidity'].min():.0f}% – {weather_df['humidity'].max():.0f}%")
print(f"Nulls       : {weather_df.isnull().sum().sum()}")
print(f"\nSaved to    : {OUT_FILE}")
