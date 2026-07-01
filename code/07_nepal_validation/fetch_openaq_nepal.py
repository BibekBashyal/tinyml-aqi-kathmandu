"""
fetch_openaq_nepal.py
=====================
Fetches Kathmandu Valley PM2.5 (and temperature/humidity where available)
from the OpenAQ v3 API and saves CSVs ready for the India→Nepal sanity check.

Usage:
    pip install requests pandas
    python fetch_openaq_nepal.py

IMPORTANT: The API key in this script was shared in a chat session.
Revoke it immediately at https://explore.openaq.org (Profile → API Keys)
and paste your NEW key below before running.

Output files (in same directory as this script):
    nepal_stations.csv          -- all stations found in Nepal
    kathmandu_stations.csv      -- Kathmandu-area stations only
    openaq_ktm_pm25.csv         -- hourly PM2.5 for all KTM stations
    openaq_ktm_merged.csv       -- PM2.5 + met (temp/RH) merged on datetime+station
    openaq_ktm_aqi_labeled.csv  -- with AQI class labels, ready for classifier test

Author: Bibek Bashyal, Kathmandu University (2026)
"""

import requests
import pandas as pd
import time
import os
import json
from datetime import datetime, timedelta

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY = "PASTE_YOUR_NEW_KEY_HERE"   # <-- replace with your regenerated key

BASE_URL = "https://api.openaq.org/v3"

# Date range for the sanity check.
# Using the Aug-Oct window that matches your field deployment,
# for the most recent available year. Adjust if needed.
DATE_FROM = "2024-08-01"
DATE_TO   = "2024-10-31"

# Kathmandu Valley bounding box (generous, includes Lalitpur/Bhaktapur)
KTM_LAT_MIN, KTM_LAT_MAX = 27.55, 27.80
KTM_LON_MIN, KTM_LON_MAX = 85.20, 85.55

# Nepal DoE AQI breakpoints (must match your thesis exactly)
AQI_BREAKPOINTS = [
    (0.0,   12.0,  "Good",                        0),
    (12.1,  35.4,  "Moderate",                    1),
    (35.5,  55.4,  "Unhealthy for Sensitive",      2),
    (55.5,  150.4, "Unhealthy",                   3),
    (150.5, 250.4, "Very Unhealthy",              4),
    (250.5, 9999,  "Hazardous",                   5),
]

HEADERS = {
    "X-API-Key": API_KEY,
    "Accept": "application/json",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get(endpoint, params=None):
    url = f"{BASE_URL}{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if resp.status_code == 401:
        raise RuntimeError("Invalid or expired API key. Regenerate at explore.openaq.org.")
    if resp.status_code == 429:
        print("  Rate limited — waiting 10 s...")
        time.sleep(10)
        return get(endpoint, params)
    resp.raise_for_status()
    return resp.json()


def pm25_to_class(pm25):
    for lo, hi, _, cls in AQI_BREAKPOINTS:
        if lo <= pm25 <= hi:
            return cls
    return 5


# ── STEP 1: Find Nepal stations ───────────────────────────────────────────────

def fetch_nepal_stations():
    print("\n── Step 1: Finding Nepal monitoring stations ──────────────────")
    results, page = [], 1
    while True:
        data = get("/locations", params={
            "countries_id": 176,   # Nepal's OpenAQ country ID
            "limit": 100,
            "page": page,
        })
        batch = data.get("results", [])
        if not batch:
            break
        results.extend(batch)
        meta = data.get("meta", {})
        found = meta.get("found", len(results))
        print(f"  Page {page}: {len(batch)} stations | total so far: {len(results)} / {found}")
        if len(results) >= found:
            break
        page += 1
        time.sleep(0.3)

    if not results:
        raise RuntimeError("No stations found. Check API key or try country search.")

    rows = []
    for loc in results:
        lat = loc.get("coordinates", {}).get("latitude")
        lon = loc.get("coordinates", {}).get("longitude")
        params_list = [s.get("parameter", {}).get("name", "?") for s in loc.get("sensors", [])]
        rows.append({
            "location_id": loc.get("id"),
            "name": loc.get("name"),
            "city": loc.get("locality", ""),
            "latitude": lat,
            "longitude": lon,
            "parameters": ", ".join(sorted(set(params_list))),
            "last_updated": loc.get("datetimeLast", {}).get("utc", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv("nepal_stations.csv", index=False)
    print(f"  Saved {len(df)} Nepal stations → nepal_stations.csv")
    return df


# ── STEP 2: Filter Kathmandu Valley stations ──────────────────────────────────

def filter_kathmandu(stations_df):
    print("\n── Step 2: Filtering Kathmandu Valley stations ────────────────")
    ktm = stations_df.dropna(subset=["latitude", "longitude"])
    ktm = ktm[
        (ktm["latitude"]  >= KTM_LAT_MIN) & (ktm["latitude"]  <= KTM_LAT_MAX) &
        (ktm["longitude"] >= KTM_LON_MIN) & (ktm["longitude"] <= KTM_LON_MAX)
    ].copy()
    print(f"  Found {len(ktm)} stations in the Kathmandu Valley bounding box:")
    for _, row in ktm.iterrows():
        print(f"    [{row['location_id']}] {row['name']}  ({row['latitude']:.4f}, {row['longitude']:.4f})")
        print(f"         Parameters: {row['parameters']}")
    ktm.to_csv("kathmandu_stations.csv", index=False)
    print(f"  Saved → kathmandu_stations.csv")
    return ktm


# ── STEP 3: Fetch sensor IDs for PM2.5, temperature, humidity ─────────────────

def get_sensors_for_location(location_id, target_params=("pm25", "temperature", "relativehumidity", "humidity")):
    data = get(f"/locations/{location_id}")
    loc = data.get("results", [{}])[0] if data.get("results") else {}
    sensors = loc.get("sensors", [])
    found = {}
    for s in sensors:
        pname = s.get("parameter", {}).get("name", "").lower().replace(" ", "")
        sid   = s.get("id")
        for tp in target_params:
            if tp in pname and tp not in found:
                found[tp] = sid
    return found


# ── STEP 4: Download measurements for a sensor ────────────────────────────────

def fetch_measurements(sensor_id, date_from, date_to, param_name):
    print(f"    Fetching {param_name} (sensor {sensor_id}): {date_from} → {date_to}")
    records, page = [], 1
    while True:
        try:
            data = get(f"/sensors/{sensor_id}/measurements", params={
                "datetime_from": f"{date_from}T00:00:00Z",
                "datetime_to":   f"{date_to}T23:59:59Z",
                "limit": 1000,
                "page": page,
            })
        except requests.HTTPError as e:
            print(f"      HTTP {e.response.status_code} — skipping")
            break

        batch = data.get("results", [])
        if not batch:
            break
        for m in batch:
            records.append({
                "datetime": m.get("period", {}).get("datetimeFrom", {}).get("utc",
                            m.get("date", {}).get("utc", "")),
                "value": m.get("value"),
            })
        meta = data.get("meta", {})
        found = meta.get("found", len(records))
        if len(records) >= found:
            break
        page += 1
        time.sleep(0.2)

    df = pd.DataFrame(records)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime", "value"])
        df = df.rename(columns={"value": param_name})
    return df


# ── STEP 5: Build merged KTM dataset with AQI labels ─────────────────────────

def fetch_all_ktm_data(ktm_df, date_from, date_to):
    print(f"\n── Step 3: Downloading measurements for {len(ktm_df)} stations ──")
    all_pm25 = []

    for _, row in ktm_df.iterrows():
        loc_id   = int(row["location_id"])
        loc_name = row["name"]
        print(f"\n  [{loc_id}] {loc_name}")

        sensors = get_sensors_for_location(loc_id)
        print(f"    Sensors found: {sensors}")

        if "pm25" not in sensors:
            print(f"    No PM2.5 sensor — skipping this station")
            continue

        # PM2.5
        pm25_df = fetch_measurements(sensors["pm25"], date_from, date_to, "PM2.5")
        if pm25_df.empty:
            print(f"    No PM2.5 data in range — skipping")
            continue

        pm25_df["location_id"] = loc_id
        pm25_df["station"]     = loc_name
        pm25_df["latitude"]    = row["latitude"]
        pm25_df["longitude"]   = row["longitude"]

        # Temperature (optional)
        for tkey in ("temperature",):
            if tkey in sensors:
                t_df = fetch_measurements(sensors[tkey], date_from, date_to, "temperature")
                if not t_df.empty:
                    pm25_df = pd.merge(pm25_df, t_df, on="datetime", how="left")

        # Humidity (optional)
        for hkey in ("relativehumidity", "humidity"):
            if hkey in sensors:
                h_df = fetch_measurements(sensors[hkey], date_from, date_to, "humidity")
                if not h_df.empty:
                    pm25_df = pd.merge(pm25_df, h_df, on="datetime", how="left")
                break

        print(f"    ✓ {len(pm25_df)} rows | cols: {list(pm25_df.columns)}")
        all_pm25.append(pm25_df)
        time.sleep(0.5)

    if not all_pm25:
        print("\nNo PM2.5 data found for any Kathmandu station in this date range.")
        return pd.DataFrame()

    merged = pd.concat(all_pm25, ignore_index=True)
    merged = merged.sort_values(["station", "datetime"])

    # Save raw merged
    merged.to_csv("openaq_ktm_pm25.csv", index=False)
    print(f"\n── Saved {len(merged)} rows → openaq_ktm_pm25.csv")

    return merged


# ── STEP 6: Apply AQI labels and final clean ──────────────────────────────────

def label_and_clean(df):
    print("\n── Step 4: Applying AQI labels ────────────────────────────────")
    df = df.dropna(subset=["PM2.5"])
    df = df[df["PM2.5"] >= 0].copy()
    df["aqi_class"] = df["PM2.5"].apply(pm25_to_class)

    print("Class distribution:")
    print(df["aqi_class"].value_counts().sort_index().to_string())
    print(f"\nPM2.5 stats:\n{df['PM2.5'].describe().to_string()}")
    if "temperature" in df.columns:
        print(f"\nTemperature stats:\n{df['temperature'].describe().to_string()}")
    if "humidity" in df.columns:
        print(f"\nHumidity stats:\n{df['humidity'].describe().to_string()}")

    df.to_csv("openaq_ktm_aqi_labeled.csv", index=False)
    print(f"\n── Saved {len(df)} labeled rows → openaq_ktm_aqi_labeled.csv")
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if API_KEY == "PASTE_YOUR_NEW_KEY_HERE":
        print("ERROR: Replace API_KEY at the top of this script with your new key.")
        print("Revoke the old key at https://explore.openaq.org → Profile → API Keys")
        exit(1)

    print(f"OpenAQ Nepal Fetcher")
    print(f"Date range: {DATE_FROM} → {DATE_TO}")
    print(f"Kathmandu bbox: lat [{KTM_LAT_MIN}–{KTM_LAT_MAX}], lon [{KTM_LON_MIN}–{KTM_LON_MAX}]")

    stations  = fetch_nepal_stations()
    ktm_stns  = filter_kathmandu(stations)

    if ktm_stns.empty:
        print("\nNo stations found in Kathmandu bounding box.")
        print("Check nepal_stations.csv and adjust bounding box if needed.")
        exit(0)

    raw = fetch_all_ktm_data(ktm_stns, DATE_FROM, DATE_TO)

    if not raw.empty:
        labeled = label_and_clean(raw)
        print("\nDone. Files written:")
        for f in ["nepal_stations.csv", "kathmandu_stations.csv",
                  "openaq_ktm_pm25.csv", "openaq_ktm_aqi_labeled.csv"]:
            if os.path.exists(f):
                size = os.path.getsize(f)
                print(f"  {f}  ({size/1024:.1f} KB)")
    else:
        print("\nNo data downloaded. Try widening the date range or bounding box.")
        print("See kathmandu_stations.csv for what stations were found.")
