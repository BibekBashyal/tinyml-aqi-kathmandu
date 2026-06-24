# Dataset Sources

This file documents all data sources used in this project.
Raw data files are NOT committed to this repository. Follow the instructions below to reproduce the dataset.

---

## 1. India CPCB Air Quality Dataset (Primary — Pollution data)

- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/abhisheksjha/time-series-air-quality-data-of-india-2010-2023
- **License:** CC BY-NC-SA 4.0
- **Contents:** Hourly PM2.5, PM10, NO2, SO2, CO, Ozone from 453 stations across 241 Indian cities (2010–2023)
- **Columns used:** `datetime`, `station`, `city`, `state`, `PM2.5`, `PM10`
- **Download:** Log in to Kaggle and download `air-quality-data-of-india-2010-2023.zip`
- **Save as:** `data/raw/india_cpcb_2020_2023.csv`
- **Access date:** June 2026

---

## 2. Open-Meteo Historical Weather API (Weather data — Temperature + Humidity)

- **Source:** Open-Meteo
- **URL:** https://open-meteo.com/en/docs/historical-weather-api
- **License:** CC BY 4.0
- **Contents:** Hourly temperature (°C) and relative humidity (%) for any global location
- **API endpoint:** `https://archive-api.open-meteo.com/v1/archive`
- **Download:** Run `python code/01_data_prep/download_openmeteo.py` — no API key required
- **Save as:** `data/raw/openmeteo_weather_india.csv`
- **Access date:** June 2026

---

## 3. Kathmandu Valley Field Dataset (Primary validation — collected by author)

- **Source:** Author-collected using ESP32 + PMS5003 devices
- **License:** CC BY-NC-SA 4.0 (to be released on Zenodo)
- **Contents:** PM2.5, PM10, temperature, humidity, AQI class label — 3 Kathmandu locations
- **Locations:** Koteshwor roadside, Dhulikhel residential, KU campus elevated
- **Collection period:** October 2026
- **Save as:** `data/field/device1_roadside_oct2026.csv`, `device2_residential_oct2026.csv`, `device3_campus_oct2026.csv`

---

## How to cite Open-Meteo

> Zippenfenig, P. (2023). Open-Meteo.com Weather API. Zenodo. https://doi.org/10.5281/zenodo.7970649

## How to cite the CPCB dataset

> Jha, A. (2023). Time Series Air Quality Data of India (2010-2023). Kaggle. https://www.kaggle.com/datasets/abhisheksjha/time-series-air-quality-data-of-india-2010-2023. License: CC BY-NC-SA 4.0.
