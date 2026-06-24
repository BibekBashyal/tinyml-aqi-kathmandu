# TinyML-Based AQI Classification on ESP32
## Cross-Location Generalization in the Kathmandu Valley

**Author:** Bibek Bashyal  
**Programme:** M.E. in Computer Engineering, Kathmandu University  
**Status:** Work in progress — Master's thesis 2026

---

## What this project does

This thesis designs, trains, and deploys a TinyML pipeline for offline multi-class Air Quality Index (AQI) classification on an ESP32 microcontroller. The system classifies air quality into six categories (Good → Hazardous) using readings from a Plantower PMS5003 sensor, with no internet connectivity required.

The core research question: **how well does a model trained at one Kathmandu Valley location generalize to other locations** — and what lightweight on-device corrections can close that gap?

---

## Repository structure

```
tinyml-aqi-kathmandu/
├── data/
│   ├── raw/            # Original downloaded datasets (not committed — see DATA_SOURCES.md)
│   ├── processed/      # Cleaned, merged, labeled datasets
│   └── field/          # SD card logs from Kathmandu field deployment
├── code/
│   ├── 01_data_prep/   # Download, merge, clean, label
│   ├── 02_eda/         # Exploratory data analysis notebooks
│   ├── 03_baseline_models/  # LR, RF, XGBoost baselines
│   ├── 04_tinyml_model/     # MLP training, TFLite conversion, INT8 quantization
│   ├── 05_cross_location/   # Cross-location generalization experiments
│   └── 06_analysis/         # Final figures and result tables
├── firmware/
│   ├── data_logger/    # Phase 1: sensor reading + SD card logging
│   └── aqi_classifier/ # Phase 2: TinyML inference on-device
├── models/             # Saved model files (.keras, .tflite, .h)
├── results/            # Metrics CSVs and figures
└── docs/               # Lab notebook, literature tracker, deployment log
```

---

## Hardware

| Component | Model | Notes |
|---|---|---|
| Microcontroller | ESP32 DevKit V1 | Primary inference platform |
| PM sensor | Plantower PMS5003 | UART connection |
| Temp/humidity | DHT22 | Digital pin |
| RTC | DS3231 | I2C, battery-backed |
| SD card | MicroSD module | SPI |
| Power monitor | INA219 | I2C, for energy profiling |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/tinyml-aqi-kathmandu.git
cd tinyml-aqi-kathmandu

# Install Python dependencies
pip install -r requirements.txt

# Download datasets (see data/raw/DATA_SOURCES.md for instructions)
python code/01_data_prep/download_openmeteo.py
```

---

## Data sources

See `data/raw/DATA_SOURCES.md` for full details, download URLs, and license information.

- **Pollution data:** India CPCB Air Quality Dataset (Kaggle, CC BY-NC-SA 4.0)
- **Weather data:** Open-Meteo Historical Weather API (CC BY 4.0)
- **Field data:** Collected by the author in Kathmandu Valley, Nepal (2026)

---

## License

Code: MIT License  
Data: See individual dataset licenses in `data/raw/DATA_SOURCES.md`

---

## Citation

If you use this work, please cite:

> Bashyal, B. (2026). *TinyML-Based AQI Classification on ESP32: Cross-Location Generalization in the Kathmandu Valley*. Master's thesis, Kathmandu University.
