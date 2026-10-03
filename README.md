# TinyML-Based AQI Classification on ESP32
## Cross-Location Generalization in the Kathmandu Valley

**Author:** Bibek Bashyal  
**Programme:** M.E. in Computer Engineering, Kathmandu University  
**Status:** Work in progress — Master's thesis 2026

---

## What this project does

This thesis designs, trains, and deploys a TinyML pipeline for offline multi-class Air Quality Index (AQI) classification on an ESP32-S3 microcontroller. The system classifies air quality into six categories (Good → Hazardous) from a Plantower PMS5003 particulate sensor and a DHT22 temperature/humidity sensor, shows the class on a 16×2 LCD, and logs every reading to a microSD card. No internet connectivity is required.

The core research question: **how well does a model trained at one location generalize to other locations** — and what lightweight on-device corrections can close that gap? Field validation uses three nodes placed around a Kathmandu bus park at increasing distance from the traffic source.

---

## Current results

Models are trained on the India CPCB + Open-Meteo merged dataset (inputs: PM2.5, PM10, temperature, humidity) and evaluated on a time-ordered held-out test split. Full tables are in `results/`.

| Model | Test accuracy | Test macro-F1 | Size |
|---|---|---|---|
| Logistic Regression | 0.9600 | 0.9393 | — |
| XGBoost (best baseline) | 0.9936 | 0.9852 | 2.5 MB (not deployable) |
| MLP float32 (TFLite) | 0.9800 | 0.9789 | 5.69 KB |
| **MLP INT8 (TFLite, deployed)** | **0.9827** | **0.9725** | **5.44 KB** |

The INT8 MLP keeps 98.7% of XGBoost's macro-F1 at roughly 1/460 of its size and is exported as a C array in `firmware/aqi_classifier/aqi_model.h`. Feature ablation and AQI-boundary robustness analyses are in `results/feature_ablation.csv` and `results/boundary_robustness.json`.

---

## Repository structure

```
tinyml-aqi-kathmandu/
├── data/
│   ├── raw/            # Original downloaded datasets (not committed — see DATA_SOURCES.md)
│   ├── processed/      # Cleaned, merged, labeled datasets (not committed)
│   └── field/          # SD card logs from the Kathmandu field deployment
├── code/
│   ├── 01_data_prep/        # Download, merge, clean, label, split
│   ├── 02_eda/              # Exploratory analysis (see README inside)
│   ├── 03_baseline_models/  # LR, RF, XGBoost baselines + feature ablation
│   ├── 04_tinyml_model/     # MLP training, TFLite conversion, INT8 quantization, C export
│   ├── 05_cross_location/   # Cross-location generalization experiments
│   ├── 06_analysis/         # AQI-boundary robustness analysis and figures
│   └── 07_nepal_validation/ # OpenAQ Nepal fetch — India-to-Kathmandu sanity check
├── firmware/
│   ├── data_logger/    # Stage 1: ESP32-S3 sensor node — bring-up and SD logging (PlatformIO)
│   └── aqi_classifier/ # Stage 2: INT8 model as C array for on-device inference
├── models/             # scaler_params.json, training history (weights not committed)
├── results/            # Metrics CSVs and figures
└── docs/               # Lab notebook, literature tracker, deployment log, AQI breakpoints
```

---

## Hardware

| Component | Model | Notes |
|---|---|---|
| Microcontroller | ESP32-S3-DevKitC-1 (N16R8) | 16 MB flash, 8 MB octal PSRAM. GPIO 33–37 are taken by PSRAM and must not be used. |
| PM sensor | Plantower PMS5003 | UART, 5 V supply. RX = GPIO18, TX = GPIO17 |
| Temp/humidity | DHT22 | GPIO2, 3.3 V |
| RTC | DS1307 Tiny RTC module | I²C 0x68, battery-backed, 5 V module |
| Display | 16×2 character LCD with PCF8574 I²C backpack | I²C 0x27, 5 V panel |
| SD card | MicroSD SPI module | CS=10, MOSI=11, SCK=12, MISO=14. **Needs 5 V** — the onboard AMS1117 regulator drops out on 3.3 V. |
| Power monitor | INA219 | I²C 0x40, for the energy benchmark |

**I²C dual-voltage note.** The ESP32-S3 drives SDA (GPIO8) and SCL (GPIO9) at 3.3 V, while the RTC and LCD modules are powered at 5 V and carry their own pull-up resistors. The bus currently works in this mixed configuration, but 5 V pull-ups exceed the ESP32-S3's 3.6 V absolute-maximum pin rating. The LCD panel does not produce readable contrast at 3.3 V (verified on the bench), so it must stay at 5 V. Measure the idle SDA/SCL voltage on the ESP32 side before extended operation; if it reads near 5 V, insert a BSS138 bidirectional level shifter between the ESP32 and the 5 V modules and remove the backpack's pull-up resistors.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/BibekBashyal/tinyml-aqi-kathmandu.git
cd tinyml-aqi-kathmandu

# Install Python dependencies
pip install -r requirements.txt

# Download datasets (see data/raw/DATA_SOURCES.md for instructions)
python code/01_data_prep/01_download_openmeteo.py
```

Firmware is a PlatformIO project; open `firmware/data_logger/` in VS Code with the PlatformIO extension and run **Upload**. The board must be connected through its **UART** USB-C port, not the native USB port (see the comments in `platformio.ini`).

---

## Data sources

See `data/raw/DATA_SOURCES.md` for full details, download URLs, and license information.

- **Pollution data:** India CPCB Air Quality Dataset (Kaggle, CC BY-NC-SA 4.0)
- **Weather data:** Open-Meteo Historical Weather API (CC BY 4.0)
- **Field data:** Collected by the author at three sites around a Kathmandu bus park, Nepal (2026)

---

## License

Code: MIT License  
Data: See individual dataset licenses in `data/raw/DATA_SOURCES.md`

---

## Citation

If you use this work, please cite:

> Bashyal, B. (2026). *TinyML-Based AQI Classification on ESP32: Cross-Location Generalization in the Kathmandu Valley*. Master's thesis, Kathmandu University.
