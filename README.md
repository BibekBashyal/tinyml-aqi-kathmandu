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
| **MLP INT8 (TFLite, deployed, per-tensor)** | **0.9874** | **0.9873** | **4.25 KB** |

The INT8 MLP matches XGBoost's macro-F1 (0.9873 vs 0.9852) at about 1/590 of its size and is exported, together with its input constants, as a C array in `firmware/aqi_classifier/aqi_model.h`. It is built by `code/04_tinyml_model/requantize_int8.py` with per-tensor weights, the only layout the ESP32's TensorFlow Lite Micro runs correctly, and a per-feature input range; the earlier per-channel model (0.9827 / 0.9725, 5.44 KB) and the reasons for the change are in `results/requantize_int8_report.json`. Feature ablation and AQI-boundary robustness analyses are in `results/feature_ablation.csv` and `results/boundary_robustness.json`.

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
│   ├── bench_bringup/  # Bench test modes for each module, plus setup notes
│   ├── data_logger/    # Stage 1: sensors, RTC timestamp, SD logging, LCD (PlatformIO)
│   └── aqi_classifier/ # Stage 2: humidity correction, scaling, INT8 TFLite Micro inference, benchmark mode
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
| Level shifter | BSS138 4-channel module | Joins the 3.3 V and 5 V I²C segments (SDA, SCL) |
| SD card | MicroSD SPI module | CS=10, MOSI=11, SCK=12, MISO=14. **Needs 5 V** — the onboard AMS1117 regulator drops out on 3.3 V. |
| Power monitor | INA219 | I²C 0x40, for the energy benchmark |

**I²C voltage design.** The ESP32-S3 GPIO is 3.3 V and not 5 V tolerant. Two modules must run at 5 V: the LCD glass is blank at 3.3 V, and the DS1307 disables its I²C interface below about 3.75 V when its coin cell is fitted. A BSS138 level shifter therefore joins a 3.3 V segment (ESP32-S3 GPIO8/GPIO9, INA219) to a 5 V segment (DS1307, LCD). Before connecting the board, check that SDA and SCL on the 3.3 V side idle at about 3.3 V.

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

Firmware is in two PlatformIO projects: `firmware/data_logger/` (stage 1) and `firmware/aqi_classifier/` (stage 2). Open one in VS Code with the PlatformIO extension and run **Upload**; see `firmware/README.md`. The board must be connected through its **UART** USB-C port, not the native USB port (see the comments in `platformio.ini`).

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
