# Lab Notebook — Bibek Bashyal

Entries are dated from the project's own git history (this repo and the firmware repo) and from the bench setup notes. Only work that actually happened is recorded; gaps are real gaps.

## 2026-06-24
- Repository initialized with the full folder structure.
- Literature tracker drafted: 16 papers across 5 topic categories.
- Dataset strategy decided: merge India CPCB (PM2.5/PM10) with Open-Meteo (temperature/humidity) by city-hour. License confirmed: CC BY-NC-SA 4.0 permits academic use with attribution.
- Wrote and ran the data-prep scripts: raw CPCB exploration, Open-Meteo download, city-hour merge, AQI labeling with a time-ordered train/val/test split.

## 2026-06-25
- Baseline models trained (Logistic Regression, Random Forest, XGBoost). XGBoost is the strongest realistic baseline: test macro-F1 0.9852. Random Forest scored 1.0 on val and test, which looks like memorization of the label rule from PM2.5/PM10 rather than a usable result.
- Feature ablation study: leave-one-out, single-feature, and noise injection. PM2.5 dominates, as expected from the AQI definition.
- TinyML MLP trained, converted to TFLite, INT8-quantized, and exported as a C array. INT8 model: accuracy 0.9827, macro-F1 0.9725, 5,568 bytes.

## 2026-06-30
- Committed result tables, scaler parameters, training history, and the quantized firmware header.
- AQI-boundary robustness analysis: direction-of-error, decision-margin sweep, bootstrap confidence intervals, and a boundary-restricted noise-injection variant (Method D). Motivation: misclassifications should cluster at class boundaries, not scatter across them.

## 2026-07-01
- Wrote the OpenAQ Nepal fetcher as an India-to-Kathmandu sanity check on the label distribution. Fixed the country ID (176 → 145) and the OpenAQ v3 `meta.found` string parsing.

## 2026-07-05
- Started the ESP32-S3 sensor-node firmware as a separate PlatformIO project (now mirrored in `firmware/data_logger/`).
- Lost several hours to Serial output vanishing: with `ARDUINO_USB_CDC_ON_BOOT=1` the output goes to the native USB port, while the cable was in the UART port. Fixed by setting the flag to 0.

## 2026-07-19
- Validated every peripheral individually behind `#define` test modes: PMS5003 over UART, DHT22, DS1307 RTC, 16×2 I²C LCD. SD card proven electrically but blocked by flaky breadboard wiring.
- Board swapped mid-session from an N8 (no PSRAM) to an N16R8 (8 MB octal PSRAM). GPIO 33–37 are consumed by PSRAM on the new board, which silently broke SPI on those pins. Moved SD to GPIO 10–14 and I²C to GPIO 8/9.
- Decision: pause and solder the connections; dupont wiring was the root cause of most failures across SD, RTC, and LCD.

## 2026-08-26
- Committed baseline-comparison plot and confusion-matrix figures to the thesis repo.
- Firmware: SD logging and DHT22 brought up on the bench node. Root cause of the long SD saga: the microSD module's AMS1117 regulator needs 5 V input, not 3.3 V. Node now logs to `/aqi_log.csv` every 3 s.

## 2026-09-12
- INA219 power monitor: first module never ACKed and shorted the bus (dead). Replacement responds intermittently; suspect unsoldered header pins. Added an INA219 test mode with a line check, pull-up probe, and GPIO sweep to find where a wire actually landed.
- Debug lesson worth keeping: sweep free GPIOs with `INPUT_PULLDOWN` to locate a mis-wired pull-up; a module's pull-ups only pull when it has power.

## 2026-10-03
- Repository consistency pass before the report is examined: README hardware table now matches the bench (ESP32-S3 N16R8, LCD, I²C dual-voltage note), placeholder clone URL fixed, bench firmware mirrored into `firmware/data_logger/`, scaffolding script removed.
- Deployment plan fixed to three sites around a Kathmandu bus park (core, nearby, residential) in both `docs/deployment_log.md` and `data/raw/DATA_SOURCES.md`. This replaces the earlier Koteshwor/Dhulikhel/KU-campus list.
- Open: level-shifter decision for the 5 V I²C modules (measure idle SDA/SCL first), scaler constants into firmware, INT8 quantize/dequantize around invoke, batch energy benchmark protocol.

---

_Add an entry every week you work on the project. Format: date, what you did, result, next step._
