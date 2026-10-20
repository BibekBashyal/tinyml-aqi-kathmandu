# Firmware: ESP32-S3-DevKitC-1 (N16R8) AQI node

Three PlatformIO projects. Open one folder at a time in VS Code with the PlatformIO extension. The USB cable goes in the board's **UART** USB-C port, not the native USB port.

| Folder | What it is | Status |
|---|---|---|
| `bench_bringup/` | Bench test firmware with one `#define` test mode per module (SD, RTC, LCD, INA219, PMS5003), plus `SETUP_NOTES.md` with every bench lesson | Used to verify each module on its own |
| `data_logger/` | Stage 1: reads PMS5003 and DHT22, timestamps with the DS1307, appends a CSV row to the microSD card every 3 s, shows values on the LCD | Compiles: 367 KB flash, 19.7 KB static RAM |
| `aqi_classifier/` | Stage 2: humidity correction, standard scaling, INT8 TensorFlow Lite Micro inference, LCD class display and SD logging. Set `BENCHMARK_MODE 1` in `src/main.cpp` for latency, RAM and batch energy measurements | Compiles: 603 KB flash, 49.7 KB static RAM |

TensorFlow Lite Micro and the model add about 236 KB of flash (classifier minus logger). That is the figure the report's 500 KB flash target refers to.

## Shared settings (`include/config.h`)

- Pins: PMS5003 on UART1 (RX GPIO18, TX GPIO17), DHT22 on GPIO2, I²C on GPIO8/GPIO9, microSD on CS 10, MOSI 11, SCK 12, MISO 14. GPIO 33 to 37 are taken by the octal PSRAM and must not be used.
- I²C runs in two voltage domains joined by a BSS138 level shifter: ESP32-S3 and INA219 on the 3.3 V side, DS1307 and LCD on the 5 V side. The clock is 10 kHz, the rate that worked on the breadboard; try 100 kHz after soldering.
- Scaler constants are copied from `models/scaler_params.json`. If the model is retrained, copy the new values in.
- Humidity correction `pm25 / (1 + alpha * (rh/100)^beta)` ships with `RH_ALPHA = 0`, a pass-through, until alpha and beta are fitted against a reference instrument.

## Model header

`aqi_classifier/aqi_model.h` is written by `code/04_tinyml_model/train_mlp.py`. It defines `models_mlp_int8_tflite[]` and `models_mlp_int8_tflite_len`. The classifier copies it into a 16-byte-aligned buffer before handing it to TensorFlow Lite Micro.

## Before flashing stage 2

1. Wire the BSS138 shifter. With the ESP32-S3 disconnected, check that SDA and SCL on the 3.3 V side idle at about 3.3 V.
2. Run an I²C scan: expect 0x68 (DS1307), 0x40 (INA219), 0x27 (LCD), and possibly 0x50 (the RTC board's EEPROM).
3. Flash stage 1 and let it log overnight. Move to stage 2 only after a clean soak.
