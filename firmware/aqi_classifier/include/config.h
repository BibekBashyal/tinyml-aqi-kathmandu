// config.h - shared pin map and constants for both firmware stages
// Lives in: firmware/data_logger/include/ and firmware/aqi_classifier/include/
// Board: ESP32-S3-DevKitC-1 (N16R8: 16 MB flash, 8 MB octal PSRAM).
// GPIO 33-37 are used by the PSRAM on this board and must not be wired.
#pragma once

// ---------------- Pins (must match wiring_diagram.png) ----------------
// PMS5003 on UART1 (Serial1), 5 V supply, 3.3 V logic
#define PMS_RX_PIN   18     // ESP32-S3 RX  <- PMS5003 TX
#define PMS_TX_PIN   17     // ESP32-S3 TX  -> PMS5003 RX
#define PMS_BAUD     9600

// DHT22 single-wire, 3.3 V, 10 k pull-up to 3V3
#define DHT_PIN      2
#define DHT_TYPE     DHT22

// I2C in two voltage domains joined by a BSS138 level shifter:
//   3.3 V side: ESP32-S3 GPIO8/9, INA219 (0x40), shifter LV
//   5 V side:   shifter HV, DS1307 RTC (0x68), LCD PCF8574 backpack (0x27)
// The LCD glass is blank at 3.3 V and the DS1307 disables I2C below about
// 3.75 V with its coin cell fitted, so both run at 5 V behind the shifter.
#define SDA_PIN      8
#define SCL_PIN      9
// 10 kHz was the rate that worked reliably on the breadboard (RTC reads timed
// out at 50-100 kHz). Retry 100000 after the harness is soldered.
#define I2C_CLOCK_HZ 10000UL
#define LCD_ADDR     0x27
#define LCD_COLS     16
#define LCD_ROWS     2

// microSD on SPI; module VCC on 5 V (its AMS1117 regulator drops out on 3V3)
#define SD_CS_PIN    10
#define SD_MOSI_PIN  11
#define SD_SCK_PIN   12
#define SD_MISO_PIN  14

// ---------------- Sampling ----------------
#define SAMPLE_INTERVAL_MS  3000UL    // one record every 3 s
#define LOG_FILE            "/aqilog.csv"

// ---------------- Humidity correction: x1c = pm25 / (1 + a*(rh/100)^b) ----
// No published alpha/beta exist for this functional form for the PMS5003.
// They can only be fitted against a reference-grade instrument (co-locating
// identical Plantower units cannot reveal their shared bias).
// alpha = 0 makes the correction a pass-through until coefficients are fitted.
#define RH_ALPHA  0.0f
#define RH_BETA   1.0f

// ---------------- Feature scaling (classifier stage only) ----------------
// Copied from models/scaler_params.json in the repo (training-set statistics).
// Order: PM2.5, PM10, temperature, humidity (same as training).
#define SCALER_FROM_JSON  1
static const float SCALER_MEAN[4] = {113.1067533325189f, 201.6154751793621f,
                                     23.049795432214065f, 74.19816404944534f};
static const float SCALER_STD[4]  = {115.17576765122364f, 176.9964349219321f,
                                     7.144550002310301f,  19.39754306254128f};
