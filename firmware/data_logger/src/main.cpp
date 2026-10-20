// Stage 1 firmware: DATA LOGGER
// Place as: firmware/data_logger/src/main.cpp   (PlatformIO project)
// Reads PMS5003 + DHT22, timestamps with DS1307, appends CSV to microSD,
// shows live values on the LCD. No ML yet - this validates the harness.

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <DHT.h>
#include <RTClib.h>
#include <Adafruit_INA219.h>
#include <hd44780.h>
#include <hd44780ioClass/hd44780_I2Cexp.h>   // PCF8574 backpack; library proven on the bench
#include "config.h"

HardwareSerial pmsSerial(1);
DHT dht(DHT_PIN, DHT_TYPE);
RTC_DS1307 rtc;
Adafruit_INA219 ina219;
hd44780_I2Cexp lcd(LCD_ADDR);   // address hardcoded: auto-detect failed on the bench

struct PmsReading { uint16_t pm25; uint16_t pm10; bool ok; };

// Parse one 32-byte PMS5003 frame on Serial1 (header 0x42 0x4D), checksum-verified.
PmsReading readPMS(uint32_t timeout_ms = 3000) {
  PmsReading r = {0, 0, false};
  uint8_t buf[32];
  uint32_t start = millis();
  while (millis() - start < timeout_ms) {
    if (pmsSerial.available() >= 32) {
      if (pmsSerial.read() != 0x42) continue;
      if (pmsSerial.peek() != 0x4D) continue;
      buf[0] = 0x42; buf[1] = pmsSerial.read();
      for (int i = 2; i < 32; i++) buf[i] = pmsSerial.read();
      uint16_t sum = 0;
      for (int i = 0; i < 30; i++) sum += buf[i];
      uint16_t chk = (buf[30] << 8) | buf[31];
      if (sum != chk) continue;
      // atmospheric-environment values: PM2.5 in bytes 12-13, PM10 in bytes 14-15
      r.pm25 = (buf[12] << 8) | buf[13];
      r.pm10 = (buf[14] << 8) | buf[15];
      r.ok = true;
      return r;
    }
    delay(10);
  }
  return r;
}

void haltWithError(const char* msg) {
  Serial.println(msg);
  lcd.clear(); lcd.print(msg);
  while (true) delay(1000);
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(I2C_CLOCK_HZ);
  pmsSerial.begin(PMS_BAUD, SERIAL_8N1, PMS_RX_PIN, PMS_TX_PIN);
  dht.begin();

  lcd.begin(LCD_COLS, LCD_ROWS); lcd.backlight();
  lcd.print("AQI Logger v1.0");

  if (!rtc.begin())    haltWithError("DS1307 not found");
  if (!rtc.isrunning()) rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  if (!ina219.begin()) Serial.println("WARN: INA219 not found (continuing)");
  SPI.begin(SD_SCK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);
  if (!SD.begin(SD_CS_PIN)) haltWithError("SD init failed");

  if (!SD.exists(LOG_FILE)) {
    File f = SD.open(LOG_FILE, FILE_WRITE);
    f.println("timestamp,pm25_raw,pm10,temp_c,rh_pct,bus_V,current_mA");
    f.close();
  }
  Serial.println("Logger ready.");
}

void loop() {
  uint32_t cycle_start = millis();

  PmsReading pms = readPMS();
  float temp = dht.readTemperature();
  float rh   = dht.readHumidity();
  DateTime now = rtc.now();
  float busV = ina219.getBusVoltage_V();
  float curr = ina219.getCurrent_mA();

  char ts[20];
  snprintf(ts, sizeof(ts), "%04d-%02d-%02d %02d:%02d:%02d",
           now.year(), now.month(), now.day(),
           now.hour(), now.minute(), now.second());

  if (pms.ok && !isnan(temp) && !isnan(rh)) {
    File f = SD.open(LOG_FILE, FILE_APPEND);
    if (f) {
      f.printf("%s,%u,%u,%.1f,%.1f,%.2f,%.1f\n",
               ts, pms.pm25, pms.pm10, temp, rh, busV, curr);
      f.close();
    } else {
      Serial.println("WARN: SD append failed");
    }
    lcd.clear();
    lcd.setCursor(0, 0); lcd.printf("PM2.5:%4u ug/m3", pms.pm25);
    lcd.setCursor(0, 1); lcd.printf("T:%4.1fC RH:%3.0f%%", temp, rh);
    Serial.printf("%s  PM2.5=%u PM10=%u T=%.1f RH=%.1f\n",
                  ts, pms.pm25, pms.pm10, temp, rh);
  } else {
    lcd.clear(); lcd.print("Sensor error");
    Serial.printf("%s  read error (pms=%d dht=%d)\n",
                  ts, pms.ok, !(isnan(temp) || isnan(rh)));
  }

  // keep a steady 3 s cadence regardless of how long the cycle took
  uint32_t elapsed = millis() - cycle_start;
  if (elapsed < SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS - elapsed);
}
