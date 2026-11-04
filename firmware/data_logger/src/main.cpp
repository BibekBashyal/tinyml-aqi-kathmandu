// Stage 1 firmware: DATA LOGGER
// Place as: firmware/data_logger/src/main.cpp   (PlatformIO project)
// Reads PMS5003 + DHT22, timestamps with DS1307, appends CSV to microSD,
// shows live values on the LCD. No ML yet - this validates the harness.

#include <Arduino.h>
#include <string.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
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
bool rtcOk = false;   // DS1307 answered at boot
bool lcdOk = false;   // LCD backpack answered at last check
static const uint8_t RTC_EEPROM_ADDR = 0x50;  // 24C32 on the DS1307 module
bool i2cAck(uint8_t addr) { Wire.beginTransmission(addr); return Wire.endTransmission() == 0; }

// ---- I2C roll call (bench 2026-11-02) ----
// Lists every address that answers, and prints again only when the set changes, so a module
// on the wrong row, unpowered, or with a non-default address (INA219 A0/A1: 0x40-0x4F) shows
// up at once. Runs only while the LCD acks: on a broken bus the verdict line says enough.
static uint8_t i2cSeen[16] = {0};   // one bit per 7-bit address
void i2cRollCall(bool busHealthy) {
  if (!busHealthy) { memset(i2cSeen, 0, sizeof(i2cSeen)); return; }   // reprint when it returns
  uint8_t now[16] = {0};
  for (uint8_t a = 0x08; a < 0x78; a++) if (i2cAck(a)) now[a >> 3] |= (uint8_t)(1 << (a & 7));
  if (memcmp(now, i2cSeen, sizeof(now)) == 0) return;
  memcpy(i2cSeen, now, sizeof(now));
  Serial.print("I2C devices:");
  bool inaAt40 = false, inaElsewhere = false;
  for (uint8_t a = 0x08; a < 0x78; a++) {
    if (!(now[a >> 3] & (1 << (a & 7)))) continue;
    const char* who = (a == LCD_ADDR) ? "LCD" : (a == 0x68) ? "DS1307" : (a == RTC_EEPROM_ADDR) ? "RTC-EEPROM"
                    : (a >= 0x40 && a <= 0x4F) ? "INA219" : "unknown";
    if (a == 0x40) inaAt40 = true; else if (a >= 0x41 && a <= 0x4F) inaElsewhere = true;
    Serial.printf(" 0x%02X=%s", a, who);
  }
  if (!inaAt40 && inaElsewhere) Serial.print("  (INA219 answers but not at 0x40: A0/A1 jumper bridged, open both pads)");
  if (!inaAt40 && !inaElsewhere) Serial.print("  (no INA219: check its VCC 3V3, GND, SDA on GPIO8 row, SCL on GPIO9 row)");
  Serial.println();
}

// ---- SD resilience (bench 2026-11-02, confirmed writes after the 2026-11-03 code review) ----
// A 5-minute mains cut on the 5 V rail took the card offline and 2.5 h of rows were lost. Every row
// now waits in a RAM ring buffer until the card has confirmed it: POSIX fflush + fsync succeeded and
// stat() shows the file grew by exactly those bytes. A failed check keeps the rows queued (a retry may
// duplicate rows, which carry their timestamp; it never loses them). The log is never opened with "w",
// the header goes into an empty file only, and after a remount the last byte tells whether a cut row
// needs a newline first. Same scheme as aqi_classifier/src/sd_store.cpp; lib/SD is patched so a block
// the card rejects is reported as a failure (lib/README.md).
static const size_t   SD_PENDING_MAX = 2000;   // ~1.7 h of rows at 3 s, 128 KB of RAM
static const size_t   SD_ROW_MAX     = 64;
static const uint32_t SD_REMOUNT_MS  = 10000;
static const size_t   SD_FLUSH_BATCH = 32;
static const char*    LOG_PATH       = "/sd" LOG_FILE;   // SD.begin() mounts the card at /sd
static const char*    LOG_HEADER     = "timestamp,pm25_raw,pm10,temp_c,rh_pct,bus_V,current_mA";
static char   sdPending[SD_PENDING_MAX][SD_ROW_MAX];
static size_t sdHead = 0, sdCount = 0, sdDropped = 0;
static uint32_t sdLastRemountMs = 0;
static bool   sdNeedNewline = false;   // the file ends inside a row
static FILE*  sdFp = nullptr;          // open for the whole mount (reopening walked the FAT chain every cycle)
static size_t sdKnownSize = 0;         // file size after our last confirmed write
bool sdOk = false;    // card mounted and every row so far confirmed on it

bool sdFileSize(size_t& size) {
  struct stat st;
  if (stat(LOG_PATH, &st) != 0) return false;
  size = (size_t)st.st_size;
  return true;
}

void sdCloseLog() {
  if (sdFp) { fclose(sdFp); sdFp = nullptr; }
}

bool sdOpenLog() {
  if (sdFp) return true;
  sdFp = fopen(LOG_PATH, "a");   // "a" never truncates
  if (sdFp && sdFileSize(sdKnownSize)) return true;
  sdCloseLog();
  return false;
}

bool sdMount() {
  SPI.begin(SD_SCK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);
  if (SD.begin(SD_CS_PIN, SPI, SD_SPI_HZ)) return true;
  SD.end(); SPI.end();
  return false;
}

// Header into an empty file, or note whether the file ends inside a row. True on success.
bool sdPrepareFile() {
  sdCloseLog();
  FILE* fp = fopen(LOG_PATH, "a+");   // creates, never truncates, allows reading the tail
  if (!fp) return false;
  bool ok = fseek(fp, 0, SEEK_END) == 0;
  long size = ok ? ftell(fp) : -1;
  if (size == 0) {
    ok = fputs(LOG_HEADER, fp) >= 0 && fputc('\n', fp) != EOF && fflush(fp) == 0 && fsync(fileno(fp)) == 0;
    sdNeedNewline = false;
  } else if (size > 0) {
    ok = fseek(fp, -1, SEEK_END) == 0;
    int last = ok ? fgetc(fp) : EOF;
    ok = ok && last != EOF;
    sdNeedNewline = ok && last != '\n';
  } else {
    ok = false;
  }
  return (fclose(fp) == 0) && ok;
}

void sdQueueRow(const char* row) {
  size_t slot = (sdHead + sdCount) % SD_PENDING_MAX;
  strlcpy(sdPending[slot], row, SD_ROW_MAX);
  if (sdCount < SD_PENDING_MAX) sdCount++;
  else { sdHead = (sdHead + 1) % SD_PENDING_MAX; sdDropped++; }   // full: forget the oldest row
}

// Drop exactly the rows the confirmed growth covers; a row cut in half stays queued.
void sdDequeueConfirmed(size_t grown, size_t batch) {
  for (size_t j = 0; j < batch; j++) {
    size_t len = strlen(sdPending[sdHead]);
    if (grown < len) { sdNeedNewline = grown > 0; return; }
    grown -= len;
    sdHead = (sdHead + 1) % SD_PENDING_MAX;
    sdCount--;
  }
}

// One batch: header or separating newline if needed, rows, fflush + fsync, then confirm via stat().
bool sdFlushBatch(FILE* fp) {
  size_t before, extra = 0, bytes = 0, batch = min(sdCount, SD_FLUSH_BATCH);
  if (fflush(fp) != 0 || !sdFileSize(before)) return false;
  if (before != sdKnownSize) return false;   // changed behind our back (card swapped): remount
  bool wrote = true;
  if (before == 0) { wrote = fputs(LOG_HEADER, fp) >= 0 && fputc('\n', fp) != EOF; extra = strlen(LOG_HEADER) + 1; }
  else if (sdNeedNewline) { wrote = fputc('\n', fp) != EOF; extra = 1; }
  for (size_t j = 0; wrote && j < batch; j++) {
    const char* row = sdPending[(sdHead + j) % SD_PENDING_MAX];
    size_t len = strlen(row);
    wrote = fwrite(row, 1, len, fp) == len;
    if (wrote) bytes += len;
  }
  size_t onCard;
  if (fflush(fp) != 0 || fsync(fileno(fp)) != 0 || !sdFileSize(onCard)) return false;   // keep all queued
  if (onCard < before + extra) return false;
  sdKnownSize = onCard;
  sdNeedNewline = false;
  sdDequeueConfirmed(onCard - before - extra, batch);
  return wrote && onCard - before - extra == bytes;
}

bool sdFlushToCard() {
  if (!sdOpenLog()) return false;
  bool ok = true;
  while (ok && sdCount) ok = sdFlushBatch(sdFp);
  if (!ok) sdCloseLog();
  return ok && sdCount == 0;
}

// Called after a failed write. Rate-limited so a dead card cannot stall the 3 s cycle.
void sdRecover() {
  if (millis() - sdLastRemountMs < SD_REMOUNT_MS) return;
  sdLastRemountMs = millis();
  sdCloseLog();   // its FatFs object dies with the unmount
  SD.end(); SPI.end(); delay(20);
  bool mounted = sdMount() && sdPrepareFile();
  if (mounted) sdOk = sdFlushToCard();
  Serial.printf("SD: %s, %u rows waiting in RAM, %u dropped since boot\n",
                !mounted ? "remount failed (card power or wires)" : sdOk ? "remounted, backlog confirmed on card" : "remounted, write not confirmed",
                (unsigned)sdCount, (unsigned)sdDropped);
}

// Every cycle: a waiting backlog is written (or the card retried) even when no new row was made.
void sdService() {
  if (sdCount == 0) return;
  if (sdOk && sdFlushToCard()) return;
  sdOk = false;
  sdRecover();
}

void sdStore(const char* row) {
  sdQueueRow(row);   // every row goes through the queue and leaves it only once confirmed
  sdService();
}

// Read the DS1307 time registers directly, with a STOP between the pointer write and the read
// (the library's repeated-start read failed on the bench), and reject impossible values so a bad
// read never becomes a CSV timestamp. Returns true and fills ts on success.
uint8_t rtcRaw[7];
const char* rtcErr = "";   // why the last RTC read was rejected
// One-shot bench diagnostic when the RTC module appears: EEPROM (24C32 at 0x50) write/read-back proves the
// bus read direction; the raw DS1307 registers then show whether the clock chip itself is sane.
void rtcModuleDiag() {
  Wire.beginTransmission(0x50); Wire.write(0x0F); Wire.write(0xF0); Wire.write(0xA5);   // byte 0x0FF0 := 0xA5
  int w = Wire.endTransmission(true); delay(10);
  Wire.beginTransmission(0x50); Wire.write(0x0F); Wire.write(0xF0); int p = Wire.endTransmission(true);
  int got = -1; if (p == 0 && Wire.requestFrom(0x50, 1) == 1) got = Wire.read();
  Serial.printf("RTC module diag: EEPROM write=%d ptr=%d readback=0x%02X -> %s\n", w, p, got,
                got == 0xA5 ? "bus read direction OK" : "EEPROM read FAILED (wiring / SDA return path)");
  Wire.beginTransmission(0x68); Wire.write(0x00);
  if (Wire.endTransmission(true) == 0 && Wire.requestFrom(0x68, 8) == 8) {
    uint8_t r[8]; for (int i = 0; i < 8; i++) r[i] = Wire.read();
    Serial.printf("DS1307 regs 0-7: %02X %02X %02X %02X %02X %02X %02X %02X (CH=%d)\n",
                  r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], (r[0] >> 7) & 1);
  } else Serial.println("DS1307 register read FAILED");
}
uint32_t rtcBadReads = 0;
static uint8_t bcd2bin(uint8_t v) { return (v >> 4) * 10 + (v & 0x0F); }
bool readRtcValidated(char* ts, size_t n) {
  Wire.beginTransmission(0x68); Wire.write(0x00);
  if (Wire.endTransmission(true) != 0) { rtcErr = "no ack"; return false; }
  if (Wire.requestFrom(0x68, 7) != 7)  { rtcErr = "short read"; return false; }
  for (int i = 0; i < 7; i++) rtcRaw[i] = Wire.read();
  rtcErr = "values out of range";
  int sec = bcd2bin(rtcRaw[0] & 0x7F), min = bcd2bin(rtcRaw[1]), hour = bcd2bin(rtcRaw[2] & 0x3F);
  int day = bcd2bin(rtcRaw[4]), month = bcd2bin(rtcRaw[5]), year = 2000 + bcd2bin(rtcRaw[6]);
  bool ok = sec < 60 && min < 60 && hour < 24 && day >= 1 && day <= 31 && month >= 1 && month <= 12 &&
            year >= 2026 && year <= 2040 && !(rtcRaw[0] & 0x80);   // bit 7 of reg 0 = clock halted
  if (!ok) return false;
  snprintf(ts, n, "%04d-%02d-%02d %02d:%02d:%02d", year, month, day, hour, min, sec);
  return true;
}
// Electrical state of a bus line, used only while the LCD is missing: "up" = external pull-up present
// (shifter 3.3 V side reaches the pin), "float" = nothing connected, "low" = held at ground.
const char* lineState(int pin) {
  pinMode(pin, INPUT_PULLDOWN); delay(2); int pd = digitalRead(pin);
  pinMode(pin, INPUT_PULLUP);   delay(2); int pu = digitalRead(pin);
  pinMode(pin, INPUT);
  return (pd && pu) ? "up" : (!pd && pu) ? "float" : "low";
}
bool inaOk = false;   // INA219 answered at last check
bool inaShort = false;   // shunt looks wired across a rail (bus ~0 V, large current)

// ---- INA219 (bench 2026-11-02) ----
// The shunt (0.1 ohm, VIN+ to VIN-) must sit IN SERIES with the ESP32's 5 V supply:
// adapter 5 V -> VIN+, VIN- -> ESP32 5V pin. Wired across a rail (VIN- on GND or 3V3)
// it shorts that rail and heats the regulator; that is the pattern the guard below flags.
static const float INA_SHORT_BUS_V_MAX = 1.0f;    // a load's bus voltage is never this low
static const float INA_SHORT_CURR_MA   = 300.0f;  // ...while this much current flows

void inaStart() {
  ina219.setCalibration_32V_1A();   // 40 uA steps; ESP32 + sensors stay well under 1 A
  Serial.println("INA219 found (32 V / 1 A range)");
}

// Read the INA219 and detect a shunt wired across a supply. Returns false when the
// readings are not plausible so the row shows nan instead of a wrong number.
void inaRead(float& busV, float& curr) {
  busV = ina219.getBusVoltage_V();
  curr = ina219.getCurrent_mA();
  inaShort = (busV < INA_SHORT_BUS_V_MAX) && (fabsf(curr) > INA_SHORT_CURR_MA);
  if (inaShort) Serial.printf("WARN: INA219 bus %.2f V with %.0f mA -> shunt is ACROSS a rail. Unplug VIN+/VIN- now\n", busV, curr);
}

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
      // header found: wait until the remaining 30 bytes have arrived, otherwise
      // read() returns -1 for the missing bytes and the checksum always fails
      uint32_t hdr_ms = millis();
      while (pmsSerial.available() < 30 && millis() - hdr_ms < 300) delay(2);
      if (pmsSerial.available() < 30) continue;
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

// Serial command "T=YYYY-MM-DD HH:MM:SS" sets the RTC from a laptop, no recompile needed.
void handleSerialTimeSet() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n'); line.trim();
  int Y, M, D, h, m, s;
  if (sscanf(line.c_str(), "T=%d-%d-%d %d:%d:%d", &Y, &M, &D, &h, &m, &s) != 6) {
    Serial.println("ignored: expected T=YYYY-MM-DD HH:MM:SS"); return;
  }
  if (!rtcOk) { Serial.println("ignored: no RTC"); return; }
  rtc.adjust(DateTime(Y, M, D, h, m, s));
  char back[20];
  bool ok = readRtcValidated(back, sizeof(back));
  Serial.printf("RTC set to %04d-%02d-%02d %02d:%02d:%02d, read back: %s\n", Y, M, D, h, m, s, ok ? back : "INVALID");
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(I2C_CLOCK_HZ);
  pmsSerial.begin(PMS_BAUD, SERIAL_8N1, PMS_RX_PIN, PMS_TX_PIN);
  dht.begin();

  lcdOk = (lcd.begin(LCD_COLS, LCD_ROWS) == 0);
  if (!lcdOk) Serial.println("WARN: LCD not found on I2C (check shifter HV side / LCD 5 V)");
  lcd.backlight(); lcd.print("AQI Logger v1.0");

  rtcOk = rtc.begin();
  if (rtcOk) rtcModuleDiag();
  if (!rtcOk) {
    if (RTC_REQUIRED) haltWithError("DS1307 not found");
    Serial.println("WARN: DS1307 not found, timestamps are uptime seconds");
    lcd.setCursor(0, 1); lcd.print("RTC missing!");
    delay(2000);
  } else {
    // Code review 2026-11-03: never rewrite the clock at boot. RTClib's repeated-start read fails on
    // this bus and returned garbage, so a correct clock was overwritten with the stale build time.
    // A clock without valid time gives uptime stamps until the laptop sends T=.
    char ts[20];
    if (readRtcValidated(ts, sizeof(ts))) Serial.printf("DS1307 time %s\n", ts);
    else Serial.printf("DS1307 time not valid (%s): send T=YYYY-MM-DD HH:MM:SS. Rows use uptime stamps until then\n", rtcErr);
  }
  inaOk = ina219.begin();
  if (inaOk) inaStart(); else Serial.println("WARN: INA219 not found (continuing)");
  i2cRollCall(lcdOk);
  // keep retrying the card instead of halting, so a reseated card or wire is picked up without a reset
  while (true) {
    if (sdMount() && sdPrepareFile()) { sdOk = true; break; }
    SD.end(); SPI.end();
    pinMode(SD_MISO_PIN, INPUT_PULLDOWN); delay(2);
    bool misoHigh = digitalRead(SD_MISO_PIN);   // a powered, seated card holds MISO high
    pinMode(SD_MISO_PIN, INPUT);
    Serial.printf("SD init failed: MISO %s. Retrying in 3 s\n",
                  misoHigh ? "high -> card powered, check CS/SCK/MOSI wires"
                           : "low -> no card, card not seated, module unpowered, or MISO wire open");
    lcd.clear(); lcd.print("SD init failed");
    lcd.setCursor(0, 1); lcd.print(misoHigh ? "chk CS/SCK/MOSI" : "card/power/MISO");
    delay(3000);
  }

  Serial.println("Logger ready.");
}

void loop() {
  uint32_t cycle_start = millis();
  // RTC missing at boot (flaky bus): keep looking for it, so timestamps recover without a reset
  if (!rtcOk && rtc.begin()) { rtcOk = true; Serial.println("DS1307 found, switching to RTC timestamps"); rtcModuleDiag(); }
  if (!inaOk && ina219.begin()) { inaOk = true; inaStart(); }
  // LCD lost power or bus (shows blocks): re-initialise it as soon as it answers again
  bool lcdAck = i2cAck(LCD_ADDR);
  i2cRollCall(lcdAck);
  if (lcdAck && !lcdOk) { lcdOk = (lcd.begin(LCD_COLS, LCD_ROWS) == 0); if (lcdOk) { lcd.backlight(); Serial.println("LCD back on the bus"); } }
  if (!lcdAck) {
    lcdOk = false;
    Wire.end();
    const char* sdaSt = lineState(SDA_PIN); const char* sclSt = lineState(SCL_PIN);
    Wire.begin(SDA_PIN, SCL_PIN); Wire.setClock(I2C_CLOCK_HZ);
    Serial.printf("I2C: no LCD ack. SDA %s, SCL %s -> %s\n", sdaSt, sclSt,
                  (strcmp(sdaSt, "low") == 0 || strcmp(sclSt, "low") == 0) ? "a bus wire sits on GND or a short" :
                  (strcmp(sdaSt, "float") == 0 || strcmp(sclSt, "float") == 0) ? "GPIO8/9 to shifter LV1/LV2 open, or shifter LV unpowered" :
                  "3.3 V side fine: shifter HV supply, HV1/HV2 wires, or LCD 5 V");
  }
  handleSerialTimeSet();

  PmsReading pms = readPMS();
  float temp = dht.readTemperature();
  float rh   = dht.readHumidity();
  float busV = NAN, curr = NAN;
  if (inaOk) inaRead(busV, curr);

  char ts[20];
  bool tsFromRtc = rtcOk && readRtcValidated(ts, sizeof(ts));
  if (!tsFromRtc) {
    snprintf(ts, sizeof(ts), "uptime_%lus", millis() / 1000UL);
    if (rtcOk) {
      rtcBadReads++;
      // bench 2026-11-02: the module fell off the bus 15 s after detection and the flag stayed
      // true for minutes. Three straight failures with no ack = lost; re-detection takes over.
      if (rtcBadReads >= 3 && !i2cAck(0x68)) {
        rtcOk = false; rtcBadReads = 0;
        Serial.println("DS1307 lost from the bus (no ack), back to uptime stamps");
      } else if (rtcBadReads % 10 == 1) {
        Serial.printf("RTC read failed: %s", rtcErr);
        if (strcmp(rtcErr, "values out of range") == 0)
          Serial.printf(", raw %02X %02X %02X %02X %02X %02X %02X",
                        rtcRaw[0], rtcRaw[1], rtcRaw[2], rtcRaw[3], rtcRaw[4], rtcRaw[5], rtcRaw[6]);
        Serial.println(" (using uptime)");
      }
    }
  } else {
    rtcBadReads = 0;
  }

  // A PMS frame is enough for a row: a failed DHT read is written as nan (decision 2026-11-01),
  // so dust data is never lost when one sensor glitches.
  bool dhtOk = !isnan(temp) && !isnan(rh);
  if (pms.ok) {
    char row[SD_ROW_MAX];
    snprintf(row, sizeof(row), "%s,%u,%u,%.1f,%.1f,%.2f,%.1f\n",
             ts, pms.pm25, pms.pm10, temp, rh, busV, curr);
    sdStore(row);   // queued in RAM until the card confirms it
    lcd.clear();
    // bench 2026-11-02: clock time on the top line (HH:MM from the RTC stamp, --:-- without RTC)
    char hhmm[6] = "--:--";
    if (tsFromRtc) strlcpy(hhmm, ts + 11, sizeof(hhmm));
    lcd.setCursor(0, 0); lcd.printf("PM2.5:%4u %s", pms.pm25, hhmm);
    lcd.setCursor(0, 1);
    if (dhtOk) lcd.printf("T:%4.1fC RH:%3.0f%%", temp, rh);
    else       lcd.print("T: ---  RH: ---");
    if (!sdOk || sdCount) { lcd.setCursor(15, 1); lcd.print("!"); }   // rows waiting in RAM
    if (inaShort) { lcd.setCursor(0, 1); lcd.print("INA SHORT unplug"); }
    // bench 2026-11-01: when the RTC or DHT is missing, say whether the RTC module's EEPROM
    // (same VCC/GND/SDA/SCL as the DS1307) still answers and whether the DHT data line still
    // sees its 10 k pull-up: up = pull-up powered, float = DATA wire or 3V3 open, low = short
    char diag[48] = "";
    if (!rtcOk || !dhtOk) {
      snprintf(diag, sizeof(diag), "  [rtc_eeprom=%d dht_line=%s]",
               i2cAck(RTC_EEPROM_ADDR), dhtOk ? "ok" : lineState(DHT_PIN));
    }
    char ina[40] = "";
    if (inaOk) snprintf(ina, sizeof(ina), " ina=%.2fV/%.0fmA", busV, curr);
    Serial.printf("%s  PM2.5=%u PM10=%u T=%.1f RH=%.1f  bus: lcd=%d rtc=%d sd=%d%s%s%s\n",
                  ts, pms.pm25, pms.pm10, temp, rh, lcdOk, rtcOk, sdOk, ina, dhtOk ? "" : "  (DHT failed)", diag);
  } else {
    sdService();   // a backlog still drains while the PMS5003 is out
    lcd.clear(); lcd.print("PMS5003 error");
    if (!sdOk || sdCount) { lcd.setCursor(15, 1); lcd.print("!"); }
    Serial.printf("%s  read error (pms=%d dht=%d)  bus: lcd=%d rtc=%d sd=%d\n",
                  ts, pms.ok, !(isnan(temp) || isnan(rh)), lcdOk, rtcOk, sdOk);
  }

  // keep a steady 3 s cadence regardless of how long the cycle took
  uint32_t elapsed = millis() - cycle_start;
  if (elapsed < SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS - elapsed);
}
