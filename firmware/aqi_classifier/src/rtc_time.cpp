#include "rtc_time.h"
#include <string.h>
#include "node_hw.h"

bool rtcOk = false;
static const int RTC_YEAR_MIN = 2026;   // a year outside this window means the clock lost its time
static const int RTC_YEAR_MAX = 2040;
static uint8_t  rtcRaw[7];
static const char* rtcErr = "";   // why the last read was rejected
static uint32_t rtcBadReads = 0;

static uint8_t bcd2bin(uint8_t v) { return (v >> 4) * 10 + (v & 0x0F); }
static bool clockHalted() { return rtcRaw[0] & 0x80; }   // CH, bit 7 of register 0

// One-shot bench diagnostic when the module appears: EEPROM (24C32 at 0x50) write/read-back proves
// the bus read direction; the raw DS1307 registers then show whether the clock chip itself is sane.
static void rtcModuleDiag() {
  Wire.beginTransmission(RTC_EEPROM_ADDR); Wire.write(0x0F); Wire.write(0xF0); Wire.write(0xA5);   // byte 0x0FF0 := 0xA5
  int w = Wire.endTransmission(true); delay(10);
  Wire.beginTransmission(RTC_EEPROM_ADDR); Wire.write(0x0F); Wire.write(0xF0); int p = Wire.endTransmission(true);
  int got = -1; if (p == 0 && Wire.requestFrom(RTC_EEPROM_ADDR, (uint8_t)1) == 1) got = Wire.read();
  Serial.printf("RTC module diag: EEPROM write=%d ptr=%d readback=0x%02X -> %s\n", w, p, got,
                got == 0xA5 ? "bus read direction OK" : "EEPROM read FAILED (wiring / SDA return path)");
  Wire.beginTransmission(RTC_ADDR); Wire.write(0x00);
  if (Wire.endTransmission(true) == 0 && Wire.requestFrom(RTC_ADDR, (uint8_t)8) == 8) {
    uint8_t r[8]; for (int i = 0; i < 8; i++) r[i] = Wire.read();
    Serial.printf("DS1307 regs 0-7: %02X %02X %02X %02X %02X %02X %02X %02X (CH=%d)\n",
                  r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], (r[0] >> 7) & 1);
  } else Serial.println("DS1307 register read FAILED");
}

static bool readRtcValidated(char* ts, size_t n) {
  Wire.beginTransmission(RTC_ADDR); Wire.write(0x00);
  if (Wire.endTransmission(true) != 0) { rtcErr = "no ack"; return false; }
  if (Wire.requestFrom(RTC_ADDR, (uint8_t)7) != 7) { rtcErr = "short read"; return false; }
  for (int i = 0; i < 7; i++) rtcRaw[i] = Wire.read();
  bool allOnes = true;
  for (int i = 0; i < 7; i++) allOnes = allOnes && rtcRaw[i] == 0xFF;
  if (allOnes) { rtcErr = "all-ones read (SDA stuck high)"; return false; }   // a bus fault, not a lost time
  rtcErr = "values out of range";
  int sec = bcd2bin(rtcRaw[0] & 0x7F), min = bcd2bin(rtcRaw[1]), hour = bcd2bin(rtcRaw[2] & 0x3F);
  int day = bcd2bin(rtcRaw[4]), month = bcd2bin(rtcRaw[5]), year = 2000 + bcd2bin(rtcRaw[6]);
  bool ok = sec < 60 && min < 60 && hour < 24 && day >= 1 && day <= 31 && month >= 1 && month <= 12 &&
            year >= RTC_YEAR_MIN && year <= RTC_YEAR_MAX && !clockHalted();
  if (!ok) return false;
  snprintf(ts, n, "%04d-%02d-%02d %02d:%02d:%02d", year, month, day, hour, min, sec);
  return true;
}

// Code review 2026-11-03: the boot check no longer rewrites the clock. It used RTClib's
// repeated-start read, which fails on this bus and returns garbage, so a correct clock was
// overwritten with the (stale) build time. A clock without valid time now gives uptime stamps
// and a prompt; the laptop sets it with T= (scratchpad autoset_cap.py does that automatically).
bool rtcTimeLost() { return strcmp(rtcErr, "values out of range") == 0; }
const char* rtcLastError() { return rtcErr; }

static void explainInvalidTime() {
  if (!rtcTimeLost())
    Serial.printf("DS1307 read failed (%s): bus problem, rows use uptime stamps until reads succeed\n", rtcErr);
  else if (clockHalted())
    Serial.println("DS1307 clock halted (CH=1, time lost): send T=YYYY-MM-DD HH:MM:SS. Rows use uptime stamps until then");
  else
    Serial.println("DS1307 time out of range (time lost): send T=YYYY-MM-DD HH:MM:SS. Rows use uptime stamps until then");
}

void rtcBegin() {
  rtcOk = rtc.begin();
  if (!rtcOk) {
    if (RTC_REQUIRED) haltWithError("DS1307 not found");
    Serial.println("WARN: DS1307 not found, timestamps are uptime seconds");
    lcd.setCursor(0, 1); lcd.print("RTC missing!");
    delay(2000);
    return;
  }
  rtcModuleDiag();
  char ts[20];
  if (readRtcValidated(ts, sizeof(ts))) Serial.printf("DS1307 time %s\n", ts);
  else explainInvalidTime();
}

void rtcService() {
  if (rtcOk || !rtc.begin()) return;
  rtcOk = true;
  Serial.println("DS1307 found, switching to RTC timestamps");
  rtcModuleDiag();
}

// After three straight failures with no ack the flag drops (bench 2026-11-02: the module fell off
// the bus 15 s after detection and the flag stayed true for minutes); re-detection takes over.
static void noteBadRead() {
  rtcBadReads++;
  if (rtcBadReads >= 3 && !i2cAck(RTC_ADDR)) {
    rtcOk = false; rtcBadReads = 0;
    Serial.println("DS1307 lost from the bus (no ack), back to uptime stamps");
    return;
  }
  if (rtcBadReads % 10 != 1) return;
  if (rtcTimeLost())
    Serial.printf("RTC raw %02X %02X %02X %02X %02X %02X %02X: ",
                  rtcRaw[0], rtcRaw[1], rtcRaw[2], rtcRaw[3], rtcRaw[4], rtcRaw[5], rtcRaw[6]);
  explainInvalidTime();
}

bool rtcTimestamp(char* ts, size_t n) {
  if (rtcOk && readRtcValidated(ts, n)) { rtcBadReads = 0; return true; }
  snprintf(ts, n, "uptime_%lus", millis() / 1000UL);
  if (rtcOk) noteBadRead();
  return false;
}

void rtcSetFromSerial(const char* line) {
  int Y, M, D, h, m, s;
  if (sscanf(line, "T=%d-%d-%d %d:%d:%d", &Y, &M, &D, &h, &m, &s) != 6) {
    Serial.println("ignored: expected T=YYYY-MM-DD HH:MM:SS"); return;
  }
  if (!rtcOk) { Serial.println("ignored: no RTC"); return; }
  rtc.adjust(DateTime(Y, M, D, h, m, s));
  char back[20];
  bool ok = readRtcValidated(back, sizeof(back));
  Serial.printf("RTC set to %04d-%02d-%02d %02d:%02d:%02d, read back: %s\n", Y, M, D, h, m, s, ok ? back : "INVALID");
}
