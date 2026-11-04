#include "sensors.h"
#include <math.h>
#include "node_hw.h"

static const int      PMS_FRAME_BYTES   = 32;
static const uint8_t  PMS_HEADER_1      = 0x42;
static const uint8_t  PMS_HEADER_2      = 0x4D;
static const uint32_t PMS_BODY_WAIT_MS  = 300;   // the 30 bytes after the header arrive within ~32 ms

// Read the 30 bytes after a 0x42 0x4D header and verify the checksum.
static bool readFrameBody(uint8_t buf[PMS_FRAME_BYTES]) {
  uint32_t hdr_ms = millis();
  while (pmsSerial.available() < PMS_FRAME_BYTES - 2 && millis() - hdr_ms < PMS_BODY_WAIT_MS) delay(2);
  if (pmsSerial.available() < PMS_FRAME_BYTES - 2) return false;
  for (int i = 2; i < PMS_FRAME_BYTES; i++) buf[i] = pmsSerial.read();
  uint16_t sum = 0;
  for (int i = 0; i < PMS_FRAME_BYTES - 2; i++) sum += buf[i];
  return sum == (uint16_t)((buf[30] << 8) | buf[31]);
}

PmsReading readPMS(uint32_t timeout_ms) {
  PmsReading r = {0, 0, false};
  while (pmsSerial.available()) pmsSerial.read();   // drop the stale backlog
  uint8_t buf[PMS_FRAME_BYTES];
  uint32_t start = millis();
  while (millis() - start < timeout_ms) {
    if (pmsSerial.available() < 2) { delay(10); continue; }
    if (pmsSerial.read() != PMS_HEADER_1) continue;
    if (pmsSerial.peek() != PMS_HEADER_2) continue;
    buf[0] = PMS_HEADER_1; buf[1] = pmsSerial.read();
    if (!readFrameBody(buf)) continue;
    // atmospheric-environment values: PM2.5 in bytes 12-13, PM10 in bytes 14-15
    r.pm25 = (buf[12] << 8) | buf[13];
    r.pm10 = (buf[14] << 8) | buf[15];
    r.ok = true;
    return r;
  }
  return r;
}

float humidityCorrect(float pm25_raw, float rh) {
  if (RH_ALPHA == 0.0f || isnan(rh)) return pm25_raw;
  return pm25_raw / (1.0f + RH_ALPHA * powf(rh / 100.0f, RH_BETA));
}
