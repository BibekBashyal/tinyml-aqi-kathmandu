#include "serial_cmd.h"
#include <stdlib.h>
#include <string.h>
#include "rtc_time.h"
#include "inference.h"
#include "sensors.h"
#include "benchmark.h"

static const size_t   LINE_MAX_BYTES = 128;
static const uint32_t TEST_IDLE_MS   = 10000;

static char     lineBuf[LINE_MAX_BYTES];
static size_t   lineLen = 0;
static bool     lineOverflow = false;
static uint32_t lastVectorMs = 0;
static bool     testActive = false;

// "<id>,pm25,pm10,temp,rh" after the "F=" prefix. strtod then a float cast: the laptop does the
// same (float(s) then float32), so both sides start from the identical float32 value.
static bool parseVector(const char* p, unsigned long& id, float v[kNumFeatures]) {
  char* end;
  id = strtoul(p, &end, 10);
  if (end == p || *end != ',') return false;
  p = end + 1;
  for (int i = 0; i < kNumFeatures; i++) {
    double d = strtod(p, &end);
    if (end == p) return false;
    v[i] = (float)d;
    if (i < kNumFeatures - 1) { if (*end != ',') return false; p = end + 1; }
    else if (*end != '\0') return false;
  }
  return true;
}

static void testVector(const char* line) {
  unsigned long id = 0; float v[kNumFeatures], probs[kNumClasses]; uint32_t us = 0;
  if (!parseVector(line + 2, id, v)) {
    if (id) Serial.printf("C=%lu,-2\n", id); else Serial.println("ignored: expected F=<id>,pm25,pm10,temp,rh");
    return;
  }
  if (!testActive) { testActive = true; Serial.println("TEST: sensor cycle paused while F= vectors arrive"); }
  lastVectorMs = millis();
  float raw[kNumFeatures] = { humidityCorrect(v[0], v[3]), v[1], v[2], v[3] };   // same path as a live cycle
  int cls = classify(raw, probs, &us);
  if (cls < 0) { Serial.printf("C=%lu,-1\n", id); return; }
  Serial.printf("C=%lu,%d,%.8f,%lu", id, cls, probs[cls], (unsigned long)us);
  for (int i = 0; i < kNumClasses; i++) Serial.printf(",%.8f", probs[i]);
  Serial.println();
}

static void dispatch(const char* line) {
  if (strncmp(line, "T=", 2) == 0) { rtcSetFromSerial(line); return; }
  if (strncmp(line, "F=", 2) == 0) { testVector(line); return; }
  if (strcmp(line, "BENCH") == 0)  { runBenchmarks(); return; }
  if (strcmp(line, "MODEL") == 0)  {
    Serial.printf("M=%u,%08lx,%08lx\n", (unsigned)tflmModelBytes(), (unsigned long)tflmModelCrc32(), (unsigned long)tflmInputCrc32());
    return;
  }
  Serial.println("ignored: commands are T=YYYY-MM-DD HH:MM:SS | F=<id>,pm25,pm10,temp,rh | BENCH | MODEL");
}

void serialService() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen && !lineOverflow) { lineBuf[lineLen] = '\0'; dispatch(lineBuf); }
      lineLen = 0; lineOverflow = false;
      continue;
    }
    if (lineLen < LINE_MAX_BYTES - 1) lineBuf[lineLen++] = c;
    else lineOverflow = true;   // drop the whole overlong line, not just its head
  }
}

bool serialTestActive() {
  if (testActive && millis() - lastVectorMs >= TEST_IDLE_MS) {
    testActive = false;
    Serial.println("TEST: no vectors for 10 s, sensor cycle resumed");
  }
  return testActive;
}
