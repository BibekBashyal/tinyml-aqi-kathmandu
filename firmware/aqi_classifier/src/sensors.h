// sensors.h - PMS5003 frame parser and the PM2.5 humidity correction.
#pragma once
#include <Arduino.h>

struct PmsReading { uint16_t pm25; uint16_t pm10; bool ok; };

// The PMS5003 sends a frame every 0.2-2.3 s while a cycle takes 3 s, so the UART buffer fills
// and its oldest frame is ~8 s stale (code review 2026-11-03). readPMS drops the backlog and
// returns the first checksum-verified frame that arrives afterwards, or ok=false on timeout.
static const uint32_t PMS_FRAME_TIMEOUT_MS = 3000;   // > the sensor's 2.3 s slowest frame interval
PmsReading readPMS(uint32_t timeout_ms = PMS_FRAME_TIMEOUT_MS);
// x1c = pm25 / (1 + a*(rh/100)^b). RH_ALPHA = 0 in config.h makes this an exact pass-through
// (no powf call) until the coefficients are fitted; a nan humidity leaves PM2.5 unchanged.
float humidityCorrect(float pm25_raw, float rh);
