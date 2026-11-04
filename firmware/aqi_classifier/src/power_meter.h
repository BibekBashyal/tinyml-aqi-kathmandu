// power_meter.h - INA219 on the 3.3 V side, shunt in series with the ESP32's 5 V input.
// Wiring: adapter 5 V -> VIN+, VIN- -> ESP32 5V pin, so the bus voltage is the board's supply and
// the current is everything the ESP32 and its 3.3 V loads draw. Wired ACROSS a rail instead
// (VIN- on GND or on 3V3) the shunt shorts that rail and heats the regulator; inaRead flags it.
// Registers are read directly with every transfer checked, and the current is computed from the
// shunt voltage (no calibration register, no library rewrite of it before each read).
#pragma once
#include <Arduino.h>

extern bool inaOk;      // INA219 answered at the last check
extern bool inaShort;   // shunt looks wired across a rail, or saturated

struct InaSample { float busV; float current_mA; };

void inaBegin();                            // detect at boot and configure
void inaService();                          // re-detect every cycle after a dropout
void inaRead(float& busV, float& curr_mA);  // status/log reading; nan when absent or the read failed
bool inaSample(InaSample& s);               // one checked read of shunt + bus registers
bool inaSetAveraging(bool avg128);          // benchmark mode (PGA /2, 128-sample averages) or normal
