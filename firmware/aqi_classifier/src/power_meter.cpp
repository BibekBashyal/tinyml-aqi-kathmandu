#include "power_meter.h"
#include <math.h>
#include "node_hw.h"

static const uint8_t  INA_ADDR         = 0x40;
static const uint8_t  REG_CONFIG       = 0x00;
static const uint8_t  REG_SHUNT        = 0x01;   // signed, 10 uV per bit (PGA /8: +/-320 mV)
static const uint8_t  REG_BUS          = 0x02;   // bits 15..3, 4 mV per bit
// Config word: 32 V bus range | PGA | bus ADC | shunt ADC | shunt+bus continuous.
// Run time: PGA /8 (+/-320 mV = 3.2 A, so the short guard sees large currents), 12-bit single samples (532 us).
// Benchmark: PGA /2 (+/-80 mV = 800 mA, smaller offset error for the ~50-250 mA node), 128-sample averages (68 ms).
static const uint16_t CONFIG_NORMAL    = 0x399F;
static const uint16_t CONFIG_AVG128    = 0x2FFF;
static const uint16_t REG_ALL_ONES     = 0xFFFF;   // what a read returns when SDA is stuck high
static const float    SHUNT_OHM        = 0.1f;   // R100 on the module
static const float    SHUNT_LSB_MV     = 0.01f;
static const float    BUS_LSB_V        = 0.004f;
static const float    SHUNT_FULL_MV    = 319.9f; // saturation of the +/-320 mV range (3.2 A)
static const float    SHORT_BUS_V_MAX  = 1.0f;   // a load's bus voltage is never this low...
static const float    SHORT_CURR_MA    = 300.0f; // ...while this much current flows
static const float    IMPLAUSIBLE_MA   = 1000.0f;// the node never draws this; VIN- on 3V3 does
static const uint8_t  LOST_AFTER_FAILS = 3;

bool inaOk = false;
bool inaShort = false;
static uint8_t inaFails = 0;

static bool readReg(uint8_t reg, uint16_t& v) {
  Wire.beginTransmission(INA_ADDR); Wire.write(reg);
  if (Wire.endTransmission(true) != 0) return false;
  if (Wire.requestFrom(INA_ADDR, (uint8_t)2) != 2) return false;
  v = (uint16_t)(Wire.read() << 8); v |= (uint16_t)Wire.read();
  return true;
}

static bool writeReg(uint8_t reg, uint16_t v) {
  Wire.beginTransmission(INA_ADDR); Wire.write(reg); Wire.write((uint8_t)(v >> 8)); Wire.write((uint8_t)v);
  return Wire.endTransmission(true) == 0;
}

// Config writes are read back: a glitch must not leave the benchmark's PGA /2 (800 mA full scale)
// active, where the short guard could no longer see a rail-level fault.
static const int CONFIG_ATTEMPTS = 3;
static bool writeConfig(uint16_t v) {
  for (int i = 0; i < CONFIG_ATTEMPTS; i++) {
    uint16_t back;
    if (writeReg(REG_CONFIG, v) && readReg(REG_CONFIG, back) && back == v) return true;
  }
  return false;
}

static bool inaStart() {
  if (!i2cAck(INA_ADDR) || !writeConfig(CONFIG_NORMAL)) return false;
  inaFails = 0;
  Serial.println("INA219 found (0.1 ohm shunt, +/-3.2 A, current from shunt voltage)");
  return true;
}

void inaBegin() {
  inaOk = inaStart();
  if (!inaOk) Serial.println("WARN: INA219 not found (continuing)");
}

void inaService() {
  if (!inaOk) inaOk = inaStart();
}

bool inaSample(InaSample& s) {
  uint16_t shunt, bus;
  if (!readReg(REG_SHUNT, shunt) || !readReg(REG_BUS, bus)) return false;
  if (bus == REG_ALL_ONES) return false;   // 32.76 V: a garbage transfer, not a measurement
  float shunt_mV = (int16_t)shunt * SHUNT_LSB_MV;
  s.current_mA = shunt_mV / SHUNT_OHM;
  s.busV = (bus >> 3) * BUS_LSB_V;
  inaShort = fabsf(shunt_mV) >= SHUNT_FULL_MV || fabsf(s.current_mA) > IMPLAUSIBLE_MA ||
             (s.busV < SHORT_BUS_V_MAX && fabsf(s.current_mA) > SHORT_CURR_MA);
  return true;
}

void inaRead(float& busV, float& curr_mA) {
  busV = NAN; curr_mA = NAN;
  if (!inaOk) return;
  InaSample s;
  if (!inaSample(s)) {
    if (++inaFails >= LOST_AFTER_FAILS) {
      inaOk = false; inaShort = false;
      Serial.println("INA219 lost from the bus (no ack), bus_V/current_mA logged as nan");
    }
    return;
  }
  inaFails = 0;
  busV = s.busV; curr_mA = s.current_mA;
  if (inaShort)
    Serial.printf("WARN: INA219 bus %.2f V with %.0f mA -> shunt across a rail or a short. Unplug VIN+/VIN- now\n",
                  busV, curr_mA);
}

bool inaSetAveraging(bool avg128) {
  if (!inaOk) return false;
  if (writeConfig(avg128 ? CONFIG_AVG128 : CONFIG_NORMAL)) return true;
  if (!avg128) { inaOk = false; inaShort = false; }   // unknown config: drop it, inaService() rewrites it
  return false;
}
