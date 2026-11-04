// node_hw.h - the node's hardware objects and the I2C bus diagnostics shared by every module.
// Two voltage domains: ESP32-S3 + INA219 on 3.3 V, DS1307 + LCD on 5 V behind a BSS138 shifter
// (SDA on shifter channel 3, SCL on channel 2; channel 1 is flaky and stays empty).
#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <DHT.h>
#include <RTClib.h>
#include <hd44780.h>
#include <hd44780ioClass/hd44780_I2Cexp.h>   // PCF8574 backpack; library proven on the bench
#include "config.h"

extern HardwareSerial pmsSerial;
extern DHT dht;
extern RTC_DS1307 rtc;
extern hd44780_I2Cexp lcd;
extern bool lcdOk;   // LCD backpack answered at the last check

static const uint8_t RTC_ADDR        = 0x68;
static const uint8_t RTC_EEPROM_ADDR = 0x50;   // 24C32 on the DS1307 module

bool i2cAck(uint8_t addr);
// Electrical state of a bus line, used only while the LCD is missing: "up" = external pull-up
// present, "float" = nothing connected, "low" = held at ground.
const char* lineState(int pin);
// Lists every address that answers, again only when the set changes (runs while the LCD acks).
void i2cRollCall(bool busHealthy);
// Once per cycle: re-initialise the LCD when it returns, print the bus verdict when it is gone.
bool lcdService();
void haltWithError(const char* msg);
