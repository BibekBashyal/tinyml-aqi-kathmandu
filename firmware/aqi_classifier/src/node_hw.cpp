#include "node_hw.h"
#include <string.h>

HardwareSerial pmsSerial(1);
DHT dht(DHT_PIN, DHT_TYPE);
RTC_DS1307 rtc;
hd44780_I2Cexp lcd(LCD_ADDR);   // address hardcoded: auto-detect failed on the bench
bool lcdOk = false;

bool i2cAck(uint8_t addr) { Wire.beginTransmission(addr); return Wire.endTransmission() == 0; }

const char* lineState(int pin) {
  pinMode(pin, INPUT_PULLDOWN); delay(2); int pd = digitalRead(pin);
  pinMode(pin, INPUT_PULLUP);   delay(2); int pu = digitalRead(pin);
  pinMode(pin, INPUT);
  return (pd && pu) ? "up" : (!pd && pu) ? "float" : "low";
}

// ---- I2C roll call (bench 2026-11-02) ----
// A module on the wrong row, unpowered, or with a non-default address (INA219 A0/A1: 0x40-0x4F)
// shows up at once. Skipped on a broken bus: the verdict line says enough there.
static uint8_t i2cSeen[16] = {0};   // one bit per 7-bit address

static const char* i2cName(uint8_t a) {
  if (a == LCD_ADDR) return "LCD";
  if (a == RTC_ADDR) return "DS1307";
  if (a == RTC_EEPROM_ADDR) return "RTC-EEPROM";
  if (a >= 0x40 && a <= 0x4F) return "INA219";
  return "unknown";
}

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
    if (a == 0x40) inaAt40 = true; else if (a >= 0x41 && a <= 0x4F) inaElsewhere = true;
    Serial.printf(" 0x%02X=%s", a, i2cName(a));
  }
  if (!inaAt40 && inaElsewhere) Serial.print("  (INA219 answers but not at 0x40: A0/A1 jumper bridged, open both pads)");
  if (!inaAt40 && !inaElsewhere) Serial.print("  (no INA219: check its VCC 3V3, GND, SDA on the LV3 row, SCL on the LV2 row)");
  Serial.println();
}

// Why the LCD does not answer, from the electrical state of the two lines on the ESP32 side.
// Note (bench 2026-11-02): with the INA219's own pull-ups on the 3.3 V side, "up" no longer
// proves the shifter's LV side is powered; "float" still proves the GPIO8/9 wires are open.
static void printBusVerdict() {
  Wire.end();
  const char* sdaSt = lineState(SDA_PIN); const char* sclSt = lineState(SCL_PIN);
  Wire.begin(SDA_PIN, SCL_PIN); Wire.setClock(I2C_CLOCK_HZ);
  const char* why = (strcmp(sdaSt, "low") == 0 || strcmp(sclSt, "low") == 0) ? "a bus wire sits on GND or a short"
                  : (strcmp(sdaSt, "float") == 0 || strcmp(sclSt, "float") == 0) ? "GPIO8/9 to shifter LV3/LV2 open, or shifter LV unpowered"
                  : "3.3 V side fine: shifter HV supply, HV3/HV2 wires, or LCD 5 V";
  Serial.printf("I2C: no LCD ack. SDA %s, SCL %s -> %s\n", sdaSt, sclSt, why);
}

bool lcdService() {
  bool lcdAck = i2cAck(LCD_ADDR);
  i2cRollCall(lcdAck);
  if (lcdAck && !lcdOk) {   // LCD lost power or bus (shows blocks): re-initialise as soon as it answers
    lcdOk = (lcd.begin(LCD_COLS, LCD_ROWS) == 0);
    if (lcdOk) { lcd.backlight(); Serial.println("LCD back on the bus"); }
  }
  if (!lcdAck) { lcdOk = false; printBusVerdict(); }
  return lcdAck;
}

void haltWithError(const char* msg) {
  Serial.println(msg);
  lcd.clear(); lcd.print(msg);
  while (true) delay(1000);
}
