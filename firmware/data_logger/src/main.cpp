#include <Arduino.h>          // Required in PlatformIO (Arduino IDE adds this automatically)
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <SPI.h>
#include <SD.h>
#include <RTClib.h>
// ESP-IDF SD-over-SPI driver -- used only for the robust format routine below.
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include "sdmmc_cmd.h"
#include "esp_vfs_fat.h"
#include <hd44780.h>
#include <hd44780ioClass/hd44780_I2Cexp.h>   // auto-detects the PCF8574 backpack
#include "DHT.h"

// --- PIN CONFIGURATION (valid for ESP32-S3) ---
#define DHTPIN 2
#define DHTTYPE DHT22
#define PMS_RX 18
#define PMS_TX 17
#define I2C_SDA 8    // moved off 35/36 -- those are used by octal PSRAM on 8MB-PSRAM S3 modules
#define I2C_SCL 9

// microSD over SPI. THIS BOARD IS AN N16R8 (has 8MB OCTAL PSRAM), so GPIO 33-37
// are consumed by the PSRAM and CANNOT be used -- that broke SCK/MISO on 35/37.
// GPIO 10-14 are safe on EVERY S3 variant (never flash, never PSRAM). This is
// also the pin set that produced the card's 0x01 response earlier.
#define SD_CS   10
#define SD_MOSI 11
#define SD_SCK  12
#define SD_MISO 14

#define LOG_FILE "/aqi_log.csv"
#define CSV_HEADER "uptime_ms,pm1_0,pm2_5,pm10,temp_c,humidity_pct"

// --- OBJECT INITIALIZATION ---
Adafruit_INA219 ina219;
DHT dht(DHTPIN, DHTTYPE);
RTC_DS1307 rtc;       // DS1307 Tiny RTC on I2C (addr 0x68), SDA=8 / SCL=9
hd44780_I2Cexp lcd(0x27);   // 16x2 I2C LCD at 0x27 (found by scan; skip auto-detect)

// RTC_TEST_MODE: bring up the DS1307 by itself. Finds it on I2C, sets the clock
// to build time on first run, then prints the time every second. Set false to
// return to the normal node.
#define RTC_TEST_MODE false

// INA219_TEST_MODE: bring up the INA219 power monitor BY ITSELF. It was hanging
// the I2C bus before, so this scans the bus fresh every second — reseat wires
// and watch the readings change live. Set false to return to the normal node.
#define INA219_TEST_MODE false

// LCD_TEST_MODE: bring up the 16x2 I2C LCD together with the RTC. Shows the live
// clock on the screen. Set false to return to the normal node.
#define LCD_TEST_MODE false

// SD_FORMAT_MODE: ONE-SHOT. Formats the SD card to FAT32 with the ESP-IDF driver
// (format_if_mount_failed) -- far more robust than the Arduino SD format_if_empty.
// For when you have no card reader to format on a PC. After it prints PASS, set
// this back to false. Takes priority over all other modes.
#define SD_FORMAT_MODE false

// --- TIMING ---
unsigned long lastLog = 0;
const int logInterval = 3000; // Log data every 3 seconds

// --- STATE ---
bool sdReady = false;         // false = log to serial only, node keeps running
int lastPm1 = -1;             // -1 = no PMS frame seen yet this interval
int lastPm25 = -1;
int lastPm10 = -1;

// Mounts the card and makes sure the CSV exists with a header row.
// Returns false (without halting) if the card is missing — the node still runs
// and prints to serial, it just doesn't persist.
// Sends a raw CMD0 (GO_IDLE_STATE) over SPI and returns the card's R1 byte.
// 0x01 = card answered and entered idle  -> these pins are correct.
// 0xFF = card said nothing at all.
// Runs at 400 kHz, the rate the SD spec requires during initialization.
static uint8_t sdRawCmd0(int cs, int sck, int mosi, int miso) {
  SPI.end();
  SPI.begin(sck, miso, mosi, cs);
  pinMode(cs, OUTPUT);
  digitalWrite(cs, HIGH);

  SPI.beginTransaction(SPISettings(400000, MSBFIRST, SPI_MODE0));

  // >=74 clocks with CS high: the card needs these to wake into SPI mode.
  for (int i = 0; i < 10; i++) SPI.transfer(0xFF);

  digitalWrite(cs, LOW);
  delayMicroseconds(100);

  static const uint8_t CMD0[] = {0x40, 0x00, 0x00, 0x00, 0x00, 0x95};  // 0x95 = valid CRC7
  for (uint8_t i = 0; i < sizeof(CMD0); i++) SPI.transfer(CMD0[i]);

  uint8_t r1 = 0xFF;
  for (int i = 0; i < 16; i++) {            // card may stall a few bytes before replying
    r1 = SPI.transfer(0xFF);
    if (r1 != 0xFF) break;
  }

  digitalWrite(cs, HIGH);
  SPI.transfer(0xFF);
  SPI.endTransaction();
  SPI.end();
  return r1;
}

// The user reports the wiring matches the pin map, yet CMD0 goes unanswered.
// Rather than trust the silkscreen, brute-force the orderings and let the card
// tell us which one it actually likes. MISO is held at 13 (already proven live).
void probeSDPinOrders() {
  const int P[3] = {SD_CS, SD_MOSI, SD_SCK};   // 10, 11, 12
  const int perms[6][3] = {                    // {cs, mosi, sck}
    {P[0], P[1], P[2]}, {P[0], P[2], P[1]},
    {P[1], P[0], P[2]}, {P[1], P[2], P[0]},
    {P[2], P[0], P[1]}, {P[2], P[1], P[0]},
  };

  Serial.print("SD      | --- raw CMD0 pin-order probe (MISO fixed at ");
  Serial.print(SD_MISO);
  Serial.println(") ---");
  for (int i = 0; i < 6; i++) {
    int cs = perms[i][0], mosi = perms[i][1], sck = perms[i][2];
    uint8_t r1 = sdRawCmd0(cs, sck, mosi, SD_MISO);

    Serial.print("SD      | CS="); Serial.print(cs);
    Serial.print(" MOSI="); Serial.print(mosi);
    Serial.print(" SCK="); Serial.print(sck);
    Serial.print("  -> R1=0x");
    if (r1 < 0x10) Serial.print('0');
    Serial.print(r1, HEX);
    // Only 0x01 is a real CMD0 answer. 0xFF is an idle-high line (no reply) and
    // 0x00 is a line stuck at ground -- NEITHER means the card is talking.
    if (r1 == 0x01)      Serial.println("  *** CARD RESPONDED — THIS IS THE CORRECT ORDER ***");
    else if (r1 == 0xFF) Serial.println("  (no reply - line idle high)");
    else if (r1 == 0x00) Serial.println("  (MISO STUCK LOW - not a reply, check the MISO wire)");
    else                 Serial.println("  (unexpected reply)");
    delay(50);
  }
  Serial.println("SD      | --- end probe ---");
}

// The SD module's pull-up resistors hold MISO high whenever the module has
// power. So: sweep every free GPIO, and any pin that stays high under an
// internal pulldown has something actively driving it -- that is where the
// MISO wire physically landed. Finds a wire plugged into the wrong hole, and
// distinguishes "wrong pin" from "module has no power at all".
void huntForMISO() {
  // Everything free on this board. Excludes 2 (DHT), 17/18 (PMS), 10/11/12 (SD
  // outputs), 19/20 (USB), 26-32 (flash), 43/44 (UART0), 0/3/45/46 (strapping).
  const int candidates[] = {4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 21,
                            35, 36, 37, 38, 39, 40, 41, 42, 47, 48};
  const int n = sizeof(candidates) / sizeof(candidates[0]);

  // Hold CS low, or the module's MISO buffer stays disabled and nothing is driven.
  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, LOW);
  delay(5);

  Serial.println("SD      | --- hunting for a driven-high pin (the MISO wire) ---");
  int found = 0;
  for (int i = 0; i < n; i++) {
    int pin = candidates[i];
    pinMode(pin, INPUT_PULLDOWN);   // fight the pin low; only a real driver wins
    delay(5);
    if (digitalRead(pin)) {
      Serial.print("SD      | GPIO ");
      Serial.print(pin);
      Serial.println(" is DRIVEN HIGH  <-- something is connected here");
      found++;
    }
    pinMode(pin, INPUT);
  }

  if (found == 0) {
    Serial.println("SD      | NO pin driven high (with CS low). MISO wire is broken,");
    Serial.println("SD      | or the card is not answering -- likely under-volted.");
  }
  Serial.println("SD      | --- end hunt ---");
  digitalWrite(SD_CS, HIGH);
}

bool initSD() {
  Serial.println("SD      | Probing card...");
  huntForMISO();

  // CS must be LOW to read MISO meaningfully. On the HW-125 breakout the 74LVC125
  // buffer driving MISO has its output-enable tied to CS, so with CS high the
  // line is high-impedance BY DESIGN and reads as "floating" even on a perfectly
  // healthy module. Selecting the card first is the only way this test means
  // anything.
  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, LOW);
  delay(5);

  pinMode(SD_MISO, INPUT_PULLUP);
  delay(5);
  bool misoWithPullup = digitalRead(SD_MISO);
  pinMode(SD_MISO, INPUT_PULLDOWN);
  delay(5);
  bool misoWithPulldown = digitalRead(SD_MISO);
  pinMode(SD_MISO, INPUT);
  digitalWrite(SD_CS, HIGH);   // release the card again

  Serial.print("SD      | MISO test (CS held low): pullup="); Serial.print(misoWithPullup);
  Serial.print(" pulldown="); Serial.print(misoWithPulldown);
  if (misoWithPullup && misoWithPulldown) {
    Serial.println("  -> DRIVEN HIGH (card powered + MISO connected)");
  } else if (misoWithPullup && !misoWithPulldown) {
    Serial.println("  -> FLOATING (MISO not connected, or card has NO POWER)");
  } else {
    Serial.println("  -> STUCK LOW (MISO shorted to GND, or wired to the wrong pin)");
  }

  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);

  // Internal pull-up on MISO. The SD spec wants a pull-up on the DAT/MISO line;
  // bare modules often omit it. CMD0's single reply byte survives without one, but
  // the longer init data reads (CMD8/ACMD41/OCR, boot sector) are marginal -- this
  // firms them up. Set AFTER SPI.begin so it isn't overwritten by the matrix setup.
  pinMode(SD_MISO, INPUT_PULLUP);

  // CS idle-high before any traffic. Some breakouts never release the bus
  // otherwise, and the card then ignores CMD0.
  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);
  delay(10);

  // 400 kHz -- the rate the raw CMD0 probe gets a clean 0x01 at. 1 MHz was still
  // too fast to complete the full init over the jumper wires to the module.
  // Last arg (format_if_empty=true) makes the ESP32 format the card to FAT itself
  // if it mounts but has a broken/missing filesystem.
  if (!SD.begin(SD_CS, SPI, 200000, "/sd", 5, true)) {
    Serial.println("SD      | Card mount FAILED");
    probeSDPinOrders();
    return false;
  }

  if (SD.cardType() == CARD_NONE) {
    Serial.println("SD      | No card detected in the slot");
    return false;
  }

  // Write the header once, on first creation only, so appends across reboots
  // don't sprinkle header rows through the middle of the dataset.
  if (!SD.exists(LOG_FILE)) {
    File f = SD.open(LOG_FILE, FILE_WRITE);
    if (!f) {
      Serial.println("SD      | Could not create " LOG_FILE);
      return false;
    }
    f.println(CSV_HEADER);
    f.close();
    Serial.println("SD      | Created " LOG_FILE);
  }

  Serial.print("SD      | Ready — ");
  Serial.print(SD.cardSize() / (1024ULL * 1024ULL));
  Serial.println(" MB card, appending to " LOG_FILE);
  return true;
}

// One row per call. Opened and closed each time so a power cut costs at most
// the current row rather than the whole file.
void logRow(int pm1, int pm25, int pm10, float t, float h) {
  File f = SD.open(LOG_FILE, FILE_APPEND);
  if (!f) {
    Serial.println("SD      | Append failed — card removed?");
    sdReady = false;
    return;
  }

  f.print(millis());  f.print(',');
  // -1 means "no PMS frame yet" — write an empty cell (pandas reads it as NaN)
  // instead of a fake -1 concentration.
  if (pm1 >= 0)  f.print(pm1);
  f.print(',');
  if (pm25 >= 0) f.print(pm25);
  f.print(',');
  if (pm10 >= 0) f.print(pm10);
  f.print(',');

  // NaN from a failed DHT read becomes an empty cell, which pandas reads as NaN.
  if (!isnan(t)) f.print(t, 1);
  f.print(',');
  if (!isnan(h)) f.print(h, 1);
  f.println();

  f.close();
}

// Set false to skip the SD probe entirely and run as a serial-only node.
// Useful to prove the sensors still work when the card is misbehaving.
#define ENABLE_SD true

// SD_TEST_MODE: bench-test the SD module BY ITSELF. Skips the sensors entirely
// and re-runs the full SD diagnostic every 3s, so you can reseat a wire and watch
// the reading change live instead of resetting between every attempt.
// Set back to false to return to the normal 3-sensor node.
#define SD_TEST_MODE false

// STATIC_PIN_TEST: hold CS/MOSI/SCK at a fixed level for 6s at a time so a
// multimeter can actually read them. SPI normally toggles far too fast to probe.
// Lets you follow a signal ESP32 -> wire -> module header -> 74LVC125 output and
// find exactly where it stops. Set false for normal operation.
#define STATIC_PIN_TEST false

// LOOPBACK_TEST: proves the ESP32's SPI peripheral + the MOSI and MISO pins work,
// with NO module and NO card involved. Jumper GPIO 11 (MOSI) straight to GPIO 14
// (MISO), remove everything else. SPI sends known bytes; if they read back, the
// ESP32 side is perfect and the fault is downstream. If not, it's the board/pins.
#define LOOPBACK_TEST false

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== AQI NODE BOOT ===");

  if (SD_FORMAT_MODE) {
    Serial.println("MODE    | SD FORMAT (one-shot, ESP-IDF FAT32)");
    return;   // loop() drives the format
  }

  if (LCD_TEST_MODE) {
    Serial.println("MODE    | LCD + RTC TEST -- I2C on SDA=8 / SCL=9");
    return;   // loop() drives the LCD test
  }

  if (RTC_TEST_MODE) {
    Serial.println("MODE    | DS1307 RTC TEST -- I2C on SDA=8 / SCL=9");
    return;   // loop() drives the RTC test
  }

  if (INA219_TEST_MODE) {
    Serial.println("MODE    | INA219 POWER MONITOR TEST -- I2C on SDA=8 / SCL=9");
    Serial.println("MODE    | Rescans every second; reseat wires while it runs.\n");
    return;   // loop() drives the INA219 test
  }

  // --- INA219 disabled while debugging its hardware (was hanging the I2C bus) ---
  // Wire.begin(I2C_SDA, I2C_SCL);
  // if (!ina219.begin()) {
  //   Serial.println("INA219 NOT FOUND");
  // } else {
  //   ina219.setCalibration_32V_2A();
  //   Serial.println("INA219 Reset & Initialized.");
  // }

  if (SD_TEST_MODE) {
    Serial.println("MODE    | SD BENCH TEST — sensors disabled, SD retried every 3s");
    Serial.println("MODE    | Reseat wires while it runs; the readings update live.\n");
    return;   // nothing else to bring up; loop() drives the SD test
  }

  Serial1.begin(9600, SERIAL_8N1, PMS_RX, PMS_TX);
  dht.begin();
  Serial.println("SENSORS | PMS5003 + DHT22 initialized");

  // SD goes LAST: if the card wedges the SPI bus, at least we know from the
  // lines above that everything else came up cleanly.
  sdReady = ENABLE_SD ? initSD() : false;
  if (!ENABLE_SD) Serial.println("SD      | Disabled at compile time");

  Serial.println("=== SETUP COMPLETE ===\n");
}

// --- RAW FULL-INIT PROBE -------------------------------------------------
// CMD0 passes but the Arduino driver's mount fails with no detail. So walk the
// whole init sequence by hand and print every response byte: the exact step
// that fails names the fault (CMD8=MOSI data integrity, ACMD41=power/card,
// CMD17 read=MISO data integrity).
static uint8_t rawCmd(uint8_t cmd, uint32_t arg, uint8_t crc) {
  SPI.transfer(0xFF);                       // 8 idle clocks between commands
  SPI.transfer(0x40 | cmd);
  SPI.transfer((arg >> 24) & 0xFF);
  SPI.transfer((arg >> 16) & 0xFF);
  SPI.transfer((arg >> 8) & 0xFF);
  SPI.transfer(arg & 0xFF);
  SPI.transfer(crc);
  uint8_t r1 = 0xFF;
  for (int i = 0; i < 16 && r1 == 0xFF; i++) r1 = SPI.transfer(0xFF);
  return r1;
}

void sdRawInitProbe() {
  Serial.println("SD      | --- raw full-init probe (200 kHz) ---");
  SPI.end();
  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  pinMode(SD_MISO, INPUT_PULLUP);
  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);
  delay(10);

  SPI.beginTransaction(SPISettings(200000, MSBFIRST, SPI_MODE0));
  for (int i = 0; i < 10; i++) SPI.transfer(0xFF);   // 80 wake-up clocks, CS high
  digitalWrite(SD_CS, LOW);

  // CMD0: enter SPI mode
  uint8_t r1 = rawCmd(0, 0, 0x95);
  Serial.printf("SD      | CMD0  (go idle)      -> R1=0x%02X %s\n", r1,
                r1 == 0x01 ? "OK" : "*** FAIL ***");

  // CMD8: voltage check, echoes 0x1AA back. Distinguishes SDv2 from v1.
  r1 = rawCmd(8, 0x000001AA, 0x87);
  uint8_t echo[4];
  for (int i = 0; i < 4; i++) echo[i] = SPI.transfer(0xFF);
  Serial.printf("SD      | CMD8  (voltage/echo) -> R1=0x%02X echo=%02X %02X %02X %02X %s\n",
                r1, echo[0], echo[1], echo[2], echo[3],
                (r1 == 0x01 && echo[2] == 0x01 && echo[3] == 0xAA) ? "OK (SDv2)"
                : (r1 & 0x04) ? "(illegal cmd -> SDv1 card)" : "*** BAD ECHO — MOSI/MISO corrupting data ***");

  // ACMD41 loop: tell the card to leave idle. Can take hundreds of ms.
  int tries = 0;
  unsigned long t0 = millis();
  do {
    rawCmd(55, 0, 0x65);                    // CMD55 = "next cmd is app-specific"
    r1 = rawCmd(41, 0x40000000, 0x77);      // HCS bit: we support SDHC
    tries++;
  } while (r1 != 0x00 && millis() - t0 < 2000);
  Serial.printf("SD      | ACMD41 (leave idle)  -> R1=0x%02X after %d tries, %lums %s\n",
                r1, tries, millis() - t0,
                r1 == 0x00 ? "OK (card ready)" : "*** FAIL — card never left idle (power?) ***");

  if (r1 == 0x00) {
    // CMD58: read OCR, bit30 = SDHC/block addressing
    r1 = rawCmd(58, 0, 0xFD);
    uint8_t ocr[4];
    for (int i = 0; i < 4; i++) ocr[i] = SPI.transfer(0xFF);
    Serial.printf("SD      | CMD58 (read OCR)     -> R1=0x%02X OCR=%02X%02X%02X%02X %s\n",
                  r1, ocr[0], ocr[1], ocr[2], ocr[3],
                  (ocr[0] & 0x40) ? "(SDHC, block-addressed)" : "(SDSC, byte-addressed)");

    // CMD17: read sector 0. Exercises a real 512-byte MISO burst -- the thing
    // the single-byte CMD0 reply never tests.
    r1 = rawCmd(17, 0, 0xFF);
    if (r1 != 0x00) {
      Serial.printf("SD      | CMD17 (read sec 0)   -> R1=0x%02X *** READ CMD REJECTED ***\n", r1);
    } else {
      uint8_t tok = 0xFF;
      t0 = millis();
      while (tok == 0xFF && millis() - t0 < 500) tok = SPI.transfer(0xFF);
      if (tok != 0xFE) {
        Serial.printf("SD      | CMD17 data token      -> 0x%02X *** NO DATA BLOCK ARRIVED ***\n", tok);
      } else {
        uint8_t buf[512];
        for (int i = 0; i < 512; i++) buf[i] = SPI.transfer(0xFF);
        SPI.transfer(0xFF); SPI.transfer(0xFF);   // discard CRC16
        Serial.printf("SD      | CMD17 (read sec 0)   -> token OK, tail=%02X %02X %s\n",
                      buf[510], buf[511],
                      (buf[510] == 0x55 && buf[511] == 0xAA)
                        ? "*** 55AA BOOT SIGNATURE — FULL READ PATH WORKS ***"
                        : "*** 512B read OK but bad signature — data corruption on MISO ***");
      }
    }
  }

  digitalWrite(SD_CS, HIGH);
  SPI.transfer(0xFF);
  SPI.endTransaction();
  SPI.end();
  Serial.println("SD      | --- end raw probe ---");
}

// Full round trip: mount, write a file, read it back, compare. Mounting alone
// only proves the card answered CMD0 -- this proves it can actually store data.
void sdBenchTest() {
  Serial.println("\n################ SD BENCH TEST ################");
  sdRawInitProbe();

  // The raw probe just reset the card to idle behind the SD library's back. Drop
  // any stale mount so initSD()'s SD.begin() does a genuine full re-mount.
  SD.end();

  if (initSD()) {
    Serial.println("SD      | MOUNT OK — now testing a real write/read...");

    const char *TEST_PATH = "/sd_selftest.txt";
    const char *PAYLOAD   = "tinyml-aqi-node selftest";

    File f = SD.open(TEST_PATH, FILE_WRITE);
    if (!f) {
      Serial.println("SD      | *** MOUNTED BUT CANNOT OPEN A FILE FOR WRITING ***");
    } else {
      f.println(PAYLOAD);
      f.close();

      f = SD.open(TEST_PATH, FILE_READ);
      if (!f) {
        Serial.println("SD      | *** WROTE THE FILE BUT CANNOT READ IT BACK ***");
      } else {
        String back = f.readStringUntil('\n');
        f.close();
        back.trim();

        if (back == PAYLOAD) {
          Serial.println("SD      | *** PASS — WRITE AND READ-BACK VERIFIED ***");
          Serial.println("SD      | *** THE MODULE AND CARD ARE FULLY WORKING ***");
        } else {
          Serial.print("SD      | *** DATA CORRUPTED. wrote=\""); Serial.print(PAYLOAD);
          Serial.print("\" read=\""); Serial.print(back); Serial.println("\" ***");
        }
      }
      SD.remove(TEST_PATH);   // leave the card as we found it
    }
  } else {
    Serial.println("SD      | FAILED — see the probe lines above for why.");
  }

  Serial.println("################ END BENCH TEST ###############\n");
}

// Drives CS/MOSI/SCK all HIGH, then all LOW, holding each state long enough to
// measure. Probe the SAME pin at three points and see where the voltage dies:
//   1. the ESP32 GPIO      -> is the chip driving it?
//   2. the module's header -> did the wire carry it?  (continuity already says yes)
//   3. the 74LVC125 output -> did the buffer pass it?  <-- the chip under suspicion
void staticPinTest() {
  const int pins[3]  = {SD_CS, SD_MOSI, SD_SCK};
  const char *names[3] = {"CS  (GPIO 10)", "MOSI(GPIO 11)", "SCK (GPIO 12)"};

  for (int i = 0; i < 3; i++) pinMode(pins[i], OUTPUT);

  for (int level = 1; level >= 0; level--) {
    for (int i = 0; i < 3; i++) digitalWrite(pins[i], level);

    Serial.println("\n--------------------------------------------------");
    Serial.print("HOLDING CS, MOSI and SCK ");
    Serial.print(level ? "HIGH (expect ~3.3 V)" : "LOW  (expect ~0.0 V)");
    Serial.println(" for 6 seconds");
    for (int i = 0; i < 3; i++) {
      Serial.print("   "); Serial.print(names[i]);
      Serial.println(level ? "  -> should read ~3.3 V" : "  -> should read ~0.0 V");
    }
    Serial.println("   MEASURE NOW (black probe on GND)");
    Serial.println("--------------------------------------------------");
    delay(6000);
  }
}

// Sends known bytes over SPI with MOSI(11) jumpered to MISO(14). A working
// peripheral reads back exactly what it sent. This isolates the ESP32 SPI engine
// and those two pins from the module, the card, and everything else.
// Bit-bang loopback: drive GPIO 11 HIGH/LOW as a PLAIN output and read GPIO 14
// as a PLAIN input. No SPI peripheral involved. This separates "is the pin/wire
// good?" from "is the SPI engine routing correctly?" -- if this passes but the
// SPI loopback fails, the pins are fine and it's an SPI-config problem.
// Back on the real MOSI pin (11) now that we know the wiring can be reseated
// reliably. Jumper GPIO 11 -> GPIO 14.
#define TEST_OUT SD_MOSI
#define TEST_IN  SD_MISO

// Hold GPIO 11 steadily HIGH so the multimeter can read the pin directly.
void gpioLoopbackTest() {
  static bool once = false;
  pinMode(TEST_OUT, OUTPUT);
  digitalWrite(TEST_OUT, HIGH);
  if (!once) {
    once = true;
    Serial.println("GPIO-LB  | GPIO 11 held HIGH. Probe the GPIO 11 pin directly (black on GND):");
    Serial.println("GPIO-LB  |   ~3.3 V -> GPIO 11 output is fine (was a seating fluke)");
    Serial.println("GPIO-LB  |   ~0 V   -> GPIO 11 output is DEAD; we remap MOSI to GPIO 4");
  }
  delay(2000);
}

void loopbackTest() {
  static bool inited = false;
  if (!inited) {
    inited = true;
    Serial.println("\nLOOPBACK | Jumper GPIO 11 (MOSI) -> GPIO 14 (MISO). Module OFF.");
  }

  gpioLoopbackTest();   // plain-GPIO check first

  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);   // then the SPI-peripheral check

  const uint8_t tests[] = {0xA5, 0x5A, 0x00, 0xFF, 0x3C};
  int pass = 0;
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
  Serial.println("LOOPBACK | ---------------------------");
  for (uint8_t i = 0; i < sizeof(tests); i++) {
    uint8_t got = SPI.transfer(tests[i]);
    bool ok = (got == tests[i]);
    if (ok) pass++;
    Serial.print("LOOPBACK | sent 0x"); if (tests[i]<0x10) Serial.print('0');
    Serial.print(tests[i], HEX); Serial.print("  got 0x");
    if (got<0x10) Serial.print('0'); Serial.print(got, HEX);
    Serial.println(ok ? "  OK" : "  <-- MISMATCH");
  }
  SPI.endTransaction();

  if (pass == sizeof(tests))
    Serial.println("LOOPBACK | *** ALL PASS — ESP32 SPI + MOSI/MISO pins are GOOD ***");
  else if (pass == 0)
    Serial.println("LOOPBACK | *** ALL FAIL — no jumper, or a dead pin/SPI ***");
  else
    Serial.println("LOOPBACK | *** PARTIAL — intermittent connection ***");
  delay(3000);
}

// Finds the INA219 on I2C, then streams voltage/current once a second. The chip
// was hanging the bus before -- so every pass starts with a fresh bus scan, and
// a 50ms Wire timeout keeps a stuck SDA from freezing the whole node.
void ina219Test() {
  static bool ready = false;

  if (!ready) {
    // Pre-flight with PLAIN GPIO reads before the Wire peripheral touches the
    // pins. An I2C bus only works if both lines idle HIGH through the pull-ups;
    // a line held LOW wedges Wire inside its first transaction and freezes the
    // whole node -- which is exactly how the INA219 hung the bus originally.
    Wire.end();
    pinMode(I2C_SDA, INPUT_PULLUP);
    pinMode(I2C_SCL, INPUT_PULLUP);
    delay(2);
    bool sdaHigh = digitalRead(I2C_SDA);
    bool sclHigh = digitalRead(I2C_SCL);
    Serial.printf("INA219  | line check: SDA(GPIO%d)=%s  SCL(GPIO%d)=%s\n",
                  I2C_SDA, sdaHigh ? "HIGH ok" : "STUCK LOW",
                  I2C_SCL, sclHigh ? "HIGH ok" : "STUCK LOW");

    // Power check: the breakout's own 10k pull-ups to VCC only pull when the
    // module HAS power. Fight each line low with the internal pulldown -- a
    // powered module's 10k wins and the line stays HIGH; a dead or unplugged
    // module loses and the line drops LOW. (Same trick that found the SD fault.)
    pinMode(I2C_SDA, INPUT_PULLDOWN);
    pinMode(I2C_SCL, INPUT_PULLDOWN);
    delay(2);
    bool sdaDriven = digitalRead(I2C_SDA);
    bool sclDriven = digitalRead(I2C_SCL);
    pinMode(I2C_SDA, INPUT_PULLUP);
    pinMode(I2C_SCL, INPUT_PULLUP);
    Serial.printf("INA219  | pull-up check: SDA=%s  SCL=%s\n",
                  sdaDriven ? "module pull-up present" : "NO PULL-UP (wire loose or module unpowered)",
                  sclDriven ? "module pull-up present" : "NO PULL-UP (wire loose or module unpowered)");

    // Module is confirmed powered but GPIO 8/9 don't feel its pull-ups -- so
    // find where the SDA/SCL wires actually landed. Sweep every free pin with
    // an internal pulldown: the module's 10k pull-ups win on whichever pins the
    // wires are really plugged into. (GPIO 2 = DHT, 10-14 = SD, both have their
    // own pull-ups -- they're expected to show up; ignore them.)
    if (!sdaDriven || !sclDriven) {
      const int cand[] = {4, 5, 6, 7, 8, 9, 13, 15, 16, 21,
                          35, 36, 37, 38, 39, 40, 41, 42, 47, 48};
      Serial.print("INA219  | hunting for the I2C wires -- driven-high free pins:");
      int found = 0;
      for (unsigned i = 0; i < sizeof(cand) / sizeof(cand[0]); i++) {
        pinMode(cand[i], INPUT_PULLDOWN);
        delay(3);
        if (digitalRead(cand[i])) { Serial.printf(" GPIO%d", cand[i]); found++; }
        pinMode(cand[i], INPUT);
      }
      if (!found) Serial.print(" NONE -- wires not reaching any pin (broken wire or wrong module holes)");
      Serial.println();
    }

    if (!sdaHigh) {
      // Classic recovery: a slave stuck mid-transfer releases SDA after seeing
      // clock pulses. Bit-bang 9 clocks on SCL, then a STOP.
      Serial.println("INA219  | SDA stuck -> trying 9-clock bus recovery...");
      pinMode(I2C_SCL, OUTPUT);
      for (int i = 0; i < 9; i++) {
        digitalWrite(I2C_SCL, LOW);  delayMicroseconds(50);
        digitalWrite(I2C_SCL, HIGH); delayMicroseconds(50);
      }
      pinMode(I2C_SDA, OUTPUT);      // STOP condition: SDA low->high while SCL high
      digitalWrite(I2C_SDA, LOW);  delayMicroseconds(50);
      digitalWrite(I2C_SDA, HIGH); delayMicroseconds(50);
      pinMode(I2C_SDA, INPUT_PULLUP);
      pinMode(I2C_SCL, INPUT_PULLUP);
      delay(2);
      sdaHigh = digitalRead(I2C_SDA);
      Serial.printf("INA219  | after recovery: SDA=%s\n", sdaHigh ? "HIGH ok" : "STILL STUCK");
    }

    if (!sdaHigh || !sclHigh) {
      Serial.println("INA219  | bus unusable. Causes in order of likelihood:");
      Serial.println("INA219  |   1. SDA/SCL wires swapped at one end");
      Serial.println("INA219  |   2. a wire in the wrong hole (shorted to GND)");
      Serial.println("INA219  |   3. INA219 has no VCC (unpowered chip clamps the line)");
      Serial.println("INA219  |   4. damaged INA219 holding the line");
      delay(1000);
      return;   // rescan next pass -- reseat wires and watch this change
    }

    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(10000);       // slow, breadboard-tolerant, same as RTC/LCD
    Wire.setTimeOut(50);        // a wedged bus errors out instead of hanging

    Serial.println("INA219  | scanning the I2C bus:");
    int found = 0;
    uint8_t inaAddr = 0;
    for (uint8_t addr = 1; addr < 127; addr++) {
      Wire.beginTransmission(addr);
      if (Wire.endTransmission() == 0) {
        Serial.print("INA219  |   device at 0x");
        if (addr < 16) Serial.print('0');
        Serial.print(addr, HEX);
        if (addr >= 0x40 && addr <= 0x4F) { Serial.print("  <-- INA219 range"); inaAddr = addr; }
        else if (addr == 0x68) Serial.print("  (RTC)");
        else if (addr == 0x27 || addr == 0x3F) Serial.print("  (LCD)");
        Serial.println();
        found++;
      }
    }
    if (found == 0) {
      Serial.println("INA219  |   NOTHING on the bus -> SDA/SCL swapped, loose, or no power");
      delay(1000);
      return;   // keep rescanning until it shows up
    }
    if (inaAddr == 0) {
      Serial.println("INA219  |   bus is alive but no device in 0x40-0x4F -> INA219 not answering");
      delay(1000);
      return;
    }

    // Default constructor expects 0x40. If A0/A1 solder jumpers moved it, say so.
    if (inaAddr != 0x40) {
      Serial.print("INA219  | found at 0x");
      Serial.print(inaAddr, HEX);
      Serial.println(" not 0x40 -- A0/A1 jumpers are bridged; update the constructor");
    }

    if (!ina219.begin()) {
      Serial.println("INA219  | begin() FAILED even though the address answers");
      delay(1000);
      return;
    }
    ina219.setCalibration_32V_2A();
    ready = true;
    Serial.println("INA219  | initialized! streaming readings...");
  }

  float busV    = ina219.getBusVoltage_V();
  float shuntmV = ina219.getShuntVoltage_mV();
  float mA      = ina219.getCurrent_mA();
  float mW      = ina219.getPower_mW();

  Serial.printf("INA219  | bus=%.3fV shunt=%.2fmV load=%.3fV current=%.1fmA power=%.1fmW\n",
                busV, shuntmV, busV + shuntmV / 1000.0f, mA, mW);
  // With nothing on VIN+/VIN- these read ~0 -- that still proves I2C works.
  delay(1000);
}

// Finds the DS1307 on I2C, sets it to build time the first time (or if the clock
// was never running / lost its battery), then prints the time once a second.
void rtcTest() {
  static bool ready = false;

  if (!ready) {
    Wire.begin(I2C_SDA, I2C_SCL);
    // 10 kHz -- the slowest practical I2C rate, maximally tolerant of long,
    // loosely-seated breadboard wiring. Reads were timing out at 50 kHz.
    Wire.setClock(10000);
    if (!rtc.begin()) {
      Serial.println("RTC     | DS1307 NOT FOUND at 0x68 -- scanning the whole I2C bus:");
      int found = 0;
      for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
          Serial.print("RTC     |   device responds at 0x");
          if (addr < 16) Serial.print('0');
          Serial.println(addr, HEX);
          found++;
        }
      }
      if (found == 0)
        Serial.println("RTC     |   NOTHING on the bus -> SDA/SCL not connected, swapped, or no power");
      delay(1500);
      return;   // keep retrying each loop until it appears
    }
    ready = true;
    Serial.println("RTC     | DS1307 found!");

    if (!rtc.isrunning()) {
      // Clock halted -> first power-up with a fresh/empty battery. Seed it with
      // the time this firmware was compiled (good to a few seconds).
      Serial.println("RTC     | clock was NOT running -> setting it to build time");
      rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    } else {
      Serial.println("RTC     | clock already running (battery kept the time)");
    }
  }

  DateTime now = rtc.now();
  char buf[24];
  snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d",
           now.year(), now.month(), now.day(),
           now.hour(), now.minute(), now.second());
  Serial.print("RTC     | "); Serial.println(buf);
  delay(1000);
}

// Brings up the LCD and the RTC on the shared I2C bus, then shows the live clock
// on the screen (and mirrors it to serial). Proves the display works and that it
// coexists with the RTC on the same two wires.
void lcdTest() {
  static bool ready = false;

  if (!ready) {
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(10000);   // same slow, wiring-tolerant rate the RTC needs

    // hd44780 returns 0 on success; it auto-detects the backpack address.
    int status = lcd.begin(16, 2);
    if (status) {
      Serial.print("LCD     | init FAILED (code ");
      Serial.print(status);
      Serial.println(", -4 = no device found). Scanning the I2C bus:");
      int found = 0;
      for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
          Serial.print("LCD     |   device at 0x");
          if (addr < 16) Serial.print('0');
          Serial.print(addr, HEX);
          if (addr == 0x68) Serial.print("  (the RTC)");
          else if (addr == 0x50) Serial.print("  (RTC's EEPROM)");
          else if (addr == 0x27 || addr == 0x3F) Serial.print("  <-- the LCD!");
          Serial.println();
          found++;
        }
      }
      if (found == 0)
        Serial.println("LCD     |   NOTHING on the bus -> wiring/power problem");
      delay(1500);
      return;
    }
    Serial.println("LCD     | display found and initialized!");

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Hello World");
    Serial.println("LCD     | showing: Hello World");
    ready = true;
  }

  delay(1000);
}

// One-shot: format the card via the Arduino SD library (format_if_empty=true),
// whose card init is reliable on this hardware. f_mkfs on a large card can take
// 30-60s -- SD.begin BLOCKS the whole time, so be patient and DON'T touch wires.
void sdFormatViaIDF() {
  static bool done = false;
  if (done) { delay(2000); return; }
  done = true;

  Serial.println("\n############ SD FORMAT (Arduino SD) ############");
  Serial.println("SDFMT   | Formatting to FAT32. Can take 30-60s. Do NOT touch wires.");

  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  pinMode(SD_MISO, INPUT_PULLUP);
  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);
  delay(10);

  // 6th arg = format_if_empty. On an unformatted card this triggers f_mkfs (FAT32)
  // then remounts. Blocks here until the format finishes.
  if (!SD.begin(SD_CS, SPI, 200000, "/sd", 5, true)) {
    Serial.println("SDFMT   | *** mount/format FAILED -- reseat card, keep still, reset ***");
    Serial.println("############ END ############");
    return;
  }

  Serial.println("SDFMT   | *** CARD MOUNTED (formatted if it was blank) ***");
  Serial.printf("SDFMT   | card size: %lluMB\n", SD.cardSize() / (1024ULL * 1024ULL));

  File f = SD.open("/aqi_selftest.txt", FILE_WRITE);
  if (!f) {
    Serial.println("SDFMT   | mounted but CANNOT open a file for writing");
  } else {
    f.println("tinyml-aqi-node format selftest");
    f.close();
    f = SD.open("/aqi_selftest.txt", FILE_READ);
    String back = f ? f.readStringUntil('\n') : String();
    if (f) f.close();
    back.trim();
    if (back == "tinyml-aqi-node format selftest")
      Serial.println("SDFMT   | *** PASS -- WRITE + READ-BACK VERIFIED. CARD IS READY. ***");
    else
      Serial.println("SDFMT   | wrote but read-back mismatch");
    SD.remove("/aqi_selftest.txt");
  }
  Serial.println("SDFMT   | done. Set SD_FORMAT_MODE=false and reflash for normal use.");
  Serial.println("############ END ############");
}

void loop() {
  if (SD_FORMAT_MODE) {
    sdFormatViaIDF();
    return;
  }

  if (LCD_TEST_MODE) {
    lcdTest();
    return;
  }

  if (RTC_TEST_MODE) {
    rtcTest();
    return;
  }

  if (INA219_TEST_MODE) {
    ina219Test();
    return;
  }

  if (LOOPBACK_TEST) {
    loopbackTest();
    return;
  }

  if (STATIC_PIN_TEST) {
    staticPinTest();
    return;
  }

  if (SD_TEST_MODE) {
    sdBenchTest();
    delay(3000);
    return;
  }

  // TASK 1: Constant PMS5003 Buffer Management
  if (Serial1.available() >= 32) {
    if (Serial1.read() == 0x42 && Serial1.peek() == 0x4D) {
      Serial1.read(); // Consume 0x4D
      uint8_t buffer[30];
      Serial1.readBytes(buffer, 30);
      lastPm1  = (buffer[8]  << 8) | buffer[9];
      lastPm25 = (buffer[10] << 8) | buffer[11];
      lastPm10 = (buffer[12] << 8) | buffer[13];
      Serial.print("[PMS] PM1.0: "); Serial.print(lastPm1);
      Serial.print(" | PM2.5: ");    Serial.print(lastPm25);
      Serial.print(" | PM10: ");     Serial.print(lastPm10);
      Serial.println(" ug/m3");
    }
  }

  // TASK 2: Scheduled Data Reporting (Power & Climate)
  if (millis() - lastLog >= logInterval) {
    lastLog = millis();

    // --- INA219 readings disabled while debugging its hardware ---
    // float volts = ina219.getBusVoltage_V();
    // delay(10);
    // float mAmps = ina219.getCurrent_mA();
    // delay(10);
    // float shunt_mV = ina219.getShuntVoltage_mV();
    // float loadV = volts + (shunt_mV / 1000);

    float h = dht.readHumidity();
    float t = dht.readTemperature();

    Serial.println("\n========================================");

    // --- INA219 power output disabled while debugging its hardware ---
    // if (isnan(mAmps)) {
    //   Serial.println("POWER   | INA219 Reading Error (Check I2C)");
    // } else {
    //   Serial.print("POWER   | "); Serial.print(loadV); Serial.print(" V | ");
    //   Serial.print(mAmps); Serial.println(" mA");
    // }

    if (isnan(h) || isnan(t)) {
      Serial.println("CLIMATE | DHT22 Error");
    } else {
      Serial.print("CLIMATE | Temp: "); Serial.print(t, 1); Serial.print("C | ");
      Serial.print("Hum: "); Serial.print(h, 1); Serial.println("%");
    }

    // Log every interval. Missing PMS frames become empty cells rather than
    // blocking the row — lets climate logging run before the PMS is wired.
    if (sdReady) {
      logRow(lastPm1, lastPm25, lastPm10, t, h);
      Serial.println("SD      | Row written");
    } else {
      Serial.println("SD      | Not logging (card unavailable)");
    }

    Serial.println("========================================\n");
  }
}