# ESP32-S3 Firmware Setup Notes

## >>> WHERE WE ARE (pick up here after soldering) <<<
Last session: validated every peripheral individually, then paused to SOLDER
(breadboard/dupont connections were too flaky/noisy to be reliable — that was the
single root cause of almost every problem, across SD, RTC, and LCD).

| Peripheral | Status | Notes |
|---|---|---|
| PMS5003 (PM1/2.5/10) | ✅ WORKING | UART, GPIO 17/18 |
| DHT22 (temp/hum) | ✅ WORKING | GPIO 2, 3.3V |
| DS1307 RTC | ✅ WORKING | I2C 0x68, needs 10 kHz + 5V (see its section) |
| 16x2 I2C LCD | ✅ WORKING | I2C 0x27, **needs 5V panel** (see its section) |
| microSD | ⚠️ PENDING SOLDER | everything proven good; only flaky wiring blocks it |
| INA219 | ⏸️ disabled | never ACKed; debug in isolation later |

**NEXT AFTER SOLDERING:** write the integrated firmware — read all sensors, show
paged data (PM / temp+hum / time) on the LCD, timestamp with the RTC, log to SD.
The individual test modes are all in `src/main.cpp` behind `#define ..._TEST_MODE`
flags. Set them all `false` for the normal node.

Test-mode flags currently in main.cpp (only ONE should be true at a time):
`SD_TEST_MODE`, `STATIC_PIN_TEST`, `LOOPBACK_TEST`, `RTC_TEST_MODE`, `LCD_TEST_MODE`.

## Board — NOTE: TWO different boards were used this session
- **Original board:** ESP32-S3-DevKitC-1 **N8** — 8MB QD flash, **NO PSRAM**.
  On this one GPIO 33-37 are free.
- **Current board (swapped in mid-session):** ESP32-S3 with **8MB embedded OCTAL
  PSRAM** (16MB flash). `esptool flash_id` -> "Features: ... Embedded PSRAM 8MB".
  **On THIS board GPIO 33-37 are consumed by the octal PSRAM and CANNOT be used.**
  This is why SCK=35 / MISO=37 failed on it. Its USB serial id is `5B8F093666`;
  the original was `5B5E083401`.
- **Lesson: check `esptool flash_id` for PSRAM before picking GPIOs.** If PSRAM is
  present, avoid 33-37. GPIO 10-14 are safe on BOTH boards (never flash/PSRAM).
- MAC (original): 90:70:69:09:04:50

### !!! THE TWO USB PORTS — READ THIS FIRST !!!
The DevKitC-1 has **two USB-C ports** and they are NOT interchangeable:

| Port | Chip | Device | What `Serial` needs |
|---|---|---|---|
| **UART** (what we use) | CH343 bridge, usb id `1a86:55d3` | `/dev/ttyACM0` | `ARDUINO_USB_CDC_ON_BOOT=0` |
| **USB** (native) | ESP32-S3 itself | separate `/dev/ttyACM*` | `ARDUINO_USB_CDC_ON_BOOT=1` |

We are on the **UART port**, so `platformio.ini` sets `CDC_ON_BOOT=0`.

**This mismatch burned hours.** With `CDC_ON_BOOT=1` while cabled to the UART port,
every `Serial.print()` was sent out the *native* port, which had no cable — so it
vanished. Meanwhile the ROM boot banner and `ESP_LOG`/`log_e()` output use UART0
and DID arrive. The result looked exactly like a hung board or a broken sensor:
boot banner appears, then total silence, while the code was actually running fine.

**Symptom -> check this first:** ROM banner arrives but no `Serial.print()` output
=> you are on the wrong USB port for your `CDC_ON_BOOT` setting.

Confirm which port you're on:
```bash
udevadm info -q property -n /dev/ttyACM0 | grep ID_VENDOR_ID
# 1a86 = CH343 bridge = UART port    -> CDC_ON_BOOT=0
# 303a = Espressif    = native USB   -> CDC_ON_BOOT=1
```

### Reserved / unusable GPIOs
- **GPIO 33-37: OCTAL PSRAM — unusable ON THE CURRENT (PSRAM) BOARD.** Free on the
  original N8 board. When in doubt, don't use them. (See the board note above.)
- GPIO 26-32: SPI flash
- GPIO 19/20: native USB D-/D+
- GPIO 0, 3, 45, 46: strapping pins
- **GPIO 10-14 are safe on both boards** (never flash/PSRAM) — use these for SPI/SD.
- **ESP32-S3 GPIO is NOT 5V-tolerant.** Signal pins are 3.3V. The exception in this
  build is the shared I2C (LCD forces the bus to 5V) — see the LCD 5V note; a level
  shifter is the proper fix.

## Environment
- Using PlatformIO in VS Code (Ubuntu)
- `pio` CLI lives at `~/.platformio/penv/bin/pio` (not on PATH in a plain shell)
- Serial device enumerates as `/dev/ttyACM0` (native USB CDC)
- Arduino IDE 2.3.10 AppImage also available as fallback
  - Needs: sudo apt install libfuse2
  - Needs: sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0

## platformio.ini (current)
```ini
[env:esp32-s3-devkitc-1]
platform = espressif32
board = esp32-s3-devkitc-1
framework = arduino
monitor_speed = 115200
; cabled to the UART (CH343) port -> Serial must be UART0, so CDC_ON_BOOT=0
build_flags =
    -DARDUINO_USB_CDC_ON_BOOT=0
; ports pinned by hardware id so ACM0/ACM1 renumbering can't break uploads
upload_port  = /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8F093666-if00
monitor_port = /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8F093666-if00
lib_deps =
    adafruit/Adafruit INA219
    adafruit/DHT sensor library
    adafruit/Adafruit Unified Sensor
    adafruit/RTClib
    duinowitchery/hd44780
```
NOTE: `upload_port`/`monitor_port` are pinned to the CURRENT board's serial id
(`5B8F...`). If you go back to the original board, update these to `5B5E083401`
(or just delete both lines to auto-detect).

## Pin map — FULL INTEGRATED NODE (what to solder to)
| Peripheral | Signal -> ESP32-S3 | Power | Notes |
|---|---|---|---|
| DHT22 | DATA -> GPIO 2 | 3.3V | temp/humidity |
| PMS5003 | sensor TX -> GPIO 18, sensor RX -> GPIO 17 | 5V | Serial1 @ 9600, crossover |
| DS1307 RTC | SDA -> GPIO 8, SCL -> GPIO 9 | 5V | I2C 0x68, shares bus w/ LCD |
| 16x2 LCD | SDA -> GPIO 8, SCL -> GPIO 9 | 5V | I2C 0x27, shares bus w/ RTC |
| microSD | CS 10, MOSI 11, SCK 12, MISO 14 | 3.3V | bare module; PENDING solder |

- **I2C bus (RTC + LCD) runs at 5V** because the LCD panel needs 5V. This puts 5V on
  the ESP's SDA/SCL (out of spec, accepted for now). **Proper fix: I2C level shifter**
  (5V side: LCD+RTC; 3.3V side: ESP). A shifter also re-enables the RTC's battery.
- Power rails: **5V** = PMS5003, RTC, LCD. **3.3V** = DHT22, SD. **GND common to all.**

### microSD wiring
- Card must be **FAT32, <= 32 GB**. exFAT / 64 GB+ will not mount.
- Bare 3.3V module (no regulator/level-shifter): VCC = **3.3V**. (The blue HW-125
  breakout with AMS1117+74LVC125 wants 5V instead — but we settled on the bare one.)
- SD cards spike >100 mA on mount — keep the 5V rail with headroom.
- One card mounted successfully (7695 MB) but couldn't write -> use
  `format_if_empty=true` in `SD.begin(...)` (already set) so the ESP formats it, or
  format FAT32 on a laptop.

### Common ground is mandatory
Any external supply MUST share ground with the ESP32. UART/I2C/SPI all need a
shared reference. Classic symptom of a missing common ground: the PMS5003 fan
spins (it has local power) but no UART data ever arrives.

## Run commands (Ctrl+Shift+P → Command Palette)
- PlatformIO: Build
- PlatformIO: Upload
- PlatformIO: Serial Monitor

## Run commands (terminal, if toolbar/palette not working)
```bash
pio run                    # build
pio run -t upload          # upload
pio device monitor         # serial monitor (115200, Ctrl+C to exit)
pio run -t upload && pio device monitor   # upload + monitor in one step
```
If `pio` is not found, use the full path: `~/.platformio/penv/bin/pio`

## S3 gotchas
- Serial blank but ROM banner shows: wrong USB port for the CDC_ON_BOOT setting
  (see "THE TWO USB PORTS" up top). This is the #1 time-waster.
- Upload "port is busy": the serial monitor holds the port. Close it first.
- On the UART (CH343) port, auto-reset works (`Hard resetting via RTS pin`), so no
  BOOT/RESET dance needed. (The native-USB port with TinyUSB does need it.)
- IntelliSense red squiggles but build passes: stale `.vscode/c_cpp_properties.json`.
  Fix: `pio project init --ide vscode`, then reload VS Code window.

## Data logging
Every 3s the node appends one row to `/aqi_log.csv` on the microSD card:

```
uptime_ms,pm1_0,pm2_5,pm10,temp_c,humidity_pct
```

- Header is written once, on file creation only — appends across reboots do not
  re-insert it, so the CSV stays directly loadable by pandas.
- A failed DHT read leaves the temp/humidity cells **empty**, which pandas parses
  as NaN. Rows with no PMS frame yet are skipped entirely.
- The file is opened/appended/closed per row, so a power cut loses at most one row.
- If the card is missing or fails, `sdReady` goes false and the node keeps running
  and printing to serial — it just stops persisting. It never halts.
- `uptime_ms` is `millis()` since boot, NOT wall-clock. There is no RTC on the node.
  If absolute timestamps are needed for the thesis, add NTP-over-WiFi or a DS3231.

## STATUS
- [x] Build SUCCESS — clean (RAM 6.0%, Flash 10.5%)
- [x] Upload + serial output working
- [x] **PMS5003 (RX 18 / TX 17) — VERIFIED WORKING.** PM1.0 / PM2.5 / PM10 all parsed.
- [x] **DHT22 (GPIO 2) — VERIFIED WORKING.** e.g. Temp 29.6C / Hum 75.7%.
- [~] **microSD (CS 10 / MOSI 11 / SCK 12 / MISO 14) — ROOT CAUSE: FLAKY WIRING,
      not any dead part.** After exhaustive testing, everything works in
      isolation; the connections just will not hold.
      DEFINITIVE evidence (4 consecutive MISO reads, 3s apart, nothing touched):
        STUCK LOW -> FLOATING -> FLOATING -> idle high(0xFF)
      The reading bounces on its own => the contact is making/breaking physically.
      What was PROVEN good, so DO NOT re-debug these:
        - ESP32 GPIO 10/11/12/14 all output/read correctly (multimeter: 3.3V at
          each pin when driven; GPIO4->GPIO14 loopback verified end-to-end)
        - The bare 3.3V module powers correctly (3.3V at its VCC)
        - MISO reads DRIVEN HIGH when the wire actually seats
        - CMD0 hand-built to spec at 400 kHz; SPI engine fine (loopback)
      NOTE: an earlier note here blamed a "dead 74LVC125" -- that was WRONG, it was
      the same flaky wiring. Two modules and two cards all "failed" for this reason.
      FIX (mechanical, not firmware):
        1. Skip the breadboard -- plug module dupont wires straight onto the ESP32
           header pins. Breadboard spring contacts are the likely weak link.
        2. Replace the jumper wires with a fresh set that clicks in.
        3. For the real deployment: SOLDER it. Dupont-on-breadboard will not stay
           reliable for multi-day data collection.
      **Workaround in use meanwhile: `tools/serial_logger.py`** captures the same
      CSV schema over USB, with real UTC timestamps (firmware has no RTC).
- [ ] **INA219 — CODE COMMENTED OUT in main.cpp. Hardware never ACKed, not a code bug.**
      An I2C scanner found NOTHING at any address on both (8,9) and (35,36), which
      rules out a pin/code cause. When miswired it hung the I2C bus hard enough to
      trip `Guru Meditation: Interrupt wdt timeout` and crash-loop the board.
      Remaining suspects: chip powered from Vin+ instead of the VCC header pin;
      VCC not actually at 3.3V; ground not common; SDA/SCL loose or swapped; dead board.
      **Debug it in ISOLATION** — only VCC->3V3, GND->GND, SDA->8, SCL->9, nothing
      else connected — and re-run the scanner looking for 0x40 before reintegrating.
      To re-enable: uncomment the `Wire.begin` + `ina219.begin()` block in setup()
      and the power-read/print block in loop().

## DS1307 RTC (HW-111 "Tiny RTC") — WORKING
- I2C on **SDA=8 / SCL=9**, address **0x68** (onboard AT24C32 EEPROM is at 0x50).
- **MUST slow I2C to 10 kHz** (`Wire.setClock(10000)`). At 100/50 kHz reads time out
  (`i2cWriteReadNonStop Error 263`) on breadboard wiring. Soldered wiring may allow
  faster — retest after soldering, but 10 kHz is safe.
- Time is seeded from build time (`DateTime(F(__DATE__), F(__TIME__))`) on first run
  / when not running. Set the exact time later (NTP-over-WiFi or manual).
- **DS1307 battery gotcha:** the chip DISABLES I2C when VCC < ~1.25 x battery voltage.
  With a 3V cell + VCC=3.3V, threshold ~3.75V > 3.3V -> I2C DEAD. Tested with battery
  REMOVED. **Running the I2C bus at 5V (as we now do for the LCD) FIXES this** — at 5V
  the battery can stay in. DS3231 would be the cleaner, more accurate long-term chip.

## 16x2 I2C LCD (RG1602A / PCF8574 backpack) — WORKING
- I2C on **SDA=8 / SCL=9** (SAME bus as the RTC), address **0x27**.
- Library: `duinowitchery/hd44780`. **Hardcode the address** — `hd44780_I2Cexp lcd(0x27);`
  The no-arg auto-detect returned -4 (ENXIO) here; giving it 0x27 worked.
- **The LCD PANEL needs ~5V.** At 3.3V the PCF8574 answers on I2C and init "succeeds",
  but the glass shows NOTHING (not even black boxes at any contrast) — 3.3V is too
  low to drive the pixels. At 5V the black boxes appear and text shows.
- **Contrast:** small blue pot on the backpack, often MULTI-TURN (10-15 turns). Sweep
  it fully: blank -> text -> solid black boxes. Text sits just before the boxes.
- **5V-on-I2C caveat:** running the LCD at 5V pulls SDA/SCL to 5V (backpack pull-ups),
  which the S3 pins tolerate at only ~0.3 mA through the clamp diodes — accepted for
  now, but a level shifter is the proper fix. This is why RTC+LCD share a 5V bus.

## Hard-won gotchas
- **INA219 has TWO separate sides.** The I2C control header (VCC/GND/SDA/SCL) powers
  the chip. The measurement terminals (Vin+/Vin-) go in *series with the load*.
  Powering the chip from Vin+ is a common and fatal mistake.
- **Serial monitor holds /dev/ttyACM0 exclusively.** Uploads fail with "port is busy"
  until you close the monitor terminal (Ctrl+C, then any key).
- `Guru Meditation Error: Interrupt wdt timeout` here meant a hung I2C bus (held
  SDA/SCL), which also corrupted DHT reads into `0.0C / 0.0%`. Not a software bug.
