// Stage 2 firmware: AQI CLASSIFIER (v1.0 target)
// Place as: firmware/aqi_classifier/src/main.cpp   (PlatformIO project)
// Pipeline: sensors -> humidity correction -> standardize -> INT8 TFLM
// inference -> LCD display + SD log. Set BENCHMARK_MODE 1 for the
// latency / RAM / batch-energy measurements, 0 for normal soak operation.
//
// Requires: aqi_model.h (the xxd export from train_mlp.py, in this folder),
// which defines  unsigned char models_mlp_int8_tflite[];
//                unsigned int  models_mlp_int8_tflite_len;
// and the real SCALER_MEAN / SCALER_STD values in config.h.

#define BENCHMARK_MODE 0

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <DHT.h>
#include <RTClib.h>
#include <Adafruit_INA219.h>
#include <hd44780.h>
#include <hd44780ioClass/hd44780_I2Cexp.h>   // PCF8574 backpack; library proven on the bench
#include "config.h"
#include "aqi_model.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

HardwareSerial pmsSerial(1);
DHT dht(DHT_PIN, DHT_TYPE);
RTC_DS1307 rtc;
Adafruit_INA219 ina219;
hd44780_I2Cexp lcd(LCD_ADDR);   // address hardcoded: auto-detect failed on the bench

const char* AQI_SHORT[6] = {"Good", "Moderate", "USG",
                            "Unhealthy", "V.Unhealthy", "Hazardous"};

// ---------------- TFLM globals ----------------
constexpr int kArenaSize = 10 * 1024;          // 10 KB tensor arena
static uint8_t tensor_arena[kArenaSize];
static tflite::MicroErrorReporter micro_error;
static tflite::AllOpsResolver resolver;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor *in_t = nullptr, *out_t = nullptr;
static uint32_t heap_before_model = 0, heap_after_model = 0;
// TFLite Micro reads the flatbuffer with aligned loads; the xxd array has no
// alignment guarantee, so the model is copied into a 16-byte-aligned buffer.
alignas(16) static uint8_t model_buf[8 * 1024];

struct PmsReading { uint16_t pm25; uint16_t pm10; bool ok; };

PmsReading readPMS(uint32_t timeout_ms = 3000) {
  PmsReading r = {0, 0, false};
  uint8_t buf[32];
  uint32_t start = millis();
  while (millis() - start < timeout_ms) {
    if (pmsSerial.available() >= 32) {
      if (pmsSerial.read() != 0x42) continue;
      if (pmsSerial.peek() != 0x4D) continue;
      buf[0] = 0x42; buf[1] = pmsSerial.read();
      for (int i = 2; i < 32; i++) buf[i] = pmsSerial.read();
      uint16_t sum = 0;
      for (int i = 0; i < 30; i++) sum += buf[i];
      if (sum != (uint16_t)((buf[30] << 8) | buf[31])) continue;
      r.pm25 = (buf[12] << 8) | buf[13];
      r.pm10 = (buf[14] << 8) | buf[15];
      r.ok = true;
      return r;
    }
    delay(10);
  }
  return r;
}

// ---------------- Pre-processing ----------------
float humidityCorrect(float pm25_raw, float rh) {
  // x1c = pm25 / (1 + a*(rh/100)^b); alpha=0 -> pass-through until fitted
  return pm25_raw / (1.0f + RH_ALPHA * powf(rh / 100.0f, RH_BETA));
}

void standardize(const float raw[4], float z[4]) {
  for (int i = 0; i < 4; i++) z[i] = (raw[i] - SCALER_MEAN[i]) / SCALER_STD[i];
}

// ---------------- Inference (handles INT8 or float I/O tensors) ----------
int runInference(const float z[4], float* top_score) {
  if (in_t->type == kTfLiteInt8) {
    const float s = in_t->params.scale;
    const int   zp = in_t->params.zero_point;
    for (int i = 0; i < 4; i++) {
      int q = (int)lroundf(z[i] / s) + zp;
      in_t->data.int8[i] = (int8_t)constrain(q, -128, 127);
    }
  } else {
    for (int i = 0; i < 4; i++) in_t->data.f[i] = z[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) return -1;

  int best = 0; float best_s = -1e9f;
  for (int i = 0; i < 6; i++) {
    float sc = (out_t->type == kTfLiteInt8)
      ? (out_t->data.int8[i] - out_t->params.zero_point) * out_t->params.scale
      : out_t->data.f[i];
    if (sc > best_s) { best_s = sc; best = i; }
  }
  if (top_score) *top_score = best_s;
  return best;
}

#if BENCHMARK_MODE
// ---------------- Benchmarks (Chapter VII protocol) ----------------
// Latency: mean +/- std over 100 runs.
// Energy: busy window = exactly N back-to-back inferences; idle window = same
// duration with the CPU waiting (delay) between INA219 reads. The INA219 is
// read at the same rate in both windows, so its own cost cancels.
// Reported per inference: TOTAL = E_busy / N, EXTRA = (P_busy - P_idle) * T / N.
// Wiring: INA219 VIN+/VIN- in series with the board's 5 V input, no power
// drawn from USB during the run, PMS5003 outside the measured line or off.
struct PowerWindow { float energy_mJ; float secs; };
const uint32_t SAMPLE_US = 2000;            // one INA219 reading every 2 ms

float readPower_mW() { return ina219.getBusVoltage_V() * ina219.getCurrent_mA(); }

PowerWindow measureBusy(int n, const float z[4]) {
  float e = 0.0f, p_prev = readPower_mW();
  uint32_t t0 = micros(), t_last = t0;
  for (int i = 0; i < n; i++) {
    runInference(z, nullptr);
    uint32_t now = micros();
    if ((uint32_t)(now - t_last) >= SAMPLE_US) {
      float p = readPower_mW();
      e += 0.5f * (p + p_prev) * ((now - t_last) / 1.0e6f);   // mW * s = mJ
      p_prev = p; t_last = now;
    }
  }
  uint32_t t_end = micros();
  float p = readPower_mW();
  e += 0.5f * (p + p_prev) * ((t_end - t_last) / 1.0e6f);     // close last gap
  PowerWindow w = { e, (t_end - t0) / 1.0e6f };
  return w;
}

PowerWindow measureIdle(float secs) {
  float e = 0.0f, p_prev = readPower_mW();
  uint32_t t0 = micros(), t_last = t0, span = (uint32_t)(secs * 1.0e6f);
  while ((uint32_t)(micros() - t0) < span) {
    delay(SAMPLE_US / 1000);                  // CPU waits, does not spin
    uint32_t now = micros();
    float p = readPower_mW();
    e += 0.5f * (p + p_prev) * ((now - t_last) / 1.0e6f);
    p_prev = p; t_last = now;
  }
  PowerWindow w = { e, (t_last - t0) / 1.0e6f };
  return w;
}

void runBenchmarks() {
  float z[4] = {0.1f, 0.2f, -0.3f, 0.5f};   // arbitrary standardized input

  // --- Latency, 100 runs ---
  float lat[100];
  for (int i = 0; i < 100; i++) {
    uint32_t t0 = micros(); runInference(z, nullptr);
    lat[i] = (micros() - t0) / 1000.0f;
  }
  float mean = 0, var = 0;
  for (int i = 0; i < 100; i++) mean += lat[i];
  mean /= 100;
  for (int i = 0; i < 100; i++) var += (lat[i]-mean)*(lat[i]-mean);
  float sd = sqrtf(var / 99);
  Serial.printf("LATENCY: %.3f +/- %.3f ms (n=100)\n", mean, sd);

  // --- RAM ---
  // The arena is a static array and TFLM never allocates at run time, so the
  // heap delta stays near zero. Report arena use + static RAM (build report).
  Serial.printf("RAM: tensor arena used %u of %u bytes; heap delta %d bytes\n",
                (unsigned)interpreter->arena_used_bytes(), (unsigned)kArenaSize,
                (int)heap_before_model - (int)heap_after_model);

  // --- Energy per inference, batch protocol ---
  const int N = 1000, REPEATS = 10;
  float tot[REPEATS], ext[REPEATS], tot_mean = 0, ext_mean = 0;
  for (int r = 0; r < REPEATS; r++) {
    PowerWindow busy = measureBusy(N, z);
    PowerWindow idle = measureIdle(busy.secs);
    float p_busy = busy.energy_mJ / busy.secs;          // mW
    float p_idle = idle.energy_mJ / idle.secs;          // mW
    tot[r] = busy.energy_mJ / N;                        // mJ per inference
    ext[r] = (p_busy - p_idle) * busy.secs / N;         // mJ per inference
    tot_mean += tot[r]; ext_mean += ext[r];
    Serial.printf("  run %d: %d inferences in %.3f s, busy %.1f mW, idle %.1f mW"
                  " -> total %.5f mJ, extra %.5f mJ per inference\n",
                  r + 1, N, busy.secs, p_busy, p_idle, tot[r], ext[r]);
  }
  tot_mean /= REPEATS; ext_mean /= REPEATS;
  float tot_var = 0, ext_var = 0;
  for (int r = 0; r < REPEATS; r++) {
    tot_var += (tot[r] - tot_mean) * (tot[r] - tot_mean);
    ext_var += (ext[r] - ext_mean) * (ext[r] - ext_mean);
  }
  Serial.printf("ENERGY total: %.5f +/- %.5f mJ per inference (N=%d x %d)\n",
                tot_mean, sqrtf(tot_var / (REPEATS - 1)), N, REPEATS);
  Serial.printf("ENERGY extra: %.5f +/- %.5f mJ per inference (N=%d x %d)\n",
                ext_mean, sqrtf(ext_var / (REPEATS - 1)), N, REPEATS);
}
#endif

void haltWithError(const char* msg) {
  Serial.println(msg);
  lcd.clear(); lcd.print(msg);
  while (true) delay(1000);
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(I2C_CLOCK_HZ);
  pmsSerial.begin(PMS_BAUD, SERIAL_8N1, PMS_RX_PIN, PMS_TX_PIN);
  dht.begin();
  lcd.begin(LCD_COLS, LCD_ROWS); lcd.backlight(); lcd.print("AQI Classifier");

  if (!SCALER_FROM_JSON)
    haltWithError("Paste scaler_params!");   // refuse to run on placeholder scaler

  if (!rtc.begin())    haltWithError("DS1307 not found");
  if (!ina219.begin()) Serial.println("WARN: INA219 not found");
  SPI.begin(SD_SCK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);
  if (!SD.begin(SD_CS_PIN)) haltWithError("SD init failed");
  if (!SD.exists(LOG_FILE)) {
    File f = SD.open(LOG_FILE, FILE_WRITE);
    f.println("timestamp,pm25_raw,pm25_corr,pm10,temp_c,rh_pct,aqi_class,latency_ms");
    f.close();
  }

  // --- TFLM init ---
  heap_before_model = ESP.getFreeHeap();
  if (models_mlp_int8_tflite_len > sizeof(model_buf)) haltWithError("Model too big");
  memcpy(model_buf, models_mlp_int8_tflite, models_mlp_int8_tflite_len);
  model = tflite::GetModel(model_buf);
  if (model->version() != TFLITE_SCHEMA_VERSION) haltWithError("Model schema!");
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize, &micro_error);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) haltWithError("Arena too small");
  in_t = interpreter->input(0);
  out_t = interpreter->output(0);
  heap_after_model = ESP.getFreeHeap();
  Serial.printf("TFLM ready. Arena used: %u bytes\n",
                (unsigned)interpreter->arena_used_bytes());

#if BENCHMARK_MODE
  runBenchmarks();
  lcd.clear(); lcd.print("Benchmarks done");
  while (true) delay(1000);
#endif
}

void loop() {
  uint32_t cycle_start = millis();

  PmsReading pms = readPMS();
  float temp = dht.readTemperature();
  float rh   = dht.readHumidity();

  if (pms.ok && !isnan(temp) && !isnan(rh)) {
    float pm25c = humidityCorrect((float)pms.pm25, rh);
    float raw[4] = {pm25c, (float)pms.pm10, temp, rh};
    float z[4];
    standardize(raw, z);

    uint32_t t0 = micros();
    float score = 0;
    int cls = runInference(z, &score);
    float lat_ms = (micros() - t0) / 1000.0f;

    DateTime now = rtc.now();
    char ts[20];
    snprintf(ts, sizeof(ts), "%04d-%02d-%02d %02d:%02d:%02d",
             now.year(), now.month(), now.day(),
             now.hour(), now.minute(), now.second());

    if (cls >= 0) {
      lcd.clear();
      lcd.setCursor(0, 0); lcd.printf("PM2.5:%4.0f ug/m3", pm25c);
      lcd.setCursor(0, 1); lcd.print(AQI_SHORT[cls]);
      File f = SD.open(LOG_FILE, FILE_APPEND);
      if (f) {
        f.printf("%s,%u,%.1f,%u,%.1f,%.1f,%d,%.2f\n",
                 ts, pms.pm25, pm25c, pms.pm10, temp, rh, cls, lat_ms);
        f.close();
      }
      Serial.printf("%s  class=%s lat=%.2f ms\n", ts, AQI_SHORT[cls], lat_ms);
    } else {
      Serial.println("Inference failed");
    }
  } else {
    lcd.clear(); lcd.print("Sensor error");
  }

  uint32_t elapsed = millis() - cycle_start;
  if (elapsed < SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS - elapsed);
}
