#include "benchmark.h"
#include <Arduino.h>
#include <math.h>
#include "inference.h"
#include "power_meter.h"

// Protocol (code review 2026-11-03, rounds 1 and 2):
//   Latency: one COLD call first (caches hold whatever ran before, as in a 3 s sensor cycle), then
//            100 WARM calls back to back: mean +/- sd, min, max. The per-cycle latency in the CSV is
//            the deployed (cold) figure; the warm figure is the kernel's best case.
//   Energy:  ENERGY_PAIRS pairs of a BUSY window (inferences back to back, no I2C) and an IDLE
//            window (delay) of WINDOW_MS each. The INA219 runs PGA /2 with 128-sample averaging per
//            ADC (~68 ms; shunt + bus cycle ~136 ms), so the reading taken right after a window covers
//            time inside that window only. A pair is rejected when either sample is implausible
//            (bus outside 4-6 V, current outside 20-800 mA, guard tripped). With t_inf the measured
//            mean warm inference time:
//              TOTAL = P_busy * t_inf          EXTRA = (P_busy - P_idle) * t_inf
//            at the board's 5 V input (includes the 3.3 V regulator's loss). The rail3v3 line gives the
//            same figures for the 3.3 V rail, I * 3.3 V * t_inf, for comparison with chip-level papers.
// Wiring: shunt in series with the board's 5 V input, board powered by the adapter, not USB.
static const int      LAT_RUNS     = 100;
static const uint32_t WINDOW_MS    = 400;    // > 204 ms: the averaged reading lies fully inside it
static const int      ENERGY_PAIRS = 25;
static const int      WARMUP_PAIRS = 1;      // first pair after the ADC mode change is discarded
static const float    MIN_IDLE_MA  = 20.0f;  // an ESP32-S3 at 240 MHz idles well above this
static const float    MAX_NODE_MA  = 800.0f; // PGA /2 full scale
static const float    MIN_BUS_V    = 4.0f;
static const float    MAX_BUS_V    = 6.0f;
static const float    RAIL_V       = 3.3f;
static const float    BENCH_RAW[kNumFeatures] = {40.0f, 50.0f, 21.0f, 75.0f};   // typical Kathmandu bench row

struct EnergyRun {
  int pairs, rejected;
  float pBusy[ENERGY_PAIRS], pIdle[ENERGY_PAIRS], iBusy[ENERGY_PAIRS], iIdle[ENERGY_PAIRS];
  double infUs; uint32_t infCount; float busV;
};

static void meanSd(const float* v, int n, float& mean, float& sd) {
  mean = 0; for (int i = 0; i < n; i++) mean += v[i];
  mean /= n;
  float var = 0; for (int i = 0; i < n; i++) var += (v[i] - mean) * (v[i] - mean);
  sd = n > 1 ? sqrtf(var / (n - 1)) : 0.0f;
}

static float median(const float* v, int n) {
  float s[ENERGY_PAIRS];
  for (int i = 0; i < n; i++) {   // insertion sort, n <= 25
    float x = v[i]; int j = i - 1;
    while (j >= 0 && s[j] > x) { s[j + 1] = s[j]; j--; }
    s[j + 1] = x;
  }
  return n % 2 ? s[n / 2] : 0.5f * (s[n / 2 - 1] + s[n / 2]);
}

static void benchLatency(const float u[kNumFeatures]) {
  float lat[LAT_RUNS], probs[kNumClasses];
  uint32_t t0 = micros(); runInference(u, probs);
  uint32_t coldUs = micros() - t0;   // printed after the warm loop, so the print cannot evict the caches
  for (int i = 0; i < LAT_RUNS; i++) {
    t0 = micros(); runInference(u, probs);
    lat[i] = (micros() - t0) / 1000.0f;
  }
  float mean, sd, lo = lat[0], hi = lat[0];
  for (int i = 1; i < LAT_RUNS; i++) { lo = min(lo, lat[i]); hi = max(hi, lat[i]); }
  meanSd(lat, LAT_RUNS, mean, sd);
  Serial.printf("LATENCY cold: %.4f ms (first call, caches hold what ran before)\n", coldUs / 1000.0f);
  Serial.printf("LATENCY: %.4f +/- %.4f ms (n=%d, min %.4f, max %.4f)\n", mean, sd, LAT_RUNS, lo, hi);
}

static void benchRam() {
  // The arena is static and TFLM never allocates at run time, so the heap delta stays near zero;
  // static RAM comes from the build report.
  Serial.printf("RAM: tensor arena used %u of %u bytes; model %u bytes in flash; heap delta %d bytes\n",
                (unsigned)tflmArenaUsed(), (unsigned)tflmArenaSize(), (unsigned)tflmModelBytes(), tflmHeapDelta());
}

static bool plausible(const InaSample& s) {
  return !inaShort && s.busV >= MIN_BUS_V && s.busV <= MAX_BUS_V &&
         s.current_mA >= MIN_IDLE_MA && s.current_mA < MAX_NODE_MA;
}

// One BUSY window then one IDLE window; false when either INA219 read failed.
static bool energyPair(const float u[kNumFeatures], InaSample& busy, InaSample& idle, double& infUs, uint32_t& n) {
  float probs[kNumClasses];
  uint32_t t0 = millis();
  while (millis() - t0 < WINDOW_MS) { uint32_t a = micros(); runInference(u, probs); infUs += micros() - a; n++; }
  bool ok = inaSample(busy);
  delay(WINDOW_MS);
  return inaSample(idle) && ok;
}

static void collectEnergy(const float u[kNumFeatures], EnergyRun& r) {
  r.pairs = 0; r.rejected = 0; r.infUs = 0; r.infCount = 0; r.busV = 0;
  for (int k = 0; k < WARMUP_PAIRS + ENERGY_PAIRS; k++) {
    double us = 0; uint32_t n = 0; InaSample busy, idle;
    bool read = energyPair(u, busy, idle, us, n);
    if (k < WARMUP_PAIRS) continue;
    if (!read || !plausible(busy) || !plausible(idle)) { r.rejected++; continue; }
    int i = r.pairs++;
    r.pBusy[i] = busy.busV * busy.current_mA; r.pIdle[i] = idle.busV * idle.current_mA;
    r.iBusy[i] = busy.current_mA;             r.iIdle[i] = idle.current_mA;
    r.infUs += us; r.infCount += n; r.busV += idle.busV;
  }
  if (r.pairs) r.busV /= r.pairs;
}

static void reportEnergy(const EnergyRun& r) {
  if (r.pairs < ENERGY_PAIRS / 2) {
    Serial.printf("ENERGY INVALID: only %d of %d window pairs plausible (%d rejected): board powered by USB, shunt not in series, or bus errors\n",
                  r.pairs, ENERGY_PAIRS, r.rejected);
    return;
  }
  float dP[ENERGY_PAIRS], dI[ENERGY_PAIRS], pb, pbSd, pi, piSd, d, dSd, ib, ibSd, ii, iiSd, di, diSd;
  for (int i = 0; i < r.pairs; i++) { dP[i] = r.pBusy[i] - r.pIdle[i]; dI[i] = r.iBusy[i] - r.iIdle[i]; }
  meanSd(r.pBusy, r.pairs, pb, pbSd); meanSd(r.pIdle, r.pairs, pi, piSd); meanSd(dP, r.pairs, d, dSd);
  meanSd(r.iBusy, r.pairs, ib, ibSd); meanSd(r.iIdle, r.pairs, ii, iiSd); meanSd(dI, r.pairs, di, diSd);
  double tInfS = r.infUs / r.infCount / 1e6;
  double duty = r.infUs / (r.pairs * (double)WINDOW_MS * 1000.0) * 100.0;
  Serial.printf("ENERGY setup: %d pairs x %lu ms windows (%d rejected), INA219 PGA /2 128-sample averaging, %.0f inferences per window, inference duty %.1f %%\n",
                r.pairs, (unsigned long)WINDOW_MS, r.rejected, (double)r.infCount / r.pairs, duty);
  Serial.printf("ENERGY power: busy %.2f +/- %.2f mW, idle %.2f +/- %.2f mW, bus %.3f V, idle current %.1f mA\n", pb, pbSd, pi, piSd, r.busV, ii);
  Serial.printf("ENERGY total: %.6f mJ per inference (P_busy x t_inf, t_inf %.4f ms)\n", pb * tInfS, tInfS * 1000.0);
  Serial.printf("ENERGY extra: %.6f +/- %.6f mJ per inference ((P_busy - P_idle) x t_inf, sd over %d pairs, median %.6f)\n",
                d * tInfS, dSd * tInfS, r.pairs, median(dP, r.pairs) * tInfS);
  Serial.printf("ENERGY rail3v3: total %.6f mJ, extra %.6f mJ per inference (I x 3.3 V x t_inf)\n",
                ib * RAIL_V * tInfS, di * RAIL_V * tInfS);
}

static void benchEnergy(const float u[kNumFeatures]) {
  if (!inaOk) { Serial.println("ENERGY: skipped, INA219 not on the bus"); return; }
  if (!inaSetAveraging(true)) { Serial.println("ENERGY: skipped, INA219 config write failed"); return; }
  static EnergyRun r;   // ~400 B of float arrays: kept off the loop task's stack
  collectEnergy(u, r);
  if (!inaSetAveraging(false)) Serial.println("WARN: INA219 normal config not restored, meter dropped for re-detection");
  reportEnergy(r);
}

void runBenchmarks() {
  Serial.println("BENCH start");
  float u[kNumFeatures];
  modelInput(BENCH_RAW, u);
  benchLatency(u);
  benchRam();
  benchEnergy(u);
  Serial.println("BENCH done");
}
