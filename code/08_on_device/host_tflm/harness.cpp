// Host build of the firmware's inference path against the board's own TensorFlow Lite Micro sources
// (TensorFlowLite_ESP32 1.0.0 from firmware/aqi_classifier/.pio/libdeps), compiled with the same
// arithmetic flags as platformio.ini (-DTFLITE_SINGLE_ROUNDING=1 -ffp-contract=off).
// Mirrors firmware/aqi_classifier/src/inference.cpp: same op resolver, same 10 KB arena, same input
// arithmetic u = (raw - AQI_INPUT_MEAN) * AQI_INPUT_INV_SCALE, model read in place from aqi_model.h.
// Usage: harness <raw float32 N x 4> <out u float32 N x 4> <out probs float32 N x 6>
// Origin: written during the 2026-11-03 code review to prove the rounding-mode finding.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "aqi_model.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

static constexpr int kFeatures = 4, kClasses = 6;
static uint8_t arena[10 * 1024];

int main(int argc, char** argv) {
  if (argc != 4) { fprintf(stderr, "usage: harness raw.bin u.bin probs.bin\n"); return 2; }
  const tflite::Model* model = tflite::GetModel(aqi_model);
  static tflite::MicroErrorReporter reporter;
  static tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddQuantize(); resolver.AddFullyConnected(); resolver.AddSoftmax(); resolver.AddDequantize();
  static tflite::MicroInterpreter interp(model, resolver, arena, sizeof(arena), &reporter);
  if (interp.AllocateTensors() != kTfLiteOk) { fprintf(stderr, "AllocateTensors failed\n"); return 1; }
  TfLiteTensor* in = interp.input(0);
  TfLiteTensor* out = interp.output(0);

  FILE* f = fopen(argv[1], "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  std::vector<float> raw;
  float row[kFeatures];
  while (fread(row, sizeof(float), kFeatures, f) == kFeatures) raw.insert(raw.end(), row, row + kFeatures);
  fclose(f);

  FILE* fu = fopen(argv[2], "wb");
  FILE* fp = fopen(argv[3], "wb");
  size_t n = raw.size() / kFeatures;
  for (size_t r = 0; r < n; r++) {
    float u[kFeatures];
    for (int i = 0; i < kFeatures; i++) u[i] = (raw[r * kFeatures + i] - AQI_INPUT_MEAN[i]) * AQI_INPUT_INV_SCALE[i];
    memcpy(in->data.f, u, sizeof(u));
    if (interp.Invoke() != kTfLiteOk) { fprintf(stderr, "Invoke failed on row %zu\n", r); return 1; }
    fwrite(u, sizeof(float), kFeatures, fu);
    fwrite(out->data.f, sizeof(float), kClasses, fp);
  }
  fclose(fu); fclose(fp);
  printf("rows %zu, arena used %zu\n", n, interp.arena_used_bytes());
  return 0;
}
