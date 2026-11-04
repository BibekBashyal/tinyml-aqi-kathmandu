#include "inference.h"
#include <string.h>
#include "config.h"
#include "aqi_model.h"   // aqi_model[], aqi_model_len, AQI_INPUT_MEAN/INV_SCALE: from requantize_int8.py

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

const char* AQI_SHORT[kNumClasses] = {"Good", "Moderate", "USG", "Unhealthy", "V.Unhealthy", "Hazardous"};

static constexpr int    kOpCount      = 4;   // QUANTIZE, FULLY_CONNECTED (x3), SOFTMAX, DEQUANTIZE
static constexpr int    kFcWeights    = 1;   // FULLY_CONNECTED inputs: 0 activations, 1 weights, 2 bias
static constexpr size_t kModelAlign   = 16;  // TFLM reads the flatbuffer with aligned loads

static uint8_t tensor_arena[TFLM_ARENA_BYTES];
static tflite::MicroErrorReporter micro_error;
// Only the ops this graph uses: the full AllOpsResolver added ~240 KB of flash.
static tflite::MicroMutableOpResolver<kOpCount> resolver;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor *in_t = nullptr, *out_t = nullptr;
static uint32_t heap_before = 0, heap_after = 0;

// This TFLM build has no per-channel FULLY_CONNECTED kernel: it would apply the first weight scale
// to every channel and still report success, silently running a different network (code review
// 2026-11-03). Refuse such a model with the real reason instead.
static bool fcWeightsPerTensor(char* err, size_t n) {
  const tflite::SubGraph* graph = model->subgraphs()->Get(0);
  for (const tflite::Operator* op : *graph->operators()) {
    const tflite::OperatorCode* code = model->operator_codes()->Get(op->opcode_index());
    if (tflite::GetBuiltinCode(code) != tflite::BuiltinOperator_FULLY_CONNECTED) continue;
    const tflite::Tensor* w = graph->tensors()->Get(op->inputs()->Get(kFcWeights));
    const tflite::QuantizationParameters* q = w->quantization();
    if (q == nullptr || q->scale() == nullptr || q->scale()->size() != 1) {
      snprintf(err, n, "FC weights per-channel");
      return false;
    }
  }
  return true;
}

static bool tensorIs(const TfLiteTensor* t, int elements) {
  return t->type == kTfLiteFloat32 && t->dims->size == 2 && t->dims->data[0] == 1 &&
         t->dims->data[1] == elements && t->bytes == elements * sizeof(float);
}

bool tflmInit(char* err, size_t n) {
  heap_before = ESP.getFreeHeap();
  if (((uintptr_t)aqi_model % kModelAlign) != 0) { snprintf(err, n, "Model not aligned"); return false; }
  model = tflite::GetModel(aqi_model);   // read in place from flash, no RAM copy
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    snprintf(err, n, "Model schema %lu != %d", (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  if (!fcWeightsPerTensor(err, n)) return false;
  if (resolver.AddQuantize() != kTfLiteOk || resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk || resolver.AddDequantize() != kTfLiteOk) {
    snprintf(err, n, "Op resolver full"); return false;
  }
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TFLM_ARENA_BYTES, &micro_error);
  interpreter = &static_interpreter;
  // fails for arena exhaustion AND for any kernel Prepare() error; TFLM prints which on the console
  if (interpreter->AllocateTensors() != kTfLiteOk) { snprintf(err, n, "AllocateTensors failed"); return false; }
  in_t  = interpreter->input(0);
  out_t = interpreter->output(0);
  if (!tensorIs(in_t, kNumFeatures) || !tensorIs(out_t, kNumClasses)) {
    snprintf(err, n, "Model I/O not float[1,%d]->[1,%d]", kNumFeatures, kNumClasses);
    return false;
  }
  heap_after = ESP.getFreeHeap();
  return true;
}

// Subtract and multiply only: both are correctly rounded IEEE float32 operations on the ESP32-S3
// FPU and in numpy, so the laptop reference reproduces this input bit for bit.
void modelInput(const float raw[kNumFeatures], float u[kNumFeatures]) {
  for (int i = 0; i < kNumFeatures; i++) u[i] = (raw[i] - AQI_INPUT_MEAN[i]) * AQI_INPUT_INV_SCALE[i];
}

int runInference(const float u[kNumFeatures], float probs[kNumClasses]) {
  if (!interpreter) return -1;
  memcpy(in_t->data.f, u, kNumFeatures * sizeof(float));
  if (interpreter->Invoke() != kTfLiteOk) return -1;
  int best = 0;
  for (int i = 0; i < kNumClasses; i++) {
    probs[i] = out_t->data.f[i];
    if (probs[i] > probs[best]) best = i;   // first maximum wins, same as numpy argmax
  }
  return best;
}

int classify(const float raw[kNumFeatures], float probs[kNumClasses], uint32_t* latency_us) {
  float u[kNumFeatures];
  modelInput(raw, u);
  uint32_t t0 = micros();
  int cls = runInference(u, probs);
  if (latency_us) *latency_us = micros() - t0;
  return cls;
}

size_t tflmArenaUsed()  { return interpreter ? interpreter->arena_used_bytes() : 0; }
size_t tflmArenaSize()  { return TFLM_ARENA_BYTES; }
size_t tflmModelBytes() { return aqi_model_len; }
int    tflmHeapDelta()  { return (int)heap_before - (int)heap_after; }

static uint32_t crc32Update(uint32_t crc, const uint8_t* p, size_t n) {   // zlib CRC-32, bitwise
  for (size_t i = 0; i < n; i++) {
    crc ^= p[i];
    for (int b = 0; b < 8; b++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return crc;
}

uint32_t tflmModelCrc32() { return ~crc32Update(0xFFFFFFFFu, aqi_model, aqi_model_len); }

// Everything the F= path applies before the model: input scaling and the humidity correction.
uint32_t tflmInputCrc32() {
  static const float rh[2] = {RH_ALPHA, RH_BETA};
  uint32_t crc = crc32Update(0xFFFFFFFFu, (const uint8_t*)AQI_INPUT_MEAN, sizeof(AQI_INPUT_MEAN));
  crc = crc32Update(crc, (const uint8_t*)AQI_INPUT_INV_SCALE, sizeof(AQI_INPUT_INV_SCALE));
  return ~crc32Update(crc, (const uint8_t*)rh, sizeof(rh));
}
