// inference.h - the INT8 MLP (TensorFlow Lite Micro) behind one function call.
// Features in training order: PM2.5 (humidity-corrected), PM10, temperature, humidity.
#pragma once
#include <Arduino.h>

static constexpr int kNumFeatures = 4;
static constexpr int kNumClasses  = 6;

extern const char* AQI_SHORT[kNumClasses];   // Good, Moderate, USG, Unhealthy, V.Unhealthy, Hazardous

bool   tflmInit(char* err, size_t n);   // false with the reason in err (the node then halts)
// Raw features -> model input u = (raw - AQI_INPUT_MEAN) * AQI_INPUT_INV_SCALE (aqi_model.h), float32.
void   modelInput(const float raw[kNumFeatures], float u[kNumFeatures]);
// Model input -> class index with the scores in probs; -1 when Invoke() failed.
int    runInference(const float u[kNumFeatures], float probs[kNumClasses]);
// Raw features -> class; latency_us covers runInference only (copy in, Invoke, argmax).
int    classify(const float raw[kNumFeatures], float probs[kNumClasses], uint32_t* latency_us);
size_t tflmArenaUsed();
size_t tflmArenaSize();
size_t tflmModelBytes();
int    tflmHeapDelta();   // free heap before minus after the interpreter set-up
// zlib CRC-32 of the model bytes, and of AQI_INPUT_MEAN, AQI_INPUT_INV_SCALE, RH_ALPHA, RH_BETA (float32 bytes):
// the laptop computes the same with zlib.crc32 to prove the board runs the model on disk.
uint32_t tflmModelCrc32();
uint32_t tflmInputCrc32();
