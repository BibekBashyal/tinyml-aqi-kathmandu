"""
tflite_export.py
================
INT8 export for the ESP32, shared by train_mlp.py and requantize_int8.py.

Two findings from the 2026-11-03 code review shape this module:

1. Per-tensor weights. TF 2.x quantizes FULLY_CONNECTED weights per output channel by default.
   The TensorFlow Lite Micro build the firmware links (TensorFlowLite_ESP32 1.0.0, kernels from
   2020-2022) only implements per-tensor FULLY_CONNECTED: it applies the first weight scale to
   every channel and AllocateTensors() still succeeds, so a per-channel model silently runs as a
   different network on the board. assert_per_tensor_fc() makes such an export fail.

2. Per-feature input range. The model's QUANTIZE op gives all four inputs ONE int8 scale. With
   standard-scaled inputs the shared range is set by the widest feature (humidity, PM10), leaving
   PM2.5 about 3.6 ug/m3 per step at the 12 ug/m3 Good/Moderate boundary, and the INT8 macro-F1
   then swings between 0.92 and 0.98 with the choice of calibration rows. fold_input_range() maps
   each feature's training range onto [-1, 1] and folds the inverse into the first Dense layer,
   so the float network is unchanged while each feature uses the whole int8 grid. Calibrating on
   the full training split then makes the export deterministic (no random calibration subset).

The device computes the model input as (raw - INPUT_MEAN) * INPUT_INV_SCALE in float32 (subtract
and multiply are correctly rounded IEEE operations on both the ESP32-S3 FPU and numpy), so the
laptop reference in code/08_on_device can reproduce the board's input bit for bit.

Author: Bibek Bashyal, Kathmandu University 2026
"""

import contextlib
import io
import json
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

FULLY_CONNECTED = 9   # tflite BuiltinOperator code
WEIGHTS_INPUT = 1     # FULLY_CONNECTED inputs: 0 = activations, 1 = weights, 2 = bias
FIRST_DENSE = "dense_1"


# ── Model surgery ─────────────────────────────────────────────────────────────
def fold_input_range(model: tf.keras.Model, z_train: np.ndarray) -> tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    """Return (folded model, centre, half_range) with folded(u) == model(z) for u = (z - centre) / half_range.

    centre and half_range are per feature, from the training split in standardized units.
    """
    z = z_train.astype(np.float64)
    zmin, zmax = z.min(axis=0), z.max(axis=0)
    centre, half_range = (zmin + zmax) / 2.0, (zmax - zmin) / 2.0
    if np.any(half_range <= 0):
        raise ValueError(f"a feature is constant in the training split: half ranges {half_range}")
    folded = tf.keras.models.clone_model(model)
    folded.set_weights(model.get_weights())
    kernel, bias = [w.astype(np.float64) for w in model.get_layer(FIRST_DENSE).get_weights()]
    # dense(z) = z @ K + b with z = centre + half_range * u  ->  u @ (half_range[:, None] * K) + (b + centre @ K)
    folded.get_layer(FIRST_DENSE).set_weights([(half_range[:, None] * kernel).astype(np.float32),
                                               (bias + centre @ kernel).astype(np.float32)])
    return folded, centre, half_range


def deployment_input_params(mean: np.ndarray, scale: np.ndarray, centre: np.ndarray,
                            half_range: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Constants for u = (raw - INPUT_MEAN) * INPUT_INV_SCALE, as float32 exactly as the firmware stores them.

    z = (raw - mean) / scale and u = (z - centre) / half_range  =>  u = (raw - (mean + centre * scale)) / (scale * half_range)
    """
    input_mean = (mean + centre * scale).astype(np.float32)
    input_inv_scale = (1.0 / (scale * half_range)).astype(np.float32)
    return input_mean, input_inv_scale


def model_input(raw: np.ndarray, input_mean: np.ndarray, input_inv_scale: np.ndarray) -> np.ndarray:
    """The firmware's float32 input arithmetic: (raw - mean) * inv_scale, element-wise in float32."""
    raw32 = np.asarray(raw, dtype=np.float32)
    return ((raw32 - input_mean.astype(np.float32)) * input_inv_scale.astype(np.float32)).astype(np.float32)


# ── Conversion ────────────────────────────────────────────────────────────────
def convert_int8_per_tensor(model: tf.keras.Model, calibration: np.ndarray) -> bytes:
    """Full-integer INT8 model with float32 I/O and per-tensor FC weights, calibrated on every row given."""
    rows = calibration.astype(np.float32)

    def representative_dataset():
        for row in rows:
            yield [row.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    converter._experimental_disable_per_channel_quantization_for_dense_layers = True
    with contextlib.redirect_stdout(io.StringIO()):
        tflite_bytes = converter.convert()
    assert_per_tensor_fc(tflite_bytes)
    return tflite_bytes


def _builtin_code(model: schema.Model, op: schema.Operator) -> int:
    code = model.OperatorCodes(op.OpcodeIndex())
    # schema 3a moved large codes to BuiltinCode(); small ones may sit in the deprecated field
    return max(code.BuiltinCode(), code.DeprecatedBuiltinCode())


def fc_weight_scale_counts(tflite_bytes: bytes) -> list[int]:
    """Number of quantization scales on each FULLY_CONNECTED weight tensor, in graph order."""
    model = schema.Model.GetRootAsModel(tflite_bytes, 0)
    graph = model.Subgraphs(0)
    counts = []
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        if _builtin_code(model, op) != FULLY_CONNECTED:
            continue
        quant = graph.Tensors(op.Inputs(WEIGHTS_INPUT)).Quantization()
        counts.append(quant.ScaleLength() if quant is not None else 0)
    return counts


def assert_per_tensor_fc(tflite_bytes: bytes) -> list[int]:
    """Raise if any FULLY_CONNECTED weight tensor carries more than one scale."""
    counts = fc_weight_scale_counts(tflite_bytes)
    if not counts:
        raise ValueError("no FULLY_CONNECTED ops found: not the AQI MLP")
    if any(c != 1 for c in counts):
        raise ValueError(
            f"FULLY_CONNECTED weights are per-channel (scales per layer: {counts}); the ESP32 "
            "TFLite Micro build only runs per-tensor FC correctly. Convert with "
            "converter._experimental_disable_per_channel_quantization_for_dense_layers = True")
    return counts


# ── C header ──────────────────────────────────────────────────────────────────
HEADER_TEMPLATE = """// aqi_model.h
// Auto-generated by code/04_tinyml_model/{generator} - do not edit by hand.
// Project: TinyML-Based AQI Classification on ESP32
// Author: Bibek Bashyal, Kathmandu University 2026
//
// Source: models/mlp_int8.tflite ({size} bytes), INT8, FULLY_CONNECTED weights quantized
// PER TENSOR (scales per layer: {scales}), calibrated on the full training split.
// Output: 6-class softmax - Good / Moderate / USG / Unhealthy / Very Unhealthy / Hazardous
//
// Model input, in this order: [0] PM2.5 (humidity-corrected)  [1] PM10  [2] temperature  [3] humidity
//   u[i] = (raw[i] - AQI_INPUT_MEAN[i]) * AQI_INPUT_INV_SCALE[i]      (float32)
// This maps each feature's training range onto [-1, 1] (standard scaling from
// models/scaler_params.json folded with a per-feature range), so every feature uses the whole
// int8 input grid. Values outside the training range saturate at the grid edge.
//
// const puts the model in flash (memory-mapped .rodata) instead of DRAM; alignas(16) satisfies
// the aligned flatbuffer loads TensorFlow Lite Micro performs, so no RAM copy is needed.

#ifndef AQI_MODEL_H
#define AQI_MODEL_H

static const float AQI_INPUT_MEAN[4]      = {{{mean}}};
static const float AQI_INPUT_INV_SCALE[4] = {{{inv}}};

alignas(16) static const unsigned char aqi_model[] = {{
{body}
}};
static const unsigned int aqi_model_len = {size};

#endif  // AQI_MODEL_H
"""


def _c_float(v: np.float32) -> str:
    """Shortest decimal that round-trips to this float32, as a C float literal."""
    return np.format_float_scientific(np.float32(v), unique=True, trim="-").replace("e+", "e") + "f"


def write_c_header(tflite_bytes: bytes, header_path: str, generator: str,
                   input_mean: np.ndarray, input_inv_scale: np.ndarray) -> str:
    """Write the model as a 16-byte-aligned const C array plus its input constants."""
    scales = assert_per_tensor_fc(tflite_bytes)
    lines = []
    for start in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[start:start + 12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    text = HEADER_TEMPLATE.format(
        generator=generator, size=len(tflite_bytes), scales=scales, body="\n".join(lines),
        mean=", ".join(_c_float(v) for v in input_mean), inv=", ".join(_c_float(v) for v in input_inv_scale))
    with open(header_path, "w") as f:
        f.write(text)
    return header_path


ARTIFACTS = {
    "tflite": "models/mlp_int8.tflite",
    "folded": "models/mlp_float32_rangefold.keras",
    "params": "models/int8_input_params.json",
    "header": "firmware/aqi_classifier/aqi_model.h",
}


def export_int8_artifacts(root: str, folded: tf.keras.Model, tflite_bytes: bytes, input_mean: np.ndarray,
                          input_inv_scale: np.ndarray, centre: np.ndarray, half_range: np.ndarray,
                          features: list[str], generator: str) -> dict[str, str]:
    """Write the four files that must always change together; returns their absolute paths."""
    paths = {k: os.path.join(root, v) for k, v in ARTIFACTS.items()}
    assert_per_tensor_fc(tflite_bytes)
    with open(paths["tflite"], "wb") as f:
        f.write(tflite_bytes)
    folded.save(paths["folded"])
    with open(paths["params"], "w") as f:
        json.dump({"features": features, "formula": "u = (raw - input_mean) * input_inv_scale, float32",
                   "input_mean": input_mean.tolist(), "input_inv_scale": input_inv_scale.tolist(),
                   "standardized_centre": centre.tolist(), "standardized_half_range": half_range.tolist(),
                   "generator": generator}, f, indent=2)
    write_c_header(tflite_bytes, paths["header"], generator, input_mean, input_inv_scale)
    return paths


def read_c_header_inputs(header_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse AQI_INPUT_MEAN / AQI_INPUT_INV_SCALE back out of aqi_model.h (float32)."""
    text = open(header_path).read()

    def array(name: str) -> np.ndarray:
        # match the declaration only, not the name in the header comment
        found = re.search(rf"\b{name}\[4\]\s*=\s*\{{([^}}]*)\}}", text)
        if found is None:
            raise ValueError(f"{name}[4] = {{...}} not found in {header_path}")
        return np.array([np.float32(float(v.strip().rstrip("f"))) for v in found.group(1).split(",")], dtype=np.float32)

    return array("AQI_INPUT_MEAN"), array("AQI_INPUT_INV_SCALE")
