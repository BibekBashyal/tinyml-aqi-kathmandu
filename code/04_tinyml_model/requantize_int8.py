"""
requantize_int8.py
==================
Builds the deployed INT8 model from the trained float32 MLP (models/mlp_float32.keras) without
retraining. Why and how: see tflite_export.py. In short, the ESP32's TensorFlow Lite Micro only
runs per-tensor FULLY_CONNECTED correctly, and with one shared int8 input scale the result
depended on which calibration rows were drawn. This script:

  1. folds a per-feature input range into the first Dense layer (float network unchanged),
  2. converts to INT8 with per-tensor weights, calibrated on the FULL training split (deterministic),
  3. evaluates on the test split with the firmware's own float32 input arithmetic.

Outputs:
  models/mlp_int8.tflite                         the deployed model
  models/mlp_int8_perchannel_2026-06-25.tflite   the previous per-channel model, kept for the record
  models/mlp_float32_rangefold.keras             float model with the input range folded in
  models/int8_input_params.json                  input constants (also embedded in aqi_model.h)
  firmware/aqi_classifier/aqi_model.h            model + input constants for the firmware
  results/tinyml_model_comparison.csv            INT8 row updated (float32 rows untouched)
  results/requantize_int8_report.json            before/after numbers
  results/figures/tinyml_model_comparison.png, confusion_matrix_mlp_int8.png

Usage:
    .venv/bin/python code/04_tinyml_model/requantize_int8.py

Author: Bibek Bashyal, Kathmandu University 2026
"""

import json
import logging
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tflite_export import (  # noqa: E402
    convert_int8_per_tensor, deployment_input_params, export_int8_artifacts, fc_weight_scale_counts,
    fold_input_range, model_input,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("requantize")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES = ["PM2.5", "PM10", "temperature", "humidity"]
TARGET = "aqi_class"
NUM_CLASSES = 6
INT8_VARIANT = "MLP INT8 (TFLite, quantized)"
P = {k: os.path.join(ROOT, v) for k, v in {
    "keras": "models/mlp_float32.keras",
    "f32": "models/mlp_float32.tflite",
    "int8": "models/mlp_int8.tflite",
    "int8_old": "models/mlp_int8_perchannel_2026-06-25.tflite",
    "scaler": "models/scaler_params.json",
    "train": "data/processed/train.csv",
    "val": "data/processed/val.csv",
    "test": "data/processed/test.csv",
    "results": "results/tinyml_model_comparison.csv",
    "report": "results/requantize_int8_report.json",
}.items()}


def sklearn_transform(x32: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """StandardScaler.transform on a float32 array, as train_mlp.py applied it: '-= mean' then '/= scale'."""
    centred = (x32.astype(np.float64) - mean).astype(np.float32)
    return (centred.astype(np.float64) / scale).astype(np.float32)


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(P[name])
    return df[FEATURES].to_numpy().astype(np.float32), df[TARGET].to_numpy(dtype=np.int32)


def load_scaler(x_train_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sc = json.load(open(P["scaler"]))
    if sc["features"] != FEATURES:
        raise ValueError(f"scaler feature order {sc['features']} != {FEATURES}")
    mean, scale = np.array(sc["mean"]), np.array(sc["scale"])
    x64 = x_train_raw.astype(np.float64)   # train_mlp.py fits the scaler on float32 values
    if not (np.allclose(x64.mean(axis=0), mean, rtol=1e-6) and np.allclose(x64.std(axis=0), scale, rtol=1e-6)):
        raise ValueError("scaler_params.json does not match the train split: re-run train_mlp.py")
    return mean, scale


def predict(tflite_bytes: bytes, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classes and per-row laptop latency with the TFLite reference kernels.

    The firmware is built with -DTFLITE_SINGLE_ROUNDING=1, which makes the board's TFLM kernels
    compute exactly what TF 2.21's reference kernels compute; code/08_on_device/host_tflm proves it
    by running the board's own TFLM sources on the host (bit-identical on every test row)."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes,
                                 experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]
    preds, lat = np.empty(len(x), dtype=np.int32), np.empty(len(x))
    for i, row in enumerate(x):
        interp.set_tensor(inp, row.reshape(1, -1))
        t0 = time.perf_counter()
        interp.invoke()
        lat[i] = (time.perf_counter() - t0) * 1000.0
        preds[i] = int(np.argmax(interp.get_tensor(out)[0]))
    return preds, lat


def scores(y: np.ndarray, preds: np.ndarray, lat: np.ndarray | None = None) -> dict:
    r = {"accuracy": round(float(accuracy_score(y, preds)), 4),
         "macro_f1": round(float(f1_score(y, preds, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)), 4),
         "per_class_f1": [round(float(v), 4) for v in f1_score(y, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)]}
    if lat is not None:
        r.update(latency_mean_ms=round(float(lat.mean()), 3), latency_std_ms=round(float(lat.std()), 3))
    return r


def update_results_csv(int8: dict, size: int) -> pd.DataFrame:
    df = pd.read_csv(P["results"])
    row = df["variant"] == INT8_VARIANT
    if row.sum() != 1:
        raise ValueError(f"expected one '{INT8_VARIANT}' row in {P['results']}")
    df = df.astype({c: object for c in df.columns})
    df.loc[row, ["accuracy", "macro_f1", "size_bytes", "size_kb", "platform", "latency_mean_ms", "latency_std_ms"]] = [
        int8["accuracy"], int8["macro_f1"], size, round(size / 1024, 2),
        "Laptop (TFLite INT8 per-tensor, reference kernels) → ESP32", int8["latency_mean_ms"], int8["latency_std_ms"]]
    df.to_csv(P["results"], index=False)
    return df


def regenerate_figures(df: pd.DataFrame, y_test: np.ndarray, int8_preds: np.ndarray, f1_drop: float, size_reduction: float) -> None:
    import train_mlp  # module-level code only sets seeds, logging and output folders
    plot_df = df.copy()
    for c in ["macro_f1", "size_kb"]:
        plot_df[c] = plot_df[c].astype(float)
    train_mlp.plot_model_comparison(plot_df, f1_drop, size_reduction)
    train_mlp.plot_confusion_matrix_int8(y_test, int8_preds)


def keep_old_model() -> bytes:
    """The per-channel model from 2026-06-25, preserved once under its own name."""
    if not os.path.exists(P["int8_old"]):
        current = open(P["int8"], "rb").read()
        if max(fc_weight_scale_counts(current)) == 1:
            raise FileNotFoundError(f"{P['int8_old']} missing and models/mlp_int8.tflite is already per-tensor")
        shutil.copyfile(P["int8"], P["int8_old"])
        log.info(f"kept the per-channel model as {os.path.relpath(P['int8_old'], ROOT)}")
    return open(P["int8_old"], "rb").read()


def main() -> None:
    os.chdir(ROOT)   # train_mlp's figure paths are relative to the repo root
    (x_train_raw, _), (x_val_raw, y_val), (x_test_raw, y_test) = load_split("train"), load_split("val"), load_split("test")
    mean, scale = load_scaler(x_train_raw)
    z_train, z_test = sklearn_transform(x_train_raw, mean, scale), sklearn_transform(x_test_raw, mean, scale)
    z_val = sklearn_transform(x_val_raw, mean, scale)

    folded, centre, half_range = fold_input_range(tf.keras.models.load_model(P["keras"]), z_train)
    in_mean, in_inv = deployment_input_params(mean, scale, centre, half_range)
    u_train = model_input(x_train_raw, in_mean, in_inv)
    new_bytes = convert_int8_per_tensor(folded, u_train)
    log.info(f"per-tensor model: {len(new_bytes)} bytes, FC weight scales per layer {fc_weight_scale_counts(new_bytes)}, "
             f"calibrated on {len(u_train):,} training rows")

    old_bytes = keep_old_model()
    f32_bytes = open(P["f32"], "rb").read()
    f32 = scores(y_test, *predict(f32_bytes, z_test))
    f32_val = scores(y_val, predict(f32_bytes, z_val)[0])
    old = scores(y_test, *predict(old_bytes, z_test))
    new_preds, new_lat = predict(new_bytes, model_input(x_test_raw, in_mean, in_inv))
    new = scores(y_test, new_preds, new_lat)
    new_val = scores(y_val, predict(new_bytes, model_input(x_val_raw, in_mean, in_inv))[0])
    for name, r in (("float32 TFLite", f32), ("INT8 per-channel (old)", old), ("INT8 per-tensor (new)", new)):
        log.info(f"  {name:24s} accuracy {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f}  per-class {r['per_class_f1']}")
    log.info(f"  validation split: float32 macro-F1 {f32_val['macro_f1']:.4f}, new INT8 macro-F1 {new_val['macro_f1']:.4f}")

    written = export_int8_artifacts(ROOT, folded, new_bytes, in_mean, in_inv, centre, half_range, FEATURES, "requantize_int8.py")
    log.info("wrote " + ", ".join(os.path.relpath(v, ROOT) for v in written.values()))

    f1_drop = round(f32["macro_f1"] - new["macro_f1"], 4)
    size_reduction = round((1 - len(new_bytes) / os.path.getsize(P["f32"])) * 100, 1)
    df = update_results_csv(new, len(new_bytes))
    regenerate_figures(df, y_test, new_preds, f1_drop, size_reduction)

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": ("ESP32 TFLite Micro runs only per-tensor FULLY_CONNECTED; the per-channel model ran mis-scaled on the "
                   "board. One shared int8 input scale also made INT8 macro-F1 depend on the calibration draw."),
        "method": "per-feature input range folded into dense_1, per-tensor INT8 weights, calibration on the full training split",
        "calibration_rows": int(len(u_train)),
        "float32_tflite": f32,
        "float32_tflite_validation": f32_val,
        "int8_per_channel_old": {"size_bytes": len(old_bytes), "fc_scales": fc_weight_scale_counts(old_bytes), **old},
        "int8_per_tensor_new": {"size_bytes": len(new_bytes), "fc_scales": fc_weight_scale_counts(new_bytes), **new},
        "int8_per_tensor_new_validation": new_val,
        "note_int8_above_float32": ("The INT8 model scores above the float32 model on both the test and the validation split. "
                                    "Quantization shifts the decision boundaries slightly and on this data the shift is "
                                    "favourable. Nothing was tuned for it: calibration uses the whole training split and no "
                                    "choice was made by looking at test or validation scores."),
        "board_arithmetic": ("firmware built with -DTFLITE_SINGLE_ROUNDING=1 -ffp-contract=off; host build of the board's "
                             "TFLM sources (code/08_on_device/host_tflm) gives bit-identical outputs to these numbers"),
        "test_rows": int(len(y_test)),
        "class_agreement_old_vs_new_pct": round(float(np.mean(predict(old_bytes, z_test)[0] == new_preds) * 100), 3),
        "quantization_f1_drop_new": f1_drop,
        "size_reduction_vs_float32_tflite_pct": size_reduction,
    }
    json.dump(report, open(P["report"], "w"), indent=2)
    log.info(f"report -> {os.path.relpath(P['report'], ROOT)}")


if __name__ == "__main__":
    main()
