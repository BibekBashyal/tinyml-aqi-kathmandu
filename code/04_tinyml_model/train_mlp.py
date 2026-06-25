"""
train_mlp.py
============
Complete TinyML pipeline for AQI classification on ESP32:

  Step 1 — Train small MLP neural network (float32)
  Step 2 — Convert to TensorFlow Lite (float32, no quantization)
  Step 3 — Apply post-training INT8 quantization
  Step 4 — Evaluate all three model variants side by side
  Step 5 — Export INT8 model as C byte array for ESP32 flashing

All models saved to: models/
All figures saved to: results/figures/
Results saved to:    results/tinyml_model_comparison.csv

Usage:
    pip install tensorflow --break-system-packages
    python code/04_tinyml_model/train_mlp.py

Author: Bibek Bashyal
Project: TinyML-Based AQI Classification on ESP32
Institution: Kathmandu University, 2026
"""

import os
import json
import time
import logging
import warnings
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES     = ["PM2.5", "PM10", "temperature", "humidity"]
TARGET       = "aqi_class"
NUM_CLASSES  = 6
RANDOM_STATE = 42
EPOCHS       = 100
BATCH_SIZE   = 64
PATIENCE     = 10

AQI_LABELS = [
    "Good", "Moderate", "USG",
    "Unhealthy", "Very\nUnhealthy", "Hazardous"
]

AQI_LABELS_FULL = [
    "Good", "Moderate", "Unhealthy for Sensitive Groups",
    "Unhealthy", "Very Unhealthy", "Hazardous"
]

# Target benchmarks from thesis proposal
TARGETS = {
    "macro_f1":          0.80,
    "quantization_drop": 0.05,
}

PATHS = {
    "train":    "data/processed/train.csv",
    "val":      "data/processed/val.csv",
    "test":     "data/processed/test.csv",
    "models":   "models",
    "results":  "results",
    "figures":  "results/figures",
    "firmware": "firmware/aqi_classifier",
}

for p in ["models", "results", "figures", "firmware"]:
    os.makedirs(PATHS[p], exist_ok=True)

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    log.info("Loading datasets...")
    train = pd.read_csv(PATHS["train"])
    val   = pd.read_csv(PATHS["val"])
    test  = pd.read_csv(PATHS["test"])

    X_train = train[FEATURES].values.astype(np.float32)
    y_train = train[TARGET].values.astype(np.int32)
    X_val   = val[FEATURES].values.astype(np.float32)
    y_val   = val[TARGET].values.astype(np.int32)
    X_test  = test[FEATURES].values.astype(np.float32)
    y_test  = test[TARGET].values.astype(np.int32)

    # Feature normalisation — fit on train, apply to all
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Save scaler params for ESP32 firmware use
    scaler_params = {
        "mean":  scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "features": FEATURES,
    }
    with open(os.path.join(PATHS["models"], "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f, indent=2)

    log.info(f"  Train : {X_train.shape[0]:,} rows (SMOTE balanced)")
    log.info(f"  Val   : {X_val.shape[0]:,} rows")
    log.info(f"  Test  : {X_test.shape[0]:,} rows")
    log.info(f"  Scaler params saved → models/scaler_params.json")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# ── Step 1: Build and train MLP ───────────────────────────────────────────────
def build_model():
    """
    Small MLP designed to fit on ESP32:
      - 4 inputs (PM2.5, PM10, temperature, humidity — normalised)
      - Dense(32) + BatchNorm + Dropout
      - Dense(16) + BatchNorm + Dropout
      - Dense(6, softmax)

    Parameter count kept deliberately minimal for TFLite Micro compatibility.
    """
    inputs = tf.keras.Input(shape=(4,), name="sensor_inputs")
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.Dropout(0.2, name="drop_1")(x)
    x = tf.keras.layers.Dense(16, activation="relu", name="dense_2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.Dropout(0.1, name="drop_2")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="aqi_output")(x)

    model = tf.keras.Model(inputs, outputs, name="aqi_mlp_v1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_model(X_train, y_train, X_val, y_val):
    log.info("\n── Step 1: Training MLP ──────────────────────────────")

    model = build_model()
    model.summary(print_fn=lambda x: log.info(f"  {x}"))

    total_params = model.count_params()
    log.info(f"\n  Total parameters: {total_params:,}")
    log.info(f"  Approx float32 size: {total_params * 4 / 1024:.1f} KB")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    log.info(f"\n  Training for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...")
    t0 = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    train_time = round(time.time() - t0, 1)
    epochs_run = len(history.history["loss"])
    log.info(f"\n  Training complete: {epochs_run} epochs in {train_time}s")

    model_path = os.path.join(PATHS["models"], "mlp_float32.keras")
    model.save(model_path)
    log.info(f"  Saved full model → {model_path}")

    hist_path = os.path.join(PATHS["models"], "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history.history, f)

    return model, history, train_time, epochs_run

# ── Step 2: Convert to TFLite float32 ────────────────────────────────────────
def convert_float32(model):
    log.info("\n── Step 2: TFLite Float32 Conversion ────────────────")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    out_path = os.path.join(PATHS["models"], "mlp_float32.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_bytes = len(tflite_model)
    log.info(f"  Float32 TFLite size: {size_bytes:,} bytes ({size_bytes/1024:.1f} KB)")
    log.info(f"  Saved → {out_path}")

    return tflite_model, size_bytes

# ── Step 3: INT8 post-training quantization ───────────────────────────────────
def quantize_int8(model, X_train):
    log.info("\n── Step 3: INT8 Post-Training Quantization ──────────")

    def representative_dataset():
        indices = np.random.choice(len(X_train), size=500, replace=False)
        for idx in indices:
            row = X_train[idx].reshape(1, -1).astype(np.float32)
            yield [row]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations              = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset     = representative_dataset
    converter.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type       = tf.float32
    converter.inference_output_type      = tf.float32

    tflite_quant = converter.convert()

    out_path = os.path.join(PATHS["models"], "mlp_int8.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_quant)

    size_bytes = len(tflite_quant)
    log.info(f"  INT8 quantized size: {size_bytes:,} bytes ({size_bytes/1024:.1f} KB)")
    log.info(f"  Saved → {out_path}")

    return tflite_quant, size_bytes

# ── Step 4: Evaluate all model variants ──────────────────────────────────────
def run_tflite_inference(tflite_bytes, X):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    preds     = []
    latencies = []

    for row in X:
        tensor = row.reshape(1, -1).astype(np.float32)
        interp.set_tensor(inp_details["index"], tensor)
        t0 = time.perf_counter()
        interp.invoke()
        t1 = time.perf_counter()
        output = interp.get_tensor(out_details["index"])
        preds.append(np.argmax(output))
        latencies.append((t1 - t0) * 1000)

    return np.array(preds), np.array(latencies)

def evaluate_all(model, tflite_f32, tflite_int8,
                 X_test, y_test, size_f32, size_int8):
    log.info("\n── Step 4: Evaluating All Model Variants ────────────")

    results = []

    log.info("  Evaluating: Full MLP (float32, Keras)...")
    y_pred_keras = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc_keras = accuracy_score(y_test, y_pred_keras)
    f1_keras  = f1_score(y_test, y_pred_keras, average="macro", zero_division=0)
    results.append({
        "variant":         "MLP float32 (Keras)",
        "accuracy":        round(acc_keras, 4),
        "macro_f1":        round(f1_keras, 4),
        "size_bytes":      model.count_params() * 4,
        "size_kb":         round(model.count_params() * 4 / 1024, 2),
        "platform":        "Laptop",
        "latency_mean_ms": "—",
        "latency_std_ms":  "—",
    })
    log.info(f"    Accuracy: {acc_keras:.4f}  Macro-F1: {f1_keras:.4f}")

    log.info("  Evaluating: TFLite float32...")
    y_pred_f32, lat_f32 = run_tflite_inference(tflite_f32, X_test)
    acc_f32 = accuracy_score(y_test, y_pred_f32)
    f1_f32  = f1_score(y_test, y_pred_f32, average="macro", zero_division=0)
    results.append({
        "variant":         "MLP float32 (TFLite)",
        "accuracy":        round(acc_f32, 4),
        "macro_f1":        round(f1_f32, 4),
        "size_bytes":      size_f32,
        "size_kb":         round(size_f32 / 1024, 2),
        "platform":        "Laptop (TFLite)",
        "latency_mean_ms": round(np.mean(lat_f32), 3),
        "latency_std_ms":  round(np.std(lat_f32), 3),
    })
    log.info(f"    Accuracy: {acc_f32:.4f}  Macro-F1: {f1_f32:.4f}  "
             f"Latency: {np.mean(lat_f32):.3f} ± {np.std(lat_f32):.3f} ms")

    log.info("  Evaluating: TFLite INT8 (quantized)...")
    y_pred_int8, lat_int8 = run_tflite_inference(tflite_int8, X_test)
    acc_int8 = accuracy_score(y_test, y_pred_int8)
    f1_int8  = f1_score(y_test, y_pred_int8, average="macro", zero_division=0)
    f1_drop  = round(f1_f32 - f1_int8, 4)
    size_reduction = round((1 - size_int8 / size_f32) * 100, 1)
    results.append({
        "variant":         "MLP INT8 (TFLite, quantized)",
        "accuracy":        round(acc_int8, 4),
        "macro_f1":        round(f1_int8, 4),
        "size_bytes":      size_int8,
        "size_kb":         round(size_int8 / 1024, 2),
        "platform":        "Laptop (TFLite INT8) → ESP32",
        "latency_mean_ms": round(np.mean(lat_int8), 3),
        "latency_std_ms":  round(np.std(lat_int8), 3),
    })
    log.info(f"    Accuracy: {acc_int8:.4f}  Macro-F1: {f1_int8:.4f}  "
             f"Latency: {np.mean(lat_int8):.3f} ± {np.std(lat_int8):.3f} ms")
    log.info(f"    Quantization macro-F1 drop: {f1_drop:+.4f}  "
             f"Size reduction: {size_reduction}%")

    print(f"\n  {'─'*55}")
    print(f"  Target checks:")
    print(f"  {'─'*55}")
    check = lambda ok, msg: f"  {'✓' if ok else '✗'} {msg}"
    print(check(f1_int8 >= TARGETS["macro_f1"],
                f"INT8 macro-F1 {f1_int8:.4f} {'≥' if f1_int8 >= TARGETS['macro_f1'] else '<'} target {TARGETS['macro_f1']}"))
    print(check(abs(f1_drop) <= TARGETS["quantization_drop"],
                f"Quantization drop {f1_drop:+.4f} {'≤' if abs(f1_drop) <= TARGETS['quantization_drop'] else '>'} target {TARGETS['quantization_drop']}"))

    print(f"\n  Per-class report — INT8 quantized (test set):")
    print(classification_report(
        y_test, y_pred_int8,
        target_names=AQI_LABELS_FULL,
        zero_division=0, digits=4
    ))

    return pd.DataFrame(results), y_pred_keras, y_pred_f32, y_pred_int8, f1_drop, size_reduction

# ── Step 5: Export C array ────────────────────────────────────────────────────
def export_c_array():
    log.info("\n── Step 5: Export C Array for ESP32 ────────────────")

    int8_path   = os.path.join(PATHS["models"], "mlp_int8.tflite")
    header_path = os.path.join(PATHS["firmware"], "aqi_model.h")
    os.makedirs(PATHS["firmware"], exist_ok=True)

    try:
        result = subprocess.run(
            ["xxd", "-i", int8_path],
            capture_output=True, text=True, check=True
        )
        c_array = result.stdout
        c_array = c_array.replace(
            "unsigned char mlp_int8_tflite[]",
            "const unsigned char aqi_model[]"
        ).replace(
            "unsigned int mlp_int8_tflite_len",
            "const unsigned int aqi_model_len"
        )

        header = f"""// aqi_model.h
// Auto-generated by train_mlp.py
// Project: TinyML-Based AQI Classification on ESP32
// Author: Bibek Bashyal, Kathmandu University 2026
//
// Input features (in order, normalised using scaler_params.json):
//   [0] PM2.5      [1] PM10      [2] temperature      [3] humidity
// Output: 6-class softmax — Good / Moderate / USG / Unhealthy / Very Unhealthy / Hazardous

#ifndef AQI_MODEL_H
#define AQI_MODEL_H

{c_array}

#endif // AQI_MODEL_H
"""
        with open(header_path, "w") as f:
            f.write(header)

        log.info(f"  C array exported → {header_path}")
        log.info(f"  Include in firmware: #include \"aqi_model.h\"")

    except FileNotFoundError:
        log.warning("  xxd not found — skipping C array export")
        log.warning("  Run manually: xxd -i models/mlp_int8.tflite > firmware/aqi_classifier/aqi_model.h")
    except Exception as e:
        log.warning(f"  C array export failed: {e}")

# ── Figures ───────────────────────────────────────────────────────────────────
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("MLP Training History", fontsize=13, fontweight="bold")

    for ax, metric, title in zip(
        axes,
        [("loss", "val_loss"), ("accuracy", "val_accuracy")],
        ["Loss", "Accuracy"]
    ):
        ax.plot(history.history[metric[0]], label="Train", color="#2196F3", linewidth=2)
        ax.plot(history.history[metric[1]], label="Val",   color="#FF9800", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PATHS["figures"], "mlp_training_history.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Training history → {out}")

def plot_model_comparison(results_df, f1_drop, size_reduction):
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("TinyML Model Variant Comparison", fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    variants = results_df["variant"].tolist()
    colors   = ["#2196F3", "#4CAF50", "#FF9800"]
    x        = np.arange(len(variants))

    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(x, results_df["macro_f1"], color=colors,
                   edgecolor="white", linewidth=0.5, width=0.5)
    ax1.axhline(y=TARGETS["macro_f1"], color="red",
                linestyle="--", linewidth=1.5, label=f"Target ({TARGETS['macro_f1']})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Keras\nfloat32", "TFLite\nfloat32", "TFLite\nINT8"], fontsize=9)
    ax1.set_ylabel("Macro-F1 Score", fontsize=11)
    ax1.set_title("Macro-F1 Score", fontsize=11)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=9)
    for bar, val in zip(bars, results_df["macro_f1"]):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01, f"{val:.4f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2 = fig.add_subplot(gs[1])
    size_kb = results_df["size_kb"].tolist()
    bars2 = ax2.bar(x, size_kb, color=colors,
                    edgecolor="white", linewidth=0.5, width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Keras\nfloat32", "TFLite\nfloat32", "TFLite\nINT8"], fontsize=9)
    ax2.set_ylabel("Model Size (KB)", fontsize=11)
    ax2.set_title("Model Size", fontsize=11)
    for bar, val in zip(bars2, size_kb):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3, f"{val:.1f} KB",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    int8_f1 = results_df.iloc[2]["macro_f1"]
    stats = [
        ["Metric", "Value"],
        ["INT8 Macro-F1", f"{int8_f1:.4f}"],
        ["F1 Drop (quant)", f"{f1_drop:+.4f}"],
        ["Size reduction", f"{size_reduction}%"],
        ["INT8 size", f"{results_df.iloc[2]['size_kb']:.1f} KB"],
        ["Target F1", f"≥ {TARGETS['macro_f1']}"],
        ["Status", "✓ PASS" if int8_f1 >= TARGETS["macro_f1"] else "✗ FAIL"],
    ]
    table = ax3.table(
        cellText=stats[1:], colLabels=stats[0],
        cellLoc="center", loc="center",
        bbox=[0.05, 0.1, 0.9, 0.85]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table[0, 0].set_facecolor("#E3F2FD")
    table[0, 1].set_facecolor("#E3F2FD")
    ax3.set_title("Summary Statistics", fontsize=11)

    plt.tight_layout()
    out = os.path.join(PATHS["figures"], "tinyml_model_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Model comparison → {out}")

def plot_confusion_matrix_int8(y_test, y_pred_int8):
    cm     = confusion_matrix(y_test, y_pred_int8)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("MLP INT8 — Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".1f"],
        ["Counts", "Row % (Recall per class)"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=AQI_LABELS, yticklabels=AQI_LABELS,
            ax=ax, linewidths=0.5
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    plt.tight_layout()
    out = os.path.join(PATHS["figures"], "confusion_matrix_mlp_int8.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  INT8 confusion matrix → {out}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  TinyML AQI Classifier — MLP Training Pipeline")
    log.info("  Bibek Bashyal — Kathmandu University 2026")
    log.info(f"  TensorFlow version: {tf.__version__}")
    log.info("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data()
    model, history, train_time, epochs_run = train_model(X_train, y_train, X_val, y_val)
    tflite_f32, size_f32   = convert_float32(model)
    tflite_int8, size_int8 = quantize_int8(model, X_train)

    results_df, y_pred_keras, y_pred_f32, y_pred_int8, f1_drop, size_reduction = \
        evaluate_all(model, tflite_f32, tflite_int8,
                     X_test, y_test, size_f32, size_int8)

    export_c_array()

    results_df.to_csv(
        os.path.join(PATHS["results"], "tinyml_model_comparison.csv"),
        index=False
    )

    log.info("\nGenerating figures...")
    plot_training_history(history)
    plot_model_comparison(results_df, f1_drop, size_reduction)
    plot_confusion_matrix_int8(y_test, y_pred_int8)

    print(f"\n{'='*60}")
    print("  TINYML PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Variant':<32} {'Accuracy':>10} {'Macro-F1':>10} {'Size':>10}")
    print(f"  {'-'*64}")
    for _, row in results_df.iterrows():
        size_str = f"{row['size_kb']:.1f} KB"
        print(f"  {row['variant']:<32} {row['accuracy']:>10.4f} {row['macro_f1']:>10.4f} {size_str:>10}")

    print(f"\n  Training time        : {train_time}s  ({epochs_run} epochs)")
    print(f"  Quantization F1 drop : {f1_drop:+.4f}")
    print(f"  Size reduction       : {size_reduction}%")

    int8_f1 = results_df.iloc[2]["macro_f1"]
    print(f"\n  Target macro-F1 ≥ {TARGETS['macro_f1']} : {'✓ PASS' if int8_f1 >= TARGETS['macro_f1'] else '✗ FAIL'} ({int8_f1:.4f})")
    print(f"  Quant drop ≤ {TARGETS['quantization_drop']}     : {'✓ PASS' if abs(f1_drop) <= TARGETS['quantization_drop'] else '✗ FAIL'} ({f1_drop:+.4f})")

    print(f"\n  Files saved:")
    print(f"    models/mlp_float32.keras")
    print(f"    models/mlp_float32.tflite")
    print(f"    models/mlp_int8.tflite")
    print(f"    models/scaler_params.json")
    print(f"    firmware/aqi_classifier/aqi_model.h")
    print(f"    results/tinyml_model_comparison.csv")
    print(f"\n  Next: Flash mlp_int8.tflite to ESP32")
    print(f"        Measure latency, RAM, Flash, energy on-device")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
