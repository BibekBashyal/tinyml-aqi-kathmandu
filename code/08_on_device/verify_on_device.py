"""
verify_on_device.py
===================
On-device verification of the INT8 AQI classifier (Chapter VII protocol).

Streams raw feature vectors to the ESP32-S3 running firmware/aqi_classifier with the serial command
F=<id>,pm25,pm10,temp,rh and compares each reply C=<id>,<class>,<conf>,<lat_us>,<p0>..<p5> with the
same INT8 model run by the TensorFlow Lite interpreter (reference kernels) on the laptop.

Expected result: identical classes AND bit-identical probabilities on every row, because
  - the board is checked first (serial MODEL) to run exactly models/mlp_int8.tflite and the input
    constants in aqi_model.h (zlib CRC-32 of both);
  - every value is sent as the shortest decimal of its float32; the board parses it with strtod and a
    float cast, the laptop with float() and np.float32, the same decimal->double->float chain;
  - humidity correction and input scaling use the constants the firmware was built with (config.h,
    aqi_model.h), in the same float32 operations (the firmware is built with -ffp-contract=off);
  - the firmware is built with -DTFLITE_SINGLE_ROUNDING=1, so its TFLM integer kernels round like
    TF 2.21's reference kernels (code/08_on_device/host_tflm proves this on the host).
Any non-identical row is therefore a finding, not noise.

Vector sources:
  field   data/field/run1_2026-11-01_aligned.csv   real PMS5003 / DHT22 rows from the node
  public  data/processed/test.csv                  --per-class rows per class (seed 42); 0 = whole split

Outputs: a complete run writes results/on_device_verification.csv and _summary.json. Any other run
(interrupted, aborted) writes results/on_device_verification_<time>_<status>.* and leaves the last
complete results untouched. Every vector gets a CSV row with its status (ok/timeout/infer/parse/bad).

Usage (stop any serial capture on the port first):
    .venv/bin/python code/08_on_device/verify_on_device.py [--port /dev/ttyACM0] [--field-step 1]
                                                           [--per-class 200] [--max-field 0]

Author: Bibek Bashyal, Kathmandu University 2026
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import warnings
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "04_tinyml_model"))
from esp_serial import EspSerial  # noqa: E402
from tflite_export import model_input, read_c_header_inputs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("verify")

MODEL = os.path.join(ROOT, "models", "mlp_int8.tflite")
HEADER = os.path.join(ROOT, "firmware", "aqi_classifier", "aqi_model.h")
CONFIG = os.path.join(ROOT, "firmware", "aqi_classifier", "include", "config.h")
FIELD = os.path.join(ROOT, "data", "field", "run1_2026-11-01_aligned.csv")
PUBLIC = os.path.join(ROOT, "data", "processed", "test.csv")
RESULTS = os.path.join(ROOT, "results")
OUT_STEM = "on_device_verification"

NUM_CLASSES = 6
LABELS = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]
PROB_STEP = 1.0 / 256.0           # the model's int8 softmax output resolution
PROB_SUM_TOL = 0.02               # 6 outputs each rounded by <= 1/512: sum within 3/256 of 1
MAX_LATENCY_US = 1_000_000
REPLY_TIMEOUT_S = 8.0             # first reply may wait for the current 3 s sensor cycle
MODEL_TIMEOUT_S = 8.0
ATTEMPTS = 2
MAX_CONSECUTIVE_FAILURES = 20
PROGRESS_EVERY = 500
CSV_FIELDS = ["id", "source", "status", "pm25", "pm10", "temp_c", "rh_pct", "label", "py_class", "dev_class",
              "agree", "py_conf", "dev_conf", "max_abs_prob_diff", "latency_us"]
FAILURE_COUNTER = {"timeout": "timeouts", "infer": "infer_errors", "parse": "parse_errors", "bad": "bad_replies"}


# ── Firmware constants and the laptop reference ───────────────────────────────
def config_float(name: str) -> float:
    found = re.search(rf"#define\s+{name}\s+([-+0-9.eE]+)f?", open(CONFIG).read())
    if found is None:
        raise ValueError(f"{name} not found in {CONFIG}")
    return float(found.group(1))


def header_model_bytes() -> bytes:
    text = open(HEADER).read()
    body = text[text.index("aqi_model[] = {") + len("aqi_model[] = {"):]
    body = body[:body.index("};")]
    return bytes(int(tok, 16) for tok in re.findall(r"0x[0-9a-fA-F]{2}", body))


def expected_fingerprint() -> tuple[int, int, int]:
    """(model bytes, CRC-32 of the model, CRC-32 of the input path constants) as the firmware computes them:
    AQI_INPUT_MEAN, AQI_INPUT_INV_SCALE, RH_ALPHA, RH_BETA as little-endian float32."""
    model_bytes = open(MODEL, "rb").read()
    mean, inv = read_c_header_inputs(HEADER)
    rh = np.array([config_float("RH_ALPHA"), config_float("RH_BETA")], dtype="<f4")
    inputs = mean.astype("<f4").tobytes() + inv.astype("<f4").tobytes() + rh.tobytes()
    return len(model_bytes), zlib.crc32(model_bytes), zlib.crc32(inputs)


class Reference:
    """The laptop-side INT8 model fed with the firmware's exact float32 pre-processing."""

    def __init__(self) -> None:
        model_bytes = open(MODEL, "rb").read()
        if header_model_bytes() != model_bytes:
            raise ValueError("aqi_model.h does not contain models/mlp_int8.tflite: rebuild the header and reflash")
        self.in_mean, self.in_inv = read_c_header_inputs(HEADER)
        self.alpha, self.beta = np.float32(config_float("RH_ALPHA")), np.float32(config_float("RH_BETA"))
        if self.alpha != 0:
            log.warning("RH_ALPHA != 0: powf on the board and numpy may differ by one ulp")
        self.interp = tf.lite.Interpreter(model_content=model_bytes,
                                          experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]["index"]
        self.out = self.interp.get_output_details()[0]["index"]

    def humidity_correct(self, pm25: np.float32, rh: np.float32) -> np.float32:
        if self.alpha == 0 or np.isnan(rh):
            return pm25
        return np.float32(pm25 / (np.float32(1) + self.alpha * np.power(rh / np.float32(100), self.beta)))

    def predict(self, v: np.ndarray) -> tuple[int, np.ndarray]:
        raw = np.array([self.humidity_correct(v[0], v[3]), v[1], v[2], v[3]], dtype=np.float32)
        self.interp.set_tensor(self.inp, model_input(raw, self.in_mean, self.in_inv).reshape(1, -1))
        self.interp.invoke()
        probs = self.interp.get_tensor(self.out)[0].astype(np.float64)
        return int(np.argmax(probs)), probs


# ── Vectors ───────────────────────────────────────────────────────────────────
def as_sent(x: float) -> tuple[str, np.float32]:
    """Shortest decimal of the float32 value and the float32 both sides parse from it."""
    text = np.format_float_positional(np.float32(x), unique=True, trim="-")
    return text, np.float32(float(text))


def load_vectors(field_step: int, per_class: int, max_field: int) -> list[tuple[str, list[float], int]]:
    rows = []
    fld = pd.read_csv(FIELD).dropna(subset=["pm25_raw", "pm10", "temp_c", "rh_pct"]).iloc[::max(1, field_step)]
    if max_field:
        fld = fld.head(max_field)
    for vals in fld[["pm25_raw", "pm10", "temp_c", "rh_pct"]].to_numpy():
        rows.append(("field", vals.tolist(), -1))
    public = pd.read_csv(PUBLIC)
    groups = [public] if per_class == 0 else [g.sample(n=min(per_class, len(g)), random_state=42)
                                              for _, g in public.groupby("aqi_class")]
    for take in groups:
        for *vals, cls in take[["PM2.5", "PM10", "temperature", "humidity", "aqi_class"]].to_numpy():
            rows.append(("public", [float(v) for v in vals], int(cls)))
    return rows


# ── Device exchange ───────────────────────────────────────────────────────────
@dataclass
class Counters:
    timeouts: int = 0
    bad_replies: int = 0
    infer_errors: int = 0
    parse_errors: int = 0
    stale_replies: int = 0
    noise: list = field(default_factory=list)


def reply_is_consistent(cls: int, conf: float, lat_us: int, probs: np.ndarray) -> bool:
    """A reply the real firmware can produce: int8 softmax steps, argmax = class, sane latency."""
    steps = probs / PROB_STEP
    return (0 <= cls < NUM_CLASSES and bool(np.all(np.abs(steps - np.round(steps)) < 1e-6))
            and int(np.argmax(probs)) == cls and conf == probs[cls] and abs(probs.sum() - 1.0) < PROB_SUM_TOL
            and 0 < lat_us < MAX_LATENCY_US)


def parse_reply(line: str, want_id: int, n: Counters) -> tuple[str, tuple | None]:
    """('ok', (cls, conf, lat_us, probs)) | ('stale'|'infer'|'parse'|'bad', None)."""
    parts = line[2:].split(",")
    try:
        rid = int(parts[0])
    except ValueError:
        return "bad", None
    if rid != want_id:
        n.stale_replies += 1
        return "stale", None
    if len(parts) == 2 and parts[1] in ("-1", "-2"):
        return ("infer" if parts[1] == "-1" else "parse"), None
    try:
        if len(parts) != 4 + NUM_CLASSES:
            raise ValueError(f"{len(parts)} fields")
        reply = (int(parts[1]), float(parts[2]), int(parts[3]), np.array([float(p) for p in parts[4:]]))
    except ValueError:
        return "bad", None
    return ("ok", reply) if reply_is_consistent(*reply) else ("bad", None)


def ask_device(port: EspSerial, next_id: list[int], texts: list[str], n: Counters) -> tuple[str, tuple | None]:
    """Send one vector (new id per attempt); returns ('ok', reply) or the last failure status."""
    status = "timeout"
    for _ in range(ATTEMPTS):
        rid = next_id[0]; next_id[0] += 1
        port.write_line(f"F={rid}," + ",".join(texts))
        deadline = time.monotonic() + REPLY_TIMEOUT_S
        while True:
            line = port.wait_for(("C=",), timeout_s=max(0.0, deadline - time.monotonic()), log=n.noise)
            status, reply = ("timeout", None) if line is None else parse_reply(line, rid, n)
            if status != "stale":
                break
        if status == "ok":
            return status, reply
        counter = FAILURE_COUNTER[status]
        setattr(n, counter, getattr(n, counter) + 1)
    return status, None


def check_board_model(port: EspSerial, n: Counters) -> dict:
    """Ask the board for its model fingerprint; raise if it is not the model on disk."""
    port.write_line("MODEL")
    line = port.wait_for(("M=",), timeout_s=MODEL_TIMEOUT_S, log=n.noise)
    if line is None:
        raise RuntimeError("no reply to MODEL: is the stage 2 classifier firmware (2026-11-03 or later) running?")
    size, model_crc, input_crc = line[2:].split(",")
    board = (int(size), int(model_crc, 16), int(input_crc, 16))
    want = expected_fingerprint()
    if board != want:
        raise RuntimeError(f"board runs a different model or input constants: board {line}, "
                           f"laptop M={want[0]},{want[1]:08x},{want[2]:08x}. Rebuild and reflash firmware/aqi_classifier")
    return {"model_bytes": board[0], "model_crc32": f"{board[1]:08x}", "input_crc32": f"{board[2]:08x}"}


# ── Run and summary ───────────────────────────────────────────────────────────
def record(i: int, src: str, sent: list, label: int, status: str, py: tuple, dev: tuple | None) -> dict:
    py_cls, py_probs = py
    rec = {"id": i, "source": src, "status": status, "pm25": sent[0][0], "pm10": sent[1][0], "temp_c": sent[2][0],
           "rh_pct": sent[3][0], "label": label, "py_class": py_cls, "py_conf": float(py_probs[py_cls])}
    if dev is not None:
        dev_cls, dev_conf, lat_us, dev_probs = dev
        rec.update({"dev_class": dev_cls, "agree": int(py_cls == dev_cls), "dev_conf": dev_conf,
                    "max_abs_prob_diff": float(np.max(np.abs(py_probs - dev_probs))), "latency_us": lat_us})
    return rec


def run(port: EspSerial, ref: Reference, vectors: list, writer: csv.DictWriter, out, n: Counters,
        records: list[dict]) -> str:
    """Appends to records as replies arrive, so the caller keeps them if the run stops early."""
    next_id, consecutive = [1], 0
    for i, (src, vals, label) in enumerate(vectors):
        sent = [as_sent(v) for v in vals]
        py = ref.predict(np.array([f for _, f in sent], dtype=np.float32))
        status, dev = ask_device(port, next_id, [t for t, _ in sent], n)
        rec = record(i, src, sent, label, status, py, dev)
        writer.writerow(rec); out.flush(); records.append(rec)
        consecutive = 0 if status == "ok" else consecutive + 1
        if consecutive >= MAX_CONSECUTIVE_FAILURES:
            log.error("%d vectors in a row got no usable reply: is the classifier firmware running?", consecutive)
            return "aborted_no_replies"
        if (i + 1) % PROGRESS_EVERY == 0:
            ok = [r for r in records if r["status"] == "ok"]
            log.info("  %d/%d  agreement %d/%d", i + 1, len(vectors), sum(r["agree"] for r in ok), len(ok))
    return "complete"


def public_scores(pub: pd.DataFrame, per_class: int) -> dict:
    labels = list(range(NUM_CLASSES))
    key = "full_test_split" if per_class == 0 else f"balanced_{per_class}_per_class"
    out = {"public_sample": key, "public_rows": int(len(pub))}
    for side, col in (("device", "dev_class"), ("laptop", "py_class")):
        out[f"public_{key}_macro_f1_{side}"] = round(float(f1_score(pub.label, pub[col], average="macro", labels=labels, zero_division=0)), 4)
        out[f"public_{key}_accuracy_{side}"] = round(float(accuracy_score(pub.label, pub[col])), 4)
    return out


def summarize(records: list[dict], vectors_total: int, n: Counters, status: str, per_class: int,
              board: dict | None) -> dict:
    s = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status, "board": board, "vectors_total": vectors_total,
         "vectors_sent": len(records), "timeouts": n.timeouts, "bad_replies": n.bad_replies, "infer_errors": n.infer_errors,
         "parse_errors": n.parse_errors, "stale_replies_discarded": n.stale_replies, "serial_noise_lines": len(n.noise)}
    df = pd.DataFrame(records, columns=CSV_FIELDS)
    ok = df[df.status == "ok"]
    s["replies_ok"] = int(len(ok))
    s["coverage_pct"] = round(100.0 * len(ok) / vectors_total, 3) if vectors_total else 0.0
    if ok.empty:
        return s
    steps = np.round(ok.max_abs_prob_diff.astype(float) / PROB_STEP).astype(int)
    s.update({
        "class_agreement_pct": round(100.0 * ok.agree.astype(int).mean(), 3),
        "disagreements": int((ok.agree.astype(int) == 0).sum()),
        "rows_with_identical_probabilities": int((ok.max_abs_prob_diff.astype(float) == 0).sum()),
        "bit_identical": bool((ok.max_abs_prob_diff.astype(float) == 0).all()),
        "max_abs_prob_diff_in_1_256_steps": int(steps.max()),
        "latency_us_mean": round(float(ok.latency_us.astype(float).mean()), 1),
        "latency_us_std": round(float(ok.latency_us.astype(float).std()), 1),
        "latency_us_p95": round(float(ok.latency_us.astype(float).quantile(0.95)), 1),
        "latency_us_max": int(ok.latency_us.astype(float).max()),
        "field_rows": int((ok.source == "field").sum()),
        "field_class_counts": {LABELS[int(k)]: int(v) for k, v in
                               ok[ok.source == "field"].dev_class.astype(int).value_counts().sort_index().items()},
    })
    pub = ok[ok.source == "public"].astype({"label": int, "dev_class": int, "py_class": int})
    if len(pub):
        s.update(public_scores(pub, per_class))
    return s


def output_paths(status: str) -> tuple[str, str]:
    stem = OUT_STEM if status == "complete" else f"{OUT_STEM}_{time.strftime('%Y-%m-%d_%H%M%S')}_{status}"
    return os.path.join(RESULTS, stem + ".csv"), os.path.join(RESULTS, stem + "_summary.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--field-step", type=int, default=1, help="use every n-th field row")
    ap.add_argument("--per-class", type=int, default=200, help="public test rows per class; 0 = the whole test split")
    ap.add_argument("--max-field", type=int, default=0, help="cap on field rows (0 = all)")
    args = ap.parse_args()

    ref = Reference()
    vectors = load_vectors(args.field_step, args.per_class, args.max_field)
    log.info("%d vectors (%d field, %d public)", len(vectors), sum(v[0] == "field" for v in vectors),
             sum(v[0] == "public" for v in vectors))
    os.makedirs(RESULTS, exist_ok=True)
    port = EspSerial(args.port)   # fails here, before any result file is touched, if the port is missing or busy
    n, records, status, board = Counters(), [], "error", None
    work_csv = os.path.join(RESULTS, f".{OUT_STEM}.partial.csv")
    if os.path.exists(work_csv):   # left by a run that crashed before its cleanup: keep it, never overwrite it
        orphan = os.path.join(RESULTS, f"{OUT_STEM}_{time.strftime('%Y-%m-%d_%H%M%S')}_orphaned.csv")
        os.replace(work_csv, orphan)
        log.warning("found rows from an earlier crashed run, kept as %s", orphan)
    try:
        board = check_board_model(port, n)
        log.info("board runs the model on disk: %s", board)
        with open(work_csv, "w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=CSV_FIELDS)
            writer.writeheader()
            status = run(port, ref, vectors, writer, out, n, records)
    except KeyboardInterrupt:
        status = "interrupted"
        log.warning("interrupted: keeping the rows received so far")
    finally:
        port.close()   # never raises
        final_csv, final_json = output_paths(status)
        if os.path.exists(work_csv):
            os.replace(work_csv, final_csv)
        summary = summarize(records, len(vectors), n, status, args.per_class, board)
        with open(final_json, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(json.dumps(summary, indent=2))
        log.info("rows -> %s\nsummary -> %s", final_csv, final_json)


if __name__ == "__main__":
    main()
