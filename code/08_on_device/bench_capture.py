"""
bench_capture.py
================
Runs the on-device benchmark protocol (serial command BENCH) on firmware/aqi_classifier and stores
the raw board output plus one parsed row in results/esp32_benchmark.csv.

Before BENCH it asks the board for its model fingerprint (serial MODEL) and stores it with the row,
so every benchmark number can be tied to the exact model and input constants it ran.

Lines parsed (formats printed by firmware/aqi_classifier/src/benchmark.cpp):
  LATENCY cold: <t> ms (...)
  LATENCY: <mean> +/- <sd> ms (n=100, min <a>, max <b>)
  RAM: tensor arena used <a> of <b> bytes; model <m> bytes in flash; heap delta <d> bytes
  ENERGY setup: <pairs> pairs x <w> ms windows (<r> rejected), ..., <k> inferences per window, inference duty <x> %
  ENERGY power: busy <p> +/- <s> mW, idle <p> +/- <s> mW, bus <v> V, idle current <i> mA
  ENERGY total: <e> mJ per inference (P_busy x t_inf, t_inf <t> ms)
  ENERGY extra: <e> +/- <s> mJ per inference (..., median <m>)
  ENERGY rail3v3: total <e> mJ, extra <e> mJ per inference (...)
  ENERGY: skipped, <reason>        |  ENERGY INVALID: <reason>

No row is appended unless the run reached "BENCH done" with a LATENCY line; energy columns stay
empty when energy was skipped or invalid, and energy_status says why.

Usage (stop any serial capture on the port first):
    .venv/bin/python code/08_on_device/bench_capture.py [--port /dev/ttyACM0] [--flash-bytes N]
                                                        [--ram-bytes N] [--note text] [--timeout 300]

Author: Bibek Bashyal, Kathmandu University 2026
"""

import argparse
import csv
import logging
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from esp_serial import EspSerial  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bench")

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")
NUM = r"(-?[\d.]+|nan|-?inf)"
PATTERNS = {
    "cold": (re.compile(rf"LATENCY cold: {NUM} ms"), ["latency_cold_ms"]),
    "latency": (re.compile(rf"LATENCY: {NUM} \+/- {NUM} ms \(n=(\d+), min {NUM}, max {NUM}\)"),
                ["latency_mean_ms", "latency_std_ms", "latency_n", "latency_min_ms", "latency_max_ms"]),
    "ram": (re.compile(r"arena used (\d+) of (\d+) bytes; model (\d+) bytes in flash; heap delta (-?\d+)"),
            ["arena_used_bytes", "arena_size_bytes", "model_bytes", "heap_delta_bytes"]),
    "setup": (re.compile(rf"ENERGY setup: (\d+) pairs x (\d+) ms windows \((\d+) rejected\).*?{NUM} inferences per window, inference duty {NUM} %"),
              ["energy_pairs", "energy_window_ms", "energy_pairs_rejected", "inferences_per_window", "inference_duty_pct"]),
    "power": (re.compile(rf"ENERGY power: busy {NUM} \+/- {NUM} mW, idle {NUM} \+/- {NUM} mW, bus {NUM} V, idle current {NUM} mA"),
              ["p_busy_mW", "p_busy_sd_mW", "p_idle_mW", "p_idle_sd_mW", "bus_V", "idle_current_mA"]),
    "total": (re.compile(rf"ENERGY total: {NUM} mJ per inference \(P_busy x t_inf, t_inf {NUM} ms\)"),
              ["energy_total_mJ", "t_inf_ms"]),
    "extra": (re.compile(rf"ENERGY extra: {NUM} \+/- {NUM} mJ per inference .*median {NUM}\)"),
              ["energy_extra_mJ", "energy_extra_sd_mJ", "energy_extra_median_mJ"]),
    "rail": (re.compile(rf"ENERGY rail3v3: total {NUM} mJ, extra {NUM} mJ per inference"),
             ["energy_total_3v3_mJ", "energy_extra_3v3_mJ"]),
    "model": (re.compile(r"^M=(\d+),([0-9a-f]{8}),([0-9a-f]{8})$"), ["board_model_bytes", "board_model_crc32", "board_input_crc32"]),
}
FIELDS = (["date"] + [c for _, cols in PATTERNS.values() for c in cols]
          + ["energy_status", "flash_bytes", "static_ram_bytes", "note", "raw_file"])


def parse(lines: list[str]) -> dict:
    row = {}
    for line in lines:
        for regex, cols in PATTERNS.values():
            m = regex.search(line)
            if m:
                row.update(zip(cols, m.groups()))
        if line.startswith(("ENERGY: skipped", "ENERGY INVALID")):
            row["energy_status"] = line.split(":", 1)[1].strip() if line.startswith("ENERGY:") else line
    if "energy_total_mJ" in row and "energy_status" not in row:
        row["energy_status"] = "ok"
    return row


MODEL_TIMEOUT_S = 8.0


def capture(port_name: str, timeout_s: float, raw: list[str]) -> tuple[list[str], bool]:
    """Lines of the requested benchmark run (plus the M= line) and whether it reached 'BENCH done'.

    Everything received goes into raw as well, for the raw file. Lines from a benchmark that was
    already running (BENCHMARK_MODE at boot) are kept in raw but not parsed: parsing starts at the
    first 'BENCH start' that follows our BENCH command."""
    run, done, started = [], False, False
    with EspSerial(port_name) as port:
        port.write_line("MODEL")
        model = port.wait_for(("M=",), timeout_s=MODEL_TIMEOUT_S, log=raw)
        if model is None:
            log.warning("no reply to MODEL: firmware older than 2026-11-03, the model fingerprint stays empty")
        else:
            raw.append(model)
            run.append(model)
        port.write_line("BENCH")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not done:
            line = port.read_line(5.0)
            if line is None:
                continue
            raw.append(line)
            log.info(line)
            started = started or line.startswith("BENCH start")
            if started:
                run.append(line)
                done = line.startswith("BENCH done")
    return run, done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--flash-bytes", type=int, default=None, help="firmware flash use from the build report")
    ap.add_argument("--ram-bytes", type=int, default=None, help="static RAM use from the build report")
    ap.add_argument("--note", default="")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds to wait for 'BENCH done'")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    raw_path = os.path.join(RESULTS, f"esp32_benchmark_{time.strftime('%Y-%m-%d_%H%M%S')}.txt")
    raw: list[str] = []
    try:
        lines, done = capture(args.port, args.timeout, raw)
    finally:   # the raw capture is kept even if the port hangs up mid-run
        with open(raw_path, "w") as f:
            f.write("\n".join(raw) + "\n")
    row = parse(lines)
    if not done or "latency_mean_ms" not in row:
        log.error("no complete benchmark (BENCH done seen: %s, LATENCY seen: %s); raw output kept in %s, no CSV row written",
                  done, "latency_mean_ms" in row, raw_path)
        sys.exit(1)

    out = {k: "" for k in FIELDS}
    out.update(row)
    out.update({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "flash_bytes": args.flash_bytes or "",
                "static_ram_bytes": args.ram_bytes or "", "note": args.note, "raw_file": os.path.basename(raw_path)})
    csv_path = os.path.join(RESULTS, "esp32_benchmark.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(out)
    log.info("raw -> %s\nrow -> %s (energy: %s)", raw_path, csv_path, out["energy_status"] or "not reported")


if __name__ == "__main__":
    main()
