"""
fake_firmware.py
================
A stand-in for firmware/aqi_classifier on a pseudo-terminal, for testing the laptop scripts
without the board. Reply formats are taken from the firmware's C sources (Serial.printf format
strings in serial_cmd.cpp and benchmark.cpp), so a format change on the board breaks these tests.

Inference uses TFLite's default kernels. For this model they are bit-identical to the reference
kernels the laptop scripts use, as the real board is when built with -DTFLITE_SINGLE_ROUNDING=1, so
replies are exact. perturb_every=N moves one probability by 1/256 on every N-th reply, to prove the
verification flags a board that is not bit-identical. wrong_model=True makes MODEL report another CRC.

Fault injection (by request number k, counted from 1): every NOISE_EVERY-th request is preceded by
a status line, every STALE_EVERY-th by a reply carrying an old id, every CORRUPT_EVERY-th first
answer is truncated, every DROP_EVERY-th first answer is not sent, every INFER_ERR_EVERY-th first
answer is C=<id>,-1. Faults hit only the first attempt for a vector; the retry is answered properly.

Author: Bibek Bashyal, Kathmandu University 2026
"""

import os
import re
import threading
import zlib

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(ROOT, "firmware", "aqi_classifier", "src")
MODEL = os.path.join(ROOT, "models", "mlp_int8.tflite")

NOISE_EVERY, STALE_EVERY, CORRUPT_EVERY, DROP_EVERY, INFER_ERR_EVERY = 7, 11, 13, 17, 19


def c_formats(filename: str) -> list[str]:
    """Serial.printf format strings of a firmware source, converted to Python %-formatting."""
    text = open(os.path.join(SRC, filename)).read()
    fmts = []
    for call in re.findall(r"Serial\.printf\((\"(?:[^\"\\]|\\.)*\"(?:\s*\"(?:[^\"\\]|\\.)*\")*)", text):
        joined = "".join(re.findall(r"\"((?:[^\"\\]|\\.)*)\"", call))
        fmts.append(joined.replace("\\n", "\n").replace("%lu", "%d").replace("%u", "%d"))
    return fmts


def fmt_with(prefix: str, filename: str) -> str:
    matches = [f for f in c_formats(filename) if f.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one printf starting with {prefix!r} in {filename}, found {matches}")
    return matches[0]


class FakeFirmware:
    def __init__(self, bench_mode: str = "ok", perturb_every: int = 0, wrong_model: bool = False,
                 ignore_vectors: bool = False, hangup_after: int = 0, boot_bench_running: bool = False) -> None:
        self.master, self.slave = os.openpty()
        self.path = os.ttyname(self.slave)
        self.bench_mode = bench_mode
        self.perturb_every = perturb_every
        self.wrong_model = wrong_model
        self.ignore_vectors = ignore_vectors   # answers MODEL, never F= (a board stuck in a long task)
        self.hangup_after = hangup_after       # close the master after N vectors: a USB disconnect
        self.boot_bench_running = boot_bench_running   # an INVALID boot benchmark finishes before MODEL is answered
        self.requests = 0
        self.faulted_vectors = set()
        self._stop = False
        self.interp = tf.lite.Interpreter(model_path=MODEL)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]["index"]
        self.out = self.interp.get_output_details()[0]["index"]
        import sys
        sys.path.insert(0, os.path.join(ROOT, "code", "04_tinyml_model"))
        from tflite_export import model_input, read_c_header_inputs
        self.model_input = model_input
        self.in_mean, self.in_inv = read_c_header_inputs(os.path.join(ROOT, "firmware", "aqi_classifier", "aqi_model.h"))
        self.model_reply = fmt_with("M=%d,%08lx,%08lx", "serial_cmd.cpp").replace("%08lx", "%08x")
        model_bytes = open(MODEL, "rb").read()
        cfg = open(os.path.join(ROOT, "firmware", "aqi_classifier", "include", "config.h")).read()
        rh = [float(re.search(rf"#define\s+{n}\s+([-+0-9.eE]+)f?", cfg).group(1)) for n in ("RH_ALPHA", "RH_BETA")]
        inputs = (self.in_mean.astype("<f4").tobytes() + self.in_inv.astype("<f4").tobytes()
                  + np.array(rh, dtype="<f4").tobytes())
        self.fingerprint = (len(model_bytes), zlib.crc32(model_bytes) ^ (1 if wrong_model else 0), zlib.crc32(inputs))
        self.c_reply_head = fmt_with("C=%d,%d,%.8f,%d", "serial_cmd.cpp")
        self.c_reply_prob = fmt_with(",%.8f", "serial_cmd.cpp")
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self._stop = True
        for fd in (self.slave, self.master):
            try:
                os.close(fd)
            except OSError:
                pass   # the master may already be closed by a simulated hang-up

    def _send(self, text: str) -> None:
        os.write(self.master, text.replace("\n", "\r\n").encode())

    def _classify(self, values: list[float]) -> tuple[int, np.ndarray, int]:
        raw = np.array([np.float32(v) for v in values], dtype=np.float32)   # RH_ALPHA = 0: pass-through
        self.interp.set_tensor(self.inp, self.model_input(raw, self.in_mean, self.in_inv).reshape(1, -1))
        self.interp.invoke()
        probs = self.interp.get_tensor(self.out)[0]
        best = 0
        for i in range(6):
            if probs[i] > probs[best]:
                best = i
        return best, probs, 180

    def _answer_vector(self, line: str) -> None:
        parts = line[2:].split(",")
        rid = int(parts[0])
        values = [float(p) for p in parts[1:]]       # strtod
        key = tuple(parts[1:])
        self.requests += 1
        k = self.requests
        if k % NOISE_EVERY == 0:
            self._send("uptime_99s  PM2.5=40 PM10=50 T=21.0 RH=75.0  class=Moderate (0.996) lat=0.18ms  bus: lcd=1 rtc=1 sd=1\n")
        if k % STALE_EVERY == 0:
            self._send(f"C={rid - 1},1,0.99609375,170" + ",0.00000000" * 6 + "\n")
        first_attempt = key not in self.faulted_vectors
        cls, probs, lat = self._classify(values)
        if self.perturb_every and k % self.perturb_every == 0:   # a 1/256 kernel difference, class unchanged
            probs = probs.copy(); other = (cls + 1) % 6; step = np.float32(1.0 / 256.0)
            runner_up = max(p for i, p in enumerate(probs) if i != cls)
            if runner_up + 2 * step < probs[cls]:   # keeps a clear winner: the class cannot change
                probs[other] += step; probs[cls] -= step
        reply = self.c_reply_head % (rid, cls, float(probs[cls]), lat) + "".join(self.c_reply_prob % float(p) for p in probs) + "\n"
        for every, action in ((CORRUPT_EVERY, "corrupt"), (DROP_EVERY, "drop"), (INFER_ERR_EVERY, "infer")):
            if first_attempt and k % every == 0:
                self.faulted_vectors.add(key)
                if action == "corrupt":
                    self._send(reply[: len(reply) // 2] + "\x80x,zz\n")
                elif action == "infer":
                    self._send(f"C={rid},-1\n")
                return
        self._send(reply)

    def _bench(self) -> None:
        self._send("BENCH start\n")
        self._send(fmt_with("LATENCY cold:", "benchmark.cpp") % (0.4127,))
        self._send(fmt_with("LATENCY: %", "benchmark.cpp") % (0.1823, 0.0041, 100, 0.1790, 0.2104))
        self._send(fmt_with("RAM:", "benchmark.cpp") % (2124, 10240, 4352, 0))
        if self.bench_mode == "truncated":
            return
        if self.bench_mode == "skipped":
            self._send("ENERGY: skipped, INA219 not on the bus\n")
        elif self.bench_mode == "invalid":
            self._send(fmt_with("ENERGY INVALID: only", "benchmark.cpp") % (3, 25, 22))
        else:
            self._send(fmt_with("ENERGY setup:", "benchmark.cpp") % (24, 400, 1, 2190.0, 99.4))
            self._send(fmt_with("ENERGY power:", "benchmark.cpp") % (412.31, 3.10, 389.75, 2.95, 5.312, 73.4))
            self._send(fmt_with("ENERGY total:", "benchmark.cpp") % (0.075163, 0.1823))
            self._send(fmt_with("ENERGY extra:", "benchmark.cpp") % (0.004113, 0.000810, 24, 0.004020))
            self._send(fmt_with("ENERGY rail3v3:", "benchmark.cpp") % (0.044161, 0.002502))
        self._send("BENCH done\n")

    def _serve(self) -> None:
        buf = b""
        while not self._stop:
            try:
                chunk = os.read(self.master, 4096)
            except OSError:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode().strip()
                if text.startswith("F=") and not self.ignore_vectors:
                    self._answer_vector(text)
                    if self.hangup_after and self.requests >= self.hangup_after:
                        self._stop = True
                        os.close(self.master)   # the slave side now sees a hang-up
                        return
                elif text == "BENCH":
                    self._bench()
                elif text == "MODEL":
                    if self.boot_bench_running:
                        mode, self.bench_mode = self.bench_mode, "invalid"
                        self._bench()                     # the tail of a boot run, before our BENCH
                        self.bench_mode = mode
                    self._send(self.model_reply % self.fingerprint)
