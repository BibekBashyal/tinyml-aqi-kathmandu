"""Tests for the laptop side of the on-device verification and benchmark (no board needed)."""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

import bench_capture
import verify_on_device as vod
from esp_serial import EspSerial
from tflite_export import assert_per_tensor_fc, read_c_header_inputs

ROOT = vod.ROOT
SMALL_RUN = ["--max-field", "40", "--per-class", "5"]   # 40 field + 30 public vectors


# ── verify_on_device ──────────────────────────────────────────────────────────
@pytest.fixture
def results_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(vod, "RESULTS", str(tmp_path))
    monkeypatch.setattr(vod, "REPLY_TIMEOUT_S", 0.5)
    monkeypatch.setattr(vod, "MODEL_TIMEOUT_S", 2.0)
    return tmp_path


def run_verify(monkeypatch, port: str) -> None:
    monkeypatch.setattr(sys, "argv", ["verify", "--port", port, *SMALL_RUN])
    vod.main()


def test_verify_end_to_end_survives_injected_faults(fake_device, results_dir, monkeypatch):
    # Arrange / Act
    run_verify(monkeypatch, fake_device.path)
    # Assert
    summary = json.load(open(results_dir / "on_device_verification_summary.json"))
    rows = pd.read_csv(results_dir / "on_device_verification.csv")
    assert summary["status"] == "complete" and summary["board"]["model_bytes"] == 4352
    assert summary["vectors_total"] == 70 and summary["replies_ok"] == 70 and summary["coverage_pct"] == 100.0
    assert summary["class_agreement_pct"] == 100.0
    assert summary["bit_identical"] and summary["rows_with_identical_probabilities"] == 70
    assert summary["stale_replies_discarded"] >= 1
    assert summary["timeouts"] >= 1 and summary["bad_replies"] >= 1 and summary["infer_errors"] >= 1
    assert len(rows) == 70 and rows["id"].is_unique and set(rows.status) == {"ok"}
    pub = rows[rows.source == "public"]
    key = summary["public_sample"]
    assert summary[f"public_{key}_macro_f1_device"] == round(f1_score(pub.label, pub.dev_class, average="macro", labels=list(range(6))), 4)
    assert summary[f"public_{key}_macro_f1_laptop"] == round(f1_score(pub.label, pub.py_class, average="macro", labels=list(range(6))), 4)


def test_verify_flags_a_board_that_is_not_bit_identical(fake_device_factory, results_dir, monkeypatch):
    dev = fake_device_factory(perturb_every=5)
    run_verify(monkeypatch, dev.path)
    summary = json.load(open(results_dir / "on_device_verification_summary.json"))
    assert summary["class_agreement_pct"] == 100.0          # classes still agree...
    assert not summary["bit_identical"]                       # ...but the probability change is caught
    assert summary["max_abs_prob_diff_in_1_256_steps"] == 1
    assert summary["rows_with_identical_probabilities"] < summary["replies_ok"]


def test_verify_refuses_a_board_with_another_model(fake_device_factory, results_dir, monkeypatch):
    dev = fake_device_factory(wrong_model=True)
    (results_dir / "on_device_verification.csv").write_text("previous complete run\n")
    with pytest.raises(RuntimeError, match="different model"):
        run_verify(monkeypatch, dev.path)
    assert (results_dir / "on_device_verification.csv").read_text() == "previous complete run\n"
    errors = list(results_dir.glob("on_device_verification_*_error_summary.json"))
    assert len(errors) == 1 and json.load(open(errors[0]))["status"] == "error"


def test_verify_abort_keeps_the_last_complete_results(fake_device_factory, results_dir, monkeypatch):
    dev = fake_device_factory(ignore_vectors=True)
    monkeypatch.setattr(vod, "REPLY_TIMEOUT_S", 0.05)
    monkeypatch.setattr(vod, "MAX_CONSECUTIVE_FAILURES", 3)
    (results_dir / "on_device_verification_summary.json").write_text('{"status": "complete"}')
    run_verify(monkeypatch, dev.path)
    assert json.load(open(results_dir / "on_device_verification_summary.json")) == {"status": "complete"}
    aborted = list(results_dir.glob("on_device_verification_*_aborted_no_replies_summary.json"))
    summary = json.load(open(aborted[0]))
    assert summary["replies_ok"] == 0 and summary["vectors_sent"] == 3 and summary["timeouts"] == 6


def test_verify_never_touches_results_when_the_port_is_missing(results_dir, monkeypatch):
    with pytest.raises(FileNotFoundError):
        run_verify(monkeypatch, "/dev/no_such_tty_for_test")
    assert list(results_dir.iterdir()) == []


def test_verify_keeps_rows_and_summary_when_the_port_hangs_up(fake_device_factory, results_dir, monkeypatch):
    dev = fake_device_factory(hangup_after=12)
    with pytest.raises(OSError):
        run_verify(monkeypatch, dev.path)
    errors = list(results_dir.glob("on_device_verification_*_error_summary.json"))
    rows = list(results_dir.glob("on_device_verification_*_error.csv"))
    assert len(errors) == 1 and len(rows) == 1
    summary = json.load(open(errors[0]))
    assert summary["status"] == "error" and summary["replies_ok"] >= 10
    assert len(pd.read_csv(rows[0])) == summary["vectors_sent"]
    assert not (results_dir / ".on_device_verification.partial.csv").exists()


def test_verify_keeps_rows_of_an_earlier_crashed_run(fake_device, results_dir, monkeypatch):
    (results_dir / ".on_device_verification.partial.csv").write_text("id,source\n0,field\n")
    run_verify(monkeypatch, fake_device.path)
    orphans = list(results_dir.glob("on_device_verification_*_orphaned.csv"))
    assert len(orphans) == 1 and orphans[0].read_text() == "id,source\n0,field\n"


def test_port_held_by_another_process_is_refused(fake_device):
    holder = subprocess.Popen(["sleep", "30"], stdin=open(fake_device.path))
    try:
        with pytest.raises(RuntimeError, match="already open"):
            EspSerial(fake_device.path)
    finally:
        holder.kill(); holder.wait()


def test_parse_reply_rejects_stale_malformed_and_inconsistent_replies():
    n = vod.Counters()
    probs = ["0.00000000"] * 4 + ["0.00390625", "0.99609375"]
    good = "C=7,5,0.99609375,180," + ",".join(probs)
    assert vod.parse_reply(good, 7, n)[0] == "ok"
    assert vod.parse_reply(good, 8, n)[0] == "stale" and n.stale_replies == 1
    assert vod.parse_reply("C=7,-1", 7, n)[0] == "infer"
    assert vod.parse_reply("C=7,-2", 7, n)[0] == "parse"
    assert vod.parse_reply("C=7,5,0.99\x80x,zz", 7, n)[0] == "bad"
    assert vod.parse_reply("C=x7,1", 7, n)[0] == "bad"
    one_digit_lost = good.replace("0.00390625", "0.0390625")               # 1/256 -> 10/256
    assert vod.parse_reply(one_digit_lost, 7, n)[0] == "bad"
    wrong_class = good.replace("C=7,5,", "C=7,1,")
    assert vod.parse_reply(wrong_class, 7, n)[0] == "bad"
    not_a_softmax_step = good.replace("0.00390625", "0.00400000")
    assert vod.parse_reply(not_a_softmax_step, 7, n)[0] == "bad"


@pytest.mark.parametrize("value", [130.10264705882352, 12.0, 0.1, 21.4, 99.99999, 1e-3])
def test_sent_text_parses_back_to_the_same_float32(value):
    text, f32 = vod.as_sent(value)
    assert np.float32(float(text)) == f32 == np.float32(value)


def test_summary_of_no_replies_does_not_crash():
    summary = vod.summarize([], 10, vod.Counters(timeouts=20), "aborted_no_replies", 200, None)
    assert summary["replies_ok"] == 0 and summary["coverage_pct"] == 0.0 and summary["timeouts"] == 20


def test_full_test_split_option_loads_every_public_row():
    vectors = vod.load_vectors(field_step=1, per_class=0, max_field=1)
    public = [v for v in vectors if v[0] == "public"]
    assert len(public) == len(pd.read_csv(vod.PUBLIC))


# ── bench_capture ─────────────────────────────────────────────────────────────
def test_bench_capture_parses_every_firmware_line(fake_bench_factory):
    dev = fake_bench_factory("ok")
    lines, done = bench_capture.capture(dev.path, 10, [])
    row = bench_capture.parse(lines)
    assert done and row["energy_status"] == "ok"
    for _, cols in bench_capture.PATTERNS.values():
        for col in cols:
            assert col in row, col
            if col not in ("board_model_crc32", "board_input_crc32"):
                float(row[col])
    expected = vod.expected_fingerprint()
    assert int(row["board_model_bytes"]) == expected[0] and int(row["board_model_crc32"], 16) == expected[1]
    assert int(row["board_input_crc32"], 16) == expected[2]   # input scaling and humidity constants
    assert float(row["energy_extra_median_mJ"]) == pytest.approx(0.004020)
    assert float(row["latency_cold_ms"]) == pytest.approx(0.4127)


@pytest.mark.parametrize("mode,status", [("skipped", "skipped, INA219 not on the bus"),
                                         ("invalid", "ENERGY INVALID: only 3 of 25")])
def test_bench_capture_records_why_energy_is_missing(fake_bench_factory, mode, status):
    dev = fake_bench_factory(mode)
    lines, done = bench_capture.capture(dev.path, 10, [])
    row = bench_capture.parse(lines)
    assert done and "energy_total_mJ" not in row
    assert row["energy_status"].startswith(status)


def test_bench_capture_parses_only_the_run_it_asked_for(fake_device_factory):
    dev = fake_device_factory(boot_bench_running=True)   # an INVALID boot run ends before MODEL is answered
    raw = []
    lines, done = bench_capture.capture(dev.path, 10, raw)
    row = bench_capture.parse(lines)
    assert done and row["energy_status"] == "ok" and float(row["energy_total_mJ"]) == pytest.approx(0.075163)
    assert any(line.startswith("ENERGY INVALID") for line in raw)   # still in the raw file


def test_bench_capture_writes_no_row_for_an_incomplete_run(fake_bench_factory, monkeypatch, tmp_path):
    dev = fake_bench_factory("truncated")
    monkeypatch.setattr(bench_capture, "RESULTS", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["bench", "--port", dev.path, "--timeout", "3"])
    with pytest.raises(SystemExit) as stop:
        bench_capture.main()
    assert stop.value.code == 1
    assert not (tmp_path / "esp32_benchmark.csv").exists()


# ── model, header and firmware consistency ────────────────────────────────────
def test_header_carries_the_deployed_model_and_its_input_constants():
    model = open(os.path.join(ROOT, "models", "mlp_int8.tflite"), "rb").read()
    assert vod.header_model_bytes() == model
    mean, inv = read_c_header_inputs(vod.HEADER)
    params = json.load(open(os.path.join(ROOT, "models", "int8_input_params.json")))
    assert np.array_equal(mean, np.array(params["input_mean"], np.float32))
    assert np.array_equal(inv, np.array(params["input_inv_scale"], np.float32))


def test_deployed_model_is_per_tensor_and_old_one_is_rejected():
    assert assert_per_tensor_fc(open(os.path.join(ROOT, "models", "mlp_int8.tflite"), "rb").read()) == [1, 1, 1]
    old = os.path.join(ROOT, "models", "mlp_int8_perchannel_2026-06-25.tflite")
    if not os.path.exists(old):
        pytest.skip("old per-channel model not present")
    with pytest.raises(ValueError, match="per-channel"):
        assert_per_tensor_fc(open(old, "rb").read())


def test_folded_float_model_matches_the_trained_one():
    import tensorflow as tf
    base = tf.keras.models.load_model(os.path.join(ROOT, "models", "mlp_float32.keras"))
    folded = tf.keras.models.load_model(os.path.join(ROOT, "models", "mlp_float32_rangefold.keras"))
    params = json.load(open(os.path.join(ROOT, "models", "int8_input_params.json")))
    centre, half = np.array(params["standardized_centre"]), np.array(params["standardized_half_range"])
    z = np.random.default_rng(0).uniform(-1.5, 4.0, size=(2000, 4)).astype(np.float32)
    p_base = base.predict(z, verbose=0)
    p_fold = folded.predict(((z - centre) / half).astype(np.float32), verbose=0)
    assert np.max(np.abs(p_base - p_fold)) < 1e-4
    assert np.mean(p_base.argmax(1) == p_fold.argmax(1)) >= 0.999   # ties at float noise level only


def ini_build_flags(path: str) -> set[str]:
    """Flags in the build_flags block of a platformio.ini, comments stripped."""
    flags, inside = set(), False
    for line in open(path):
        if re.match(r"^build_flags\s*=", line):
            inside = True
            continue
        if inside and not line.startswith((" ", "\t")):
            break
        if inside:
            flags.update(line.split(";", 1)[0].split())
    return flags


def test_firmware_build_flags_match_the_host_harness():
    ini = ini_build_flags(os.path.join(ROOT, "firmware", "aqi_classifier", "platformio.ini"))
    script = open(os.path.join(vod.HERE, "host_tflm", "build.sh")).read()
    harness = set(re.search(r"^FLAGS=\((.*?)\)$", script, re.S | re.M).group(1).split())
    for flag in ("-DTFLITE_SINGLE_ROUNDING=1", "-ffp-contract=off"):
        assert flag in ini and flag in harness, flag


def test_worst_case_csv_row_fits_the_firmware_buffer():
    main_cpp = open(os.path.join(ROOT, "firmware", "aqi_classifier", "src", "main.cpp")).read()
    fmt = re.search(r'snprintf\(row, sizeof\(row\), "([^"]+)"', main_cpp).group(1).replace("\\n", "\n").replace("%u", "%d")
    config = open(os.path.join(ROOT, "firmware", "aqi_classifier", "include", "config.h")).read()
    row_bytes = int(re.search(r"#define SD_ROW_BYTES\s+(\d+)", config).group(1))
    worst = fmt % ("uptime_4294967s", 65535, -65535.0, 65535, -40.0, 100.0, "5", "1.000", "99999.999", float("nan"), -3200.0)
    worst_rtc = fmt % ("2026-11-03 07:28:10", 65535, 65535.0, 65535, -40.0, 100.0, "5", "1.000", "99999.999", 32.76, -3276.8)
    assert len(worst) < row_bytes and len(worst_rtc) < row_bytes
