"""Bit-exactness of the board's TFLM against the laptop reference, without the board.

Builds code/08_on_device/host_tflm/harness from the firmware's own TensorFlow Lite Micro sources
(.pio/libdeps, same arithmetic flags as platformio.ini) and runs the whole test split through it.
Every input and every probability must equal the laptop reference used by verify_on_device.py.
Skipped when g++ or the PlatformIO library download is missing. First build takes ~1 minute.
"""

import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import verify_on_device as vod
from tflite_export import model_input

HOST = os.path.join(vod.HERE, "host_tflm")
LIB = os.path.join(vod.ROOT, "firmware", "aqi_classifier", ".pio", "libdeps", "esp32-s3-devkitc-1", "TensorFlowLite_ESP32", "src")


@pytest.fixture(scope="module")
def harness():
    if shutil.which("g++") is None or not os.path.isdir(LIB):
        pytest.skip("needs g++ and the TFLM library (run 'pio run' in firmware/aqi_classifier)")
    subprocess.run(["bash", os.path.join(HOST, "build.sh")], check=True, capture_output=True)
    return os.path.join(HOST, "build", "harness")


@pytest.mark.slow
def test_board_tflm_is_bit_identical_to_the_laptop_reference(harness, tmp_path):
    # Arrange
    raw = pd.read_csv(vod.PUBLIC)[["PM2.5", "PM10", "temperature", "humidity"]].to_numpy().astype(np.float32)
    raw.tofile(tmp_path / "raw.bin")
    ref = vod.Reference()
    # Act
    subprocess.run([harness, tmp_path / "raw.bin", tmp_path / "u.bin", tmp_path / "p.bin"], check=True, capture_output=True)
    board_u = np.fromfile(tmp_path / "u.bin", np.float32).reshape(-1, 4)
    board_p = np.fromfile(tmp_path / "p.bin", np.float32).reshape(-1, 6)
    laptop_u = model_input(raw, ref.in_mean, ref.in_inv)
    laptop_p = np.empty_like(board_p)
    for i, u in enumerate(laptop_u):
        ref.interp.set_tensor(ref.inp, u.reshape(1, -1))
        ref.interp.invoke()
        laptop_p[i] = ref.interp.get_tensor(ref.out)[0]
    # Assert
    assert len(board_p) == len(raw) > 20000
    assert np.array_equal(board_u.view(np.uint32), laptop_u.view(np.uint32))
    assert np.array_equal(board_p.view(np.uint32), laptop_p.view(np.uint32))
