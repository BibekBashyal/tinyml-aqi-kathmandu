import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "code", "04_tinyml_model"))

import pytest  # noqa: E402

from fake_firmware import FakeFirmware  # noqa: E402


@pytest.fixture
def fake_device_factory():
    devices = []

    def make(**kwargs):
        dev = FakeFirmware(**kwargs)
        devices.append(dev)
        return dev

    yield make
    for dev in devices:
        dev.close()


@pytest.fixture
def fake_device(fake_device_factory):
    return fake_device_factory()


@pytest.fixture
def fake_bench_factory():
    devices = []

    def make(mode):
        dev = FakeFirmware(bench_mode=mode)
        devices.append(dev)
        return dev

    yield make
    for dev in devices:
        dev.close()
