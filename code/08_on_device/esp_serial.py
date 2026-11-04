"""
esp_serial.py
=============
Minimal line-oriented serial port for the ESP32-S3-DevKitC-1 UART port (CH343 bridge).

Uses a raw os.open + termios configuration instead of pyserial: on this bridge pyserial's
open/flush sequence loses the first bytes and sometimes whole lines (bench 2026-10-30).
Opening the port does NOT reset the board (DTR/RTS are left alone), so a running
firmware keeps running. Input already queued in the kernel when the port opens is discarded,
so a reply left over from an earlier run can never be read as an answer to a new request, and the
port is opened exclusively: the open fails if another process already holds the device (checked via
/proc), and TIOCEXCL stops new openers, so a second reader such as 'pio device monitor' cannot steal
half of the replies. A disconnected device (hang-up) raises OSError instead of timing out silently.

Author: Bibek Bashyal, Kathmandu University 2026
"""

import errno
import fcntl
import os
import select
import termios
import time

READ_CHUNK = 4096
POLL_S = 0.5


def other_openers(dev: str) -> list[int]:
    """PIDs of other processes that have the device open (best effort, readable /proc entries only)."""
    target, me, pids = os.path.realpath(dev), os.getpid(), []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            fds = os.listdir(f"/proc/{entry}/fd")
            if any(os.path.realpath(f"/proc/{entry}/fd/{fd}") == target for fd in fds):
                pids.append(int(entry))
        except OSError:   # process gone or not ours
            continue
    return pids


class EspSerial:
    def __init__(self, dev: str = "/dev/ttyACM0", baud: int = 115200) -> None:
        busy = other_openers(dev)
        if busy:
            raise RuntimeError(f"{dev} is already open by process(es) {busy}: stop the serial capture or monitor first")
        self.fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(self.fd)
        attrs[0] = 0                                         # iflag: raw
        attrs[1] = 0                                         # oflag: raw
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[3] = 0                                         # lflag: no echo, no canonical mode
        attrs[4] = attrs[5] = getattr(termios, f"B{baud}")
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        termios.tcflush(self.fd, termios.TCIFLUSH)           # drop stale input from earlier runs
        fcntl.ioctl(self.fd, termios.TIOCEXCL)               # no second reader while we run
        self._buf = b""

    def __enter__(self) -> "EspSerial":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write_line(self, text: str) -> None:
        os.write(self.fd, (text.rstrip("\n") + "\n").encode())

    def read_line(self, timeout_s: float = 2.0) -> str | None:
        """One complete line (without newline) or None on timeout."""
        deadline = time.monotonic() + timeout_s
        while True:
            if b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                return line.replace(b"\r", b"").decode("latin1", "replace").strip()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self.fd], [], [], min(remaining, POLL_S))
            if not ready:
                continue
            try:
                chunk = os.read(self.fd, READ_CHUNK)
            except BlockingIOError:
                continue
            if not chunk:   # select said readable but nothing came: the tty was hung up
                raise OSError(errno.EIO, "serial port hung up (USB bridge disconnected?)")
            self._buf += chunk

    def wait_for(self, prefixes: tuple[str, ...], timeout_s: float = 5.0, log: list | None = None) -> str | None:
        """Read lines until one starts with any of prefixes; other lines go to log if given."""
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            line = self.read_line(remaining)
            if line is None:
                return None
            if line.startswith(prefixes):
                return line
            if log is not None:
                log.append(line)

    def close(self) -> None:
        """Idempotent and never raises: the exclusive flag dies with the tty at the last close."""
        fd, self.fd = self.fd, -1
        if fd < 0:
            return
        try:
            os.close(fd)
        except OSError:
            pass
