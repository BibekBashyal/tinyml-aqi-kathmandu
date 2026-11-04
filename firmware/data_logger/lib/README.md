# Patched framework libraries

PlatformIO uses a library in the project's `lib/` folder instead of the framework's built-in one with
the same name. Both firmware projects carry the same patched copy (a shared folder outside the project
breaks the build: the project path contains "(1)" and is passed to the shell unquoted).

## SD (from framework-arduinoespressif32 3.20017.241212, `libraries/SD`, version 2.0.0)

`src/sd_diskio.cpp`, function `sdWriteSector`: the card's data-response token (`xxx0sss1 & 0x1F`) is
`0x05` accepted, `0x0B` CRC error, `0x0D` write error (SD Physical Layer Simplified Specification,
SPI mode, data response token). Upstream compared it with `0x0A` and `0x0C`. A block the card
rejected for a CRC error, or a busy timeout where `sdWriteBytes` returns 0, therefore fell through to
`SEND_STATUS` and was reported as written. On a long unattended log that is silent data loss. The
patch retries on `0x0B` and fails on anything other than `0x05`. The same wrong constant in the
CRC-recovery branch of `sdWriteSectors` is corrected too. Found in the 2026-11-03 code review.

When the framework is updated, check whether upstream fixed it, and re-copy with the patch if not.
