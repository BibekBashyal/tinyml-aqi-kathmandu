#!/usr/bin/env bash
# Build host_tflm/harness from the board's own TFLM sources with the firmware's arithmetic flags.
# Needs g++ and the library downloaded by PlatformIO (run 'pio run' in firmware/aqi_classifier once).
# Usage: code/08_on_device/host_tflm/build.sh [build_dir]     (default: host_tflm/build)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
LIB="$ROOT/firmware/aqi_classifier/.pio/libdeps/esp32-s3-devkitc-1/TensorFlowLite_ESP32/src"
OUT="${1:-$HERE/build}"
[ -d "$LIB/tensorflow" ] || { echo "TFLM sources not found at $LIB: run 'pio run' in firmware/aqi_classifier first" >&2; exit 2; }
# keep in step with build_flags in firmware/aqi_classifier/platformio.ini
FLAGS=(-std=c++17 -O2 -w -fpermissive -fno-exceptions -ffp-contract=off -DTFLITE_SINGLE_ROUNDING=1 -DTF_LITE_STATIC_MEMORY
       -I"$LIB" -I"$LIB/third_party/gemmlowp" -I"$LIB/third_party/flatbuffers/include" -I"$LIB/third_party/ruy"
       -I"$ROOT/firmware/aqi_classifier")
g++ -Wno-template-body -x c++ -fsyntax-only /dev/null 2>/dev/null && FLAGS+=(-Wno-template-body)
# objects live in a folder keyed on everything that changes code generation, so a flag, compiler,
# library or source-list change can never reuse stale objects (code review 2026-11-03, round 3)
KEY="$({ printf '%s\n' "${FLAGS[@]}"; g++ --version | head -1; cat "$LIB/../library.properties" "$HERE/sources.txt"; } | sha1sum | cut -c1-12)"
OBJ="$OUT/obj-$KEY"
mkdir -p "$OBJ"
find "$OUT" -maxdepth 1 -type d -name 'obj*' ! -path "$OBJ" -exec rm -rf {} +
compile_one() {
  local src="$1" obj="$OBJ/$(echo "$1" | tr '/' '_').o"
  [ "$obj" -nt "$LIB/$src" ] && return 0
  g++ "${FLAGS[@]}" -c "$LIB/$src" -o "$obj"
}
export -f compile_one; export LIB OUT OBJ
export FLAGS_STR="$(printf '%q ' "${FLAGS[@]}")"
grep -v '^#' "$HERE/sources.txt" | xargs -P "$(nproc)" -I{} bash -c 'eval "FLAGS=($FLAGS_STR)"; compile_one "$1"' _ {}
g++ "${FLAGS[@]}" -c "$HERE/harness.cpp" -o "$OUT/harness.o"   # rebuilt every time: aqi_model.h may have changed
# a static archive lets the linker take only the objects the harness needs (TFLM's test helpers stay out)
rm -f "$OUT/libtflm.a" && ar rcs "$OUT/libtflm.a" "$OBJ"/*.o
g++ "$OUT/harness.o" "$OUT/libtflm.a" -o "$OUT/harness"
echo "built $OUT/harness"
