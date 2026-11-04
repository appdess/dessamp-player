#!/bin/sh
# Build the experimental CPU-optimized renderer for the HG12 RAM trial.
set -eu
: "${ZIG:?Set ZIG to the Zig 0.13.0 executable}"
[ "$#" = 1 ] || { echo 'Usage: build_renderer.sh NEW_PRIVATE_OUTPUT_DIRECTORY' >&2; exit 2; }
[ "$("$ZIG" version)" = 0.13.0 ] || { echo 'This HG12 build uses Zig 0.13.0' >&2; exit 2; }
out=$1
mkdir "$out"
here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
core=$(CDPATH='' cd -- "$here/../../../NeuralAmpModelerCore" && pwd)
bridge=$(CDPATH='' cd -- "$here/../../targets/hg12/runtime" && pwd)/fire.cpp
compile() {
    src=$1
    obj=$2
    shift 2
    "$ZIG" c++ -target arm-linux-gnueabihf.2.17 -mcpu=cortex_a17 -std=c++17 -DNDEBUG \
        -DNAM_SAMPLE_FLOAT -DEIGEN_DONT_PARALLELIZE -fvisibility=hidden -fPIC \
        -ffunction-sections -fdata-sections -I"$core" -I"$core/Dependencies/eigen" \
        -I"$core/Dependencies/nlohmann" "$@" -c "$src" -o "$out/$obj"
}
compile "$core/NAM/dsp.cpp" dsp.o -O2
compile "$core/NAM/a2_fast.cpp" a2.o -O3 -DNAM_HG12_NEON_FRAMES=3
compile "$bridge" fire.o -O2
"$ZIG" c++ -target arm-linux-gnueabihf.2.17 -mcpu=cortex_a17 -O3 -std=c++17 -DNDEBUG \
    -DNAM_SAMPLE_FLOAT -DEIGEN_DONT_PARALLELIZE -fvisibility=hidden -fPIC \
    -ffunction-sections -fdata-sections -shared \
    -Wl,--gc-sections -Wl,-z,defs -Wl,-soname,libdessfire_hg12.so.1 \
    "$out/dsp.o" "$out/a2.o" "$out/fire.o" -o "$out/libdessfire_hg12.so.1"
