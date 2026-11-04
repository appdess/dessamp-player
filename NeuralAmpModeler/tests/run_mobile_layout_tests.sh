#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="$(mktemp -d "${TMPDIR:-/tmp}/dessamp-mobile-layout.XXXXXX")"
trap 'rm -rf "$out"' EXIT
clang++ -std=c++17 -Wno-deprecated-declarations \
  -I "$repo/iPlug2/IGraphics" -I "$repo/iPlug2/IPlug" -I "$repo/iPlug2/WDL" \
  -I "$repo/iPlug2/Dependencies/IGraphics/NanoSVG/src" \
  "$repo/NeuralAmpModeler/tests/test_mobile_layout.cpp" -o "$out/check"
"$out/check"
