#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model.dess|player-manifest.json> [--snapshot] [--test-boost-bank] [--test-verdant-tone] [--expect-control-label index=label] [--expect-input-calibration-reference-dbu value] [--ir-path /absolute/cabinet.wav] [--component-binary /absolute/DessAmp]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/NeuralAmpModeler/tests/test_au_player.mm"
build_dir="${TMPDIR:-/tmp}/dessamp-au-player-test"
binary="$build_dir/test_au_player"

mkdir -p "$build_dir"
clang++ -std=c++17 -Wall -Wextra -Werror -Wno-unused-parameter \
  -I "$repo_root/NeuralAmpModelerCore" \
  -I "$repo_root/NeuralAmpModelerCore/Dependencies/nlohmann" \
  -I "$repo_root/NeuralAmpModelerCore/Dependencies/eigen" \
  -framework AudioToolbox -framework AudioUnit -framework CoreFoundation \
  "$source_file" -o "$binary"
"$binary" "$@"
