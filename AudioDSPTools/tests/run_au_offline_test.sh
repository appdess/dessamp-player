#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/AudioDSPTools/tests/test_au_offline.mm"
build_dir="${TMPDIR:-/tmp}/dessmetal-au-offline-test"
binary="$build_dir/test_au_offline"

mkdir -p "$build_dir"
clang++ -std=c++17 -Wall -Wextra -Werror \
  -framework AudioToolbox -framework AudioUnit -framework CoreFoundation \
  "$source_file" -o "$binary"
"$binary"
