#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 0 && ! ( $# -eq 2 && "$1" == "--component-binary" ) ]]; then
  echo "usage: $0 [--component-binary /absolute/DessAmp]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/NeuralAmpModeler/tests/test_au_state_migration.mm"
build_dir="${TMPDIR:-/tmp}/dessamp-au-state-migration-test"
binary="$build_dir/test_au_state_migration"

mkdir -p "$build_dir"
clang++ -std=c++17 -Wall -Wextra -Werror \
  -framework AudioToolbox -framework AudioUnit -framework CoreFoundation \
  "$source_file" -o "$binary"
"$binary" "$@"
