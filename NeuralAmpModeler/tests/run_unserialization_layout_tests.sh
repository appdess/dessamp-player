#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/NeuralAmpModeler/tests/test_unserialization_layout.cpp"
build_dir="${TMPDIR:-/tmp}/dessmetal-unserialization-layout-test"
binary="$build_dir/test_unserialization_layout"

mkdir -p "$build_dir"
clang++ -std=c++17 -Wall -Wextra -Werror "$source_file" -o "$binary"
"$binary"
