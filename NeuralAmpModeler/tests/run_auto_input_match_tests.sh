#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/dessamp-auto-input.XXXXXX")"
clang++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined \
  "$repo_root/NeuralAmpModeler/tests/test_auto_input_match.cpp" -o "$test_dir/test_auto_input_match"
"$test_dir/test_auto_input_match"
