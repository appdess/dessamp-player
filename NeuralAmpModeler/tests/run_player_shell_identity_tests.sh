#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/NeuralAmpModeler/tests/test_player_shell_identity.cpp"
build_dir="${TMPDIR:-/tmp}/dessamp-player-shell-identity-test"
binary="$build_dir/test_player_shell_identity"

mkdir -p "$build_dir"
clang++ -std=c++17 -Wall -Wextra -Werror "$source_file" -o "$binary"
"$binary"
