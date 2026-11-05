#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:---fix}"

if [[ "$mode" != "--fix" && "$mode" != "--check" ]]; then
    echo "Usage: $0 [--fix|--check]" >&2
    exit 2
fi

if ! command -v black >/dev/null; then
    echo "Black is required; install it with: python -m pip install black" >&2
    exit 127
fi

black_args=(--target-version py310 --quiet)
if [[ "$mode" == "--check" ]]; then
    black_args+=(--check --diff)
fi

mapfile -d '' python_files < <(find "${repo_root}/isaac_toolkit" "${repo_root}/tests" -name '*.py' -print0)
for python_file in "${python_files[@]}"; do
    black "${black_args[@]}" "$python_file"
done
