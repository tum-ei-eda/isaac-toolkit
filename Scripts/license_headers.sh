#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:---fix}"
template="${repo_root}/Scripts/apache-2.tmpl"

if [[ "$mode" != "--fix" && "$mode" != "--check" ]]; then
    echo "Usage: $0 [--fix|--check]" >&2
    exit 2
fi

command -v licenseheaders >/dev/null || {
    echo "licenseheaders is required; install it with: python -m pip install licenseheaders" >&2
    exit 127
}

run_licenseheaders() {
    local root="$1"
    licenseheaders -t "${template}" -y 2026 \
        -o "TUM Department of Electrical and Computer Engineering" \
        -n "ISAAC Toolkit" \
        -u "https://github.com/tum-ei-eda/isaac-toolkit.git" \
        -d "${root}/isaac_toolkit/" -E py
    licenseheaders -t "${template}" -y 2026 \
        -o "TUM Department of Electrical and Computer Engineering" \
        -n "ISAAC Toolkit" \
        -u "https://github.com/tum-ei-eda/isaac-toolkit.git" \
        -d "${root}/tests/" -E py
}

if [[ "$mode" == "--fix" ]]; then
    run_licenseheaders "$repo_root"
    echo "License headers updated. Review and commit the changes."
    exit 0
fi

tmp_root="$(mktemp -d)"
trap 'rm -rf "$tmp_root"' EXIT
cp -a "${repo_root}/isaac_toolkit" "${tmp_root}/isaac_toolkit"
cp -a "${repo_root}/tests" "${tmp_root}/tests"
run_licenseheaders "$tmp_root"
if ! diff -ru "${repo_root}/isaac_toolkit" "${tmp_root}/isaac_toolkit" || \
   ! diff -ru "${repo_root}/tests" "${tmp_root}/tests"; then
    echo "License headers need updating. Run Scripts/license_headers.sh --fix." >&2
    exit 1
fi
echo "License header check passed."
