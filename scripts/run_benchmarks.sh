#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable="${PYTHON_EXECUTABLE:-$project_root/.venv/bin/python}"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 OUTPUT_CSV [benchmark options]" >&2
    exit 64
fi

output_path="$1"
shift

cd "$project_root"
"$python_executable" scripts/check_environment.py --require-cuda
"$python_executable" benchmarks/benchmark_softmax.py \
    --output "$output_path" \
    "$@"
