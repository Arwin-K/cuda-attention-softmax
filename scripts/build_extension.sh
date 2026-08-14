#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "$project_root/.venv/bin/python" ]]; then
    python_executable="$project_root/.venv/bin/python"
else
    python_executable="${PYTHON_EXECUTABLE:-python3}"
fi

cd "$project_root"

if [[ "$(uname -s)" != "Linux" ]]; then
    "$python_executable" scripts/check_environment.py
    echo "SKIP: CUDA extension builds require Linux, an NVIDIA GPU, and CUDA."
    exit 0
fi

"$python_executable" scripts/check_environment.py --require-cuda
CUDA_ATTENTION_BUILD_CUDA=1 "$python_executable" setup.py build_ext --inplace
