#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "$project_root/.venv/bin/python" ]]; then
    python_executable="$project_root/.venv/bin/python"
else
    python_executable="${PYTHON_EXECUTABLE:-python3}"
fi

cd "$project_root"
"$python_executable" -m pytest -q tests
