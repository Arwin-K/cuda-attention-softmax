#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "$project_root/.venv/bin/python" ]]; then
    python_executable="$project_root/.venv/bin/python"
else
    python_executable="${PYTHON_EXECUTABLE:-python3}"
fi

# Hiding CUDA makes this workflow capability-independent even if a reviewer
# happens to invoke it on a GPU host. It validates the CPU/reference and static
# contracts; it is never evidence that the CUDA extension compiled or ran.
export CUDA_VISIBLE_DEVICES=""

cd "$project_root"
echo "CPU_ONLY_STEP: environment"
"$python_executable" scripts/check_environment.py

echo "CPU_ONLY_STEP: package import"
"$python_executable" -c "import cuda_attention; print('cuda_attention import: PASS')"

echo "CPU_ONLY_STEP: deterministic Colab notebook"
"$python_executable" scripts/generate_colab_notebook.py --check

echo "CPU_ONLY_STEP: complete CPU/static test suite"
"$python_executable" -m pytest -q tests

echo "CPU_ONLY_STEP: imported result schema"
"$python_executable" scripts/audit_results.py \
    results/runs/2026-08-23_tesla-t4_ca87722/artifacts >/dev/null

echo "CPU_ONLY_STEP: figure provenance"
"$python_executable" -c "import json; from pathlib import Path; from scripts.generate_result_provenance import generate; root=Path('results/runs/2026-08-23_tesla-t4_ca87722'); assert json.loads((root/'figure_provenance.json').read_text()) == generate(root/'artifacts'); print('figure provenance: PASS')"

echo "CPU_ONLY_REPRODUCIBILITY: PASS"
