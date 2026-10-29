#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable="${PYTHON_EXECUTABLE:-$project_root/.venv/bin/python}"
if [[ ! -x "$python_executable" ]]; then
    python_executable="python3"
fi

run="$project_root/results/runs/2026-08-23_tesla-t4_ca87722"
if [[ $# -gt 1 ]]; then
    echo "usage: $0 [NEW_OUTPUT_DIRECTORY]" >&2
    exit 64
fi
if [[ $# -eq 1 ]]; then
    output_dir="$1"
    if [[ -e "$output_dir" ]]; then
        echo "output directory already exists: $output_dir" >&2
        exit 73
    fi
    mkdir -p "$output_dir"
else
    output_dir="$(mktemp -d "${TMPDIR:-/tmp}/cuda-softmax-reproduction.XXXXXX")"
fi
output_dir="$(cd "$output_dir" && pwd)"
export MPLCONFIGDIR="$output_dir/.matplotlib"

cd "$project_root"
"$python_executable" scripts/audit_results.py "$run/artifacts" --output "$output_dir/schema_audit.json" >/dev/null
"$python_executable" scripts/generate_result_provenance.py "$run/artifacts" --output "$output_dir/figure_provenance.json"
diff -u "$run/schema_audit.json" "$output_dir/schema_audit.json"
diff -u "$run/figure_provenance.json" "$output_dir/figure_provenance.json"

"$python_executable" benchmarks/compare_speedups.py \
    --softmax-raw "$run/artifacts/benchmarks/raw/softmax_raw.csv" \
    --attention-raw "$run/artifacts/attention/attention_raw.csv" \
    --output "$output_dir/speedups.csv"
"$python_executable" scripts/generate_figures.py \
    --summary "$run/artifacts/benchmarks/summaries/softmax_summary.csv" \
    --speedup-comparison "$output_dir/speedups.csv" \
    --output-directory "$output_dir/figures"
"$python_executable" benchmarks/generate_tables.py \
    --softmax-summary "$run/artifacts/benchmarks/summaries/softmax_summary.csv" \
    --attention-raw "$run/artifacts/attention/attention_raw.csv" \
    --speedup-comparison "$output_dir/speedups.csv" \
    --output-directory "$output_dir/tables"

echo "Published evidence audited and analysis regenerated: $output_dir"
