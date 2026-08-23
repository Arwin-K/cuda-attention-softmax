#!/usr/bin/env sh
set -eu

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/.." && pwd)
OUTPUT_DIRECTORY=${1:-"$PROJECT_ROOT/results/raw/nsight"}
PYTHON_EXECUTABLE=${PYTHON:-python}
REPORT_BASE="$OUTPUT_DIRECTORY/fused_causal_softmax"
REPORT_PATH="$REPORT_BASE.ncu-rep"
CSV_PATH="$OUTPUT_DIRECTORY/fused_causal_softmax_raw.csv"

mkdir -p "$OUTPUT_DIRECTORY"
if ! command -v ncu >/dev/null 2>&1; then
    printf '%s\n' "NSIGHT COMPUTE UNAVAILABLE: ncu is not installed" >&2
    exit 2
fi
if [ -e "$REPORT_PATH" ] || [ -e "$CSV_PATH" ]; then
    printf '%s\n' "Refusing to overwrite existing Nsight artifacts" >&2
    exit 2
fi

# The explicit repository path fixes imports when ncu launches Python from a
# profiler-owned working directory, including the observed Colab failure mode.
PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    ncu \
    --set basic \
    --target-processes all \
    --kernel-name 'regex:.*fused_causal_softmax_kernel.*' \
    --export "$REPORT_BASE" \
    "$PYTHON_EXECUTABLE" "$SCRIPT_DIRECTORY/profile_cuda.py"

ncu --import "$REPORT_PATH" --csv --page raw > "$CSV_PATH"
printf '%s\n' "$REPORT_PATH" "$CSV_PATH"
