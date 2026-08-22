"""Benchmark one controlled custom-kernel launch configuration on CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_softmax import run_custom_case
from benchmarks.config import SUPPORTED_BLOCK_SIZES, softmax_benchmark_registry
from benchmarks.summarize_results import (
    load_raw_benchmark_csv,
    select_launch_configuration,
    summarize_raw_records,
)
from cuda_attention.benchmark import (
    collect_cuda_run_metadata,
    raw_benchmark_records,
    write_raw_benchmark_csv,
)
from cuda_attention.operator import CudaExtensionUnavailableError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--block-size", type=int, choices=SUPPORTED_BLOCK_SIZES)
    mode.add_argument("--select-from", type=Path, nargs=3, metavar=("CSV128", "CSV256", "CSV512"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.select_from is not None:
        try:
            summaries = []
            for path in arguments.select_from:
                summaries.extend(summarize_raw_records(load_raw_benchmark_csv(path)))
            selection = select_launch_configuration(summaries)
        except (FileNotFoundError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 0

    if not torch.cuda.is_available():
        print("launch tuning requires Linux with an NVIDIA GPU", file=sys.stderr)
        return 2

    block_size = arguments.block_size
    assert block_size is not None
    metadata = collect_cuda_run_metadata(PROJECT_ROOT)
    records: list[dict[str, object]] = []
    try:
        for config in softmax_benchmark_registry(block_size=block_size):
            samples_us = run_custom_case(config)
            records.extend(
                raw_benchmark_records(
                    metadata=metadata,
                    implementation_description=(
                        "warp-reduction custom CUDA fused scale + mask + softmax; "
                        f"block_size={block_size}"
                    ),
                    sequence_length=config.sequence_length,
                    rows=config.rows,
                    columns=config.columns,
                    dtype=config.dtype,
                    warmups=config.warmups,
                    iterations=config.iterations,
                    samples_us=samples_us,
                    launch_block_size=block_size,
                    compile_warmups=0,
                )
            )
    except CudaExtensionUnavailableError as error:
        print(str(error), file=sys.stderr)
        return 2

    write_raw_benchmark_csv(arguments.output, records)
    print(f"wrote {len(records)} raw samples to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
