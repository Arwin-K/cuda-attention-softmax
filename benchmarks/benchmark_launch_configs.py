"""Benchmark one controlled custom-kernel launch configuration on CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_softmax import run_custom_case
from benchmarks.config import SUPPORTED_BLOCK_SIZES, softmax_benchmark_registry
from cuda_attention.benchmark import (
    collect_cuda_run_metadata,
    raw_benchmark_records,
    write_raw_benchmark_csv,
)
from cuda_attention.operator import CudaExtensionUnavailableError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, choices=SUPPORTED_BLOCK_SIZES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        print("launch tuning requires Linux with an NVIDIA GPU", file=sys.stderr)
        return 2

    metadata = collect_cuda_run_metadata(PROJECT_ROOT)
    records: list[dict[str, object]] = []
    try:
        for config in softmax_benchmark_registry(block_size=arguments.block_size):
            samples_us = run_custom_case(config)
            records.extend(
                raw_benchmark_records(
                    metadata=metadata,
                    implementation_description=(
                        "warp-reduction custom CUDA fused scale + mask + softmax; "
                        f"block_size={arguments.block_size}"
                    ),
                    sequence_length=config.sequence_length,
                    rows=config.rows,
                    columns=config.columns,
                    dtype=config.dtype,
                    warmups=config.warmups,
                    iterations=config.iterations,
                    samples_us=samples_us,
                    launch_block_size=arguments.block_size,
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
