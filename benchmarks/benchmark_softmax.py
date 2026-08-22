"""Benchmark equivalent causal scaled-softmax implementations on CUDA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from collections.abc import Callable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.config import SoftmaxBenchmarkConfig, softmax_benchmark_registry
from cuda_attention.benchmark import (
    collect_cuda_run_metadata,
    make_score_tensor,
    raw_benchmark_records,
    time_cuda_callable,
    write_raw_benchmark_csv,
)
from cuda_attention.operator import (
    CudaExtensionUnavailableError,
    cuda_extension_available,
    fused_causal_softmax,
)
from cuda_attention.reference import causal_allowed_mask, causal_scaled_softmax


def pytorch_eager_causal_softmax(
    scores: torch.Tensor,
    scale: float,
    allowed: torch.Tensor,
) -> torch.Tensor:
    """Compose the same scale, mask, and softmax work in eager PyTorch."""

    return torch.softmax((scores * scale).masked_fill(~allowed, -torch.inf), dim=-1)


def prepare_eager_case(
    config: SoftmaxBenchmarkConfig,
) -> tuple[torch.Tensor, Callable[[], torch.Tensor]]:
    """Allocate controlled CUDA inputs outside the timed operation."""

    scores = make_score_tensor(
        config.sequence_length,
        batch_heads=config.batch_heads,
        seed=config.seed,
        dtype=torch.float32,
        device="cuda",
    )
    allowed = causal_allowed_mask(
        config.rows,
        config.sequence_length,
        device="cuda",
    )
    scale = 1.0 / math.sqrt(64)
    return scores, lambda: pytorch_eager_causal_softmax(scores, scale, allowed)


def run_eager_case(config: SoftmaxBenchmarkConfig) -> list[float]:
    """Measure one eager case and return raw CUDA-event samples."""

    _, operation = prepare_eager_case(config)
    return time_cuda_callable(
        operation,
        warmups=config.warmups,
        iterations=config.iterations,
    )


def prepare_custom_case(
    config: SoftmaxBenchmarkConfig,
) -> tuple[torch.Tensor, Callable[[], torch.Tensor]]:
    """Allocate the same controlled scores for the fused custom operation."""

    if not cuda_extension_available():
        raise CudaExtensionUnavailableError(
            "custom benchmark requires the compiled cuda_attention._C extension"
        )
    scores = make_score_tensor(
        config.sequence_length,
        batch_heads=config.batch_heads,
        seed=config.seed,
        dtype=torch.float32,
        device="cuda",
    )
    scale = 1.0 / math.sqrt(64)
    return scores, lambda: fused_causal_softmax(scores, scale)


def run_custom_case(config: SoftmaxBenchmarkConfig) -> list[float]:
    """Validate once, then measure only the fused custom operation."""

    scores, operation = prepare_custom_case(config)
    scale = 1.0 / math.sqrt(64)
    actual = operation()
    expected = causal_scaled_softmax(scores, scale)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.cuda.synchronize()
    return time_cuda_callable(
        operation,
        warmups=config.warmups,
        iterations=config.iterations,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument(
        "--implementation",
        choices=("eager", "custom", "both"),
        default="both",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA benchmark requires Linux with an NVIDIA GPU.", file=sys.stderr)
        return 2

    registry = softmax_benchmark_registry()
    if arguments.sequence_length is not None:
        registry = (
            SoftmaxBenchmarkConfig(
                sequence_length=arguments.sequence_length,
                warmups=arguments.warmups or registry[0].warmups,
                iterations=arguments.iterations or registry[0].iterations,
            ),
        )

    implementations = {
        "eager": (("PyTorch eager scale + causal mask + softmax", run_eager_case),),
        "custom": (("warp-reduction custom CUDA fused scale + mask + softmax", run_custom_case),),
        "both": (
            ("PyTorch eager scale + causal mask + softmax", run_eager_case),
            ("warp-reduction custom CUDA fused scale + mask + softmax", run_custom_case),
        ),
    }
    metadata = collect_cuda_run_metadata(PROJECT_ROOT)
    records: list[dict[str, object]] = []
    try:
        for config in registry:
            for description, runner in implementations[arguments.implementation]:
                samples_us = runner(config)
                records.extend(
                    raw_benchmark_records(
                        metadata=metadata,
                        implementation_description=description,
                        sequence_length=config.sequence_length,
                        rows=config.rows,
                        columns=config.columns,
                        dtype=config.dtype,
                        warmups=config.warmups,
                        iterations=config.iterations,
                        samples_us=samples_us,
                    )
                )
    except CudaExtensionUnavailableError as error:
        print(str(error), file=sys.stderr)
        return 2
    if arguments.output is not None:
        write_raw_benchmark_csv(arguments.output, records)
        print(f"wrote {len(records)} raw samples to {arguments.output}")
    else:
        for record in records:
            print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
