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
from cuda_attention.benchmark import make_score_tensor, time_cuda_callable
from cuda_attention.reference import causal_allowed_mask


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--iterations", type=int)
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

    for config in registry:
        samples_us = run_eager_case(config)
        print(
            json.dumps(
                {
                    "implementation": "pytorch_eager",
                    "sequence_length": config.sequence_length,
                    "rows": config.rows,
                    "columns": config.columns,
                    "samples_us": samples_us,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
