"""PyTorch Profiler regions for isolated softmax and complete attention."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from cuda_attention.attention import (
    custom_causal_attention,
    explicit_causal_attention,
    sdpa_causal_attention,
)
from cuda_attention.operator import DEFAULT_BLOCK_SIZE, fused_causal_softmax
from cuda_attention.reference import causal_scaled_softmax


PROFILE_REGION_NAMES = (
    "softmax/pytorch_eager",
    "softmax/custom_cuda",
    "attention/explicit_eager",
    "attention/custom_cuda",
    "attention/pytorch_sdpa",
)


@dataclass(frozen=True)
class ProfilerConfig:
    """A representative workload small enough to inspect in one trace."""

    sequence_length: int = 512
    batch: int = 1
    heads: int = 8
    head_dimension: int = 64
    warmups: int = 5
    repeats: int = 10
    block_size: int = DEFAULT_BLOCK_SIZE

    def __post_init__(self) -> None:
        if min(
            self.sequence_length,
            self.batch,
            self.heads,
            self.head_dimension,
            self.repeats,
        ) <= 0:
            raise ValueError("profile dimensions and repeats must be positive")
        if self.warmups < 0:
            raise ValueError("profile warmups must be nonnegative")


ProfileOperation = Callable[[], object]


def prepare_profile_operations(
    config: ProfilerConfig,
    *,
    device: torch.device | str,
    include_custom: bool,
) -> dict[str, ProfileOperation]:
    """Allocate shared inputs and return named operations for one profile run.

    Attention paths receive the same Q/K/V tensors. The isolated softmax paths
    receive the same already-materialized score matrix, so their profiler
    regions do not accidentally include QK^T.
    """

    generator = torch.Generator(device=device).manual_seed(2026)
    shape = (
        config.batch,
        config.heads,
        config.sequence_length,
        config.head_dimension,
    )
    query = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    key = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    value = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    scores = (query @ key.transpose(-2, -1)).reshape(-1, config.sequence_length)
    scale = 1.0 / math.sqrt(config.head_dimension)

    operations: dict[str, ProfileOperation] = {
        "softmax/pytorch_eager": lambda: causal_scaled_softmax(scores, scale),
        "attention/explicit_eager": lambda: explicit_causal_attention(
            query, key, value
        ),
        "attention/pytorch_sdpa": lambda: sdpa_causal_attention(query, key, value),
    }
    if include_custom:
        operations["softmax/custom_cuda"] = lambda: fused_causal_softmax(
            scores,
            scale,
            block_size=config.block_size,
        )
        operations["attention/custom_cuda"] = lambda: custom_causal_attention(
            query,
            key,
            value,
            block_size=config.block_size,
        )
    return {
        name: operations[name]
        for name in PROFILE_REGION_NAMES
        if name in operations
    }


def execute_profile_regions(
    operations: Mapping[str, ProfileOperation],
    *,
    repeats: int,
) -> dict[str, object]:
    """Execute named regions while preserving their last result.

    ``record_function`` creates human-readable parent ranges. CUDA kernels are
    asynchronous, so these labels describe logical ownership; device duration
    is obtained from the profiler's CUDA events, not Python wall-clock time.
    """

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    results: dict[str, object] = {}
    for _ in range(repeats):
        for name, operation in operations.items():
            with record_function(name):
                results[name] = operation()
    return results


def capture_cuda_profile(config: ProfilerConfig):
    """Warm up and capture one CUDA profile containing every comparison path."""

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch Profiler CUDA capture requires an NVIDIA GPU")
    operations = prepare_profile_operations(
        config,
        device="cuda",
        include_custom=True,
    )
    for _ in range(config.warmups):
        for operation in operations.values():
            operation()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as captured_profile:
        execute_profile_regions(operations, repeats=config.repeats)
        torch.cuda.synchronize()
    return captured_profile
