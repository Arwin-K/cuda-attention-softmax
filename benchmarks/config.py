"""Authoritative shapes and controls for fused-softmax benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


BENCHMARK_SEQUENCE_LENGTHS = (128, 255, 512, 768, 1024, 1536, 2048)
DEFAULT_BATCH_HEADS = 8
DEFAULT_DTYPE = "float32"
DEFAULT_WARMUPS = 25
DEFAULT_ITERATIONS = 100
DEFAULT_SEED = 2026

# Aggregate summaries are derived from raw per-iteration CUDA-event samples;
# they are never entered manually into this configuration.


@dataclass(frozen=True)
class SoftmaxBenchmarkConfig:
    """One controlled flattened-score benchmark case.

    Keeping rows derived from ``batch_heads * sequence_length`` ensures every
    implementation measures the same amount of attention-score work.
    """

    sequence_length: int
    batch_heads: int = DEFAULT_BATCH_HEADS
    dtype: str = DEFAULT_DTYPE
    warmups: int = DEFAULT_WARMUPS
    iterations: int = DEFAULT_ITERATIONS
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        for name, value in (
            ("sequence_length", self.sequence_length),
            ("batch_heads", self.batch_heads),
            ("warmups", self.warmups),
            ("iterations", self.iterations),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.dtype != "float32":
            raise ValueError("the primary benchmark currently requires float32")

    @property
    def rows(self) -> int:
        return self.batch_heads * self.sequence_length

    @property
    def columns(self) -> int:
        return self.sequence_length


def softmax_benchmark_registry() -> tuple[SoftmaxBenchmarkConfig, ...]:
    """Return the complete primary benchmark matrix in stable order."""

    return tuple(
        SoftmaxBenchmarkConfig(sequence_length=sequence_length)
        for sequence_length in BENCHMARK_SEQUENCE_LENGTHS
    )
