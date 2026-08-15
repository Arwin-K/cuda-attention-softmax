"""CPU-safe checks for benchmark controls and derived shapes."""

import pytest
import torch

from benchmarks.config import (
    BENCHMARK_SEQUENCE_LENGTHS,
    SoftmaxBenchmarkConfig,
    softmax_benchmark_registry,
)
from cuda_attention.benchmark import time_cuda_callable


def test_softmax_registry_contains_required_shapes_in_order() -> None:
    registry = softmax_benchmark_registry()

    assert tuple(case.sequence_length for case in registry) == (
        128,
        255,
        512,
        768,
        1024,
        1536,
        2048,
    )
    assert tuple(case.sequence_length for case in registry) == BENCHMARK_SEQUENCE_LENGTHS
    assert all(case.rows == 8 * case.sequence_length for case in registry)
    assert all(case.columns == case.sequence_length for case in registry)
    assert all(case.dtype == "float32" for case in registry)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sequence_length": 0},
        {"sequence_length": 128, "batch_heads": 0},
        {"sequence_length": 128, "warmups": 0},
        {"sequence_length": 128, "iterations": 0},
        {"sequence_length": 128, "seed": -1},
        {"sequence_length": 128, "dtype": "float64"},
    ],
)
def test_softmax_config_rejects_invalid_controls(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SoftmaxBenchmarkConfig(**overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(("warmups", "iterations"), [(0, 1), (1, 0), (-1, 1)])
def test_cuda_timer_rejects_nonpositive_counts(warmups: int, iterations: int) -> None:
    with pytest.raises(ValueError):
        time_cuda_callable(lambda: None, warmups=warmups, iterations=iterations)


def test_cuda_timer_does_not_fall_back_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="NVIDIA GPU"):
        time_cuda_callable(lambda: None, warmups=1, iterations=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_timer_returns_one_positive_sample_per_iteration() -> None:
    tensor = torch.ones(1024, device="cuda")

    samples = time_cuda_callable(
        lambda: tensor.add_(1.0),
        warmups=2,
        iterations=3,
    )

    assert len(samples) == 3
    assert all(sample > 0.0 for sample in samples)
