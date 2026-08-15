"""CPU-safe checks for benchmark controls and derived shapes."""

import pytest

from benchmarks.config import (
    BENCHMARK_SEQUENCE_LENGTHS,
    SoftmaxBenchmarkConfig,
    softmax_benchmark_registry,
)


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
