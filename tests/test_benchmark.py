"""CPU-safe checks for benchmark controls and derived shapes."""

import csv
from unittest.mock import patch

import pytest
import torch

from benchmarks.config import (
    BENCHMARK_SEQUENCE_LENGTHS,
    SoftmaxBenchmarkConfig,
    softmax_benchmark_registry,
)
from benchmarks.benchmark_softmax import (
    prepare_custom_case,
    pytorch_eager_causal_softmax,
)
from benchmarks.summarize_results import (
    load_raw_benchmark_csv,
    validate_comparison_pair,
)
from cuda_attention.benchmark import (
    RAW_BENCHMARK_FIELDS,
    raw_benchmark_records,
    time_cuda_callable,
    write_raw_benchmark_csv,
)
from cuda_attention.operator import CudaExtensionUnavailableError
from cuda_attention.reference import causal_allowed_mask, causal_scaled_softmax


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


def test_eager_benchmark_operation_matches_reference_on_cpu() -> None:
    generator = torch.Generator().manual_seed(37)
    scores = torch.randn(12, 6, generator=generator)
    allowed = causal_allowed_mask(12, 6)

    actual = pytorch_eager_causal_softmax(scores, scale=0.125, allowed=allowed)
    expected = causal_scaled_softmax(scores, scale=0.125)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_custom_benchmark_never_substitutes_an_unbuilt_extension() -> None:
    config = SoftmaxBenchmarkConfig(sequence_length=128)

    with (
        patch("benchmarks.benchmark_softmax.cuda_extension_available", return_value=False),
        pytest.raises(CudaExtensionUnavailableError, match="compiled"),
    ):
        prepare_custom_case(config)


def test_raw_benchmark_csv_preserves_samples_and_provenance(tmp_path) -> None:
    metadata = {
        "git_commit": "a" * 40,
        "gpu_name": "test GPU",
        "compute_capability": "9.0",
        "pytorch_version": "test torch",
        "cuda_version": "test CUDA",
        "timestamp": "2026-08-15T00:00:00+00:00",
    }
    records = raw_benchmark_records(
        metadata=metadata,
        implementation_description="test implementation",
        sequence_length=128,
        rows=1024,
        columns=128,
        dtype="float32",
        warmups=2,
        iterations=3,
        samples_us=[1.0, 2.0, 3.0],
    )
    output_path = tmp_path / "raw.csv"

    write_raw_benchmark_csv(output_path, records)

    with output_path.open(newline="", encoding="utf-8") as output_file:
        saved = list(csv.DictReader(output_file))
    assert tuple(saved[0]) == RAW_BENCHMARK_FIELDS
    assert [row["sample_us"] for row in saved] == ["1.0", "2.0", "3.0"]
    assert all(row["git_commit"] == "a" * 40 for row in saved)
    assert all(row["gpu_name"] == "test GPU" for row in saved)


def test_raw_record_count_must_match_iterations() -> None:
    with pytest.raises(ValueError, match="sample count"):
        raw_benchmark_records(
            metadata={},
            implementation_description="test",
            sequence_length=128,
            rows=1024,
            columns=128,
            dtype="float32",
            warmups=2,
            iterations=2,
            samples_us=[1.0],
        )


def test_comparison_preflight_requires_matching_controls_and_distinct_commits(
    tmp_path,
) -> None:
    common_metadata = {
        "gpu_name": "test GPU",
        "compute_capability": "9.0",
        "pytorch_version": "test torch",
        "cuda_version": "test CUDA",
        "timestamp": "2026-08-20T00:00:00+00:00",
    }
    baseline = raw_benchmark_records(
        metadata={**common_metadata, "git_commit": "a" * 40},
        implementation_description="row serial",
        sequence_length=128,
        rows=1024,
        columns=128,
        dtype="float32",
        warmups=2,
        iterations=2,
        samples_us=[4.0, 5.0],
    )
    candidate = raw_benchmark_records(
        metadata={**common_metadata, "git_commit": "b" * 40},
        implementation_description="block parallel",
        sequence_length=128,
        rows=1024,
        columns=128,
        dtype="float32",
        warmups=2,
        iterations=2,
        samples_us=[2.0, 3.0],
    )
    baseline_path = tmp_path / "baseline.csv"
    candidate_path = tmp_path / "candidate.csv"
    write_raw_benchmark_csv(baseline_path, baseline)
    write_raw_benchmark_csv(candidate_path, candidate)

    report = validate_comparison_pair(
        load_raw_benchmark_csv(baseline_path),
        load_raw_benchmark_csv(candidate_path),
    )

    assert report["baseline_commit"] == "a" * 40
    assert report["candidate_commit"] == "b" * 40
    assert report["controlled_cases"] == 1


def test_comparison_preflight_rejects_missing_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_raw_benchmark_csv(tmp_path / "missing.csv")
