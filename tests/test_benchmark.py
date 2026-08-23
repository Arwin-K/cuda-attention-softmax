"""CPU-safe checks for benchmark controls and derived shapes."""

import csv
from unittest.mock import patch

import pytest
import torch

from benchmarks.config import (
    AttentionBenchmarkConfig,
    BENCHMARK_SEQUENCE_LENGTHS,
    SUPPORTED_BLOCK_SIZES,
    SoftmaxBenchmarkConfig,
    attention_benchmark_registry,
    softmax_benchmark_registry,
)
from benchmarks.benchmark_attention import (
    ATTENTION_RAW_FIELDS,
    attention_raw_records,
    checkpoint_attention_records,
    prepare_attention_operations,
    validate_attention_operations,
    write_attention_raw_csv,
)
from benchmarks.benchmark_softmax import (
    compile_causal_softmax,
    prepare_custom_case,
    pytorch_eager_causal_softmax,
)
from benchmarks.compare_speedups import (
    compare_kernel_and_attention_speedups,
    summarize_attention_records,
)
from benchmarks.summarize_results import (
    SUMMARY_FIELDS,
    load_raw_benchmark_csv,
    percentile,
    select_launch_configuration,
    summarize_raw_records,
    validate_comparison_pair,
    validate_framework_comparison,
    write_summary_csv,
)
from cuda_attention.benchmark import (
    RAW_BENCHMARK_FIELDS,
    raw_benchmark_records,
    run_untimed_warmups,
    time_cuda_callable,
    write_raw_benchmark_csv,
)
from cuda_attention.operator import CudaExtensionUnavailableError
from cuda_attention.plotting import load_summary_csv, metric_series
from cuda_attention.reference import causal_allowed_mask, causal_scaled_softmax
from scripts.generate_figures import generate_figures


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
    assert all(case.block_size == 256 for case in registry)


def test_attention_registry_uses_planned_complete_attention_shapes() -> None:
    registry = attention_benchmark_registry()

    assert tuple(case.sequence_length for case in registry) == BENCHMARK_SEQUENCE_LENGTHS
    assert all(case.shape == (1, 8, case.sequence_length, 64) for case in registry)
    assert all(case.dtype == "float32" for case in registry)


def test_attention_noncustom_paths_share_inputs_and_match_on_cpu() -> None:
    config = AttentionBenchmarkConfig(
        sequence_length=7,
        heads=2,
        head_dimension=8,
        warmups=1,
        iterations=2,
    )
    operations = prepare_attention_operations(
        config,
        device="cpu",
        include_custom=False,
    )

    validate_attention_operations(operations)

    assert set(operations) == {"explicit_eager", "pytorch_sdpa"}


def test_attention_raw_csv_preserves_full_workload_provenance(tmp_path) -> None:
    config = AttentionBenchmarkConfig(
        sequence_length=4,
        heads=2,
        head_dimension=8,
        warmups=1,
        iterations=2,
        block_size=128,
    )
    metadata = {
        "git_commit": "d" * 40,
        "gpu_name": "fixture GPU",
        "compute_capability": "7.5",
        "pytorch_version": "fixture torch",
        "cuda_version": "fixture CUDA",
        "timestamp": "2026-08-23T00:00:00+00:00",
    }
    records = attention_raw_records(
        metadata=metadata,
        config=config,
        implementation="custom_cuda",
        samples_us=[1.0, 2.0],
    )
    output = tmp_path / "attention.csv"

    write_attention_raw_csv(output, records)

    with output.open(newline="", encoding="utf-8") as output_file:
        saved = list(csv.DictReader(output_file))
    assert tuple(saved[0]) == ATTENTION_RAW_FIELDS
    assert [row["sample_us"] for row in saved] == ["1.0", "2.0"]
    assert all(row["git_commit"] == "d" * 40 for row in saved)
    assert all(row["block_size"] == "128" for row in saved)


def test_attention_checkpoint_rewrites_complete_accumulated_sample_set(tmp_path) -> None:
    output = tmp_path / "attention.csv"
    first = [{field: "" for field in ATTENTION_RAW_FIELDS}]
    first[0]["sample_index"] = 0
    second = [*first, {**first[0], "sample_index": 1}]

    checkpoint_attention_records(output, first)
    checkpoint_attention_records(output, second)

    with output.open(newline="", encoding="utf-8") as output_file:
        saved = list(csv.DictReader(output_file))
    assert [row["sample_index"] for row in saved] == ["0", "1"]


def test_kernel_and_attention_speedups_use_matched_median_ratios() -> None:
    provenance = {
        "git_commit": "a" * 40,
        "gpu_name": "fixture GPU",
        "compute_capability": "7.5",
        "pytorch_version": "fixture torch",
        "cuda_version": "fixture CUDA",
    }
    softmax = [
        {
            **provenance,
            "implementation_description": "PyTorch eager scale + causal mask + softmax",
            "sequence_length": 128,
            "median_us": 8.0,
        },
        {
            **provenance,
            "implementation_description": "warp-reduction custom CUDA fused scale + mask + softmax",
            "sequence_length": 128,
            "median_us": 2.0,
        },
    ]
    attention = [
        {
            **provenance,
            "implementation": "explicit_eager",
            "sequence_length": 128,
            "median_us": 20.0,
        },
        {
            **provenance,
            "implementation": "custom_cuda",
            "sequence_length": 128,
            "median_us": 10.0,
        },
    ]

    comparison = compare_kernel_and_attention_speedups(
        softmax,
        attention,
        required_lengths=(128,),
    )

    assert comparison[0]["kernel_speedup"] == 4.0
    assert comparison[0]["attention_speedup"] == 2.0
    assert comparison[0]["translation_ratio"] == 0.5


def test_kernel_attention_comparison_rejects_mismatched_gpu() -> None:
    common = {
        "git_commit": "a" * 40,
        "compute_capability": "7.5",
        "pytorch_version": "torch",
        "cuda_version": "CUDA",
        "sequence_length": 128,
        "median_us": 1.0,
    }
    softmax = [
        {**common, "gpu_name": "GPU A", "implementation_description": name}
        for name in (
            "PyTorch eager scale + causal mask + softmax",
            "warp-reduction custom CUDA fused scale + mask + softmax",
        )
    ]
    attention = [
        {**common, "gpu_name": "GPU B", "implementation": name}
        for name in ("explicit_eager", "custom_cuda")
    ]

    with pytest.raises(ValueError, match="matched provenance"):
        compare_kernel_and_attention_speedups(
            softmax,
            attention,
            required_lengths=(128,),
        )


def test_attention_summary_aggregates_raw_samples() -> None:
    base = {field: "" for field in ATTENTION_RAW_FIELDS}
    base.update(
        {
            "git_commit": "b" * 40,
            "implementation": "custom_cuda",
            "sequence_length": "128",
            "batch": "1",
            "heads": "8",
            "head_dimension": "64",
            "dtype": "float32",
            "block_size": "128",
            "warmups": "1",
            "iterations": "3",
            "gpu_name": "fixture GPU",
            "compute_capability": "7.5",
            "pytorch_version": "torch",
            "cuda_version": "CUDA",
            "timestamp": "2026-08-23T00:00:00+00:00",
        }
    )
    rows = [
        {**base, "sample_index": str(index), "sample_us": str(sample)}
        for index, sample in enumerate((2.0, 4.0, 8.0))
    ]

    summary = summarize_attention_records(rows)

    assert summary[0]["median_us"] == 4.0
    assert summary[0]["p25_us"] == 3.0
    assert summary[0]["p75_us"] == 6.0


@pytest.mark.parametrize("block_size", SUPPORTED_BLOCK_SIZES)
def test_softmax_registry_applies_one_controlled_block_size(block_size: int) -> None:
    registry = softmax_benchmark_registry(block_size=block_size)

    assert all(case.block_size == block_size for case in registry)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sequence_length": 0},
        {"sequence_length": 128, "batch_heads": 0},
        {"sequence_length": 128, "warmups": 0},
        {"sequence_length": 128, "iterations": 0},
        {"sequence_length": 128, "seed": -1},
        {"sequence_length": 128, "dtype": "float64"},
        {"sequence_length": 128, "block_size": 64},
        {"sequence_length": 128, "block_size": 1024},
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


def test_explicit_untimed_warmups_have_a_separate_call_count() -> None:
    calls: list[int] = []

    result = run_untimed_warmups(
        lambda: calls.append(len(calls)) or len(calls),
        iterations=2,
    )

    assert result == 2
    assert calls == [0, 1]


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


def test_compiled_baseline_matches_same_expression_on_cpu() -> None:
    generator = torch.Generator().manual_seed(76)
    scores = torch.randn(12, 6, generator=generator)
    allowed = causal_allowed_mask(12, 6)
    compiled = compile_causal_softmax(backend="eager")

    actual = compiled(scores, 0.125, allowed)
    expected = pytorch_eager_causal_softmax(scores, 0.125, allowed)

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
    assert all(row["launch_block_size"] == "" for row in saved)
    assert all(row["compile_warmups"] == "0" for row in saved)


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


def test_framework_preflight_requires_equal_work_for_all_three_paths() -> None:
    metadata = {
        "git_commit": "f" * 40,
        "gpu_name": "fixture GPU",
        "compute_capability": "9.0",
        "pytorch_version": "fixture torch",
        "cuda_version": "fixture CUDA",
        "timestamp": "2026-08-22T00:00:00+00:00",
    }
    descriptions = (
        ("PyTorch eager scale + causal mask + softmax", None, 0),
        ("torch.compile scale + causal mask + softmax", None, 1),
        (
            "warp-reduction custom CUDA fused scale + mask + softmax; block_size=256",
            256,
            0,
        ),
    )
    records = []
    for description, block_size, compile_warmups in descriptions:
        records.extend(
            raw_benchmark_records(
                metadata=metadata,
                implementation_description=description,
                sequence_length=128,
                rows=1024,
                columns=128,
                dtype="float32",
                warmups=2,
                iterations=1,
                samples_us=[1.0],
                launch_block_size=block_size,
                compile_warmups=compile_warmups,
            )
        )
    string_records = [
        {key: str(value) for key, value in record.items()} for record in records
    ]

    report = validate_framework_comparison(string_records)

    assert report["frameworks"] == ["compiled", "custom", "eager"]
    assert report["controlled_cases"] == 1


def test_framework_preflight_rejects_incomplete_path_set() -> None:
    records = [
        {
            "git_commit": "f" * 40,
            "implementation_description": "PyTorch eager scale + causal mask + softmax",
            "sequence_length": "128",
            "rows": "1024",
            "columns": "128",
            "dtype": "float32",
            "launch_block_size": "",
            "compile_warmups": "0",
            "warmups": "2",
            "iterations": "1",
            "gpu_name": "fixture GPU",
            "compute_capability": "9.0",
            "pytorch_version": "fixture torch",
            "cuda_version": "fixture CUDA",
        }
    ]

    with pytest.raises(ValueError, match="eager, compiled, and custom"):
        validate_framework_comparison(records)


def test_summary_statistics_and_throughput_come_from_raw_samples(tmp_path) -> None:
    metadata = {
        "git_commit": "c" * 40,
        "gpu_name": "fixture GPU",
        "compute_capability": "9.0",
        "pytorch_version": "fixture torch",
        "cuda_version": "fixture CUDA",
        "timestamp": "2026-08-20T00:00:00+00:00",
    }
    raw = raw_benchmark_records(
        metadata=metadata,
        implementation_description="synthetic fixture only",
        sequence_length=2,
        rows=4,
        columns=2,
        dtype="float32",
        warmups=1,
        iterations=4,
        samples_us=[1.0, 2.0, 3.0, 4.0],
    )

    summaries = summarize_raw_records(
        [{key: str(value) for key, value in record.items()} for record in raw]
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["median_us"] == 2.5
    assert summary["p25_us"] == 1.75
    assert summary["p75_us"] == 3.25
    assert summary["elements_per_second"] == 3_200_000.0

    output_path = tmp_path / "summary.csv"
    write_summary_csv(output_path, summaries)
    with output_path.open(newline="", encoding="utf-8") as output_file:
        saved = list(csv.DictReader(output_file))
    assert tuple(saved[0]) == SUMMARY_FIELDS
    assert saved[0]["git_commit"] == "c" * 40


def test_percentile_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 0.5)


def _launch_summary(block_size: int, sequence_length: int, median_us: float) -> dict[str, object]:
    return {
        "git_commit": "e" * 40,
        "gpu_name": "fixture GPU",
        "compute_capability": "9.0",
        "pytorch_version": "fixture torch",
        "cuda_version": "fixture CUDA",
        "launch_block_size": str(block_size),
        "sequence_length": str(sequence_length),
        "median_us": median_us,
    }


def test_launch_selection_uses_complete_per_shape_relative_latencies() -> None:
    summaries = [
        _launch_summary(128, 128, 1.0),
        _launch_summary(256, 128, 1.1),
        _launch_summary(512, 128, 1.4),
        _launch_summary(128, 2048, 8.0),
        _launch_summary(256, 2048, 6.0),
        _launch_summary(512, 2048, 7.0),
        _launch_summary(128, 768, 4.0),
        _launch_summary(256, 768, 3.0),
        _launch_summary(512, 768, 3.5),
    ]

    selection = select_launch_configuration(summaries)

    assert selection["selected_block_size"] == 256
    assert selection["per_sequence_wins"] == {128: 1, 256: 2, 512: 0}


def test_launch_selection_rejects_missing_configuration() -> None:
    summaries = [
        _launch_summary(128, 128, 1.0),
        _launch_summary(256, 128, 1.1),
    ]

    with pytest.raises(ValueError, match="complete 128/256/512"):
        select_launch_configuration(summaries)


def test_plot_series_remain_commit_specific_and_shape_ordered(tmp_path) -> None:
    records = [
        {
            "git_commit": "b" * 40,
            "implementation_description": "block parallel",
            "sequence_length": "512",
            "median_us": "3.0",
            "elements_per_second": "4.0",
        },
        {
            "git_commit": "b" * 40,
            "implementation_description": "block parallel",
            "sequence_length": "128",
            "median_us": "1.0",
            "elements_per_second": "2.0",
        },
    ]

    series = metric_series(records, "median_us")

    assert len(series) == 1
    assert series[0].label.endswith("(bbbbbbbb)")
    assert series[0].x == (128, 512)
    assert series[0].y == (1.0, 3.0)

    empty_path = tmp_path / "empty.csv"
    empty_path.write_text(
        "git_commit,implementation_description,sequence_length,median_us,elements_per_second\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no measurements"):
        load_summary_csv(empty_path)


def test_figure_generation_maps_summary_to_expected_outputs(tmp_path) -> None:
    summary_path = tmp_path / "summary.csv"
    summary_path.write_text(
        "git_commit,implementation_description,sequence_length,median_us,elements_per_second\n"
        + f"{'d' * 40},fixture only,128,1.0,2.0\n",
        encoding="utf-8",
    )
    output_directory = tmp_path / "figures"

    with (
        patch("scripts.generate_figures.plot_latency") as latency,
        patch("scripts.generate_figures.plot_throughput") as throughput,
    ):
        generated = generate_figures(summary_path, output_directory)

    assert generated == (
        output_directory / "softmax_latency.png",
        output_directory / "softmax_throughput.png",
    )
    assert latency.call_args.args[0] == throughput.call_args.args[0]
    latency.assert_called_once_with(latency.call_args.args[0], generated[0])
    throughput.assert_called_once_with(throughput.call_args.args[0], generated[1])
