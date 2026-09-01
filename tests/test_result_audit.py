from pathlib import Path

from benchmarks.compare_speedups import (
    compare_kernel_and_attention_speedups,
    load_attention_raw_csv,
    summarize_attention_records,
    write_speedup_comparison,
)
from benchmarks.generate_tables import generate_results_tables
from benchmarks.summarize_results import load_raw_benchmark_csv, summarize_raw_records
from scripts.audit_results import audit_artifacts, missing_fields, percentile


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "results/runs/2026-08-23_tesla-t4_ca87722/artifacts"


def test_missing_fields_reports_actionable_gap():
    assert missing_fields(["git_commit", "latency_us"], ["git_commit", "gpu_name"]) == [
        "gpu_name"
    ]


def test_percentile_matches_linear_interpolation():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.25) == 1.75
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.75) == 3.25


def test_imported_experiment_passes_complete_schema_audit():
    report = audit_artifacts(ARTIFACTS)
    assert report["status"] == "PASS", [
        item for item in report["checks"] if item["status"] == "FAIL"
    ]


def test_published_raw_data_regenerates_comparison_and_tables(tmp_path):
    softmax = summarize_raw_records(
        load_raw_benchmark_csv(ARTIFACTS / "benchmarks/raw/softmax_raw.csv")
    )
    attention = summarize_attention_records(
        load_attention_raw_csv(ARTIFACTS / "attention/attention_raw.csv")
    )
    comparison = compare_kernel_and_attention_speedups(softmax, attention)
    comparison_path = tmp_path / "speedup_comparison.csv"
    write_speedup_comparison(comparison_path, comparison)

    outputs = generate_results_tables(
        softmax_summary_path=(
            ARTIFACTS / "benchmarks/summaries/softmax_summary.csv"
        ),
        attention_raw_path=ARTIFACTS / "attention/attention_raw.csv",
        speedup_comparison_path=comparison_path,
        output_directory=tmp_path / "tables",
    )

    assert len(comparison) == 7
    assert all(path.is_file() for path in outputs)
