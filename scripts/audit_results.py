#!/usr/bin/env python3
"""Audit measured artifacts without manufacturing or silently repairing data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


LENGTHS = {128, 255, 512, 768, 1024, 1536, 2048}
PROVENANCE = {
    "git_commit",
    "gpu_name",
    "compute_capability",
    "pytorch_version",
    "cuda_version",
    "timestamp",
}
SOFTMAX_RAW = PROVENANCE | {
    "implementation",
    "implementation_description",
    "sequence_length",
    "rows",
    "columns",
    "dtype",
    "scale",
    "warmups",
    "iterations",
    "sample_index",
    "latency_us",
    "launch_block_size",
    "compile_warmups",
}
SOFTMAX_SUMMARY = (SOFTMAX_RAW - {"sample_index", "latency_us", "scale"}) | {
    "median_us",
    "p25_us",
    "p75_us",
    "elements_per_second",
}
ATTENTION_RAW = PROVENANCE | {
    "implementation",
    "batch",
    "heads",
    "sequence_length",
    "head_dimension",
    "dtype",
    "warmups",
    "iterations",
    "sample_index",
    "latency_us",
}
ATTENTION_SUMMARY = (ATTENTION_RAW - {"sample_index", "latency_us"}) | {
    "median_us",
    "p25_us",
    "p75_us",
    "speedup_vs_explicit_eager",
    "speedup_vs_sdpa",
}
CORRECTNESS = {
    "git_commit",
    "sequence_length",
    "rows",
    "input_family",
    "dtype",
    "scale",
    "max_absolute_error",
    "max_relative_error",
    "max_row_sum_error",
    "masked_positions_exactly_zero",
    "contains_nan",
    "contains_inf",
    "shape_correct",
    "dtype_correct",
    "device_correct",
    "pass_fail",
    "rtol",
    "atol",
    "gpu_name",
    "compute_capability",
    "timestamp",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def missing_fields(columns: Iterable[str], required: Iterable[str]) -> list[str]:
    return sorted(set(required) - set(columns))


def percentile(values: list[float], fraction: float) -> float:
    """Match NumPy's default linear interpolation without requiring NumPy."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def check(self, name: str, condition: bool, detail: object) -> None:
        self.checks.append(
            {"name": name, "status": "PASS" if condition else "FAIL", "detail": detail}
        )

    @property
    def passed(self) -> bool:
        return all(item["status"] == "PASS" for item in self.checks)


def _schema(audit: Audit, name: str, path: Path, required: set[str]) -> list[dict[str, str]]:
    columns, rows = read_csv(path)
    missing = missing_fields(columns, required)
    audit.check(f"{name} schema", not missing, {"missing": missing, "rows": len(rows)})
    audit.check(f"{name} nonempty", bool(rows), {"rows": len(rows)})
    return rows


def _audit_samples(
    audit: Audit,
    name: str,
    rows: list[dict[str, str]],
    key_fields: tuple[str, ...],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in key_fields)].append(row)
    problems: list[str] = []
    for key, group in groups.items():
        expected = int(group[0]["iterations"])
        indices = sorted(int(row["sample_index"]) for row in group)
        if len(group) != expected or indices != list(range(expected)):
            problems.append(f"{key}: rows={len(group)}, iterations={expected}")
        if any(float(row["latency_us"]) <= 0 for row in group):
            problems.append(f"{key}: non-positive latency")
    audit.check(f"{name} sample counts", not problems, problems or f"{len(groups)} complete groups")
    return groups


def _audit_summary(
    audit: Audit,
    name: str,
    groups: dict[tuple[str, ...], list[dict[str, str]]],
    summary_rows: list[dict[str, str]],
    key_fields: tuple[str, ...],
) -> None:
    summaries = {tuple(row[field] for field in key_fields): row for row in summary_rows}
    problems: list[str] = []
    for key, group in groups.items():
        if key not in summaries:
            problems.append(f"{key}: missing summary")
            continue
        values = [float(row["latency_us"]) for row in group]
        expected = {
            "median_us": statistics.median(values),
            "p25_us": percentile(values, 0.25),
            "p75_us": percentile(values, 0.75),
        }
        for field, value in expected.items():
            actual = float(summaries[key][field])
            if not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-9):
                problems.append(f"{key}: {field}={actual}, recomputed={value}")
    extra = set(summaries) - set(groups)
    problems.extend(f"{key}: summary without raw group" for key in sorted(extra))
    audit.check(f"{name} summary recomputation", not problems, problems or f"{len(groups)} groups match")


def audit_artifacts(root: Path) -> dict[str, object]:
    root = root.resolve()
    audit = Audit()
    manifest_path = root / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    commit = manifest["git_commit"]

    manifest_paths = [root / item for item in manifest["csv_files"]]
    missing_manifest_paths = [str(path.relative_to(root)) for path in manifest_paths if not path.is_file()]
    audit.check("manifest CSV inventory", not missing_manifest_paths, missing_manifest_paths or f"{len(manifest_paths)} files")

    softmax = _schema(audit, "softmax raw", root / "benchmarks/raw/softmax_raw.csv", SOFTMAX_RAW)
    softmax_summary = _schema(audit, "softmax summary", root / "benchmarks/summaries/softmax_summary.csv", SOFTMAX_SUMMARY | {"speedup_vs_eager"})
    historical = _schema(audit, "historical raw", root / "benchmarks/raw/historical_raw.csv", SOFTMAX_RAW)
    historical_summary = _schema(audit, "historical summary", root / "benchmarks/summaries/historical_summary.csv", SOFTMAX_SUMMARY | {"speedup_vs_row_serial"})
    launch = _schema(audit, "launch raw", root / "benchmarks/raw/launch_configuration_raw.csv", SOFTMAX_RAW)
    launch_summary = _schema(audit, "launch summary", root / "benchmarks/summaries/launch_configuration_summary.csv", SOFTMAX_SUMMARY)
    attention = _schema(audit, "attention raw", root / "attention/attention_raw.csv", ATTENTION_RAW)
    attention_summary = _schema(audit, "attention summary", root / "attention/attention_summary.csv", ATTENTION_SUMMARY)
    correctness = _schema(audit, "correctness", root / "correctness/correctness_results.csv", CORRECTNESS)

    softmax_groups = _audit_samples(audit, "softmax", softmax, ("implementation", "sequence_length"))
    historical_groups = _audit_samples(audit, "historical", historical, ("implementation", "sequence_length"))
    launch_groups = _audit_samples(audit, "launch", launch, ("launch_block_size", "sequence_length"))
    attention_groups = _audit_samples(audit, "attention", attention, ("implementation", "sequence_length"))
    _audit_summary(audit, "softmax", softmax_groups, softmax_summary, ("implementation", "sequence_length"))
    _audit_summary(audit, "historical", historical_groups, historical_summary, ("implementation", "sequence_length"))
    _audit_summary(audit, "launch", launch_groups, launch_summary, ("launch_block_size", "sequence_length"))
    _audit_summary(audit, "attention", attention_groups, attention_summary, ("implementation", "sequence_length"))

    current_rows = softmax + launch + attention + correctness
    audit.check("current commit provenance", {row["git_commit"] for row in current_rows} == {commit}, sorted({row["git_commit"] for row in current_rows}))
    environment = {(row["gpu_name"], row["compute_capability"]) for row in current_rows}
    audit.check("current GPU provenance", environment == {(manifest["gpu"], manifest["compute_capability"])}, sorted(environment))

    audited_lengths = {int(row["sequence_length"]) for row in softmax}
    audit.check("softmax shape registry", audited_lengths == LENGTHS, sorted(audited_lengths))
    audit.check("softmax implementations", {row["implementation"] for row in softmax} == {"pytorch_eager", "torch_compile", "custom_cuda"}, sorted({row["implementation"] for row in softmax}))
    audit.check("launch configurations", {int(row["launch_block_size"]) for row in launch} == {128, 256, 512}, sorted({int(row["launch_block_size"]) for row in launch}))
    audit.check("attention implementations", {row["implementation"] for row in attention} == {"explicit_pytorch", "custom_softmax_attention", "pytorch_sdpa"}, sorted({row["implementation"] for row in attention}))
    audit.check("correctness cases", len(correctness) == 88 and all(row["pass_fail"] == "PASS" for row in correctness), {"rows": len(correctness), "failed": sum(row["pass_fail"] != "PASS" for row in correctness)})

    profiler_fields, profiler_rows = read_csv(root / "profiler/pytorch_profiler_events.csv")
    profiler_required = {"operator", "count", "self_cpu_time_total_us", "cpu_time_total_us", "self_device_time_total_us"}
    audit.check("PyTorch profiler schema", not missing_fields(profiler_fields, profiler_required) and bool(profiler_rows), {"missing": missing_fields(profiler_fields, profiler_required), "rows": len(profiler_rows)})
    nsight_fields, nsight_rows = read_csv(root / "profiler/nsight/ncu_raw_export.csv")
    nsight_required = {"Kernel Name", "Block Size", "Grid Size", "gpu__time_duration.sum", "launch__registers_per_thread", "launch__shared_mem_per_block_dynamic", "gpu__dram_throughput.sum.pct_of_peak_sustained_elapsed", "sm__throughput.sum.pct_of_peak_sustained_elapsed"}
    data_rows = [row for row in nsight_rows if row.get("ID")]
    audit.check("Nsight schema", not missing_fields(nsight_fields, nsight_required) and bool(data_rows), {"missing": missing_fields(nsight_fields, nsight_required), "data_rows": len(data_rows)})

    return {
        "schema_version": 1,
        "artifact_root": root.name,
        "git_commit": commit,
        "status": "PASS" if audit.passed else "FAIL",
        "checks": audit.checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_artifacts(args.artifact_root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
