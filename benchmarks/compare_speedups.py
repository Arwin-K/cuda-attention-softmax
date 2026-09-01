"""Compare measured softmax speedup with measured full-attention speedup."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys
from collections.abc import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_attention import ATTENTION_RAW_FIELDS
from benchmarks.config import BENCHMARK_SEQUENCE_LENGTHS
from benchmarks.summarize_results import (
    load_raw_benchmark_csv,
    percentile,
    summarize_raw_records,
)


PUBLISHED_ATTENTION_RAW_FIELDS = (
    "implementation",
    "git_commit",
    "batch",
    "heads",
    "sequence_length",
    "head_dimension",
    "dtype",
    "warmups",
    "iterations",
    "sample_index",
    "latency_us",
    "gpu_name",
    "compute_capability",
    "pytorch_version",
    "cuda_version",
    "timestamp",
)
PUBLISHED_ATTENTION_IMPLEMENTATIONS = {
    "explicit_pytorch": "explicit_eager",
    "custom_softmax_attention": "custom_cuda",
    "pytorch_sdpa": "pytorch_sdpa",
}


ATTENTION_SUMMARY_FIELDS = (
    "git_commit",
    "implementation",
    "sequence_length",
    "batch",
    "heads",
    "head_dimension",
    "dtype",
    "block_size",
    "warmups",
    "iterations",
    "median_us",
    "p25_us",
    "p75_us",
    "gpu_name",
    "compute_capability",
    "pytorch_version",
    "cuda_version",
    "timestamp",
)

SPEEDUP_COMPARISON_FIELDS = (
    "git_commit",
    "sequence_length",
    "kernel_baseline",
    "kernel_candidate",
    "kernel_speedup",
    "attention_baseline",
    "attention_candidate",
    "attention_speedup",
    "translation_ratio",
    "gpu_name",
    "compute_capability",
    "pytorch_version",
    "cuda_version",
)


def load_attention_raw_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"attention raw CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        fields = tuple(reader.fieldnames or ())
        if fields not in {ATTENTION_RAW_FIELDS, PUBLISHED_ATTENTION_RAW_FIELDS}:
            raise ValueError("unexpected attention raw schema")
        rows = list(reader)
    if not rows:
        raise ValueError("attention raw CSV contains no samples")
    if fields == PUBLISHED_ATTENTION_RAW_FIELDS:
        implementations = {row["implementation"] for row in rows}
        unknown = implementations - set(PUBLISHED_ATTENTION_IMPLEMENTATIONS)
        if unknown:
            raise ValueError(
                f"unexpected published attention implementations: {sorted(unknown)}"
            )
        rows = [
            {
                **{
                    field: row[field]
                    for field in ATTENTION_RAW_FIELDS
                    if field not in {"implementation", "block_size", "sample_us"}
                },
                "implementation": PUBLISHED_ATTENTION_IMPLEMENTATIONS[
                    row["implementation"]
                ],
                "block_size": "",
                "sample_us": row["latency_us"],
            }
            for row in rows
        ]
    return rows


def summarize_attention_records(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    group_fields = tuple(
        field
        for field in ATTENTION_SUMMARY_FIELDS
        if field not in {"median_us", "p25_us", "p75_us"}
    )
    grouped: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        sample = float(row["sample_us"])
        if sample <= 0.0:
            raise ValueError("attention timing samples must be positive")
        grouped.setdefault(key, []).append(sample)

    summaries: list[dict[str, object]] = []
    for key, samples in grouped.items():
        group = dict(zip(group_fields, key, strict=True))
        if len(samples) != int(group["iterations"]):
            raise ValueError("attention raw sample count does not match iterations")
        summaries.append(
            {
                **group,
                "median_us": statistics.median(samples),
                "p25_us": percentile(samples, 0.25),
                "p75_us": percentile(samples, 0.75),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (int(row["sequence_length"]), str(row["implementation"])),
    )


def _softmax_role(description: str) -> str | None:
    if description.startswith("PyTorch eager"):
        return "baseline"
    if description.startswith("warp-reduction custom CUDA"):
        return "candidate"
    return None


def compare_kernel_and_attention_speedups(
    softmax_summaries: Sequence[Mapping[str, object]],
    attention_summaries: Sequence[Mapping[str, object]],
    *,
    required_lengths: Sequence[int] = BENCHMARK_SEQUENCE_LENGTHS,
) -> list[dict[str, object]]:
    """Compute matched speedups and refuse incompatible experimental evidence."""

    softmax: dict[tuple[int, str], Mapping[str, object]] = {}
    for row in softmax_summaries:
        role = _softmax_role(str(row["implementation_description"]))
        if role is not None:
            key = (int(row["sequence_length"]), role)
            if key in softmax:
                raise ValueError("duplicate softmax summary path")
            softmax[key] = row
    attention: dict[tuple[int, str], Mapping[str, object]] = {}
    attention_roles = {"explicit_eager": "baseline", "custom_cuda": "candidate"}
    for row in attention_summaries:
        role = attention_roles.get(str(row["implementation"]))
        if role is not None:
            key = (int(row["sequence_length"]), role)
            if key in attention:
                raise ValueError("duplicate attention summary path")
            attention[key] = row

    comparison: list[dict[str, object]] = []
    provenance_fields = (
        "git_commit",
        "gpu_name",
        "compute_capability",
        "pytorch_version",
        "cuda_version",
    )
    for sequence_length in required_lengths:
        required = [
            softmax.get((sequence_length, "baseline")),
            softmax.get((sequence_length, "candidate")),
            attention.get((sequence_length, "baseline")),
            attention.get((sequence_length, "candidate")),
        ]
        if any(row is None for row in required):
            raise ValueError(f"missing matched speedup path for S={sequence_length}")
        kernel_baseline, kernel_candidate, attention_baseline, attention_candidate = required
        assert all(row is not None for row in required)
        provenance = {
            tuple(str(row[field]) for field in provenance_fields)
            for row in required
            if row is not None
        }
        if len(provenance) != 1:
            raise ValueError("kernel/attention comparison requires matched provenance")
        kernel_speedup = float(kernel_baseline["median_us"]) / float(
            kernel_candidate["median_us"]
        )
        attention_speedup = float(attention_baseline["median_us"]) / float(
            attention_candidate["median_us"]
        )
        provenance_values = next(iter(provenance))
        comparison.append(
            {
                **dict(zip(provenance_fields, provenance_values, strict=True)),
                "sequence_length": sequence_length,
                "kernel_baseline": "pytorch_eager",
                "kernel_candidate": "custom_cuda",
                "kernel_speedup": kernel_speedup,
                "attention_baseline": "explicit_eager",
                "attention_candidate": "custom_cuda",
                "attention_speedup": attention_speedup,
                "translation_ratio": attention_speedup / kernel_speedup,
            }
        )
    return comparison


def write_speedup_comparison(
    output_path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SPEEDUP_COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--softmax-raw", type=Path, required=True)
    parser.add_argument("--attention-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        softmax = summarize_raw_records(load_raw_benchmark_csv(arguments.softmax_raw))
        attention = summarize_attention_records(
            load_attention_raw_csv(arguments.attention_raw)
        )
        comparison = compare_kernel_and_attention_speedups(softmax, attention)
        write_speedup_comparison(arguments.output, comparison)
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"wrote {len(comparison)} matched speedup rows to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
