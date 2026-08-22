"""Validate raw benchmark artifacts before computing comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cuda_attention.benchmark import RAW_BENCHMARK_FIELDS


CONTROL_FIELDS = (
    "sequence_length",
    "rows",
    "columns",
    "dtype",
    "launch_block_size",
    "compile_warmups",
    "warmups",
    "iterations",
    "gpu_name",
    "compute_capability",
    "pytorch_version",
    "cuda_version",
)

SUMMARY_FIELDS = (
    "git_commit",
    "implementation_description",
    "sequence_length",
    "rows",
    "columns",
    "dtype",
    "launch_block_size",
    "compile_warmups",
    "warmups",
    "iterations",
    "median_us",
    "p25_us",
    "p75_us",
    "elements_per_second",
    "gpu_name",
    "compute_capability",
    "pytorch_version",
    "cuda_version",
    "timestamp",
)

SUMMARY_GROUP_FIELDS = tuple(
    field
    for field in SUMMARY_FIELDS
    if field not in {"median_us", "p25_us", "p75_us", "elements_per_second"}
)


def load_raw_benchmark_csv(path: Path) -> list[dict[str, str]]:
    """Load a nonempty raw artifact only when its schema is complete."""

    if not path.is_file():
        raise FileNotFoundError(f"raw benchmark CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != RAW_BENCHMARK_FIELDS:
            raise ValueError(f"unexpected raw benchmark schema in {path}")
        records = list(reader)
    if not records:
        raise ValueError(f"raw benchmark CSV contains no samples: {path}")
    return records


def validate_comparison_pair(
    baseline: list[dict[str, str]],
    candidate: list[dict[str, str]],
) -> dict[str, object]:
    """Require different commits under identical workload/environment controls."""

    baseline_commits = {record["git_commit"] for record in baseline}
    candidate_commits = {record["git_commit"] for record in candidate}
    if len(baseline_commits) != 1 or len(candidate_commits) != 1:
        raise ValueError("each comparison artifact must contain exactly one Git commit")
    if baseline_commits == candidate_commits:
        raise ValueError("baseline and candidate must come from different Git commits")

    baseline_controls = {
        tuple(record[field] for field in CONTROL_FIELDS) for record in baseline
    }
    candidate_controls = {
        tuple(record[field] for field in CONTROL_FIELDS) for record in candidate
    }
    if baseline_controls != candidate_controls:
        raise ValueError("baseline and candidate controls or environments do not match")

    return {
        "baseline_commit": next(iter(baseline_commits)),
        "candidate_commit": next(iter(candidate_commits)),
        "controlled_cases": len(baseline_controls),
        "status": "ready_for_summary",
    }


def percentile(samples: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile over finite timing samples."""

    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be between zero and one")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_raw_records(
    records: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Aggregate raw timing samples without discarding their provenance."""

    grouped: dict[tuple[str, ...], list[float]] = {}
    for record in records:
        key = tuple(record[field] for field in SUMMARY_GROUP_FIELDS)
        sample = float(record["sample_us"])
        if sample <= 0.0:
            raise ValueError("raw CUDA timing samples must be positive")
        grouped.setdefault(key, []).append(sample)

    summaries: list[dict[str, object]] = []
    for key, samples in grouped.items():
        group = dict(zip(SUMMARY_GROUP_FIELDS, key, strict=True))
        iterations = int(group["iterations"])
        if len(samples) != iterations:
            raise ValueError("raw sample count does not match recorded iterations")
        median_us = statistics.median(samples)
        elements = int(group["rows"]) * int(group["columns"])
        summaries.append(
            {
                **group,
                "median_us": median_us,
                "p25_us": percentile(samples, 0.25),
                "p75_us": percentile(samples, 0.75),
                "elements_per_second": elements * 1_000_000.0 / median_us,
            }
        )
    return sorted(
        summaries,
        key=lambda record: (
            int(record["sequence_length"]),
            str(record["git_commit"]),
            str(record["implementation_description"]),
        ),
    )


def write_summary_csv(
    output_path: Path,
    records: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def select_launch_configuration(
    summaries: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Select among complete 128/256/512 results without overweighting long rows.

    For each sequence length, latency is divided by the fastest configuration's
    latency at that same length. The configuration with the lowest median
    relative latency wins. This gives every planned shape one vote-like weight
    while retaining the magnitude of slowdowns.
    """

    expected_sizes = {128, 256, 512}
    by_size: dict[int, dict[int, float]] = {}
    provenance: set[tuple[str, ...]] = set()
    for record in summaries:
        raw_size = str(record.get("launch_block_size", ""))
        if not raw_size:
            raise ValueError("launch selection requires a block size on every row")
        block_size = int(raw_size)
        sequence_length = int(record["sequence_length"])
        latency = float(record["median_us"])
        if latency <= 0.0:
            raise ValueError("launch selection requires positive median latency")
        if sequence_length in by_size.setdefault(block_size, {}):
            raise ValueError("launch selection requires one summary per size and shape")
        by_size[block_size][sequence_length] = latency
        provenance.add(
            tuple(
                str(record[field])
                for field in (
                    "git_commit",
                    "gpu_name",
                    "compute_capability",
                    "pytorch_version",
                    "cuda_version",
                )
            )
        )

    if set(by_size) != expected_sizes:
        raise ValueError("launch selection requires complete 128/256/512 results")
    if len(provenance) != 1:
        raise ValueError("launch selection requires one Git and GPU environment")
    shape_sets = {tuple(sorted(results)) for results in by_size.values()}
    if len(shape_sets) != 1:
        raise ValueError("launch configurations must cover identical sequence lengths")

    sequence_lengths = next(iter(shape_sets))
    relative_latency: dict[int, list[float]] = {size: [] for size in expected_sizes}
    wins = {size: 0 for size in expected_sizes}
    for sequence_length in sequence_lengths:
        best_latency = min(
            by_size[size][sequence_length] for size in expected_sizes
        )
        winners = [
            size
            for size in expected_sizes
            if by_size[size][sequence_length] == best_latency
        ]
        for size in winners:
            wins[size] += 1
        for size in expected_sizes:
            relative_latency[size].append(
                by_size[size][sequence_length] / best_latency
            )

    scores = {
        size: statistics.median(relative_latency[size]) for size in expected_sizes
    }
    selected = min(expected_sizes, key=lambda size: (scores[size], size))
    return {
        "selected_block_size": selected,
        "median_relative_latency": scores,
        "per_sequence_wins": wins,
        "sequence_lengths": list(sequence_lengths),
        "selection_rule": "lowest median per-shape relative latency",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        baseline = load_raw_benchmark_csv(arguments.baseline)
        candidate = load_raw_benchmark_csv(arguments.candidate)
        report = validate_comparison_pair(baseline, candidate)
        summaries = summarize_raw_records(baseline) + summarize_raw_records(candidate)
        write_summary_csv(arguments.output, summaries)
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps({**report, "summary_rows": len(summaries)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
