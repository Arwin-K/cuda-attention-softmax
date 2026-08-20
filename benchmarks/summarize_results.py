"""Validate raw benchmark artifacts before computing comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cuda_attention.benchmark import RAW_BENCHMARK_FIELDS


CONTROL_FIELDS = (
    "sequence_length",
    "rows",
    "columns",
    "dtype",
    "warmups",
    "iterations",
    "gpu_name",
    "compute_capability",
    "pytorch_version",
    "cuda_version",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        report = validate_comparison_pair(
            load_raw_benchmark_csv(arguments.baseline),
            load_raw_benchmark_csv(arguments.candidate),
        )
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
