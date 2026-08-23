#!/usr/bin/env python3
"""Generate deterministic figure-to-source provenance for a measured run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


FIGURE_SOURCES = {
    "fig_historical_speedup": [
        "benchmarks/raw/historical_raw.csv",
        "benchmarks/summaries/historical_summary.csv",
    ],
    "fig_kernel_vs_attention_speedup": [
        "benchmarks/raw/softmax_raw.csv",
        "benchmarks/summaries/softmax_summary.csv",
        "attention/attention_raw.csv",
        "attention/attention_summary.csv",
        "attention/amdahl_analysis.csv",
    ],
    "fig_launch_configuration": [
        "benchmarks/raw/launch_configuration_raw.csv",
        "benchmarks/summaries/launch_configuration_summary.csv",
    ],
    "fig_profiler_breakdown": [
        "profiler/pytorch_profiler_events.csv",
        "profiler/nsight/ncu_raw_export.csv",
    ],
    "fig_softmax_latency": [
        "benchmarks/raw/softmax_raw.csv",
        "benchmarks/summaries/softmax_summary.csv",
    ],
    "fig_softmax_throughput": [
        "benchmarks/raw/softmax_raw.csv",
        "benchmarks/summaries/softmax_summary.csv",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commits_in_csv(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "git_commit" not in (reader.fieldnames or []):
            return set()
        return {row["git_commit"] for row in reader if row.get("git_commit")}


def generate(artifact_root: Path) -> dict[str, object]:
    root = artifact_root.resolve()
    experiment = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
    declared = set(experiment["figures"])
    expected = {
        f"figures/{stem}.{suffix}"
        for stem in FIGURE_SOURCES
        for suffix in ("pdf", "png")
    }
    if declared != expected:
        missing = sorted(expected - declared)
        extra = sorted(declared - expected)
        raise ValueError(f"figure inventory mismatch: missing={missing}, extra={extra}")

    records: list[dict[str, object]] = []
    for figure_path in sorted(declared):
        figure = root / figure_path
        stem = figure.stem
        source_paths = FIGURE_SOURCES[stem]
        sources = []
        commits: set[str] = set()
        for source_path in source_paths:
            source = root / source_path
            if not source.is_file():
                raise FileNotFoundError(source)
            sources.append({"path": source_path, "sha256": sha256(source)})
            commits.update(commits_in_csv(source))
        if not figure.is_file():
            raise FileNotFoundError(figure)
        records.append(
            {
                "figure": figure_path,
                "figure_sha256": sha256(figure),
                "measured_git_commits": sorted(commits or {experiment["git_commit"]}),
                "sources": sources,
            }
        )

    return {
        "schema_version": 1,
        "artifact_root": root.name,
        "experiment_manifest_sha256": sha256(root / "experiment_manifest.json"),
        "run_git_commit": experiment["git_commit"],
        "figures": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = generate(args.artifact_root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"WROTE: {args.output} ({len(report['figures'])} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
