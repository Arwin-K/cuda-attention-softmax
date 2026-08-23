#!/usr/bin/env python3
"""Derive public quantitative claims and verify website-facing prose."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PUBLIC_DOCUMENTS = {
    "README.md": ["historical_row_to_warp", "custom_vs_eager", "custom_attention_vs_eager"],
    "docs/blog_post.md": ["historical_row_to_warp", "warp_vs_shared", "custom_vs_eager", "custom_attention_vs_eager"],
    "docs/mini_paper.md": ["historical_row_to_warp", "warp_vs_shared", "custom_vs_eager", "custom_attention_vs_eager"],
    "docs/research_journey.md": ["historical_row_to_warp", "warp_vs_shared", "custom_vs_eager", "custom_attention_vs_eager"],
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value_range(values: list[float]) -> dict[str, float]:
    return {"minimum": min(values), "maximum": max(values)}


def derive_claims(artifact_root: Path) -> dict[str, dict[str, object]]:
    root = artifact_root.resolve()
    correctness = rows(root / "correctness/correctness_results.csv")
    historical = rows(root / "benchmarks/summaries/historical_summary.csv")
    softmax = rows(root / "benchmarks/summaries/softmax_summary.csv")
    attention = rows(root / "attention/attention_summary.csv")
    amdahl = rows(root / "attention/amdahl_analysis.csv")
    launch = json.loads((root / "benchmarks/summaries/launch_selection.json").read_text())
    profiler = rows(root / "profiler/pytorch_profiler_events.csv")
    nsight_all = rows(root / "profiler/nsight/ncu_raw_export.csv")
    nsight = [row for row in nsight_all if row.get("ID")]

    by_historical = {
        (row["implementation"], int(row["sequence_length"])): float(row["median_us"])
        for row in historical
    }
    sequence_lengths = sorted({length for _, length in by_historical})
    row_to_warp = [
        by_historical[("row_serial", length)] / by_historical[("warp_reduction", length)]
        for length in sequence_lengths
    ]
    shared_to_warp = [
        by_historical[("block_shared_tree", length)] / by_historical[("warp_reduction", length)]
        for length in sequence_lengths
    ]
    custom_softmax = [row for row in softmax if row["implementation"] == "custom_cuda"]
    custom_to_eager = [float(row["speedup_vs_eager"]) for row in custom_softmax]
    softmax_by_key = {
        (row["implementation"], row["sequence_length"]): float(row["median_us"])
        for row in softmax
    }
    custom_compile_wins = sum(
        softmax_by_key[("custom_cuda", str(length))]
        < softmax_by_key[("torch_compile", str(length))]
        for length in sequence_lengths
    )
    custom_attention = [
        row for row in attention if row["implementation"] == "custom_softmax_attention"
    ]
    attention_to_eager = [float(row["speedup_vs_explicit_eager"]) for row in custom_attention]
    attention_by_key = {
        (row["implementation"], row["sequence_length"]): float(row["median_us"])
        for row in attention
    }
    custom_to_sdpa = [
        attention_by_key[("custom_softmax_attention", str(length))]
        / attention_by_key[("pytorch_sdpa", str(length))]
        for length in sequence_lengths
    ]
    profiler_device = {
        row["operator"]: float(row["self_device_time_total_us"])
        for row in profiler
        if float(row["self_device_time_total_us"]) > 0
    }

    claims: dict[str, dict[str, object]] = {
        "correctness": {
            "passed_cases": sum(row["pass_fail"] == "PASS" for row in correctness),
            "total_cases": len(correctness),
            "maximum_absolute_error": max(float(row["max_absolute_error"]) for row in correctness),
            "evidence": ["correctness/correctness_results.csv"],
        },
        "historical_row_to_warp": {
            **value_range(row_to_warp),
            "display": f"{min(row_to_warp):.2f}--{max(row_to_warp):.2f}x",
            "evidence": ["benchmarks/summaries/historical_summary.csv"],
        },
        "warp_vs_shared": {
            **value_range(shared_to_warp),
            "display": f"{min(shared_to_warp):.2f}--{max(shared_to_warp):.2f}x",
            "evidence": ["benchmarks/summaries/historical_summary.csv"],
        },
        "custom_vs_eager": {
            **value_range(custom_to_eager),
            "display": f"{min(custom_to_eager):.2f}--{max(custom_to_eager):.2f}x",
            "custom_wins": len(custom_to_eager),
            "evidence": ["benchmarks/summaries/softmax_summary.csv"],
        },
        "custom_vs_compile": {
            "custom_wins": custom_compile_wins,
            "comparisons": len(sequence_lengths),
            "evidence": ["benchmarks/summaries/softmax_summary.csv"],
        },
        "launch_selection": {
            "selected_block_size": launch["selected_block_size"],
            "per_sequence_wins": launch["per_sequence_wins"],
            "evidence": ["benchmarks/summaries/launch_selection.json"],
        },
        "custom_attention_vs_eager": {
            **value_range(attention_to_eager),
            "display": f"{min(attention_to_eager):.2f}--{max(attention_to_eager):.2f}x",
            "evidence": ["attention/attention_summary.csv"],
        },
        "custom_attention_vs_sdpa": {
            **value_range(custom_to_sdpa),
            "display": f"{min(custom_to_sdpa):.2f}--{max(custom_to_sdpa):.2f}x",
            "sdpa_wins": sum(value > 1.0 for value in custom_to_sdpa),
            "comparisons": len(custom_to_sdpa),
            "evidence": ["attention/attention_summary.csv"],
        },
        "kernel_vs_attention_translation": {
            "kernel_speedup_greater": sum(
                float(row["measured_softmax_speedup"])
                > float(row["measured_attention_speedup"])
                for row in amdahl
            ),
            "comparisons": len(amdahl),
            "evidence": ["attention/amdahl_analysis.csv"],
        },
        "pytorch_profiler": {
            "fused_kernel_us": profiler_device["(anonymous namespace)::fused_causal_softmax_kernel(float const*, float*, long, long, float)"],
            "custom_attention_us": profiler_device["attention::custom_softmax_attention"],
            "explicit_attention_us": profiler_device["attention::explicit_pytorch"],
            "sdpa_us": profiler_device["attention::pytorch_sdpa"],
            "evidence": ["profiler/pytorch_profiler_events.csv"],
        },
        "nsight": {
            "profiled_launches": len(nsight),
            "duration_us": value_range([float(row["gpu__time_duration.sum"]) for row in nsight]),
            "dram_peak_percent": value_range([float(row["gpu__dram_throughput.sum.pct_of_peak_sustained_elapsed"]) for row in nsight]),
            "sm_peak_percent": value_range([float(row["sm__throughput.sum.pct_of_peak_sustained_elapsed"]) for row in nsight]),
            "evidence": ["profiler/nsight/ncu_raw_export.csv"],
        },
    }
    return claims


def audit_documents(repository_root: Path, claims: dict[str, dict[str, object]]) -> list[str]:
    failures = []
    for document, claim_ids in PUBLIC_DOCUMENTS.items():
        text = (repository_root / document).read_text(encoding="utf-8")
        for claim_id in claim_ids:
            expected = str(claims[claim_id]["display"])
            if expected not in text:
                failures.append(f"{document}: missing derived {claim_id} display {expected}")
        if "T4" not in text:
            failures.append(f"{document}: missing T4 scope")
        if "SDPA" not in text:
            failures.append(f"{document}: missing SDPA comparison")
    return failures


def build_report(repository_root: Path, artifact_root: Path) -> dict[str, object]:
    claims = derive_claims(artifact_root)
    failures = audit_documents(repository_root, claims)
    return {
        "schema_version": 1,
        "measured_git_commit": json.loads((artifact_root / "experiment_manifest.json").read_text())["git_commit"],
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "claims": claims,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.repository_root.resolve(), args.artifact_root.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
