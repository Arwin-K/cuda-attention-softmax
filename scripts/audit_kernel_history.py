#!/usr/bin/env python3
"""Verify documented CUDA milestones and the single-source invariant."""

from __future__ import annotations

from pathlib import Path
import subprocess


MILESTONES = {
    "c96a49b9": "add CUDA source skeleton and launch interface",
    "e309f1f9": "implement initial row-serial fused causal softmax kernel",
    "e31e6387": "add stable maximum scan to CUDA kernel",
    "d29490c4": "add causal masking and scaling inside CUDA kernel",
    "27eb32b2": "add exponential sum and normalization to CUDA kernel",
    "27586189": "rewrite kernel mapping to one CUDA block per softmax row",
    "1a4ecbd4": "distribute row elements with thread-strided access",
    "7c7991c8": "implement shared-memory maximum reduction",
    "02a4c519": "implement shared-memory sum reduction",
    "4b5880a9": "add warp and lane helper utilities to CUDA code",
    "d763b22d": "implement warp-level maximum reduction with shuffle operations",
    "4b2bfd49": "implement warp-level sum reduction with shuffle operations",
    "885c07b0": "replace shared-memory block reductions with warp reductions",
    "9ac83c64": "add benchmark support for configurable block sizes",
    "9841d5fd": "fix CUDA 12.8 infinity constant include",
}


def git_subject(root: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "show", "-s", "--format=%s", revision],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def audit(root: Path) -> list[str]:
    failures = []
    for revision, expected in MILESTONES.items():
        actual = git_subject(root, revision)
        if actual != expected:
            failures.append(f"{revision}: expected {expected!r}, found {actual!r}")

    tracked_result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [Path(line) for line in tracked_result.stdout.splitlines()]
    cuda_sources = sorted(path for path in tracked if path.suffix == ".cu")
    expected_source = Path("csrc/fused_causal_softmax.cu")
    if cuda_sources != [expected_source]:
        failures.append(f"expected one tracked CUDA source {expected_source}, found {cuda_sources}")

    forbidden = {"v1", "v2", "v3", "naive", "optimized", "final"}
    bad_names = sorted(path for path in tracked if path.stem.lower() in forbidden)
    if bad_names:
        failures.append(f"forbidden version-like paths: {bad_names}")
    return failures


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = audit(root)
    if failures:
        print("KERNEL_HISTORY_AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"KERNEL_HISTORY_AUDIT: PASS ({len(MILESTONES)} milestones, one .cu source)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
