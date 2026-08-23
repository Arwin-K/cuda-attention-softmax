"""PyTorch Profiler regions for isolated softmax and complete attention."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import csv
from dataclasses import dataclass
import math
import json
from pathlib import Path
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cuda_attention.attention import (
    custom_causal_attention,
    explicit_causal_attention,
    sdpa_causal_attention,
)
from cuda_attention.operator import DEFAULT_BLOCK_SIZE, fused_causal_softmax
from cuda_attention.benchmark import collect_cuda_run_metadata
from cuda_attention.reference import causal_scaled_softmax


PROFILE_REGION_NAMES = (
    "softmax/pytorch_eager",
    "softmax/custom_cuda",
    "attention/explicit_eager",
    "attention/custom_cuda",
    "attention/pytorch_sdpa",
)

PROFILER_SUMMARY_FIELDS = (
    "git_commit",
    "region",
    "calls",
    "cpu_time_total_us",
    "self_cpu_time_total_us",
    "cuda_time_total_us",
    "self_cuda_time_total_us",
    "sequence_length",
    "batch",
    "heads",
    "head_dimension",
    "block_size",
    "warmups",
    "repeats",
    "gpu_name",
    "compute_capability",
    "pytorch_version",
    "cuda_version",
    "timestamp",
)


@dataclass(frozen=True)
class ProfilerConfig:
    """A representative workload small enough to inspect in one trace."""

    sequence_length: int = 512
    batch: int = 1
    heads: int = 8
    head_dimension: int = 64
    warmups: int = 5
    repeats: int = 10
    block_size: int = DEFAULT_BLOCK_SIZE

    def __post_init__(self) -> None:
        if min(
            self.sequence_length,
            self.batch,
            self.heads,
            self.head_dimension,
            self.repeats,
        ) <= 0:
            raise ValueError("profile dimensions and repeats must be positive")
        if self.warmups < 0:
            raise ValueError("profile warmups must be nonnegative")


ProfileOperation = Callable[[], object]


def prepare_profile_operations(
    config: ProfilerConfig,
    *,
    device: torch.device | str,
    include_custom: bool,
) -> dict[str, ProfileOperation]:
    """Allocate shared inputs and return named operations for one profile run.

    Attention paths receive the same Q/K/V tensors. The isolated softmax paths
    receive the same already-materialized score matrix, so their profiler
    regions do not accidentally include QK^T.
    """

    generator = torch.Generator(device=device).manual_seed(2026)
    shape = (
        config.batch,
        config.heads,
        config.sequence_length,
        config.head_dimension,
    )
    query = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    key = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    value = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    scores = (query @ key.transpose(-2, -1)).reshape(-1, config.sequence_length)
    scale = 1.0 / math.sqrt(config.head_dimension)

    operations: dict[str, ProfileOperation] = {
        "softmax/pytorch_eager": lambda: causal_scaled_softmax(scores, scale),
        "attention/explicit_eager": lambda: explicit_causal_attention(
            query, key, value
        ),
        "attention/pytorch_sdpa": lambda: sdpa_causal_attention(query, key, value),
    }
    if include_custom:
        operations["softmax/custom_cuda"] = lambda: fused_causal_softmax(
            scores,
            scale,
            block_size=config.block_size,
        )
        operations["attention/custom_cuda"] = lambda: custom_causal_attention(
            query,
            key,
            value,
            block_size=config.block_size,
        )
    return {
        name: operations[name]
        for name in PROFILE_REGION_NAMES
        if name in operations
    }


def execute_profile_regions(
    operations: Mapping[str, ProfileOperation],
    *,
    repeats: int,
) -> dict[str, object]:
    """Execute named regions while preserving their last result.

    ``record_function`` creates human-readable parent ranges. CUDA kernels are
    asynchronous, so these labels describe logical ownership; device duration
    is obtained from the profiler's CUDA events, not Python wall-clock time.
    """

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    results: dict[str, object] = {}
    for _ in range(repeats):
        for name, operation in operations.items():
            with record_function(name):
                results[name] = operation()
    return results


def capture_cuda_profile(config: ProfilerConfig):
    """Warm up and capture one CUDA profile containing every comparison path."""

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch Profiler CUDA capture requires an NVIDIA GPU")
    operations = prepare_profile_operations(
        config,
        device="cuda",
        include_custom=True,
    )
    for _ in range(config.warmups):
        for operation in operations.values():
            operation()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as captured_profile:
        execute_profile_regions(operations, repeats=config.repeats)
        torch.cuda.synchronize()
    return captured_profile


def profiler_summary_rows(
    captured_profile,
    *,
    config: ProfilerConfig,
    metadata: Mapping[str, str],
) -> list[dict[str, object]]:
    """Extract named-region totals while retaining experimental provenance."""

    averages = {event.key: event for event in captured_profile.key_averages()}
    missing = [name for name in PROFILE_REGION_NAMES if name not in averages]
    if missing:
        raise ValueError(f"profile is missing required regions: {missing}")
    rows: list[dict[str, object]] = []
    for name in PROFILE_REGION_NAMES:
        event = averages[name]
        rows.append(
            {
                **metadata,
                "region": name,
                "calls": int(event.count),
                "cpu_time_total_us": float(event.cpu_time_total),
                "self_cpu_time_total_us": float(event.self_cpu_time_total),
                "cuda_time_total_us": float(event.device_time_total),
                "self_cuda_time_total_us": float(event.self_device_time_total),
                "sequence_length": config.sequence_length,
                "batch": config.batch,
                "heads": config.heads,
                "head_dimension": config.head_dimension,
                "block_size": config.block_size,
                "warmups": config.warmups,
                "repeats": config.repeats,
            }
        )
    return rows


def export_profile_artifacts(
    captured_profile,
    *,
    config: ProfilerConfig,
    metadata: Mapping[str, str],
    output_directory: Path,
) -> tuple[Path, Path, Path]:
    """Write a Chrome trace, region CSV, and metadata without overwriting."""

    trace_path = output_directory / "pytorch_trace.json"
    summary_path = output_directory / "pytorch_profiler_summary.csv"
    metadata_path = output_directory / "pytorch_profiler_metadata.json"
    outputs = (trace_path, summary_path, metadata_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite profiler artifacts: {existing}")

    output_directory.mkdir(parents=True, exist_ok=True)
    captured_profile.export_chrome_trace(str(trace_path))
    rows = profiler_summary_rows(
        captured_profile,
        config=config,
        metadata=metadata,
    )
    with summary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=PROFILER_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata_document = {
        **metadata,
        "sequence_length": config.sequence_length,
        "batch": config.batch,
        "heads": config.heads,
        "head_dimension": config.head_dimension,
        "block_size": config.block_size,
        "warmups": config.warmups,
        "repeats": config.repeats,
        "regions": list(PROFILE_REGION_NAMES),
    }
    with metadata_path.open("w", encoding="utf-8") as output_file:
        json.dump(metadata_document, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "raw" / "pytorch_profiler",
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()
    config = ProfilerConfig(
        sequence_length=arguments.sequence_length,
        warmups=arguments.warmups,
        repeats=arguments.repeats,
    )
    try:
        metadata = collect_cuda_run_metadata(PROJECT_ROOT)
        captured_profile = capture_cuda_profile(config)
        outputs = export_profile_artifacts(
            captured_profile,
            config=config,
            metadata=metadata,
            output_directory=arguments.output_dir,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
