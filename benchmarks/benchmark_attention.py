"""Benchmark equivalent complete causal-attention paths with CUDA events."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from collections.abc import Callable, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.config import (
    AttentionBenchmarkConfig,
    attention_benchmark_registry,
)
from cuda_attention.attention import (
    custom_causal_attention,
    explicit_causal_attention,
    sdpa_causal_attention,
)
from cuda_attention.benchmark import collect_cuda_run_metadata, time_cuda_callable
from cuda_attention.operator import CudaExtensionUnavailableError, cuda_extension_available


ATTENTION_RAW_FIELDS = (
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
    "sample_index",
    "sample_us",
    "gpu_name",
    "compute_capability",
    "pytorch_version",
    "cuda_version",
    "timestamp",
)


def make_qkv(
    config: AttentionBenchmarkConfig,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate deterministic Q/K/V once, outside every timed operation."""

    generator = torch.Generator(device=device).manual_seed(config.seed)
    tensors = tuple(
        torch.randn(
            config.shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        for _ in range(3)
    )
    return tensors  # type: ignore[return-value]


def prepare_attention_operations(
    config: AttentionBenchmarkConfig,
    *,
    device: torch.device | str = "cuda",
    include_custom: bool = True,
) -> dict[str, Callable[[], torch.Tensor]]:
    """Create complete paths sharing the same already-allocated Q/K/V tensors."""

    query, key, value = make_qkv(config, device=device)
    operations: dict[str, Callable[[], torch.Tensor]] = {
        "explicit_eager": lambda: explicit_causal_attention(query, key, value).output,
        "pytorch_sdpa": lambda: sdpa_causal_attention(query, key, value),
    }
    if include_custom:
        if not cuda_extension_available():
            raise CudaExtensionUnavailableError(
                "custom attention benchmark requires the compiled CUDA extension"
            )
        operations["custom_cuda"] = lambda: custom_causal_attention(
            query,
            key,
            value,
            block_size=config.block_size,
        ).output
    return operations


def validate_attention_operations(
    operations: Mapping[str, Callable[[], torch.Tensor]],
) -> None:
    """Reject a path before timing when it disagrees with explicit attention."""

    if "explicit_eager" not in operations:
        raise ValueError("attention validation requires explicit_eager")
    expected = operations["explicit_eager"]()
    for name, operation in operations.items():
        if name == "explicit_eager":
            continue
        torch.testing.assert_close(operation(), expected, rtol=1e-5, atol=1e-6)
    if expected.is_cuda:
        torch.cuda.synchronize()


def attention_raw_records(
    *,
    metadata: Mapping[str, str],
    config: AttentionBenchmarkConfig,
    implementation: str,
    samples_us: Sequence[float],
) -> list[dict[str, object]]:
    """Attach full-workload controls and provenance to raw CUDA samples."""

    if len(samples_us) != config.iterations:
        raise ValueError("sample count must equal configured iterations")
    return [
        {
            **metadata,
            "implementation": implementation,
            "sequence_length": config.sequence_length,
            "batch": config.batch,
            "heads": config.heads,
            "head_dimension": config.head_dimension,
            "dtype": config.dtype,
            "block_size": config.block_size if implementation == "custom_cuda" else "",
            "warmups": config.warmups,
            "iterations": config.iterations,
            "sample_index": sample_index,
            "sample_us": float(sample_us),
        }
        for sample_index, sample_us in enumerate(samples_us)
    ]


def write_attention_raw_csv(
    output_path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=ATTENTION_RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_attention_records(
    output_path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Persist every completed case so a remote-runtime loss keeps evidence."""

    write_attention_raw_csv(output_path, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--block-size", type=int, choices=(128, 256, 512), default=256)
    parser.add_argument(
        "--implementation",
        choices=("explicit_eager", "custom_cuda", "pytorch_sdpa", "all"),
        default="all",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        print("attention benchmark requires Linux with an NVIDIA GPU", file=sys.stderr)
        return 2
    if arguments.output.exists():
        print(f"refusing to overwrite existing artifact: {arguments.output}", file=sys.stderr)
        return 2

    registry = attention_benchmark_registry(block_size=arguments.block_size)
    if arguments.sequence_length is not None:
        registry = (
            AttentionBenchmarkConfig(
                sequence_length=arguments.sequence_length,
                warmups=arguments.warmups or registry[0].warmups,
                iterations=arguments.iterations or registry[0].iterations,
                block_size=arguments.block_size,
            ),
        )
    selected = (
        ("explicit_eager", "custom_cuda", "pytorch_sdpa")
        if arguments.implementation == "all"
        else (arguments.implementation,)
    )
    metadata = collect_cuda_run_metadata(PROJECT_ROOT)
    records: list[dict[str, object]] = []
    try:
        for config in registry:
            operations = prepare_attention_operations(config)
            validate_attention_operations(operations)
            for implementation in selected:
                samples = time_cuda_callable(
                    operations[implementation],
                    warmups=config.warmups,
                    iterations=config.iterations,
                )
                records.extend(
                    attention_raw_records(
                        metadata=metadata,
                        config=config,
                        implementation=implementation,
                        samples_us=samples,
                    )
                )
                checkpoint_attention_records(arguments.output, records)
    except (CudaExtensionUnavailableError, AssertionError) as error:
        print(str(error), file=sys.stderr)
        return 2

    print(f"wrote {len(records)} raw attention samples to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
