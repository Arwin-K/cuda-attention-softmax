"""Deterministic tensor and seed helpers shared by tests and experiments."""

from __future__ import annotations

import math
import random
import csv
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import torch
from torch import Tensor


DEFAULT_SEED = 0

RAW_BENCHMARK_FIELDS = (
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
    "sample_index",
    "sample_us",
    "gpu_name",
    "compute_capability",
    "pytorch_version",
    "cuda_version",
    "timestamp",
)


def current_git_commit(repository_root: Path) -> str:
    """Return the exact implementation revision attached to a measurement."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def collect_cuda_run_metadata(repository_root: Path) -> dict[str, str]:
    """Capture immutable run provenance from the active CUDA environment."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark metadata requires an NVIDIA GPU")
    device_index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    return {
        "git_commit": current_git_commit(repository_root),
        "gpu_name": torch.cuda.get_device_name(device_index),
        "compute_capability": f"{major}.{minor}",
        "pytorch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def raw_benchmark_records(
    *,
    metadata: Mapping[str, str],
    implementation_description: str,
    sequence_length: int,
    rows: int,
    columns: int,
    dtype: str,
    warmups: int,
    iterations: int,
    samples_us: Sequence[float],
    launch_block_size: int | None = None,
    compile_warmups: int = 0,
) -> list[dict[str, object]]:
    """Combine raw samples with enough context to reproduce their workload."""

    if len(samples_us) != iterations:
        raise ValueError("sample count must equal configured iterations")
    records: list[dict[str, object]] = []
    for sample_index, sample_us in enumerate(samples_us):
        records.append(
            {
                **metadata,
                "implementation_description": implementation_description,
                "sequence_length": sequence_length,
                "rows": rows,
                "columns": columns,
                "dtype": dtype,
                "launch_block_size": (
                    "" if launch_block_size is None else launch_block_size
                ),
                "compile_warmups": compile_warmups,
                "warmups": warmups,
                "iterations": iterations,
                "sample_index": sample_index,
                "sample_us": float(sample_us),
            }
        )
    return records


def write_raw_benchmark_csv(
    output_path: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    """Write raw records with a stable, reviewable column order."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RAW_BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def _validate_timing_count(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def run_untimed_warmups(
    operation: Callable[[], object],
    *,
    iterations: int,
) -> object:
    """Run explicit startup calls and return the last result without timing it."""

    if not callable(operation):
        raise TypeError("operation must be callable")
    _validate_timing_count("iterations", iterations)
    result: object = None
    for _ in range(iterations):
        result = operation()
    return result


def time_cuda_callable(
    operation: Callable[[], object],
    *,
    warmups: int,
    iterations: int,
) -> list[float]:
    """Return per-iteration CUDA-event durations in microseconds.

    CUDA launches are asynchronous with respect to Python. Events are recorded
    on the current CUDA stream around only ``operation``; synchronizing the end
    event waits for that measured work without including input construction in
    the interval. Warmups happen first so one-time initialization does not
    become a steady-state sample.
    """

    if not callable(operation):
        raise TypeError("operation must be callable")
    _validate_timing_count("warmups", warmups)
    _validate_timing_count("iterations", iterations)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA event timing requires an available NVIDIA GPU")

    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_us: list[float] = []
    for _ in range(iterations):
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples_us.append(float(start.elapsed_time(end)) * 1000.0)
    return samples_us


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _validate_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_floating_dtype(dtype: torch.dtype) -> None:
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("experiment tensors require a floating-point dtype")


def set_experiment_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python and PyTorch global random-number generators.

    This helper intentionally does not seed Apple's MPS backend as a substitute
    for CUDA. Device-local tensor factories below use an explicit generator so
    their values do not depend on unrelated global random draws.
    """

    _validate_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _make_generator(device: torch.device | str, seed: int) -> torch.Generator:
    _validate_seed(seed)
    normalized_device = torch.device(device)
    generator_device = normalized_device.type if normalized_device.index is None else normalized_device
    return torch.Generator(device=generator_device).manual_seed(seed)


def make_score_tensor(
    sequence_length: int,
    *,
    batch_heads: int = 8,
    seed: int = DEFAULT_SEED,
    magnitude: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Create reproducible flattened attention scores shaped ``[B*H*S, S]``."""

    _validate_positive_integer("sequence_length", sequence_length)
    _validate_positive_integer("batch_heads", batch_heads)
    _validate_floating_dtype(dtype)
    if not isinstance(magnitude, (int, float)) or isinstance(magnitude, bool):
        raise TypeError("magnitude must be a real number")
    if not math.isfinite(float(magnitude)) or magnitude < 0:
        raise ValueError("magnitude must be finite and nonnegative")

    generator = _make_generator(device, seed)
    return torch.randn(
        batch_heads * sequence_length,
        sequence_length,
        generator=generator,
        dtype=dtype,
        device=device,
    ) * magnitude


def make_qkv_tensors(
    batch_size: int,
    heads: int,
    sequence_length: int,
    head_dimension: int,
    *,
    seed: int = DEFAULT_SEED,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor, Tensor]:
    """Create reproducible, distinct Q/K/V tensors from one local RNG stream."""

    for name, value in (
        ("batch_size", batch_size),
        ("heads", heads),
        ("sequence_length", sequence_length),
        ("head_dimension", head_dimension),
    ):
        _validate_positive_integer(name, value)
    _validate_floating_dtype(dtype)

    generator = _make_generator(device, seed)
    shape = (batch_size, heads, sequence_length, head_dimension)
    query = torch.randn(shape, generator=generator, dtype=dtype, device=device)
    key = torch.randn(shape, generator=generator, dtype=dtype, device=device)
    value = torch.randn(shape, generator=generator, dtype=dtype, device=device)
    return query, key, value
