"""Deterministic tensor and seed helpers shared by tests and experiments."""

from __future__ import annotations

import math
import random

import torch
from torch import Tensor


DEFAULT_SEED = 0


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
