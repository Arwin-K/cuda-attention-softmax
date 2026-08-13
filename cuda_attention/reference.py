"""Transparent PyTorch reference operations used to validate future CUDA code."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor


def causal_allowed_mask(
    rows: int,
    sequence_length: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the allowed positions for flattened causal-attention score rows.

    A score tensor originally shaped ``[batch, heads, S, S]`` is flattened to
    ``[rows, S]``. Every group of ``S`` rows repeats query positions
    ``0, ..., S - 1``, so ``row_index % S`` recovers the query position. A
    boolean ``True`` means the column is visible to that query.

    Args:
        rows: Number of flattened score rows.
        sequence_length: Number of score columns and query positions.
        device: Device on which to construct the boolean mask.

    Returns:
        Boolean tensor shaped ``[rows, sequence_length]``.
    """

    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
        raise ValueError("rows must be a positive integer")
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length <= 0
    ):
        raise ValueError("sequence_length must be a positive integer")

    row_indices = torch.arange(rows, device=device)
    query_positions = (row_indices % sequence_length).unsqueeze(1)
    column_indices = torch.arange(sequence_length, device=device).unsqueeze(0)
    return column_indices <= query_positions


def stable_softmax(values: Tensor, dim: int = -1) -> Tensor:
    """Compute softmax explicitly with the subtract-maximum stability trick.

    Subtracting the maximum does not change softmax because it multiplies every
    numerator and the denominator by the same factor. It does ensure the
    largest exponent input is zero, preventing large positive logits from
    producing an overflowing exponential.

    Args:
        values: Non-empty floating-point tensor.
        dim: Dimension whose values form each softmax row.

    Returns:
        Probabilities with the same shape and dtype as ``values``.

    Raises:
        TypeError: If ``values`` is not a floating-point tensor.
        ValueError: If ``values`` is scalar or the reduction dimension is empty.
    """

    if not isinstance(values, Tensor):
        raise TypeError("values must be a torch.Tensor")
    if not values.is_floating_point():
        raise TypeError("stable_softmax requires a floating-point tensor")
    if values.ndim == 0:
        raise ValueError("stable_softmax requires at least one dimension")

    normalized_dim = dim if dim >= 0 else values.ndim + dim
    if normalized_dim < 0 or normalized_dim >= values.ndim:
        raise IndexError(
            f"dim {dim} is out of range for a {values.ndim}-dimensional tensor"
        )
    if values.shape[normalized_dim] == 0:
        raise ValueError("stable_softmax cannot reduce an empty dimension")

    maximum = torch.amax(values, dim=dim, keepdim=True)
    shifted = values - maximum
    exponentials = torch.exp(shifted)
    denominator = torch.sum(exponentials, dim=dim, keepdim=True)
    return exponentials / denominator


def causal_scaled_softmax(scores: Tensor, scale: float) -> Tensor:
    """Apply scaling, causal masking, and stable row-wise softmax.

    Args:
        scores: Floating-point tensor shaped ``[rows, sequence_length]``.
        scale: Finite positive multiplier, normally ``1 / sqrt(head_dimension)``.

    Returns:
        Causal probabilities with the same shape, dtype, and device as
        ``scores``. Positions after each flattened row's query position are
        exactly zero.
    """

    if not isinstance(scores, Tensor):
        raise TypeError("scores must be a torch.Tensor")
    if not scores.is_floating_point():
        raise TypeError("causal_scaled_softmax requires floating-point scores")
    if scores.ndim != 2:
        raise ValueError("scores must have shape [rows, sequence_length]")
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("scores must contain at least one row and column")
    if not isinstance(scale, Real) or isinstance(scale, bool):
        raise TypeError("scale must be a real number")
    if not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("scale must be finite and positive")

    rows, sequence_length = scores.shape
    allowed = causal_allowed_mask(rows, sequence_length, device=scores.device)
    scaled_scores = scores * scale
    masked_scores = scaled_scores.masked_fill(~allowed, -torch.inf)
    return stable_softmax(masked_scores, dim=-1)
