"""Transparent PyTorch reference operations used to validate future CUDA code."""

from __future__ import annotations

import torch
from torch import Tensor


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
