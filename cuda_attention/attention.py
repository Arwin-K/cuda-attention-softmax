"""Explicit scaled dot-product attention built from inspectable PyTorch steps."""

from __future__ import annotations

from dataclasses import dataclass
import math

from torch import Tensor

from .operator import DEFAULT_BLOCK_SIZE, fused_causal_softmax
from .reference import causal_scaled_softmax


@dataclass(frozen=True)
class AttentionResult:
    """Attention output plus probabilities retained for correctness inspection."""

    output: Tensor
    probabilities: Tensor


def _validate_qkv(query: Tensor, key: Tensor, value: Tensor) -> None:
    """Validate the common [batch, heads, sequence, head_dim] contract."""

    if not all(isinstance(tensor, Tensor) for tensor in (query, key, value)):
        raise TypeError("query, key, and value must be torch.Tensor objects")
    if not all(tensor.is_floating_point() for tensor in (query, key, value)):
        raise TypeError("query, key, and value must use floating-point dtypes")
    if not all(tensor.ndim == 4 for tensor in (query, key, value)):
        raise ValueError(
            "query, key, and value must have shape "
            "[batch, heads, sequence_length, head_dimension]"
        )
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError("query, key, and value must have identical dtypes")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    if any(dimension == 0 for dimension in query.shape):
        raise ValueError("query, key, and value dimensions must be nonzero")


def explicit_causal_attention(query: Tensor, key: Tensor, value: Tensor) -> AttentionResult:
    """Compute ``QK^T -> causal scaled softmax -> PV`` explicitly.

    Keeping the steps separate makes the future replacement boundary clear:
    the custom CUDA operator will replace only the scaling/masking/softmax
    stage, while PyTorch continues to perform both matrix multiplications.
    """

    _validate_qkv(query, key, value)
    batch, heads, sequence_length, head_dimension = query.shape

    scores = query @ key.transpose(-2, -1)
    flattened_scores = scores.reshape(-1, sequence_length)
    flattened_probabilities = causal_scaled_softmax(
        flattened_scores, scale=1.0 / math.sqrt(head_dimension)
    )
    probabilities = flattened_probabilities.reshape(
        batch, heads, sequence_length, sequence_length
    )
    output = probabilities @ value
    return AttentionResult(output=output, probabilities=probabilities)


def custom_causal_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> AttentionResult:
    """Compute explicit attention with only softmax replaced by custom CUDA.

    PyTorch still owns both matrix multiplications. The score tensor is flattened
    from ``[batch, heads, sequence, sequence]`` to the custom operator's
    ``[rows, sequence]`` contract, where ``row % sequence`` recovers the query
    position used by the in-kernel causal mask.
    """

    _validate_qkv(query, key, value)
    batch, heads, sequence_length, head_dimension = query.shape

    scores = query @ key.transpose(-2, -1)
    flattened_scores = scores.reshape(-1, sequence_length).contiguous()
    flattened_probabilities = fused_causal_softmax(
        flattened_scores,
        scale=1.0 / math.sqrt(head_dimension),
        block_size=block_size,
    )
    probabilities = flattened_probabilities.reshape(
        batch,
        heads,
        sequence_length,
        sequence_length,
    )
    output = probabilities @ value
    return AttentionResult(output=output, probabilities=probabilities)
