"""Independent PyTorch-composition comparisons for explicit attention."""

import math

import pytest
import torch

from cuda_attention.attention import explicit_causal_attention


def pytorch_composed_causal_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trusted comparison using PyTorch softmax rather than the manual reference."""

    sequence_length = query.shape[-2]
    head_dimension = query.shape[-1]
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dimension)
    allowed = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=query.device,
    ).tril()
    probabilities = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
    return probabilities @ value, probabilities


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((1, 1, 4, 8), torch.float32),
        ((2, 2, 5, 3), torch.float32),
        ((1, 3, 3, 4), torch.float64),
    ],
)
def test_explicit_attention_matches_independent_pytorch_composition(
    shape: tuple[int, int, int, int], dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(47)
    query = torch.randn(shape, generator=generator, dtype=dtype)
    key = torch.randn(shape, generator=generator, dtype=dtype)
    value = torch.randn(shape, generator=generator, dtype=dtype)

    actual = explicit_causal_attention(query, key, value)
    expected_output, expected_probabilities = pytorch_composed_causal_attention(
        query, key, value
    )

    torch.testing.assert_close(actual.probabilities, expected_probabilities)
    torch.testing.assert_close(actual.output, expected_output)


def test_causality_prevents_future_value_changes_from_affecting_first_output() -> None:
    query = torch.tensor([[[[1.0], [1.0], [1.0]]]])
    key = torch.tensor([[[[1.0], [2.0], [3.0]]]])
    original_value = torch.tensor([[[[5.0], [10.0], [20.0]]]])
    changed_future_value = torch.tensor([[[[5.0], [-1000.0], [1000.0]]]])

    original = explicit_causal_attention(query, key, original_value)
    changed = explicit_causal_attention(query, key, changed_future_value)

    torch.testing.assert_close(original.output[..., 0, :], changed.output[..., 0, :])
