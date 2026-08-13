"""Shape, probability, and input-contract tests for explicit attention."""

import pytest
import torch

from cuda_attention.attention import explicit_causal_attention


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 1),
        (1, 2, 4, 8),
        (2, 3, 5, 4),
    ],
)
def test_attention_shapes_and_probability_invariants(
    shape: tuple[int, int, int, int],
) -> None:
    batch, heads, sequence_length, _ = shape
    generator = torch.Generator().manual_seed(sum(shape))
    query = torch.randn(shape, generator=generator)
    key = torch.randn(shape, generator=generator)
    value = torch.randn(shape, generator=generator)

    result = explicit_causal_attention(query, key, value)

    assert result.output.shape == shape
    assert result.output.dtype == query.dtype
    assert result.probabilities.shape == (
        batch,
        heads,
        sequence_length,
        sequence_length,
    )
    assert result.probabilities.dtype == query.dtype
    torch.testing.assert_close(
        result.probabilities.sum(dim=-1), torch.ones(batch, heads, sequence_length)
    )
    future = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1
    )
    assert torch.count_nonzero(result.probabilities.masked_select(future)) == 0
    assert torch.isfinite(result.output).all()
    assert torch.isfinite(result.probabilities).all()


def test_attention_rejects_mismatched_shapes() -> None:
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 3, 8)
    value = torch.randn(1, 2, 4, 8)

    with pytest.raises(ValueError, match="identical shapes"):
        explicit_causal_attention(query, key, value)


def test_attention_rejects_non_four_dimensional_inputs() -> None:
    tensor = torch.randn(2, 4, 8)

    with pytest.raises(ValueError, match=r"\[batch, heads"):
        explicit_causal_attention(tensor, tensor, tensor)


def test_attention_rejects_mismatched_dtypes() -> None:
    query = torch.randn(1, 1, 2, 4, dtype=torch.float32)
    key = torch.randn(1, 1, 2, 4, dtype=torch.float64)
    value = torch.randn(1, 1, 2, 4, dtype=torch.float32)

    with pytest.raises(TypeError, match="identical dtypes"):
        explicit_causal_attention(query, key, value)
