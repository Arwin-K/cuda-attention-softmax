"""Shape, probability, and input-contract tests for explicit attention."""

import math

import pytest
import torch

from cuda_attention.attention import (
    custom_causal_attention,
    explicit_causal_attention,
    sdpa_causal_attention,
)
from cuda_attention.operator import cuda_extension_available
from cuda_attention.reference import causal_scaled_softmax


CUDA_ATTENTION_UNAVAILABLE_REASON = (
    None
    if torch.cuda.is_available() and cuda_extension_available()
    else "requires an NVIDIA CUDA device and compiled cuda_attention._C extension"
)


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


def test_custom_attention_replaces_only_the_softmax_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (2, 3, 5, 4)
    generator = torch.Generator().manual_seed(81)
    query = torch.randn(shape, generator=generator)
    key = torch.randn(shape, generator=generator)
    value = torch.randn(shape, generator=generator)
    observed: dict[str, object] = {}

    def trusted_softmax(scores: torch.Tensor, scale: float, block_size: int) -> torch.Tensor:
        observed.update(
            shape=tuple(scores.shape),
            contiguous=scores.is_contiguous(),
            scale=scale,
            block_size=block_size,
        )
        return causal_scaled_softmax(scores, scale)

    monkeypatch.setattr("cuda_attention.attention.fused_causal_softmax", trusted_softmax)

    actual = custom_causal_attention(query, key, value, block_size=128)
    expected = explicit_causal_attention(query, key, value)

    torch.testing.assert_close(actual.output, expected.output)
    torch.testing.assert_close(actual.probabilities, expected.probabilities)
    assert observed == {
        "shape": (2 * 3 * 5, 5),
        "contiguous": True,
        "scale": 1.0 / math.sqrt(4),
        "block_size": 128,
    }


@pytest.mark.skipif(
    CUDA_ATTENTION_UNAVAILABLE_REASON is not None,
    reason=CUDA_ATTENTION_UNAVAILABLE_REASON,
)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 31, 16),
        (1, 2, 33, 32),
        (2, 2, 64, 64),
    ],
)
def test_custom_attention_matches_explicit_reference_on_cuda(
    shape: tuple[int, int, int, int],
) -> None:
    batch, heads, sequence_length, _ = shape
    generator = torch.Generator(device="cuda").manual_seed(sum(shape) + 82)
    query = torch.randn(shape, generator=generator, device="cuda")
    key = torch.randn(shape, generator=generator, device="cuda")
    value = torch.randn(shape, generator=generator, device="cuda")

    actual = custom_causal_attention(query, key, value)
    expected = explicit_causal_attention(query, key, value)

    torch.testing.assert_close(actual.output, expected.output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        actual.probabilities,
        expected.probabilities,
        rtol=1e-5,
        atol=1e-6,
    )
    assert actual.output.shape == shape
    assert actual.output.dtype == query.dtype
    assert actual.output.device == query.device
    assert torch.isfinite(actual.output).all()
    assert torch.isfinite(actual.probabilities).all()
    torch.testing.assert_close(
        actual.probabilities.sum(dim=-1),
        torch.ones(batch, heads, sequence_length, device="cuda"),
        rtol=1e-5,
        atol=1e-6,
    )
    future = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device="cuda"),
        diagonal=1,
    )
    assert torch.count_nonzero(actual.probabilities.masked_select(future)) == 0


@pytest.mark.parametrize("sequence_length", [1, 4, 7])
def test_sdpa_causal_baseline_matches_explicit_reference(
    sequence_length: int,
) -> None:
    shape = (1, 2, sequence_length, 8)
    generator = torch.Generator().manual_seed(83 + sequence_length)
    query = torch.randn(shape, generator=generator)
    key = torch.randn(shape, generator=generator)
    value = torch.randn(shape, generator=generator)

    actual = sdpa_causal_attention(query, key, value)
    expected = explicit_causal_attention(query, key, value).output

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.shape == shape
    assert actual.dtype == query.dtype
    assert torch.isfinite(actual).all()
