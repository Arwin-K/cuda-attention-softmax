"""Independent PyTorch-composition comparisons for explicit attention."""

import math

import pytest
import torch

from cuda_attention.attention import explicit_causal_attention
from cuda_attention.benchmark import (
    make_qkv_tensors,
    make_score_tensor,
    set_experiment_seed,
)


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


def test_score_factory_is_repeatable_and_has_primary_experiment_shape() -> None:
    first = make_score_tensor(7, batch_heads=3, seed=101, dtype=torch.float64)
    repeated = make_score_tensor(7, batch_heads=3, seed=101, dtype=torch.float64)
    changed_seed = make_score_tensor(7, batch_heads=3, seed=102, dtype=torch.float64)

    assert first.shape == (21, 7)
    assert first.dtype == torch.float64
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed_seed)


def test_qkv_factory_is_repeatable_but_produces_distinct_tensors() -> None:
    first = make_qkv_tensors(2, 3, 5, 4, seed=211)
    repeated = make_qkv_tensors(2, 3, 5, 4, seed=211)

    for tensor, repeated_tensor in zip(first, repeated, strict=True):
        assert tensor.shape == (2, 3, 5, 4)
        assert torch.equal(tensor, repeated_tensor)
    query, key, value = first
    assert not torch.equal(query, key)
    assert not torch.equal(key, value)


def test_global_seed_helper_repeats_python_and_pytorch_draws() -> None:
    import random

    set_experiment_seed(307)
    first_python = random.random()
    first_torch = torch.rand(4)
    set_experiment_seed(307)

    assert random.random() == first_python
    assert torch.equal(torch.rand(4), first_torch)


@pytest.mark.parametrize("seed", [-1, True])
def test_factories_reject_invalid_seeds(seed: int) -> None:
    with pytest.raises(ValueError):
        make_score_tensor(4, seed=seed)
