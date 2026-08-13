"""Correctness tests for the transparent stable-softmax reference."""

import pytest
import torch

from cuda_attention.reference import stable_softmax


def test_equal_values_produce_uniform_probabilities() -> None:
    values = torch.tensor([[3.0, 3.0, 3.0, 3.0]], dtype=torch.float32)

    actual = stable_softmax(values)

    torch.testing.assert_close(actual, torch.full_like(values, 0.25))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_torch_softmax_and_preserves_invariants(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(5, 11, generator=generator, dtype=dtype)

    actual = stable_softmax(values, dim=-1)
    expected = torch.softmax(values, dim=-1)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(5, dtype=dtype))
    assert torch.all(actual >= 0)
    assert actual.shape == values.shape
    assert actual.dtype == values.dtype


def test_supports_a_nonfinal_reduction_dimension() -> None:
    values = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
    )

    actual = stable_softmax(values, dim=0)

    torch.testing.assert_close(actual, torch.softmax(values, dim=0))
    torch.testing.assert_close(actual.sum(dim=0), torch.ones(3, dtype=torch.float64))


@pytest.mark.parametrize(
    ("values", "error_type"),
    [
        (torch.tensor([1, 2, 3]), TypeError),
        (torch.tensor(1.0), ValueError),
        (torch.empty(2, 0), ValueError),
    ],
)
def test_rejects_inputs_without_a_valid_softmax_row(
    values: torch.Tensor, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        stable_softmax(values)
