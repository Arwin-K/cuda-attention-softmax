"""Stress tests for stable floating-point softmax behavior.

Large positive logits make a direct ``exp(x)`` implementation overflow in
FP32. Maximum subtraction keeps the largest exponent input at zero while
preserving the exact mathematical probability ratio.
"""

import pytest
import torch

from cuda_attention.reference import stable_softmax


RTOL = 1e-5
ATOL = 1e-6


@pytest.mark.parametrize("magnitude", [10.0, 100.0, 1000.0])
def test_scaled_random_logits_remain_finite_and_match_pytorch(magnitude: float) -> None:
    generator = torch.Generator().manual_seed(23)
    values = torch.randn(8, 17, generator=generator, dtype=torch.float32) * magnitude

    actual = stable_softmax(values)
    expected = torch.softmax(values, dim=-1)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(8), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "values",
    [
        torch.zeros(2, 8),
        torch.full((2, 8), 1000.0),
        torch.tensor([[1000.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[-1000.0, 0.0, 0.0, 0.0]]),
    ],
    ids=["zeros", "equal-large", "dominant-positive", "dominant-negative"],
)
def test_structured_extreme_inputs_match_pytorch(values: torch.Tensor) -> None:
    actual = stable_softmax(values)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, torch.softmax(values, dim=-1), rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(values.shape[0]),
        rtol=RTOL,
        atol=ATOL,
    )


def test_maximum_subtraction_avoids_naive_exponential_overflow() -> None:
    values = torch.tensor([[1000.0, 999.0, 998.0]], dtype=torch.float32)

    naive_exponentials = torch.exp(values)
    stable_probabilities = stable_softmax(values)

    assert torch.isinf(naive_exponentials).all()
    assert torch.isfinite(stable_probabilities).all()
    torch.testing.assert_close(stable_probabilities, torch.softmax(values, dim=-1))
