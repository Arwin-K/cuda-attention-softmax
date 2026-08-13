"""CPU reference tests around future CUDA warp/block boundaries.

Testing values immediately below, at, and above powers of two catches indexing
or reduction assumptions that convenient widths can hide.
"""

import pytest
import torch

from cuda_attention.reference import causal_allowed_mask, causal_scaled_softmax


CORRECTNESS_SEQUENCE_LENGTHS = (31, 32, 33, 63, 64, 127, 128, 255, 511, 768, 1023)


@pytest.mark.parametrize("sequence_length", CORRECTNESS_SEQUENCE_LENGTHS)
def test_reference_supports_planned_irregular_sequence_lengths(
    sequence_length: int,
) -> None:
    rows = 2 * sequence_length
    generator = torch.Generator().manual_seed(sequence_length)
    scores = torch.randn(rows, sequence_length, generator=generator)
    allowed = causal_allowed_mask(rows, sequence_length)

    actual = causal_scaled_softmax(scores, scale=0.125)
    expected = torch.softmax(
        (scores * 0.125).masked_fill(~allowed, -torch.inf), dim=-1
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        actual.sum(dim=-1), torch.ones(rows), rtol=1e-5, atol=1e-6
    )
    assert torch.count_nonzero(actual.masked_select(~allowed)) == 0
    assert torch.isfinite(actual).all()
    assert actual.shape == scores.shape
    assert actual.dtype == scores.dtype
