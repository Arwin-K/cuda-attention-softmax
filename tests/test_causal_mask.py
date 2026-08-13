"""Tests for flattened-row causal-mask semantics."""

import pytest
import torch

from cuda_attention.reference import causal_allowed_mask, causal_scaled_softmax


def test_allowed_mask_has_hand_verifiable_boundary_rows() -> None:
    mask = causal_allowed_mask(rows=5, sequence_length=4)

    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
            [True, False, False, False],
        ]
    )
    assert torch.equal(mask, expected)


def test_future_probabilities_are_exactly_zero() -> None:
    scores = torch.tensor(
        [
            [1.0, 100.0, 100.0, 100.0],
            [1.0, 2.0, 100.0, 100.0],
            [1.0, 2.0, 3.0, 100.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
    )
    allowed = causal_allowed_mask(rows=4, sequence_length=4)

    probabilities = causal_scaled_softmax(scores, scale=1.0)

    assert torch.count_nonzero(probabilities.masked_select(~allowed)) == 0
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(4))
    torch.testing.assert_close(probabilities[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_last_query_matches_unmasked_softmax() -> None:
    scores = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [3.0, 2.0, 1.0]])

    probabilities = causal_scaled_softmax(scores, scale=0.5)

    torch.testing.assert_close(probabilities[-1], torch.softmax(scores[-1] * 0.5, dim=0))


@pytest.mark.parametrize(("rows", "sequence_length"), [(0, 4), (4, 0), (-1, 4)])
def test_mask_rejects_nonpositive_shapes(rows: int, sequence_length: int) -> None:
    with pytest.raises(ValueError):
        causal_allowed_mask(rows, sequence_length)
