"""CUDA correctness gates for the optional fused operator.

These tests are collected on every platform. They skip when either a CUDA
device or the compiled extension is absent, which keeps Apple Silicon useful
for CPU development without pretending that MPS validates CUDA behavior.
"""

import math

import pytest
import torch

from cuda_attention.operator import cuda_extension_available, fused_causal_softmax
from cuda_attention.reference import causal_allowed_mask, causal_scaled_softmax


RTOL = 1e-5
ATOL = 1e-6
CORE_SEQUENCE_LENGTHS = (32, 64, 128)
IRREGULAR_SEQUENCE_LENGTHS = (31, 33, 63, 127, 255, 511, 768, 1023)
STRESS_MAGNITUDES = (10.0, 100.0, 1000.0)


def _cuda_test_unavailable_reason() -> str | None:
    if not torch.cuda.is_available():
        return "requires an NVIDIA CUDA device; MPS is not a CUDA substitute"
    if not cuda_extension_available():
        return "requires the compiled cuda_attention._C extension"
    return None


CUDA_TEST_UNAVAILABLE_REASON = _cuda_test_unavailable_reason()
pytestmark = pytest.mark.skipif(
    CUDA_TEST_UNAVAILABLE_REASON is not None,
    reason=CUDA_TEST_UNAVAILABLE_REASON or "CUDA test prerequisites unavailable",
)


@pytest.mark.parametrize("sequence_length", CORE_SEQUENCE_LENGTHS)
def test_cuda_operator_matches_pytorch_reference(sequence_length: int) -> None:
    """Compare the same FP32 scale-mask-softmax work across implementations."""

    generator = torch.Generator(device="cuda").manual_seed(sequence_length)
    rows = 2 * sequence_length
    scores = torch.randn(
        rows,
        sequence_length,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    scale = 1.0 / math.sqrt(64)

    actual = fused_causal_softmax(scores, scale)
    expected = causal_scaled_softmax(scores, scale)
    allowed = causal_allowed_mask(rows, sequence_length, device="cuda")

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(rows, device="cuda"),
        rtol=RTOL,
        atol=ATOL,
    )
    assert actual.shape == scores.shape
    assert actual.dtype == scores.dtype
    assert actual.device == scores.device
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual.masked_select(~allowed)) == 0


def _assert_cuda_matches_reference(scores: torch.Tensor, scale: float = 1.0) -> None:
    actual = fused_causal_softmax(scores, scale)
    expected = causal_scaled_softmax(scores, scale)
    allowed = causal_allowed_mask(
        scores.shape[0],
        scores.shape[1],
        device=scores.device,
    )

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(scores.shape[0], device=scores.device),
        rtol=RTOL,
        atol=ATOL,
    )
    assert actual.shape == scores.shape
    assert actual.dtype == scores.dtype
    assert actual.device == scores.device
    assert torch.all(actual >= 0)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual.masked_select(~allowed)) == 0


@pytest.mark.parametrize("magnitude", STRESS_MAGNITUDES)
def test_cuda_operator_is_stable_for_large_random_logits(magnitude: float) -> None:
    generator = torch.Generator(device="cuda").manual_seed(330)
    scores = torch.randn(128, 64, generator=generator, device="cuda") * magnitude

    _assert_cuda_matches_reference(scores)


@pytest.mark.parametrize(
    "case",
    ["zeros", "equal", "dominant-positive", "dominant-negative"],
)
def test_cuda_operator_matches_structured_stress_cases(case: str) -> None:
    scores = torch.zeros(128, 64, device="cuda")
    if case == "equal":
        scores.fill_(1000.0)
    elif case == "dominant-positive":
        scores[:, 0] = 1000.0
    elif case == "dominant-negative":
        scores[:, 0] = -1000.0

    _assert_cuda_matches_reference(scores)


@pytest.mark.parametrize("sequence_length", IRREGULAR_SEQUENCE_LENGTHS)
def test_cuda_operator_supports_irregular_sequence_lengths(
    sequence_length: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(sequence_length)
    scores = torch.randn(
        sequence_length,
        sequence_length,
        generator=generator,
        device="cuda",
    )

    _assert_cuda_matches_reference(scores, scale=0.125)


def test_cuda_operator_rejects_cpu_scores() -> None:
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        fused_causal_softmax(torch.randn(4, 4), scale=1.0)


def test_cuda_operator_rejects_noncontiguous_scores() -> None:
    scores = torch.randn(4, 4, device="cuda").transpose(0, 1)
    assert not scores.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous"):
        fused_causal_softmax(scores, scale=1.0)


def test_cuda_operator_rejects_non_fp32_scores() -> None:
    scores = torch.randn(4, 4, device="cuda", dtype=torch.float64)

    with pytest.raises(RuntimeError, match="torch.float32"):
        fused_causal_softmax(scores, scale=1.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
def test_cuda_operator_rejects_invalid_scale(scale: float) -> None:
    scores = torch.randn(4, 4, device="cuda")

    with pytest.raises(RuntimeError, match="finite and positive"):
        fused_causal_softmax(scores, scale)
