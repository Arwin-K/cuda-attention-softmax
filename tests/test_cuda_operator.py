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
# This union is the explicit regression contract from AGENTS.md. Keeping it
# visible in the CUDA gate prevents later optimization work from silently
# dropping an awkward length merely because another parametrized group changed.
REQUIRED_SEQUENCE_LENGTHS = (31, 32, 33, 63, 64, 127, 128, 255, 511, 768, 1023)
CORE_SEQUENCE_LENGTHS = (32, 64, 128)
IRREGULAR_SEQUENCE_LENGTHS = (
    1,
    2,
    3,
    31,
    33,
    63,
    127,
    255,
    257,
    511,
    513,
    768,
    1023,
    1025,
)
STRESS_MAGNITUDES = (10.0, 100.0, 1000.0)
STRESS_SEQUENCE_LENGTHS = (31, 128, 511)
ROW_WRAP_SHAPES = ((1, 31), (30, 31), (31, 31), (32, 31), (260, 257))
PARTIAL_WARP_ALLOWED_COLUMNS = (1, 2, 31, 32, 33, 63, 64, 65, 95, 96, 97)


@pytest.mark.cuda_static
def test_cuda_gate_covers_every_required_sequence_length() -> None:
    covered = set(CORE_SEQUENCE_LENGTHS) | set(IRREGULAR_SEQUENCE_LENGTHS)

    assert set(REQUIRED_SEQUENCE_LENGTHS) <= covered


def _cuda_test_unavailable_reason() -> str | None:
    if not torch.cuda.is_available():
        return "requires an NVIDIA CUDA device; MPS is not a CUDA substitute"
    if not cuda_extension_available():
        return "requires the compiled cuda_attention._C extension"
    return None


CUDA_TEST_UNAVAILABLE_REASON = _cuda_test_unavailable_reason()


@pytest.fixture(autouse=True)
def require_cuda_for_device_cases(request: pytest.FixtureRequest) -> None:
    """Skip device execution while allowing static coverage gates to run."""

    if request.node.get_closest_marker("cuda_static") is not None:
        return
    if CUDA_TEST_UNAVAILABLE_REASON is not None:
        pytest.skip(CUDA_TEST_UNAVAILABLE_REASON)


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


@pytest.mark.parametrize("sequence_length", STRESS_SEQUENCE_LENGTHS)
@pytest.mark.parametrize("magnitude", STRESS_MAGNITUDES)
def test_cuda_operator_is_stable_for_large_random_logits(
    magnitude: float,
    sequence_length: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        330 + sequence_length
    )
    scores = torch.randn(
        2 * sequence_length,
        sequence_length,
        generator=generator,
        device="cuda",
    ) * magnitude

    _assert_cuda_matches_reference(scores)


@pytest.mark.parametrize(
    "case",
    ["zeros", "equal", "dominant-positive", "dominant-negative"],
)
def test_cuda_operator_matches_structured_stress_cases(case: str) -> None:
    scores = torch.zeros(254, 127, device="cuda")
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


@pytest.mark.parametrize(("rows", "sequence_length"), ROW_WRAP_SHAPES)
def test_cuda_operator_handles_partial_and_wrapped_query_cycles(
    rows: int,
    sequence_length: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(rows + sequence_length)
    scores = torch.randn(
        rows,
        sequence_length,
        generator=generator,
        device="cuda",
    )

    _assert_cuda_matches_reference(scores, scale=0.125)


@pytest.mark.parametrize("allowed_columns", PARTIAL_WARP_ALLOWED_COLUMNS)
def test_warp_reductions_handle_partially_populated_column_groups(
    allowed_columns: int,
) -> None:
    """Exercise both sides of 32-column boundaries in the causal prefix.

    The CUDA block always consists of complete hardware warps. For a causal row,
    however, lanes whose starting column is beyond ``allowed_columns`` own no
    data. They must enter shuffle reductions with max/sum identity values while
    still executing the shuffle instructions under the full-warp mask.
    """

    sequence_length = 97
    rows = allowed_columns
    generator = torch.Generator(device="cuda").manual_seed(680 + allowed_columns)
    scores = torch.randn(
        rows,
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
