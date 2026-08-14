"""Tests for optional PyTorch/CUDA environment detection."""

from types import SimpleNamespace
from unittest.mock import patch

from cuda_attention.environment import detect_environment
from cuda_attention.operator import (
    CudaExtensionUnavailableError,
    cuda_extension_available,
    fused_causal_softmax,
)


def test_environment_snapshot_is_serializable_plain_data() -> None:
    snapshot = detect_environment()
    values = snapshot.as_dict()

    assert values["operating_system"]
    assert values["machine"]
    assert values["python_version"]
    assert isinstance(values["cuda_available"], bool)
    assert isinstance(values["cuda_device_count"], int)


def test_missing_pytorch_is_an_optional_capability() -> None:
    with (
        patch("cuda_attention.environment.platform.system", return_value="Darwin"),
        patch("cuda_attention.environment.platform.machine", return_value="arm64"),
        patch(
            "cuda_attention.environment.importlib.import_module",
            side_effect=ImportError("simulated missing PyTorch"),
        ),
    ):
        snapshot = detect_environment()

    assert snapshot.is_apple_silicon
    assert not snapshot.pytorch_available
    assert not snapshot.cuda_available
    assert snapshot.cuda_device_count == 0
    assert "simulated missing PyTorch" in (snapshot.pytorch_import_error or "")


def test_cuda_capability_comes_from_pytorch_runtime() -> None:
    fake_torch = SimpleNamespace(
        __version__="test-pytorch",
        version=SimpleNamespace(cuda="test-cuda"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2),
    )
    with patch(
        "cuda_attention.environment.importlib.import_module", return_value=fake_torch
    ):
        snapshot = detect_environment()

    assert snapshot.pytorch_available
    assert snapshot.pytorch_version == "test-pytorch"
    assert snapshot.pytorch_cuda_version == "test-cuda"
    assert snapshot.cuda_available
    assert snapshot.cuda_device_count == 2


def test_apple_silicon_is_not_reported_as_cuda() -> None:
    fake_torch = SimpleNamespace(
        __version__="test-pytorch",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
    )
    with (
        patch("cuda_attention.environment.platform.system", return_value="Darwin"),
        patch("cuda_attention.environment.platform.machine", return_value="arm64"),
        patch(
            "cuda_attention.environment.importlib.import_module",
            return_value=fake_torch,
        ),
    ):
        snapshot = detect_environment()

    assert snapshot.is_apple_silicon
    assert not snapshot.cuda_available
    assert snapshot.pytorch_cuda_version is None


def test_unbuilt_cuda_extension_is_reported_as_optional() -> None:
    assert not cuda_extension_available()


def test_custom_operator_fails_clearly_without_compiled_extension() -> None:
    with patch(
        "cuda_attention.operator.importlib.import_module",
        side_effect=ModuleNotFoundError("simulated missing extension"),
    ):
        try:
            fused_causal_softmax(None, 1.0)  # type: ignore[arg-type]
        except CudaExtensionUnavailableError as error:
            assert "Linux with an NVIDIA GPU" in str(error)
            assert "MPS is not a substitute" in str(error)
        else:
            raise AssertionError("missing extension did not raise a clear error")
