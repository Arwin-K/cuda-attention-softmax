"""Detect host and optional PyTorch/CUDA capabilities without requiring CUDA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import platform
from typing import Any


@dataclass(frozen=True)
class EnvironmentInfo:
    """A serializable snapshot of capabilities relevant to this project."""

    operating_system: str
    machine: str
    python_version: str
    is_apple_silicon: bool
    pytorch_available: bool
    pytorch_version: str | None
    pytorch_cuda_version: str | None
    cuda_available: bool
    cuda_device_count: int
    pytorch_import_error: str | None

    def as_dict(self) -> dict[str, object]:
        """Return plain data suitable for readable or machine-readable output."""

        return asdict(self)


def detect_environment() -> EnvironmentInfo:
    """Inspect the current host while treating PyTorch and CUDA as optional.

    Apple Silicon identifies the local development platform. CUDA availability
    is a separate runtime capability and is true only when PyTorch reports an
    accessible CUDA device. Apple's MPS backend is intentionally not queried as
    a CUDA substitute.
    """

    operating_system = platform.system()
    machine = platform.machine()
    is_apple_silicon = operating_system == "Darwin" and machine.lower() in {
        "arm64",
        "aarch64",
    }

    try:
        torch: Any = importlib.import_module("torch")
    except (ImportError, OSError) as error:
        return EnvironmentInfo(
            operating_system=operating_system,
            machine=machine,
            python_version=platform.python_version(),
            is_apple_silicon=is_apple_silicon,
            pytorch_available=False,
            pytorch_version=None,
            pytorch_cuda_version=None,
            cuda_available=False,
            cuda_device_count=0,
            pytorch_import_error=f"{type(error).__name__}: {error}",
        )

    cuda = getattr(torch, "cuda", None)
    cuda_available = bool(cuda is not None and cuda.is_available())
    cuda_device_count = int(cuda.device_count()) if cuda_available else 0
    torch_version = getattr(torch, "version", None)

    return EnvironmentInfo(
        operating_system=operating_system,
        machine=machine,
        python_version=platform.python_version(),
        is_apple_silicon=is_apple_silicon,
        pytorch_available=True,
        pytorch_version=str(getattr(torch, "__version__", "unknown")),
        pytorch_cuda_version=getattr(torch_version, "cuda", None),
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        pytorch_import_error=None,
    )
