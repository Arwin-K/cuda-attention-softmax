"""Guarded access to the optional compiled CUDA extension."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

from torch import Tensor


EXTENSION_MODULE = "cuda_attention._C"


class CudaExtensionUnavailableError(RuntimeError):
    """Raised when custom CUDA execution is requested without the extension."""


def cuda_extension_available() -> bool:
    """Return whether the optional compiled extension can be discovered."""

    try:
        return importlib.util.find_spec(EXTENSION_MODULE) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _load_cuda_extension() -> ModuleType:
    try:
        return importlib.import_module(EXTENSION_MODULE)
    except (ImportError, OSError) as error:
        raise CudaExtensionUnavailableError(
            "The cuda_attention CUDA extension is unavailable. CPU reference "
            "operations remain usable. Build the extension only on Linux with "
            "an NVIDIA GPU and CUDA toolkit; Apple MPS is not a substitute."
        ) from error


def fused_causal_softmax(scores: Tensor, scale: float) -> Tensor:
    """Dispatch to the compiled fused causal scaled-softmax CUDA operator."""

    extension = _load_cuda_extension()
    return extension.fused_causal_softmax(scores, scale)
