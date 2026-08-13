"""Educational fused causal-softmax research package.

Importing the package is safe when PyTorch and CUDA are unavailable. Optional
capabilities are inspected only when :func:`detect_environment` is called.
"""

from .environment import EnvironmentInfo, detect_environment

__all__ = ["EnvironmentInfo", "detect_environment"]
