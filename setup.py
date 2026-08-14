"""Build configuration for the optional PyTorch C++/CUDA extension."""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_CUDA_ENV = "CUDA_ATTENTION_BUILD_CUDA"


def cuda_build_requested() -> bool:
    """Return whether this invocation explicitly requested CUDA compilation."""

    return os.environ.get(BUILD_CUDA_ENV, "0") == "1"


def extension_configuration() -> tuple[list[object], dict[str, object]]:
    """Create extension settings only for an explicitly requested CUDA build."""

    if not cuda_build_requested():
        return [], {}

    # Importing cpp_extension can inspect CUDA-related configuration. Keeping
    # the import inside this opt-in path lets metadata and CPU workflows remain
    # usable on machines that cannot compile CUDA.
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    extension = CUDAExtension(
        name="cuda_attention._C",
        sources=[
            str(PROJECT_ROOT / "csrc" / "bindings.cpp"),
            str(PROJECT_ROOT / "csrc" / "fused_causal_softmax.cu"),
        ],
        include_dirs=[str(PROJECT_ROOT / "csrc")],
    )
    return [extension], {"build_ext": BuildExtension}


extension_modules, command_classes = extension_configuration()

setup(
    packages=find_packages(),
    ext_modules=extension_modules,
    cmdclass=command_classes,
)
