"""Small deterministic target launched by NVIDIA Nsight Compute."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cuda_attention.operator import cuda_extension_available, fused_causal_softmax


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-heads", type=int, default=8)
    parser.add_argument("--head-dimension", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=3)
    arguments = parser.parse_args()
    if min(
        arguments.sequence_length,
        arguments.batch_heads,
        arguments.head_dimension,
        arguments.block_size,
    ) <= 0 or arguments.warmups < 0:
        parser.error("dimensions must be positive and warmups must be nonnegative")
    if not torch.cuda.is_available():
        print("Nsight target requires an NVIDIA CUDA runtime", file=sys.stderr)
        return 2
    if not cuda_extension_available():
        print("Nsight target requires the built CUDA extension", file=sys.stderr)
        return 2

    generator = torch.Generator(device="cuda").manual_seed(2026)
    scores = torch.randn(
        arguments.batch_heads * arguments.sequence_length,
        arguments.sequence_length,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    scale = 1.0 / math.sqrt(arguments.head_dimension)
    operation = lambda: fused_causal_softmax(
        scores,
        scale,
        block_size=arguments.block_size,
    )
    for _ in range(arguments.warmups):
        operation()
    torch.cuda.synchronize()

    # The shell helper filters for this one measured kernel launch. Keeping
    # warmups outside the measured launch avoids collecting initialization.
    operation()
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
