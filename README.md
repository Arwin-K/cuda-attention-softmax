# CUDA Optimization of Fused Causal Softmax for Transformer Attention

An educational CUDA and ML-systems research project. The repository follows one
implementation in `csrc/fused_causal_softmax.cu` as it evolves through Git
history; it does not maintain parallel kernel versions.

## Current status

The block-parallel milestone through Commit 060 is complete: the project has
environment detection, research framing, an
explicit CPU reference, a causal attention path, correctness/stability/edge
tests, reproducible tensor helpers, opt-in extension infrastructure, a
provenance-aware benchmark harness, and the shared-memory reduction foundations
of a one-block-per-row CUDA design. The kernel source distributes scaling,
maximum and denominator reductions, exponentiation, and final normalization
across block threads while retaining exact causal zeros. It has not been
compiled or run on NVIDIA hardware, so no CUDA correctness, latency, throughput,
speedup, or profiler result exists yet.

`tests/test_cuda_operator.py` is the device correctness gate. It covers normal,
large-magnitude, structured, irregular-width, block-boundary, and flattened-row
wrap cases using fixed tolerances, probability invariants, and selected invalid-
input paths. These tests skip visibly unless both an NVIDIA CUDA device and the
compiled extension are available.

## Development platforms

Apple Silicon macOS is the local learning, documentation, CPU-reference, test,
and results-analysis environment. Importing `cuda_attention` does not require
PyTorch or CUDA, and the project never treats Apple's MPS backend as CUDA.

CUDA compilation, CUDA correctness tests, GPU benchmarks, and NVIDIA profiling
belong on Linux with an NVIDIA GPU and the CUDA toolkit. They are optional
capabilities rather than package-import requirements. Inspect the current host
without compiling anything:

```bash
python3 scripts/check_environment.py
```

Run the complete CPU-safe validation suite from a source checkout:

```bash
./scripts/run_tests.sh
```

Calling the custom operator without a compiled extension raises
`CudaExtensionUnavailableError`; it does not compile automatically or redirect
CUDA work to MPS. The trusted CPU functions remain available from
`cuda_attention.reference` and `cuda_attention.attention`.

On the NVIDIA Linux host, verify prerequisites and request the opt-in build:

```bash
python3 scripts/check_environment.py --require-cuda
./scripts/build_extension.sh
```

On macOS the build script reports `SKIP` and exits without invoking a compiler.

## Layout

- `cuda_attention/`: Python package, environment detection, and future
  CPU-friendly reference paths.
- `csrc/`: the C++/CUDA extension boundary and single evolving CUDA source file.
- `tests/`: correctness tests, written before performance claims.
- `benchmarks/`, `profiling/`, `results/`, and `figures/`: reproducible
  measurement inputs and outputs.
- `docs/` and `notebooks/`: the research record and learning material.

See `PROJECT_PLAN.md` for the ordered research plan and `AGENTS.md` for the
engineering, platform, and research-integrity constraints.
