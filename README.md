# CUDA Optimization of Fused Causal Softmax for Transformer Attention

An educational CUDA and ML-systems research project. The repository follows one
implementation in `csrc/fused_causal_softmax.cu` as it evolves through Git
history; it does not maintain parallel kernel versions.

## Current status

Day 1 is complete: the project has environment detection, research framing, an
explicit CPU reference, a causal attention path, correctness/stability/edge
tests, reproducible tensor helpers, and opt-in extension build infrastructure.
No compiled CUDA extension, completed kernel, benchmark result, or profiler
result exists yet.

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

## Layout

- `cuda_attention/`: Python package, environment detection, and future
  CPU-friendly reference paths.
- `csrc/`: the future C++/CUDA extension boundary and single CUDA source file.
- `tests/`: correctness tests, written before performance claims.
- `benchmarks/`, `profiling/`, `results/`, and `figures/`: reproducible
  measurement inputs and outputs.
- `docs/` and `notebooks/`: the research record and learning material.

See `PROJECT_PLAN.md` for the ordered research plan and `AGENTS.md` for the
engineering, platform, and research-integrity constraints.
