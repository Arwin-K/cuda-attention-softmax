# CUDA Optimization of Fused Causal Softmax for Transformer Attention

An educational CUDA and ML-systems research project. The repository follows one
implementation in `csrc/fused_causal_softmax.cu` as it evolves through Git
history; it does not maintain parallel kernel versions.

## Current status

Commit 001 establishes project structure only. No PyTorch reference, CUDA
extension, kernel, benchmark result, or profiler result exists yet.

## Layout

- `cuda_attention/`: future Python package and CPU-friendly reference paths.
- `csrc/`: the future C++/CUDA extension boundary and single CUDA source file.
- `tests/`: correctness tests, written before performance claims.
- `benchmarks/`, `profiling/`, `results/`, and `figures/`: reproducible
  measurement inputs and outputs.
- `docs/` and `notebooks/`: the research record and learning material.

See `PROJECT_PLAN.md` for the ordered research plan and `AGENTS.md` for the
engineering, platform, and research-integrity constraints.

