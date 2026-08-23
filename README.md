# CUDA Optimization of Fused Causal Softmax for Transformer Attention

An educational CUDA/ML-systems case study that evolves one fused causal
scaled-softmax kernel from serial row processing to warp-shuffle reductions.
The repository keeps one primary implementation in
`csrc/fused_causal_softmax.cu`; Git history, commit-tagged measurements, and a
full experiment archive preserve the evolution.

## Measured outcome

A complete Colab run is preserved under
[`results/runs/2026-08-23_tesla-t4_ca87722`](results/runs/2026-08-23_tesla-t4_ca87722).
It records a clean `ca87722a` checkout on one NVIDIA Tesla T4, PyTorch
2.11.0+cu128, CUDA toolkit 12.8, FP32 inputs, 25 warmups, and 100 timed samples
per implementation and shape.

- All 88 structured softmax comparisons passed at fixed `rtol=1e-5` and
  `atol=1e-6`; maximum absolute error was `3.5763e-7`.
- The warp kernel was 3.22--8.92x faster than the historical row-serial kernel
  and 1.07--1.91x faster than the shared-tree block kernel.
- Custom softmax was 1.40--3.94x faster than equivalent PyTorch eager work.
- Explicit attention using custom softmax was 1.54--2.31x faster than explicit
  eager attention, but PyTorch SDPA beat the explicit custom path at every
  tested shape.
- Launch tuning selected 128 threads by the aggregate rule, although 256
  threads won the two longest individual shapes.

These are results from one GPU session, not universal CUDA claims. See the
[paper](docs/mini_paper.md), [limitations](docs/limitations.md), and
[raw artifacts](results/runs/2026-08-23_tesla-t4_ca87722/artifacts).

## What the kernel teaches

The implementation makes stable maximum subtraction, flattened causal row
indexing, thread-strided access, warp-shuffle reductions, compact per-warp
shared state, synchronization, and final normalization inspectable. Its main
application lesson is just as important: a faster softmax does not remove the
`QK^T` and `probabilities @ V` matrix multiplications.

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

After CUDA correctness passes, launch tuning and framework timing can use:

```bash
python3 benchmarks/benchmark_launch_configs.py --block-size 128 --output results/raw/launch_128.csv
python3 benchmarks/benchmark_launch_configs.py --block-size 256 --output results/raw/launch_256.csv
python3 benchmarks/benchmark_launch_configs.py --block-size 512 --output results/raw/launch_512.csv
python3 benchmarks/benchmark_softmax.py --implementation all --output results/raw/framework_comparison.csv
```

These commands must run in one controlled NVIDIA environment. The checked-in
Colab notebook is the recommended complete workflow; never substitute CPU/MPS
timings or fabricate missing values.

## Google Colab experiment notebook

[![Open the research notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arwin-K/cuda-attention-softmax/blob/main/notebooks/04_colab_research_experiments.ipynb)

The [Colab experiment guide](docs/colab_experiments.md) explains the exact run,
recovery, evidence-validation, and ZIP handoff workflow. The notebook connects
to this repository, builds through `scripts/build_extension.sh`, uses existing
CUDA tests and benchmark entry points, and adds notebook-local orchestration for
attention, profiling, figures, tables, and research-paper artifacts. The
reusable notebook is output-free; the exact executed notebook is preserved with
the measured run.

## Research outputs

- [Markdown paper](docs/mini_paper.md) and
  [LaTeX source](docs/mini_paper.tex)
- [Educational optimization story](docs/blog_post.md)
- [Design journal](docs/design_journal.md),
  [experiment log](docs/experiment_log.md), and
  [learning journal](docs/learning_journal.md)
- [Interview and defense notes](docs/interview_notes.md)
- [112-commit public journal](WEBSITE_JOURNAL.md)

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
