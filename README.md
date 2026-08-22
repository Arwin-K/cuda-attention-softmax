# CUDA Optimization of Fused Causal Softmax for Transformer Attention

An educational CUDA and ML-systems research project. The repository follows one
implementation in `csrc/fused_causal_softmax.cu` as it evolves through Git
history; it does not maintain parallel kernel versions.

## Current status

Day 5 through Commit 080 is complete. The one-block-per-row source now performs
maximum and denominator reductions in two levels: warp-local register shuffles,
then one compact shared value per warp combined by the first warp. Scaling,
causal masking, stable exponentiation, and normalization remain inside the same
kernel. The launch accepts 128, 256, or 512 threads without duplicating source.

The benchmark path records raw CUDA-event samples, Git/hardware/software
provenance, launch block size, and compile warmups. It supports PyTorch eager,
`torch.compile`, and the custom operator for the same scale-mask-softmax work.
The 256-thread default is provisional: no launch size has been selected from
measurements.

This source has not yet been compiled or run on NVIDIA hardware. There is no
CUDA correctness, latency, throughput, speedup, or profiler result. All such
claims remain pending the documented Linux/NVIDIA execution workflow.

`tests/test_cuda_operator.py` is the device correctness gate. It covers normal,
large-magnitude, structured, irregular-width, block-boundary, and flattened-row
wrap cases using fixed tolerances, probability invariants, and selected invalid-
input paths plus causal prefixes around warp boundaries. Static CUDA contract
checks run everywhere; device cases skip visibly unless both an NVIDIA CUDA
device and the compiled extension are available.

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

After CUDA correctness passes, launch tuning and framework timing use:

```bash
python3 benchmarks/benchmark_launch_configs.py --block-size 128 --output results/raw/launch_128.csv
python3 benchmarks/benchmark_launch_configs.py --block-size 256 --output results/raw/launch_256.csv
python3 benchmarks/benchmark_launch_configs.py --block-size 512 --output results/raw/launch_512.csv
python3 benchmarks/benchmark_softmax.py --implementation all --output results/raw/framework_comparison.csv
```

These commands must run in one controlled NVIDIA environment. Do not commit
partial or fabricated artifacts.

## Google Colab experiment notebook

[![Open the research notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arwin-K/cuda-attention-softmax/blob/day-five-pt-2/notebooks/04_colab_research_experiments.ipynb)

The [Colab experiment guide](docs/colab_experiments.md) explains the exact run,
recovery, evidence-validation, and ZIP handoff workflow. The notebook connects
to this repository, builds through `scripts/build_extension.sh`, uses existing
CUDA tests and benchmark entry points, and adds notebook-local orchestration for
attention, profiling, figures, tables, and research-paper artifacts. It contains
no precomputed or fabricated measurements.

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
