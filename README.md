# CUDA Optimization of Fused Causal Softmax

[![LaTeX paper](https://github.com/Arwin-K/cuda-attention-softmax/actions/workflows/latex.yml/badge.svg)](https://github.com/Arwin-K/cuda-attention-softmax/actions/workflows/latex.yml)

An educational GPU-systems research project examining how CUDA work
decomposition, reductions, warp communication, and launch configuration affect
causal scaled softmax and end-to-end transformer attention.

The repository contains one evolving CUDA implementation:
[`csrc/fused_causal_softmax.cu`](csrc/fused_causal_softmax.cu). Earlier kernel
designs are preserved in Git history and compared using commit-tagged
measurements rather than duplicate source files.

## Research question

> How do GPU work decomposition, parallel reductions, warp-level communication,
> and launch configuration affect fused causal scaled-softmax performance, and
> how much do kernel-level improvements translate into complete transformer
> attention performance?

The operator fuses scaling, causal masking, stable maximum subtraction,
exponentiation, reduction, and normalization for FP32 score tensors shaped
`[rows, sequence_length]`. It implements the forward pass only.

## Measured result

One complete experiment is preserved in
[`results/runs/2026-08-23_tesla-t4_ca87722`](results/runs/2026-08-23_tesla-t4_ca87722).
It records a clean `ca87722a` checkout on an NVIDIA Tesla T4 using PyTorch
2.11.0+cu128 and CUDA 12.8, with 25 warmups and 100 CUDA-event samples per
implementation and shape.

- All 88 structured softmax comparisons passed at `rtol=1e-5`, `atol=1e-6`;
  maximum absolute error was `3.5763e-7`.
- The warp-reduction kernel was 3.22--8.92x faster than the historical
  row-serial kernel and 1.07--1.91x faster than the shared-tree block kernel.
- Custom softmax was 1.40--3.94x faster than equivalent PyTorch eager work.
- Explicit attention using the custom softmax was 1.54--2.31x faster than
  explicit eager attention, while PyTorch SDPA was faster at every tested
  shape.

These measurements describe one controlled T4 session, not universal CUDA
performance. See the [rendered research paper](docs/mini_paper.pdf),
[results](docs/results.md), and [limitations](docs/limitations.md) for the full
experimental context.

## Quick start: CPU reference and analysis

Python 3.10 or newer is required. CUDA is not required for package import,
reference correctness tests, result auditing, or plotting.

```bash
git clone https://github.com/Arwin-K/cuda-attention-softmax.git
cd cuda-attention-softmax
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_environment.py
./scripts/run_tests.sh
```

A minimal CPU reference example:

```python
import torch

from cuda_attention.reference import causal_scaled_softmax

scores = torch.randn(8, 8, dtype=torch.float32)
probabilities = causal_scaled_softmax(scores, scale=0.125)

assert probabilities.shape == scores.shape
torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(8))
```

The complete CPU-only reproducibility check also validates the output-free
Colab notebook and the checked-in experiment schema:

```bash
./scripts/verify_cpu_reproducibility.sh
```

## Build and test the CUDA extension

CUDA execution requires Linux, an NVIDIA GPU, a compatible driver, and the CUDA
toolkit. The build is deliberately opt-in and never runs automatically on
macOS. Apple's MPS backend is not treated as CUDA.

```bash
nvidia-smi
python scripts/check_environment.py --require-cuda
./scripts/build_extension.sh
python -m pytest -q tests/test_cuda_operator.py tests/test_attention.py
```

If the extension is unavailable, calling the custom operator raises
`CudaExtensionUnavailableError`; CPU reference functions remain usable.

## Reproduce the GPU experiment

The output-free
[`04_colab_research_experiments.ipynb`](notebooks/04_colab_research_experiments.ipynb)
is the recommended end-to-end workflow.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arwin-K/cuda-attention-softmax/blob/main/notebooks/04_colab_research_experiments.ipynb)

It verifies the environment and source revision, builds the extension, runs
correctness checks, benchmarks historical/current kernels and framework
baselines, profiles the operator, and packages raw evidence. Follow the
[experiment guide](docs/colab_experiments.md) and use a fresh output directory
for every hardware/software environment.

For shorter manual runs after CUDA correctness passes:

```bash
python benchmarks/benchmark_launch_configs.py --block-size 128 --output results/raw/launch_128.csv
python benchmarks/benchmark_softmax.py --implementation all --output results/raw/softmax.csv
python benchmarks/benchmark_attention.py --implementation all --output results/raw/attention.csv
python profiling/profile_pytorch.py --output-dir results/raw/pytorch_profiler
sh profiling/run_ncu.sh results/raw/nsight
```

Never compare rows collected under different GPU/software conditions as a
single controlled experiment. Benchmark commands record Git, device, and
software metadata with every result.

## Repository layout

```text
cuda_attention/   Python reference, operator boundary, attention, and utilities
csrc/             C++ binding and the single CUDA implementation
tests/            CPU, static-contract, and optional CUDA correctness tests
benchmarks/       Softmax, launch, framework, and attention benchmarks
profiling/        PyTorch Profiler and Nsight Compute entry points
notebooks/        Reproducible end-to-end NVIDIA experiment
results/runs/     Raw measured evidence with environment and Git provenance
docs/             Paper, methods, results, limitations, and reproduction guide
scripts/          Environment, build, validation, audit, and figure helpers
```

## Research record

- [Research paper PDF preview](docs/mini_paper.pdf)
- [LaTeX source](docs/mini_paper.tex), [bibliography](docs/references.bib), and [Markdown companion](docs/mini_paper.md)
- [Methodology](docs/methodology.md)
- [Measured results](docs/results.md)
- [Kernel evolution](docs/kernel_evolution.md)
- [Reproducibility guide](docs/reproducibility.md)
- [Experimental limitations](docs/limitations.md)
- [Raw and generated artifacts](results/runs/2026-08-23_tesla-t4_ca87722/artifacts)
- [Citation metadata](CITATION.cff)

## Scope

The custom operator supports contiguous FP32 CUDA inputs, causal masking, and
forward execution. It does not implement autograd/backward, dropout, arbitrary
masks, mixed precision, or a custom matrix multiplication. The explicit custom
attention path still uses PyTorch for `QK^T` and `probabilities @ V`; its results
must therefore be distinguished from an isolated softmax microbenchmark.

## License

This project is released under the [MIT License](LICENSE).
