# Reproducibility

Run commands from the repository root unless noted otherwise. CPU validation
and GPU measurement are separate workflows: a successful CPU run does not
claim that the CUDA extension compiled or executed.

## 1. Create the environment

Python 3.10 or newer is required.

```bash
git clone https://github.com/Arwin-K/cuda-attention-softmax.git
cd cuda-attention-softmax
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs PyTorch, pytest, and Matplotlib. It does not install an NVIDIA
driver or CUDA toolkit.

## 2. Validate the CPU-safe repository

This workflow explicitly hides CUDA, verifies package import and the generated
notebook, runs the CPU/static tests, and audits the checked-in result schema and
figure provenance:

```bash
./scripts/verify_cpu_reproducibility.sh
```

Expected final marker:

```text
CPU_ONLY_REPRODUCIBILITY: PASS
```

To run only the ordinary test suite:

```bash
python scripts/check_environment.py
./scripts/run_tests.sh
```

CUDA-specific tests skip cleanly when an NVIDIA device and compiled extension
are unavailable.

## 3. Audit the published T4 evidence

The preserved experiment is under
`results/runs/2026-08-23_tesla-t4_ca87722`. Regenerate audits into a temporary
directory so published evidence is not silently overwritten:

```bash
python scripts/audit_results.py \
  results/runs/2026-08-23_tesla-t4_ca87722/artifacts \
  --output /tmp/cuda-softmax-schema-audit.json

python scripts/generate_result_provenance.py \
  results/runs/2026-08-23_tesla-t4_ca87722/artifacts \
  --output /tmp/cuda-softmax-figure-provenance.json

diff -u \
  results/runs/2026-08-23_tesla-t4_ca87722/schema_audit.json \
  /tmp/cuda-softmax-schema-audit.json

diff -u \
  results/runs/2026-08-23_tesla-t4_ca87722/figure_provenance.json \
  /tmp/cuda-softmax-figure-provenance.json
```

Both diffs should be empty. These checks validate internal consistency and
source hashes; they do not rerun GPU timing.

The historical implementation milestones can be checked in a full Git clone:

```bash
python scripts/audit_kernel_history.py
```

## 4. Regenerate figures and tables

These commands use the preserved CSVs and can run on CPU:

```bash
python benchmarks/compare_speedups.py \
  --softmax-raw results/runs/2026-08-23_tesla-t4_ca87722/artifacts/benchmarks/raw/softmax_raw.csv \
  --attention-raw results/runs/2026-08-23_tesla-t4_ca87722/artifacts/attention/attention_raw.csv \
  --output /tmp/cuda-softmax-speedups.csv

python scripts/generate_figures.py \
  --summary results/runs/2026-08-23_tesla-t4_ca87722/artifacts/benchmarks/summaries/softmax_summary.csv \
  --speedup-comparison /tmp/cuda-softmax-speedups.csv \
  --output-directory /tmp/cuda-softmax-figures

python benchmarks/generate_tables.py \
  --softmax-summary results/runs/2026-08-23_tesla-t4_ca87722/artifacts/benchmarks/summaries/softmax_summary.csv \
  --attention-raw results/runs/2026-08-23_tesla-t4_ca87722/artifacts/attention/attention_raw.csv \
  --speedup-comparison /tmp/cuda-softmax-speedups.csv \
  --output-directory /tmp/cuda-softmax-tables
```

Use temporary output paths first, review diffs, and preserve the relationship
between figures and their source CSVs.

## 5. Build and run on NVIDIA

Requirements:

- Linux
- NVIDIA GPU and compatible driver
- CUDA toolkit with `nvcc`
- a CUDA-enabled PyTorch installation compatible with that toolkit

Verify the host and build explicitly:

```bash
nvidia-smi
python scripts/check_environment.py --require-cuda
./scripts/build_extension.sh
python -m pytest -q tests/test_cuda_operator.py tests/test_attention.py
```

Stop before benchmarking if environment, build, or correctness validation
fails. The recommended complete experiment is the output-free
[`notebooks/04_colab_research_experiments.ipynb`](../notebooks/04_colab_research_experiments.ipynb),
described in the [Colab guide](colab_experiments.md). It keeps all compared
implementations in one runtime, records raw CUDA-event samples, captures
environment/Git metadata, and produces a validation report.

Manual commands for smaller investigations are:

```bash
python benchmarks/benchmark_softmax.py --implementation all --output results/raw/softmax.csv
python benchmarks/benchmark_attention.py --implementation all --output results/raw/attention.csv
python profiling/profile_pytorch.py --output-dir results/raw/pytorch_profiler
sh profiling/run_ncu.sh results/raw/nsight
```

Create a new run directory for every distinct Git revision, GPU, driver,
PyTorch version, or CUDA version. Do not merge measurements from different
environments into a matched comparison.

## 6. Build the paper

Install a TeX distribution that provides `latexmk` and BibTeX, then run:

```bash
./scripts/build_paper.sh
```

Tectonic is also supported as a local fallback. The output is
`docs/mini_paper.pdf`, which GitHub renders as an in-browser preview. The
`LaTeX paper` GitHub Actions workflow independently compiles the manuscript and
uploads the PDF as a workflow artifact after relevant pushes or pull requests.
Quantitative edits should be traceable to the raw CSV or profiler export that
supports them.
