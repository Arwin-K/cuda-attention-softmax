# Reproducibility commands

This page separates commands by capability and states what each result means.
Run from the repository root unless a step says otherwise.

## Status legend

- **Verified on Apple Silicon:** executed successfully on 2026-08-23 during
  Day 7.
- **Recorded T4 execution:** the supplied `ca87722a` artifacts show the notebook
  stage completed on a Tesla T4; Commit 109 did not rerun it on the Mac.
- **Documented only:** requires a dependency/tool unavailable in the current
  local environment and must not be described as executed here.

## 1. Create a local development environment

**Platform:** macOS or Linux, CPU is sufficient.  
**Status:** documented only; the existing `.venv` was already present.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
```

Expected capability: PyTorch reference code, pytest, Matplotlib analysis, and
package imports. This does not install an NVIDIA driver or prove CUDA support.

## 2. Reproduce the CPU-safe repository checks

**Platform:** Apple Silicon macOS or Linux CPU.  
**Status:** verified on Apple Silicon.

```bash
./scripts/verify_cpu_reproducibility.sh
```

Expected final marker: `CPU_ONLY_REPRODUCIBILITY: PASS`. On the Commit 109 Mac
run, 142 tests passed and 69 GPU-only tests skipped. The command validates
imports, notebook determinism, CPU/static tests, schema, and provenance; it
deliberately hides CUDA and never builds the extension.

Individual checks are:

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/python scripts/generate_colab_notebook.py --check
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests
```

## 3. Audit the imported T4 evidence

**Platform:** any CPU with Python standard library; pytest for tests.  
**Status:** verified on Apple Silicon.

```bash
.venv/bin/python scripts/audit_results.py \
  results/runs/2026-08-23_tesla-t4_ca87722/artifacts \
  --output results/runs/2026-08-23_tesla-t4_ca87722/schema_audit.json

.venv/bin/python scripts/generate_result_provenance.py \
  results/runs/2026-08-23_tesla-t4_ca87722/artifacts \
  --output results/runs/2026-08-23_tesla-t4_ca87722/figure_provenance.json

.venv/bin/python scripts/audit_public_claims.py \
  results/runs/2026-08-23_tesla-t4_ca87722/artifacts \
  --output results/runs/2026-08-23_tesla-t4_ca87722/public_claim_audit.json
```

Expected status in schema and public-claim reports: `PASS`. The provenance
command should report 12 figures. These commands verify internal consistency;
they do not rerun GPU timing.

## 4. Verify Git and publication history

**Platform:** any Git checkout.  
**Status:** verified on Apple Silicon.

```bash
.venv/bin/python scripts/audit_kernel_history.py
.venv/bin/python scripts/audit_website_journal.py
git status --short
```

Expected markers: 15 kernel milestones with one tracked `.cu` source and
entries 001--112 exactly once with explicit status. A clean `git status` is
required before a new measured run.

## 5. Regenerate analysis outputs from measured CSVs

**Platform:** CPU with `requirements-dev.txt`, including Matplotlib.  
**Status:** documented only in Commit 109; the current local `.venv` does not
contain Matplotlib. The preserved figures/tables came from the T4 notebook.

Use a fresh temporary output directory so checked-in evidence is not silently
overwritten:

```bash
python3 scripts/generate_figures.py \
  --summary results/runs/2026-08-23_tesla-t4_ca87722/artifacts/benchmarks/summaries/softmax_summary.csv \
  --speedup-comparison results/runs/2026-08-23_tesla-t4_ca87722/artifacts/attention/amdahl_analysis.csv \
  --output-directory /tmp/cuda-softmax-figures

python3 benchmarks/generate_tables.py \
  --softmax-summary results/runs/2026-08-23_tesla-t4_ca87722/artifacts/benchmarks/summaries/softmax_summary.csv \
  --attention-raw results/runs/2026-08-23_tesla-t4_ca87722/artifacts/attention/attention_raw.csv \
  --speedup-comparison results/runs/2026-08-23_tesla-t4_ca87722/artifacts/attention/amdahl_analysis.csv \
  --output-directory /tmp/cuda-softmax-tables
```

Expected outputs are latency/throughput/speedup images and LaTeX-ready tables.
Compare them with the provenance manifest before replacing publication files.

## 6. Rebuild and rerun on NVIDIA

**Platform:** Linux, NVIDIA GPU, compatible driver, CUDA toolkit.  
**Status:** recorded T4 execution for measured commit `ca87722a`; not executable
on the Day 7 Mac. Follow `nvidia_handoff.md` for exact replication versus a new
latest-commit run.

Minimum gate:

```bash
nvidia-smi
python3 scripts/check_environment.py --require-cuda
./scripts/build_extension.sh
python3 -m pytest -q tests/test_cuda_operator.py tests/test_attention.py
```

Stop before timing if any gate fails. After correctness, the complete recommended
workflow is the Colab notebook:

[Open the experiment notebook in Colab](https://colab.research.google.com/github/Arwin-K/cuda-attention-softmax/blob/main/notebooks/04_colab_research_experiments.ipynb)

Manual debugging commands are:

```bash
python3 benchmarks/benchmark_softmax.py --implementation all --output results/raw/softmax_raw.csv
python3 benchmarks/benchmark_attention.py --implementation all --output results/raw/attention_raw.csv
python3 profiling/profile_pytorch.py --output-dir results/profiler/pytorch
sh profiling/run_ncu.sh
```

Expected outputs must include raw samples with Git/GPU/software fields, a
correctness pass, profiler metadata, and explicit skipped/failed status for any
optional Nsight limitation. Manual commands do not recreate the notebook's
historical checkouts and full manifest by themselves.

## 7. Build the paper

**Platform:** Overleaf or local TeX with `biber`.  
**Status:** documented only; `pdflatex` and `biber` are unavailable on the Day
7 Mac.

```bash
cd docs
pdflatex mini_paper.tex
biber mini_paper
pdflatex mini_paper.tex
pdflatex mini_paper.tex
```

Expected output: `docs/mini_paper.pdf`. Before submission, replace only the
explicit `TODO(student)` affiliation/reflection fields, apply the exact venue
class files, rerun audits, and review the final PDF. Do not replace evidence
TODOs with invented measurements.
