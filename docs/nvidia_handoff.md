# NVIDIA execution handoff checklist

Use this checklist on Linux with a real NVIDIA CUDA GPU. CPU and Apple MPS
results are not substitutes. A Tesla T4 is the closest Colab choice for exact
replication; another GPU is a useful separate replication and must be labeled
as a different environment.

## 1. Choose the research mode

- **Exact measured-revision replication:** check out
  `ca87722a00ebf585cd788c67949e7c0b32dca788` and compare with the preserved T4
  run. Do not expect later audit/documentation files in that checkout.
- **Latest-workflow rerun:** after the Day 7 branch is merged, use `main`. This
  validates the current reproducibility workflow but creates a new measured Git
  revision. Never relabel it as the earlier `ca87722a` run.

Record the selected revision before building:

```bash
git rev-parse HEAD
git status --short
```

The worktree must be clean so code and measurements have an unambiguous hash.

## 2. Verify the GPU runtime

In Colab select **Runtime > Change runtime type > T4 GPU**, then run:

```bash
nvidia-smi
nvcc --version
python3 --version
```

Stop if `nvidia-smi` cannot see a GPU. An Nsight counter-permission problem can
skip only the optional Nsight stage; lack of a CUDA GPU blocks the build,
correctness, benchmarks, and all GPU conclusions.

## 3. Clone with a plain URL

Use the literal URL below. Do not paste Markdown link syntax such as
`[https://...](https://...)` into Python or Git.

```bash
git clone https://github.com/Arwin-K/cuda-attention-softmax.git
cd cuda-attention-softmax
git checkout main
```

For the exact historical run, replace the last command with:

```bash
git checkout --detach ca87722a00ebf585cd788c67949e7c0b32dca788
```

## 4. Verify and build before measuring

```bash
python3 scripts/check_environment.py --require-cuda
python3 -m pip install -r requirements.txt -r requirements-dev.txt
./scripts/build_extension.sh
python3 -m pytest -q tests/test_cuda_operator.py tests/test_attention.py
```

Stop on a nonzero build, import failure, correctness failure, unexpected NaN or
Inf, causal-mask failure, or tolerance failure. Preserve the logs. Do not weaken
tolerances or continue to performance sections.

## 5. Recommended complete notebook run

Open the output-free notebook from GitHub:

[Open `04_colab_research_experiments.ipynb` in Colab](https://colab.research.google.com/github/Arwin-K/cuda-attention-softmax/blob/main/notebooks/04_colab_research_experiments.ipynb)

In its configuration cell:

- use `SOURCE_MODE = "GIT"`;
- use `REPO_URL = "https://github.com/Arwin-K/cuda-attention-softmax.git"`;
- use `BRANCH = "main"` after Day 7 is merged;
- set the configured sequence-length matrix, FP32, seed 1234, 25 warmups, and
  100 iterations unless declaring a different experiment;
- use a new run/artifact directory for every attempt.

The notebook intentionally refuses to overwrite an existing artifact. If a
runtime restart leaves `/content/cuda_softmax_artifacts`, either resume the
matching run or choose a new unique directory such as
`/content/cuda_softmax_artifacts_run_2`. Do not delete the only copy of a
completed run before downloading it.

Run cells top to bottom. The required order is environment, checkout, build,
CUDA correctness, launch configurations, current and historical softmax,
attention correctness/benchmarks, PyTorch Profiler, optional Nsight, figures,
tables, validation, manifest, then ZIP export.

## 6. Manual script alternative

After build and correctness succeed:

```bash
python3 benchmarks/benchmark_launch_configs.py --block-size 128 --output results/raw/launch_128.csv
python3 benchmarks/benchmark_launch_configs.py --block-size 256 --output results/raw/launch_256.csv
python3 benchmarks/benchmark_launch_configs.py --block-size 512 --output results/raw/launch_512.csv
python3 benchmarks/benchmark_softmax.py --implementation all --output results/raw/softmax_raw.csv
python3 benchmarks/benchmark_attention.py --implementation all --output results/raw/attention_raw.csv
python3 profiling/profile_pytorch.py
sh profiling/run_ncu.sh
```

These individual commands do not replace the notebook's combined validation,
historical checkout, tables, or manifest steps. Use them for debugging or a
clearly documented custom run.

## 7. Validate before interpretation

Require all mandatory manifest stages to be `PASS` or `COMPLETE`. Confirm:

- the recorded Git hash matches the intended clean checkout;
- every benchmark group has 100 samples indexed 0--99;
- GPU, compute capability, PyTorch, and CUDA fields are populated;
- all planned shapes and implementations are present;
- correctness passed before timing;
- compilation startup is separate from steady-state `torch.compile` timing;
- SDPA and explicit attention are described with their different fusion scopes;
- profiler facts use `MEASURED / INTERPRETATION / NEXT EXPERIMENT`.

If Nsight cannot access performance counters, retain its status/log as a
documented skip. Do not infer occupancy or bandwidth from a failed capture.

## 8. Preserve the handoff

Before the Colab runtime disconnects, download both:

1. `cuda_softmax_research_artifacts.zip`; and
2. the executed notebook with outputs.

Record SHA-256 values:

```bash
sha256sum cuda_softmax_research_artifacts.zip
sha256sum 04_colab_research_experiments.ipynb
```

Import the pair into a new directory named with date, GPU, and short measured
commit. Keep the exact executed notebook beside extracted artifacts. Run the
repository's schema and provenance tools before updating prose:

```bash
python3 scripts/audit_results.py path/to/run/artifacts --output path/to/run/schema_audit.json
python3 scripts/generate_result_provenance.py path/to/run/artifacts --output path/to/run/figure_provenance.json
```

## 9. Handoff report

Return these facts to the local reviewer:

```text
measured Git commit:
git dirty state:
GPU and compute capability:
PyTorch and CUDA versions:
correctness pass/fail and tolerances:
completed/skipped/failed stages:
artifact ZIP SHA-256:
executed notebook SHA-256:
known deviations from the planned protocol:
```

Do not fill a missing item from memory. Mark it missing and repeat the relevant
collection step.
