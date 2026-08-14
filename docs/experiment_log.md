# Experiment log

This log records experiments that were actually run. Planned work belongs in
`PROJECT_PLAN.md`; an unavailable GPU or missing result remains explicit rather
than being replaced with an estimate.

Use these labels consistently:

```text
HYPOTHESIS: What we predict before collecting data.
MEASUREMENT: What the command or profiler actually reported.
INTERPRETATION: A possible explanation of the measurement.
NEXT EXPERIMENT: A controlled way to test that explanation.
```

## Experiment entry template

### Experiment: `TODO: short descriptive name`

- **Date/time:** `TODO: ISO 8601 timestamp with timezone`
- **Git commit:** `TODO: full or unambiguous commit hash`
- **Hardware:** `TODO: CPU/GPU model and relevant capability, or unavailable`
- **Software:** `TODO: OS, Python, PyTorch, CUDA, compiler, and driver versions`
- **Research question:** `TODO: the specific question this run addresses`
- **HYPOTHESIS:** `TODO: prediction written before the run`
- **Independent variable:** `TODO: the one factor deliberately changed`
- **Controlled variables:** `TODO: shapes, dtype, inputs, timing, and environment`
- **Metrics:** `TODO: correctness values, latency, throughput, or profiler metrics`
- **Command/script:** `TODO: exact reproducible command`
- **Raw result file:** `TODO: repository-relative artifact path, or none`
- **MEASUREMENT:** `TODO: direct observation from the run`
- **INTERPRETATION:** `TODO: evidence-bounded explanation`
- **Limitations:** `TODO: threats to validity and missing evidence`
- **NEXT EXPERIMENT:** `TODO: follow-up that could confirm or reject the interpretation`
- **Student reflection:** `TODO(student): What surprised you or changed your understanding?`

## Day checkpoint template

Use this template at commits 016, 032, 048, 064, 080, 096, and 112.

### Day `TODO` checkpoint — Commit `TODO`

- **What was implemented:** `TODO: code, tests, tooling, and documentation`
- **What was actually measured:** `TODO: measured evidence, or explicitly none`
- **What I learned:** `TODO(student): Write this in your own words.`
- **What surprised me:** `TODO(student): Write this in your own words.`
- **Unresolved questions:** `TODO: technical and experimental questions`
- **Next day:** `TODO: what the next working day will investigate`

## Current evidence status

No performance or profiling experiment has been recorded yet.

## Day 1 checkpoint — Commit 016

- **What was implemented:** Repository organization; platform-aware optional
  imports; the research question and hypotheses; CUDA/attention background;
  learning and experiment templates; explicit stable softmax; flattened-row
  causal masking; scaling/masking/softmax composition; explicit `QK^T -> P ->
  PV` attention; CPU correctness/stability/edge tests; and reproducible seed,
  score, and Q/K/V tensor helpers.
- **What was actually measured:** No performance was measured. The CPU test
  suite was run on Apple Silicon macOS with Python 3.11 and PyTorch 2.13.0. The
  exact final test count and outcome are recorded in the Commit 016 handoff.
- **What I learned:** `TODO(student): Write this in your own words.`
- **What surprised me:** `TODO(student): Write this in your own words.`
- **Unresolved questions:** How the reference tolerances transfer to CUDA
  reduction order; how expensive the row-serial kernel will be; and which
  NVIDIA environment will produce the first build and measurement evidence.
- **Next day:** Add executable learning notebooks, the C++/CUDA extension
  boundary, guarded Mac behavior, and the first correctness-first row-serial
  CUDA implementation through Commit 032. GPU-only claims remain pending until
  those commands run on Linux with NVIDIA hardware.

## CPU reference and notebook validation — Commit 019

- **Date/time:** 2026-08-14, America/Toronto
- **Git commit:** Commit 019 — `document CPU reference methodology and learning notes`
- **Hardware:** Apple Silicon arm64 CPU; no NVIDIA GPU available
- **Software:** Darwin, Python 3.11.15, PyTorch 2.13.0, no PyTorch CUDA build
- **Research question:** Does the CPU reference provide a reproducible semantic
  and numerical oracle for future CUDA comparisons?
- **HYPOTHESIS:** The stable reference, explicit attention composition, and
  educational examples will satisfy their existing assertions on CPU.
- **Independent variable:** None; this is a validation run, not a comparative
  performance experiment.
- **Controlled variables:** Repository state through Commit 018, project virtual
  environment, deterministic test seeds, CPU execution
- **Metrics:** pytest pass/fail count; notebook code-path completion
- **Command/script:** `.venv/bin/python -m pytest -q tests` plus sequential
  execution of code cells from notebooks 01 and 02
- **Raw result file:** None; console validation only
- **MEASUREMENT:** 47 tests passed in 1.02 seconds; two notebook code paths
  executed without assertion failure.
- **INTERPRETATION:** The current CPU reference and lessons are internally
  consistent enough to serve as the next implementation oracle. This says
  nothing about CUDA correctness or performance.
- **Limitations:** No CUDA compiler, NVIDIA GPU, extension, kernel, benchmark,
  or profiler was involved. The pytest duration is not a benchmark result.
- **NEXT EXPERIMENT:** Re-run reference comparisons through the guarded custom
  operator after the CUDA extension becomes available on NVIDIA Linux.
- **Student reflection:** `TODO(student): What changed in your understanding?`

## CPU regression stabilization — Commit 020

- **Date/time:** 2026-08-14, America/Toronto
- **Git commit:** Commit 020 — `stabilize CPU reference test suite`
- **Hardware:** Apple Silicon arm64 CPU; no NVIDIA GPU available
- **Software:** Darwin, Python 3.11.15, PyTorch 2.13.0, pytest 9.1.1
- **Research question:** Does the complete CPU-safe regression suite pass through
  one reproducible source-checkout command before native extension work begins?
- **HYPOTHESIS:** Existing reference tests plus automated environment-capability
  cases will pass without treating unavailable CUDA as a failure.
- **Independent variable:** Added environment-detection regression coverage
- **Controlled variables:** CPU device, project virtual environment, repository
  state through Commit 020, deterministic test inputs
- **Metrics:** pytest pass/fail count
- **Command/script:** `./scripts/run_tests.sh`
- **Raw result file:** None; console validation only
- **MEASUREMENT:** 51 tests passed in 1.13 seconds. The real environment reported
  no available CUDA device; simulated tests covered missing PyTorch and a
  CUDA-capable PyTorch runtime.
- **INTERPRETATION:** The CPU reference and optional-capability boundary are
  stable enough to begin extension scaffolding. This does not validate C++ or
  CUDA compilation.
- **Limitations:** No Linux, NVIDIA GPU, CUDA toolkit, extension binary, or
  device execution was involved. Test duration is not a benchmark.
- **NEXT EXPERIMENT:** Perform static Mac checks on extension infrastructure,
  then compile and run it on NVIDIA Linux when available.
- **Student reflection:** `TODO(student): Record your own checkpoint response.`

## Day 2 checkpoint — Commit 032

- **What was implemented:** Two executable CPU learning notebooks; documented
  CPU-reference methodology; a stable CPU regression command; opt-in PyTorch
  C++/CUDA extension configuration; pybind11 and launcher boundaries; guarded
  Python loading; Mac-safe environment/build scripts; and the first row-serial
  fused causal scaled-softmax kernel. The kernel maps one global thread to one
  row, performs a stable allowed-column maximum, computes scaled shifted
  exponentials and their sum, normalizes allowed entries, and writes exact
  zeros for future positions. Host checks now cover device, dense contiguous
  FP32 layout, nonempty 2D shape, scale, grid bounds, device selection, current
  stream use, and immediate launch errors. CUDA-vs-PyTorch tests cover normal
  inputs at sequence lengths 32, 64, and 128 plus selected invalid calls.
- **What was actually measured:** No GPU performance, profiler data, CUDA
  compilation, or CUDA numerical result was measured. On Apple Silicon macOS,
  `./scripts/run_tests.sh` reported 53 passed and 10 CUDA-only tests skipped in
  0.94 seconds. Both notebook code paths executed without assertion failures.
  `scripts/check_environment.py` reported arm64 Darwin, Python 3.11.15,
  PyTorch 2.13.0, no PyTorch CUDA build, no CUDA device, no `nvcc`, and no
  `nvidia-smi`. `scripts/build_extension.sh` returned its expected unsupported-
  platform `SKIP`. Pytest duration is validation metadata, not a benchmark.
- **What I learned:** `TODO(student): Explain in your own words how Python,
  pybind11, a host launcher, a CUDA kernel, and a PyTorch tensor connect.`
- **What surprised me:** `TODO(student): Record your own observation; no
  personal reflection has been inferred.`
- **Unresolved questions:** Whether the extension compiles against the remote
  CUDA/PyTorch toolchain; whether the row-serial output passes fixed-tolerance
  comparisons; how extreme and irregular inputs behave on GPU; and where the
  first measured bottleneck appears.
- **Next day:** Run CUDA numerical and odd-width correctness gates when NVIDIA
  hardware is available; centralize benchmark controls; add CUDA-event timing,
  eager/custom paths, and provenance metadata; collect a baseline only from a
  real GPU; then begin the one-block-per-row shared-reduction design.
