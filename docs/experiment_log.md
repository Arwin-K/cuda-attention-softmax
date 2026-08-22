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

## Initial row-serial CUDA baseline attempt — Commit 040

- **Date/time:** 2026-08-15, America/Toronto
- **Git commit:** `8f07d76` (benchmark harness state used for the attempt)
- **Hardware:** Apple Silicon arm64 CPU; no NVIDIA GPU available
- **Software:** Darwin, Python 3.11.15, PyTorch 2.13.0 without CUDA; no `nvcc`
  or `nvidia-smi`
- **Research question:** What latency does the row-serial fused operator achieve
  relative to equivalent eager PyTorch across the primary sequence lengths?
- **HYPOTHESIS:** Because every owning thread scans its row serially, latency is
  expected to grow strongly with sequence length; this prediction has not been
  tested.
- **Independent variable:** Intended implementation path and sequence length;
  no variable was actually measured because CUDA prerequisites failed.
- **Controlled variables:** Planned FP32, `batch_heads=8`, identical seeded
  scores, scale `1/sqrt(64)`, prebuilt eager mask, 25 warmups, 100 iterations,
  and CUDA-event timing
- **Metrics:** Intended raw microsecond samples; none produced
- **Command/script:** `.venv/bin/python benchmarks/benchmark_softmax.py
  --implementation both --sequence-length 128 --warmups 2 --iterations 3
  --output /tmp/cuda_attention_day3_baseline.csv`
- **Raw result file:** None; the command exited before creating a CSV
- **MEASUREMENT:** Exit status 2 with `CUDA benchmark requires Linux with an
  NVIDIA GPU.` No CUDA work or timing occurred.
- **INTERPRETATION:** The harness enforces the platform boundary. There is no
  performance evidence from this attempt and no basis for a speedup claim.
- **Limitations:** CUDA compilation, correctness, and event timing remain
  untested. The row-serial source has not run on an NVIDIA GPU.
- **NEXT EXPERIMENT:** On NVIDIA Linux, build the extension, run the full CUDA
  correctness suite, then execute `./scripts/run_benchmarks.sh
  results/raw/row_serial_<commit>.csv --implementation both`.
- **Student reflection:** `TODO(student): Record what this blocked run taught
  you about capability checks and honest negative evidence.`

## Day 3 checkpoint — Commit 048

- **What was implemented:** CUDA stress gates for large and structured logits;
  irregular-width GPU cases through sequence length 1023; an immutable primary
  benchmark registry; warmup-aware CUDA-event timing; equivalent eager/custom
  paths with an untimed correctness precheck; raw CSV provenance containing Git,
  workload, timing, hardware, software, and timestamp fields; and a guarded
  benchmark runner. The single CUDA source evolved from global-thread row
  ownership to one block per row, thread-strided column staging, register-local
  maxima and exponential sums, synchronized shared-memory maximum and sum trees,
  and exact masked zeros. Final probability division remains on thread 0 until
  Commit 049.
- **What was actually measured:** No CUDA value or performance metric was
  measured. The baseline attempt exited 2 before timing because the Apple
  Silicon host has no NVIDIA GPU, CUDA-enabled PyTorch, `nvcc`, or `nvidia-smi`;
  no raw CSV was created. On the final Day 3 working tree,
  `./scripts/run_tests.sh` reported 68 passed and 26 CUDA-only skips in 0.77
  seconds. Python compilation and shell syntax checks passed. Test duration is
  validation metadata, not a kernel benchmark.
- **What I learned:** `TODO(student): Explain the difference among a
  thread-local partial, shared-memory tree reduction, barrier, and final block
  result in your own words.`
- **What surprised me:** `TODO(student): Record your own observation; no
  personal reaction has been inferred.`
- **Unresolved questions:** Whether any Day 2/3 CUDA source compiles against the
  remote toolchain; whether block reductions match PyTorch at fixed tolerances;
  how barrier and shared-memory costs compare with saved serial work; where the
  row-serial/block crossover lies; and whether the raw benchmark schema works
  unchanged on the target NVIDIA host.
- **Next day:** Parallelize normalization, validate and stress the complete
  block path, measure it against an actually collected row-serial baseline,
  add summary/plotting support, audit coalescing and arbitrary-width behavior,
  document the evidence, stabilize the block milestone, and begin warp/lane
  reduction foundations through Commit 064.

## Block-parallel CUDA validation attempt — Commit 050

- **Date/time:** 2026-08-20, America/Toronto
- **Git commit:** Commit 050 — `validate block-parallel kernel against PyTorch`
- **Hardware:** Apple Silicon arm64; no NVIDIA GPU
- **Software:** Darwin, Python 3.11.15, PyTorch 2.13.0 without CUDA
- **Research question:** Does the complete one-block-per-row implementation
  match the PyTorch reference across normal, stress, and irregular cases?
- **HYPOTHESIS:** Cooperative maximum, denominator, and normalization work
  preserve the reference semantics within the fixed FP32 tolerances.
- **Independent variable:** Intended custom block-parallel CUDA output versus
  PyTorch reference output
- **Controlled variables:** Existing deterministic inputs, sequence lengths,
  scales, FP32 dtype, `rtol=1e-5`, and `atol=1e-6`
- **Metrics:** Closeness, row sums, non-negativity, exact causal zeros,
  finiteness, shape, dtype, and device
- **Command/script:** `.venv/bin/python -m pytest -q
  tests/test_cuda_operator.py -rs`
- **Raw result file:** None; test console only
- **MEASUREMENT:** Every CUDA case was skipped because an NVIDIA device was
  unavailable. No custom output was produced or compared.
- **INTERPRETATION:** Test coverage and platform gating are prepared, but block-
  parallel CUDA correctness is not established.
- **Limitations:** No CUDA build, kernel launch, device synchronization, or
  numerical comparison occurred.
- **NEXT EXPERIMENT:** Build the extension on NVIDIA Linux and rerun this exact
  suite before accepting benchmark output.
- **Student reflection:** `TODO(student): Explain why a collected skip is not a
  passed CUDA correctness test.`

## Row-serial versus block-parallel benchmark attempt — Commit 053

- **Date/time:** 2026-08-20, America/Toronto
- **Git commit:** Commit 053 — `benchmark block-parallel kernel against baseline commit`
- **Hardware:** Apple Silicon arm64; no NVIDIA GPU
- **Software:** Darwin, Python 3.11.15, PyTorch 2.13.0 without CUDA
- **Research question:** How does the complete block-parallel kernel compare
  with the historical row-serial implementation under identical controls?
- **HYPOTHESIS:** Block cooperation should benefit longer rows, with a possible
  short-row penalty from barriers and reduction overhead.
- **Independent variable:** Kernel implementation Git commit
- **Controlled variables:** Required shapes, FP32, `batch_heads=8`, seeds,
  warmups, iterations, GPU/software environment, and timed boundaries
- **Metrics:** Intended median latency, quartiles, throughput, and speedup
- **Command/script:** `.venv/bin/python benchmarks/summarize_results.py
  --baseline results/raw/row_serial.csv --candidate
  results/raw/block_parallel.csv`
- **Raw result file:** None for either implementation
- **MEASUREMENT:** Comparison preflight reported the missing baseline artifact
  and exited before calculating any statistic.
- **INTERPRETATION:** The comparison remains unavailable; no crossover or
  optimization claim is supported.
- **Limitations:** Neither implementation has a CUDA correctness pass or raw
  timing artifact from a common NVIDIA environment.
- **NEXT EXPERIMENT:** On one NVIDIA host, measure the historical row-serial
  commit and the block-parallel commit with the same harness, then rerun the
  preflight and summary.
- **Student reflection:** `TODO(student): Explain why both Git revision and
  controlled environment must match a before/after comparison.`

## Block-parallel stabilization audit — Commit 060

- **Date/time:** 2026-08-20, America/Toronto
- **Git commit:** Commit 060 — `stabilize block-parallel implementation`
- **Hardware:** Apple Silicon arm64; no NVIDIA GPU
- **Software:** Darwin, Python 3.11.15, PyTorch 2.13.0 without CUDA
- **Research question:** Is the complete block-parallel source milestone
  internally coherent and locally reproducible before warp work begins?
- **HYPOTHESIS:** CPU-safe tests, static checks, imports, and artifact guards
  should pass while all CUDA-only work remains visibly skipped or unavailable.
- **Independent variable:** None; milestone regression only
- **Controlled variables:** Day 4 source through Commit 060 and project virtual
  environment
- **Metrics:** Test pass/skip counts, syntax checks, Git status, and artifact
  presence
- **Command/script:** `./scripts/run_tests.sh` plus Python/shell syntax checks
  and inspection of `results/raw`, `results/summary`, and `figures`
- **Raw result file:** None
- **MEASUREMENT:** 74 tests passed and 43 skipped. Only tracked `.gitkeep`
  placeholders exist in result/figure directories. No CUDA CSV or figure exists.
- **INTERPRETATION:** The repository milestone is locally stable. This does not
  establish CUDA compilation, correctness, or performance.
- **Limitations:** Every custom-device test skipped and no NVIDIA toolchain was
  exercised.
- **NEXT EXPERIMENT:** Build and validate this exact milestone on NVIDIA Linux;
  preserve its raw CSV before replacing shared trees with warp reductions.
- **Student reflection:** `TODO(student): Define what “stable” means here
  without treating skipped GPU work as success.`

## Day 4 checkpoint — Commit 064

- **What was implemented:** Parallel final normalization; expanded normal,
  stress, odd-width, block-boundary, and flattened-row CUDA gates; an artifact
  preflight for matched historical comparisons; median, p25, p75, and throughput
  aggregation; commit-aware latency/throughput plotting; evidence-only figure
  generation; coalescing and boundary audits; and a stabilized shared-tree block
  milestone. Warp/lane helpers now support shuffle maximum and sum reductions.
  Maximum uses one shared value per warp and a first-warp final reduction. Sum
  reduction is warp-local but deliberately retains a padded 256-entry shared
  tree until Commit 065.
- **What was actually measured:** No CUDA compilation, output, latency,
  throughput, speedup, memory metric, or profiler result was measured. The
  comparison and figure commands stopped on missing raw/summary artifacts. On
  Apple Silicon, the final applicable suite reports 74 passed and 43 CUDA-only
  skips; Python and shell syntax checks pass. Test duration is not a benchmark.
- **What I learned:** `TODO(student): Explain how a shuffle reduction moves
  register values within one warp and why a cross-warp bridge is still needed.`
- **What surprised me:** `TODO(student): Record your own observation; no
  personal reflection has been inferred.`
- **Unresolved questions:** Whether the CUDA source compiles; whether changed
  reduction order satisfies fixed tolerances; whether warp communication lowers
  latency; how much shared memory/synchronization actually changes; and where
  block or warp strategies cross over by sequence length.
- **Next day:** Compact warp sums, remove the remaining full shared-tree paths,
  validate/stress the complete warp kernel, verify fusion, attempt matched warp
  benchmarking, expose block size as an experimental variable, measure
  128/256/512 only on NVIDIA hardware, choose from evidence, and add strong
  `torch.compile` comparisons through Commit 080.

## Warp-reduction CUDA validation attempt — Commit 067

- **Date/time:** 2026-08-22, America/Toronto
- **Git commit:** Commit 067 — `validate warp-reduction kernel correctness`
- **Hardware:** Apple Silicon arm64; no NVIDIA GPU
- **Software:** Darwin, Python 3.11.15, PyTorch 2.13.0 without CUDA
- **Research question:** Does the completed compact warp-reduction kernel match
  the trusted PyTorch path for every required correctness length and input
  family?
- **HYPOTHESIS:** Replacing the shared trees with two-level shuffle reductions
  preserves causal scaled-softmax semantics within `rtol=1e-5`, `atol=1e-6`.
- **Independent variable:** Custom warp-reduction CUDA output versus PyTorch
  reference output
- **Controlled variables:** FP32, existing deterministic seeds and scales,
  required lengths, stress families, and fixed tolerances
- **Metrics:** Numerical closeness, row sums, exact masked zeros, finiteness,
  non-negativity, shape, dtype, and device
- **Command/script:** `.venv/bin/python -m pytest -q
  tests/test_cuda_operator.py -rs`
- **Raw result file:** None; correctness test console only
- **MEASUREMENT:** The CPU-safe coverage-contract test passed. All CUDA kernel
  cases skipped because no NVIDIA device is available, so no CUDA output was
  compared.
- **INTERPRETATION:** The intended CUDA regression matrix is complete and
  discoverable, but warp-kernel correctness is not established.
- **Limitations:** No extension build, kernel launch, synchronization, or device
  arithmetic occurred on this host.
- **NEXT EXPERIMENT:** Run the exact test module after building the extension in
  Google Colab, and treat any failure as a blocker before benchmarking.
- **Student reflection:** `TODO(student): Explain why preserving tolerances is
  more scientifically useful than relaxing them after an optimization.`

## Shared-tree versus warp-reduction benchmark attempt — Commit 070

- **Date/time:** 2026-08-22, America/Toronto
- **Git commit:** Commit 070 — `benchmark warp-reduction optimization against prior commit`
- **Hardware:** Apple Silicon arm64; no NVIDIA GPU
- **Software:** Darwin, Python 3.11.15, PyTorch 2.13.0 without CUDA
- **Research question:** Does compact warp communication change latency and
  throughput relative to the stabilized shared-tree block milestone?
- **HYPOTHESIS:** Warp shuffles and eight shared partials may lower reduction
  overhead, especially where row work is large enough to amortize launch and
  barrier costs.
- **Independent variable:** Git revision: block milestone `f9420de` versus the
  warp-reduction revision produced after Commit 069
- **Controlled variables:** One GPU/session, FP32, `batch_heads=8`, required
  shapes, seed, warmups, iterations, scale, input construction, and timed region
- **Metrics:** Median, p25, p75 microseconds, elements/second, and derived speedup
- **Command/script:** `.venv/bin/python benchmarks/benchmark_softmax.py
  --implementation custom --sequence-length 128 --warmups 2 --iterations 3
  --output results/raw/warp_reduction.csv`
- **Raw result file:** None
- **MEASUREMENT:** The harness exited with “CUDA benchmark requires Linux with
  an NVIDIA GPU.” It created no result artifact.
- **INTERPRETATION:** No before/after result exists. The mechanism remains a
  hypothesis until both commits are timed on the same NVIDIA environment.
- **Limitations:** This host cannot compile or execute either CUDA revision.
- **NEXT EXPERIMENT:** In one Colab GPU runtime, build/test `f9420de`, save its
  raw CSV outside the checkout, then build/test the Commit 069 revision with
  identical controls and run comparison preflight.
- **Student reflection:** `TODO(student): Explain why measuring only the current
  revision would not answer the optimization question.`

## 128-thread launch measurement attempt — Commit 072

- **Date/time:** 2026-08-22, America/Toronto
- **Git commit:** Commit 072 — `benchmark 128-thread launch configuration`
- **Hardware/software:** Apple Silicon arm64; PyTorch 2.13.0 without CUDA
- **Research question:** What latency distribution does the 128-thread launch
  produce across the required FP32 shape registry?
- **HYPOTHESIS:** Fewer threads may reduce coordination cost on short rows but
  require more strided work per thread on long rows.
- **Independent variable:** `launch_block_size=128`
- **Controlled variables:** Required shapes, `batch_heads=8`, FP32, seed,
  warmups, iterations, kernel revision, and intended NVIDIA session
- **Metrics:** Median, p25, p75 microseconds and elements/second
- **Command/script:** `.venv/bin/python benchmarks/benchmark_launch_configs.py
  --block-size 128 --output results/raw/launch_128.csv`
- **Raw result file:** None
- **MEASUREMENT:** The CUDA guard rejected the run and no CSV was created.
- **INTERPRETATION:** Nothing is known yet about the 128-thread configuration's
  latency or throughput.
- **Limitations:** No NVIDIA runtime on this host.
- **NEXT EXPERIMENT:** Run this unchanged command in the same Colab runtime as
  the 256- and 512-thread experiments.
- **Student reflection:** `TODO(student): Predict which sequence lengths might
  favor fewer threads and explain why.`

## 256-thread launch measurement attempt — Commit 073

- **Date/time:** 2026-08-22, America/Toronto
- **Git commit:** Commit 073 — `benchmark 256-thread launch configuration`
- **Hardware/software:** Apple Silicon arm64; PyTorch 2.13.0 without CUDA
- **Research question:** What latency distribution does the provisional
  256-thread launch produce across the required FP32 shape registry?
- **HYPOTHESIS:** The middle configuration may balance column parallelism with
  per-block coordination, but that balance may vary with sequence length.
- **Independent variable:** `launch_block_size=256`
- **Controlled variables:** Identical to the 128-thread experiment
- **Metrics:** Median, p25, p75 microseconds and elements/second
- **Command/script:** `.venv/bin/python benchmarks/benchmark_launch_configs.py
  --block-size 256 --output results/raw/launch_256.csv`
- **Raw result file:** None
- **MEASUREMENT:** The CUDA guard rejected the run and no CSV was created.
- **INTERPRETATION:** The existing 256-thread default is still provisional, not
  a measured selection.
- **Limitations:** No NVIDIA runtime on this host.
- **NEXT EXPERIMENT:** Run this command beside the 128- and 512-thread commands
  without changing the Colab runtime or software environment.
- **Student reflection:** `TODO(student): Explain why a familiar default is not
  evidence that it is optimal.`

## 512-thread launch measurement attempt — Commit 074

- **Date/time:** 2026-08-22, America/Toronto
- **Git commit:** Commit 074 — `benchmark 512-thread launch configuration`
- **Hardware/software:** Apple Silicon arm64; PyTorch 2.13.0 without CUDA
- **Research question:** What latency distribution does the 512-thread launch
  produce across the required FP32 shape registry?
- **HYPOTHESIS:** More threads may expose additional parallelism on long rows,
  while extra warps and idle lanes may hurt short or early causal rows.
- **Independent variable:** `launch_block_size=512`
- **Controlled variables:** Identical to the 128- and 256-thread experiments
- **Metrics:** Median, p25, p75 microseconds and elements/second
- **Command/script:** `.venv/bin/python benchmarks/benchmark_launch_configs.py
  --block-size 512 --output results/raw/launch_512.csv`
- **Raw result file:** None
- **MEASUREMENT:** The CUDA guard rejected the run and no CSV was created.
- **INTERPRETATION:** No claim can be made about increased parallelism or its
  overhead.
- **Limitations:** No NVIDIA runtime on this host.
- **NEXT EXPERIMENT:** Complete all three commands in one Colab session, then
  summarize results by shape before choosing any default.
- **Student reflection:** `TODO(student): Explain why more threads can increase
  overhead even when the maximum block size is supported.`
