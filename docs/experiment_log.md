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

One Tesla T4 Colab notebook run completed CUDA correctness, benchmark, and
PyTorch Profiler stages. Its saved notebook output is available in Git history,
but the generated raw-artifact ZIP has not been imported into this checkout.
Accordingly, execution status and failure messages are recorded below while
latency, speedup, and profiler-event values remain unavailable.

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

## Launch-default selection gate — Commit 075

- **Date/time:** 2026-08-22, America/Toronto
- **Git commit:** Commit 075 — `select launch configuration from measured results`
- **Hardware/software:** CPU-only analysis path on Apple Silicon
- **Research question:** Which supported block size gives the best robust result
  across the complete planned sequence-length set?
- **HYPOTHESIS:** The winner may be shape-dependent, so a per-shape normalized
  score is safer than summing raw latency across unlike workloads.
- **Independent variable:** Launch block size 128, 256, or 512
- **Controlled variables:** One Git revision, GPU/software environment, and the
  complete identical shape set
- **Metrics:** Median relative latency across shapes and per-shape win count
- **Command/script:** `.venv/bin/python benchmarks/benchmark_launch_configs.py
  --select-from results/raw/launch_128.csv results/raw/launch_256.csv
  results/raw/launch_512.csv --output results/summary/launch_selection.json`
- **Raw result file:** None
- **MEASUREMENT:** Selection stopped at the missing 128-thread CSV. No summary
  or selected-default artifact was written.
- **INTERPRETATION:** The data requirement works as intended; 256 remains the
  historical provisional default, not a research result.
- **Limitations:** Only synthetic unit fixtures exercised the ranking formula.
- **NEXT EXPERIMENT:** Generate all three real CSVs in one Colab session, run
  the selector, inspect per-shape tradeoffs, and only then call a size tuned.
- **Student reflection:** `TODO(student): Explain why normalizing within each
  sequence length prevents the longest workloads from deciding the result by
  scale alone.`

## Eager versus compiled versus custom comparison attempt — Commit 078

- **Date/time:** 2026-08-22, America/Toronto
- **Git commit:** Commit 078 — `compare custom CUDA against eager and compiled PyTorch`
- **Hardware/software:** Apple Silicon arm64; PyTorch 2.13.0 without CUDA
- **Research question:** How does the custom fused operator compare with eager
  composition and `torch.compile` for identical causal scaled-softmax work?
- **HYPOTHESIS:** Compilation may narrow the eager/custom gap by reducing
  framework dispatches or fusing operations, but the outcome is GPU- and
  shape-dependent.
- **Independent variable:** Eager, compiled, or custom implementation path
- **Controlled variables:** Identical deterministic inputs, causal semantics,
  scale, FP32, shape registry, one Git/GPU environment, warmups, iterations,
  allocations outside timing, and compile startup outside steady state
- **Metrics:** Median, p25, p75 microseconds and elements/second
- **Command/script:** `.venv/bin/python benchmarks/benchmark_softmax.py
  --implementation all --output results/raw/framework_comparison.csv`
- **Raw result file:** None
- **MEASUREMENT:** The CUDA guard rejected the benchmark. Framework-summary
  preflight then rejected the absent raw CSV; neither artifact was created.
- **INTERPRETATION:** No framework ranking or speedup is supported. CPU tests
  establish comparison-schema behavior and small-case compiled semantics only.
- **Limitations:** No CUDA compiler backend or custom extension ran.
- **NEXT EXPERIMENT:** Build and pass CUDA correctness in Colab, run the full
  command once, then use `--frameworks` summary preflight before interpreting.
- **Student reflection:** `TODO(student): Explain why eager, compiled, and custom
  paths must normalize the same scores with the same causal rule.`

## Day 5 checkpoint — Commit 080

- **What was implemented:** The denominator now uses the same compact two-level
  warp reduction as the maximum, and the obsolete full shared trees are absent
  from the active source. Tests explicitly cover required lengths, partial data
  around warp boundaries, and the fused kernel boundary. Block sizes 128, 256,
  and 512 flow through Python, C++, one CUDA kernel launch, benchmark controls,
  and CSV metadata. Complete-data launch selection and eager/compiled/custom
  comparison preflights are implemented. `torch.compile` startup is separated
  from steady-state CUDA timing.
- **What was actually measured:** No CUDA compilation, custom output, latency,
  throughput, speedup, launch winner, compiled CUDA behavior, or profiler metric
  was measured. All benchmark commands stopped at the non-CUDA platform guard
  and produced no artifacts. On Apple Silicon, 92 tests pass and 54 GPU-only
  cases plus nine block-size/irregular-width cases skip (63 GPU-only skips in
  total); Python compilation, shell syntax, diff checks, package/environment
  checks, and the graceful build skip pass. Test duration is not a benchmark.
- **What I learned:** `TODO(student): Explain the two-level warp reduction and
  why launch size must be selected from measurements rather than intuition.`
- **What surprised me:** `TODO(student): Record your own observation after
  reviewing the code and, later, after running Colab; none is inferred here.`
- **Unresolved questions:** Whether the extension compiles; whether all fixed-
  tolerance CUDA gates pass for 128/256/512; whether warp reductions outperform
  the shared-tree and row-serial milestones; which launch size wins by shape;
  whether `torch.compile` fuses the expression; and how custom compares with
  eager and compiled PyTorch.
- **Next day:** Integrate the custom softmax into explicit transformer attention,
  validate against PyTorch SDPA, benchmark kernel versus end-to-end behavior,
  and add PyTorch Profiler and optional Nsight Compute workflows through Commit
  096. GPU-dependent Day 6 conclusions must wait for actual NVIDIA artifacts.

## Supplemental Colab notebook handoff

- **Date/time:** 2026-08-22, America/Toronto
- **Git branch:** `day-five-pt-2`
- **Purpose:** Convert the complete remote NVIDIA procedure into one connected,
  restart-aware Colab notebook while preserving the 112 planned commit numbers.
- **What was implemented:** A deterministic notebook generator and one Colab
  notebook covering runtime validation, repository checkout, extension build,
  CUDA correctness, historical kernels, launch tuning, framework and attention
  benchmarks, profiling, plots, tables, validation, manifest generation, Drive
  backup, and ZIP export. The notebook calls the repository's existing build,
  test, and benchmark interfaces instead of embedding a second implementation.
- **What was actually measured:** No CUDA result. This development host has no
  NVIDIA CUDA runtime. Only notebook generation, JSON structure, Python syntax,
  CPU-safe tests, shell syntax, import behavior, and artifact-integrity rules
  were checked locally. The checked-in notebook has no execution outputs.
- **Interpretation:** The repository now has a single auditable execution path
  for collecting the missing evidence, but its presence is not evidence that
  the kernel compiles, is correct, or is faster on any GPU.
- **NEXT EXPERIMENT:** Push `day-five-pt-2`, open
  `notebooks/04_colab_research_experiments.ipynb` on an NVIDIA Colab runtime,
  run it top to bottom, resolve any critical validation failure, and preserve
  `cuda_softmax_research_artifacts.zip` before interpreting results.
- **Student reflection:** `TODO(student): After the Colab run, explain which
  controls made comparisons fair and which limitations remain.`

## First NVIDIA extension build attempt — supplemental Colab run

- **Date/time:** 2026-08-22T16:46:28Z
- **Git commit:** `e7ee590f16135bb9543a5d84ba5f16b5d3b63a09`
- **Hardware:** NVIDIA Tesla T4, compute capability 7.5
- **Software:** Linux x86_64, Python 3.13.15, PyTorch 2.11.0+cu128, PyTorch CUDA
  12.8, NVCC 12.8, g++ 11.4
- **Research question:** Can the merged CUDA extension compile in the Colab
  environment before numerical and performance experiments begin?
- **HYPOTHESIS:** The opt-in extension build will compile for `sm_75` using the
  CUDA toolkit supplied by Colab.
- **Independent variable:** Colab CUDA 12.8 build environment
- **Controlled variables:** Git commit, single primary CUDA source, T4 runtime,
  PyTorch installation, and repository build script
- **Metrics:** Build return code and extension import readiness
- **Command/script:** `bash scripts/build_extension.sh`
- **Raw result file:** Colab artifact `build/build_log.txt`; the user preserved
  and supplied the log for diagnosis
- **MEASUREMENT:** Dependency installation returned 0. The C++ binding compiled.
  NVCC targeted `compute_75`/`sm_75` but returned code 2 for
  `identifier "CUDART_INF_F" is undefined`; the overall build returned 1 and
  the extension was not imported.
- **INTERPRETATION:** The kernel depended on a transitive header include. This
  is a compilation portability defect, not evidence about numerical correctness
  or performance.
- **Limitations:** No kernel launched, so no CUDA correctness, latency,
  throughput, speedup, or profiling conclusion is supported.
- **NEXT EXPERIMENT:** Build the committed explicit-header fix in a fresh
  artifact directory on the T4, then run the full fixed-tolerance correctness
  gate before benchmarking.
- **Student reflection:** `TODO(student): Explain why a failed compilation is
  useful experimental evidence but cannot answer a performance hypothesis.`

## End-to-end attention execution evidence — Commit 086

- **Date/time:** 2026-08-22, after the corrected extension build
- **Git commit:** `d1b3fd38b28075c7fbfcff2b03cde4a2a6b02f1d`
- **Hardware/software:** Tesla T4, compute capability 7.5, PyTorch
  2.11.0+cu128, CUDA 12.8, Python 3.13.15
- **Research question:** How does complete explicit attention with the custom
  softmax scale across the planned sequence lengths?
- **HYPOTHESIS:** Softmax acceleration will translate to a smaller complete-
  attention improvement because both matrix multiplications remain.
- **Independent variable:** Attention implementation and sequence length
- **Controlled variables:** B=1, H=8, D=64, FP32, one T4/runtime, identical Q/K/V,
  25 warmups, 100 CUDA-event samples, allocations outside timing
- **Metrics:** Per-iteration complete-attention latency in microseconds
- **Command/script:** Colab notebook attention stage; repository equivalent is
  `python benchmarks/benchmark_attention.py --implementation all --output
  results/raw/attention_raw.csv`
- **Raw result file:** The executed notebook reported
  `attention/attention_raw.csv`, `attention_summary.csv`, and
  `attention_correctness.json` in its generated ZIP. That ZIP has not been
  supplied to this workspace, so the CSV values are not available here.
- **MEASUREMENT:** The saved notebook output reports `Attention benchmarks:
  COMPLETE` and `CUDA correctness: PASS`. No attention latency value is copied
  into the repository without the raw artifact.
- **INTERPRETATION:** Execution completion establishes that the workflow ran; it
  does not support a quantitative speedup claim without the CSV.
- **Limitations:** Raw samples and summaries are unavailable to this checkout.
- **NEXT EXPERIMENT:** Import `cuda_softmax_research_artifacts.zip`, validate its
  manifest and raw schemas, then compute attention statistics from those rows.
- **Student reflection:** `TODO(student): Explain why a completion label is not
  a substitute for raw timing samples.`

## PyTorch Profiler and Nsight attempt — Commit 091

- **Date/time:** 2026-08-22, corrected Tesla T4 Colab run
- **Git commit:** `d1b3fd38b28075c7fbfcff2b03cde4a2a6b02f1d`
- **Hardware:** NVIDIA Tesla T4, compute capability 7.5
- **Software:** Linux x86_64, Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8;
  Nsight Compute reported version 2025.1.1
- **Research question:** Which framework operations and CUDA kernels account
  for isolated-softmax and complete-attention time, and what low-level metrics
  explain the fused kernel's behavior?
- **HYPOTHESIS:** Matrix multiplications will remain prominent in complete
  attention, while kernel metrics may reveal reduction or launch overhead.
- **Independent variable:** Profiled implementation path
- **Controlled variables:** One Git checkout and T4 runtime, configured profile
  shape, FP32 inputs, and notebook profiler controls
- **Metrics:** Intended PyTorch operator/kernel durations and Nsight basic
  metrics
- **Command/script:** Executed Colab notebook Sections 13 and 14; repository
  follow-ups are `python profiling/profile_pytorch.py` and
  `sh profiling/run_ncu.sh`
- **Raw result file:** The notebook created profiler artifacts in its ZIP, but
  that ZIP is not present in this checkout. No trace, profiler CSV, `.ncu-rep`,
  or Nsight CSV is available here.

### MEASURED

The saved notebook output reports the PyTorch Profiler stage `COMPLETE`. The
optional Nsight stage found `ncu` 2025.1.1 but its generated target stopped with
`ModuleNotFoundError: No module named 'cuda_attention'`; therefore no fused
kernel launched under Nsight and no hardware metric was captured.

### INTERPRETATION

PyTorch Profiler execution succeeded, but its operator names and timings cannot
be reconstructed responsibly from a completion marker. The Nsight failure is
consistent with a missing Python import path and says nothing about kernel
occupancy, memory behavior, or instruction throughput.

### NEXT EXPERIMENT

Import the original artifact ZIP to recover the PyTorch trace and event CSV.
Then rerun the corrected Nsight target, which explicitly supplies the repository
on `PYTHONPATH`; if Colab rejects hardware counters, preserve that distinct
permission error and repeat on an unrestricted NVIDIA Linux host.

- **Limitations:** Raw PyTorch profiler values are unavailable, and Nsight
  never reached the kernel. No profiling-based bottleneck claim is supported.
- **Student reflection:** `TODO(student): After inspecting the recovered trace,
  write what surprised you. Do not infer a reflection from the completion log.`

## Day 6 checkpoint — Commit 096

- **What was implemented:** Explicit custom attention now uses the one fused
  operator only between QK^T and PV; CUDA correctness tests compare its outputs
  and probabilities with the explicit reference and PyTorch SDPA. Equivalent
  explicit-eager/custom/SDPA attention benchmarks use shared inputs, correctness
  gates, CUDA events, checkpointed raw rows, and Git/GPU/software provenance.
  Matched analysis computes kernel speedup, attention speedup, and translation
  ratio. Named PyTorch Profiler regions export trace/CSV/JSON artifacts, while a
  corrected Nsight target and shell helper preserve report/CSV evidence. Figure
  and LaTeX-table generators reject incomplete, mixed, or overwritten evidence.
- **What was actually measured:** The preserved T4 notebook run at
  `d1b3fd38b28075c7fbfcff2b03cde4a2a6b02f1d` reported a successful corrected
  extension build, 70 CUDA pytest passes, 88/88 structured FP32 cases at the
  fixed tolerances, completed softmax and attention benchmarks, a reported
  128-thread launch selection, and completed PyTorch profiling. Nsight Compute
  2025.1.1 failed at package import before kernel launch. The raw ZIP is absent,
  so no latency, throughput, speedup, or profiler duration is reported. On the
  Day 6 development host, `./scripts/run_tests.sh` reported 120 passed and 69
  skipped; CUDA-only cases skip because Apple Silicon has no NVIDIA runtime.
- **What I learned:** `TODO(student): Explain in your own words why a fast
  middle operation may produce a much smaller full-attention speedup.`
- **What surprised me:** `TODO(student): Record your own reaction to the
  correctness or launch evidence; no personal reflection is inferred.`
- **Unresolved questions:** What the missing raw sample distributions show;
  whether 128 threads wins every length or only the aggregate selector; how
  custom compares with eager, compiled PyTorch, and SDPA; how kernel speedup
  translates into full attention; which PyTorch regions dominate; and whether
  corrected Nsight capture reveals reduction, occupancy, or memory limits.
- **Next day:** Build the LaTeX-ready paper outline, blog, recruiter-facing
  README, and interview material; then audit provenance, figure-to-CSV links,
  Apple Silicon and NVIDIA handoffs, kernel history, public claims, CUDA
  comments, and final reproducibility through Commit 112. Quantitative prose
  must remain pending until raw artifacts are recovered or rerun.

## Day 7 publication checkpoint — Commit 100

- **Imported evidence:** The supplied archive and exact executed notebook are
  preserved under `results/runs/2026-08-23_tesla-t4_ca87722/`. The archive
  identifies a clean `ca87722a` run on one Tesla T4 and contains raw samples,
  summaries, figures, tables, profiler exports, and validation metadata.
- **What was actually measured:** 88/88 structured softmax cases and all seven
  attention cases passed. The warp kernel improved on row serial by 3.22--8.92x,
  on shared-tree reduction by 1.07--1.91x, and on eager softmax by 1.40--3.94x.
  Custom explicit attention improved on explicit eager by 1.54--2.31x, while
  SDPA was faster than custom at every tested length. Nsight successfully
  profiled the length-512 target in this supplied run.
- **Publication changes:** Paper, blog, README, results, discussion,
  limitations, and interview notes now share the same evidence boundary and
  name both favorable and unfavorable comparisons.
- **Interpretation:** Work decomposition had the largest historical effect;
  warp communication added a smaller consistent benefit; and unchanged or
  more broadly fused attention work limits end-to-end translation. These
  statements are restricted to the measured T4 workload.
- **Student learning:** `TODO(student): Explain which evidence-boundary rule is
  most important to you and why.`
- **Remaining audit:** Automate schemas and provenance, verify Mac and NVIDIA
  workflows, trace kernel history, audit website claims, review CUDA comments,
  and complete the final reproducibility checkpoint.

## Day 7 checkpoint — Commit 112

- **What was implemented:** The supplied T4 archive and exact executed notebook
  are preserved beside an output-free reusable notebook. The repository now has
  a Markdown and LaTeX paper, educational blog, recruiter-facing README,
  interview/defense guide, website journey, 112-entry status-aware journal,
  Apple Silicon verification workflow, NVIDIA handoff, Git kernel history, and
  automated schema, figure-provenance, public-claim, history, journal, and
  evidence/scope audits. The primary implementation remains the single
  `csrc/fused_causal_softmax.cu` file.
- **What was actually measured:** The imported clean `ca87722a` run used one
  Tesla T4, FP32, 25 warmups, and 100 timing samples per path and length. It
  passed 88/88 structured softmax cases and all seven attention cases. Warp
  reduction improved on row serial by 3.22--8.92x and shared-tree reduction by
  1.07--1.91x; custom softmax improved on eager by 1.40--3.94x; custom explicit
  attention improved on explicit eager by 1.54--2.31x; and SDPA remained faster
  than custom explicit attention at all seven lengths. PyTorch Profiler and
  Nsight artifacts are present. Day 7 performed no new GPU measurement.
- **What changed in the project's understanding:** Work decomposition produced
  the largest historical performance change; warp-local communication added a
  smaller consistent improvement; launch preference varied by length; and an
  isolated kernel gain only partially translated through unchanged matrix
  multiplications. Production SDPA's broader fusion boundary matters as much as
  the custom softmax's local speed.
- **Final local verification:** On Darwin arm64 with Python 3.11.15 and PyTorch
  2.13.0 CPU, `verify_cpu_reproducibility.sh` reported 146 passed and 69
  explicit GPU-only skips. Package import, deterministic notebook, raw schema,
  figure provenance, public claims, Git history, website index, and combined
  evidence/scope gates passed. No CUDA or MPS work ran.
- **Student understanding:** `TODO(student): Explain in your own words how the
  result changed your mental model of GPU reductions and end-to-end speedup.`
- **Unresolved questions:** Cross-architecture and independent-session
  replication; matched historical Nsight metrics; shape-aware dispatch;
  FP16/BF16 numerical policy; backward/autograd; other attention layouts and
  head dimensions; and whole-attention IO-aware fusion.
- **Next work:** Preserve this checkpoint, merge the Day 7 branch after review,
  complete the student-owned reflection/affiliation fields, compile and inspect
  the venue-formatted paper, then treat every further GPU run as a new
  commit-tagged experiment rather than overwriting this T4 evidence.
