# PROJECT_PLAN.md

# CUDA Attention Softmax — 112-Commit / 7-Day Research Plan

## Operating rule

There is one evolving CUDA implementation:

```text
csrc/fused_causal_softmax.cu
```

Git history preserves the implementation's evolution. Do not create source-file copies named V1/V2/V3/naive/optimized/final.

Each planned commit has one focused purpose. Perform only the requested commit and stop afterward.

## Website journal rule

Every commit is also a publishable learning step. Before and after each commit, add the corresponding entry to `WEBSITE_JOURNAL.md` using the exact commit number and message. Write in a candid first-person-ready journal style without inventing the student's personal reactions, measurements, or hardware results. Distinguish the planned narrative from a completed entry: planned entries describe the question being investigated; completed entries state only what was actually changed, tested, and observed.

---

# Day 1 — Research framing, stable softmax, attention fundamentals, and CPU correctness

## Commit 001

**Exact commit message:** `scaffold research-oriented CUDA project`

**Purpose:** Create the complete repository skeleton without implementing ML or CUDA logic.

**Learn:** research repo organization; separation of source, tests, benchmarks, profiling, results, notebooks, and docs.

**Expected state:** Only scaffolding and placeholder files should exist; imports/tests should not depend on CUDA yet.

**Validation:** Run basic file/tree checks and any trivial import/test infrastructure available.

**Documentation:** README placeholders and minimal docs scaffolding.

---

## Commit 002

**Exact commit message:** `add environment detection and platform-aware imports`

**Purpose:** Make the package explicitly aware of Apple Silicon development versus remote NVIDIA execution.

**Learn:** host platform detection; CUDA availability; graceful optional capabilities.

**Expected state:** Importing and environment checks should succeed on an M3 Mac and accurately report no CUDA.

**Validation:** Run environment script locally; verify package import; verify CUDA absence is not treated as a project failure.

**Documentation:** Document Mac development / NVIDIA execution workflow.

---

## Commit 003

**Exact commit message:** `document research question and initial hypotheses`

**Purpose:** Write the research question, variables, hypotheses, and success criteria before implementation.

**Learn:** research question; independent/dependent/control variables; falsifiable hypotheses.

**Expected state:** No performance claims; hypotheses clearly labeled as predictions.

**Validation:** Documentation review only.

**Documentation:** Update research_question.md.

---

## Commit 004

**Exact commit message:** `add CUDA and transformer background notes`

**Purpose:** Create beginner-level technical background needed for the study.

**Learn:** softmax; transformer attention; CUDA execution model; GPU memory; reductions; warps.

**Expected state:** Background is educational and does not claim project-specific results.

**Validation:** Documentation review.

**Documentation:** Update background.md.

---

## Commit 005

**Exact commit message:** `add learning journal and experiment log templates`

**Purpose:** Create reusable learning and experiment journal structures.

**Learn:** research logging; separating observation from interpretation; personal reflection.

**Expected state:** Templates include TODO(student) fields rather than fabricated reflection.

**Validation:** Documentation review.

**Documentation:** Update learning_journal.md and experiment_log.md.

---

## Commit 006

**Exact commit message:** `implement stable softmax reference in PyTorch`

**Purpose:** Implement a transparent numerically stable softmax in PyTorch without relying solely on torch.softmax.

**Learn:** softmax equation; subtract-max trick; reduction dimensions.

**Expected state:** Manual stable implementation matches torch.softmax within tolerance.

**Validation:** CPU tests on small random and large-value inputs.

**Documentation:** Add educational docstrings/comments.

---

## Commit 007

**Exact commit message:** `add causal mask construction to reference operator`

**Purpose:** Construct causal masking based on flattened row/query position.

**Learn:** autoregressive causality; flattened attention row layout; boolean masks.

**Expected state:** Future columns receive zero probability after softmax.

**Validation:** CPU masking tests on tiny hand-verifiable matrices.

**Documentation:** Document row_index % sequence_length.

---

## Commit 008

**Exact commit message:** `combine scaling masking and softmax reference path`

**Purpose:** Combine scaling, causal masking, and stable softmax into the complete PyTorch reference operator.

**Learn:** scaled dot-product attention normalization; operation ordering.

**Expected state:** One trusted reference function exists for future CUDA comparisons.

**Validation:** Reference equivalence and row-sum tests.

**Documentation:** Document operator semantics.

---

## Commit 009

**Exact commit message:** `add basic reference softmax correctness tests`

**Purpose:** Add focused baseline correctness tests for the reference softmax.

**Learn:** test invariants; probability normalization; reference testing.

**Expected state:** Tests fail on incorrect normalization and pass on correct implementation.

**Validation:** pytest CPU suite.

**Documentation:** No performance work.

---

## Commit 010

**Exact commit message:** `add causal masking correctness tests`

**Purpose:** Add targeted causal-mask correctness tests.

**Learn:** mask semantics; boundary positions; first/last query row behavior.

**Expected state:** All masked future probabilities are exactly/appropriately zero.

**Validation:** pytest CPU suite with explicit expected masks.

---

## Commit 011

**Exact commit message:** `add numerical stability stress tests`

**Purpose:** Stress numerical stability with increasingly large input magnitudes.

**Learn:** floating-point overflow; exp behavior; stable softmax.

**Expected state:** Reference remains finite for planned stress cases and matches PyTorch.

**Validation:** pytest normal*10, *100, *1000 and dominant-value cases.

**Documentation:** Record why these cases matter.

---

## Commit 012

**Exact commit message:** `add odd and non-power-of-two shape tests`

**Purpose:** Test odd and non-power-of-two sequence lengths before CUDA exists.

**Learn:** boundary testing; future warp/block edge cases.

**Expected state:** Reference works for 31/32/33 and other planned edge lengths.

**Validation:** pytest edge shape suite.

**Documentation:** Explain why adjacent boundary sizes are useful.

---

## Commit 013

**Exact commit message:** `implement explicit scaled dot-product attention reference`

**Purpose:** Implement explicit QK^T -> causal scaled softmax -> PV attention in PyTorch.

**Learn:** Q/K/V; tensor shapes; transpose; attention score matrix.

**Expected state:** Explicit forward attention produces correct output shapes and probabilities.

**Validation:** CPU attention smoke tests.

**Documentation:** Keep implementation explicit for learning.

---

## Commit 014

**Exact commit message:** `add attention shape and probability validation tests`

**Purpose:** Validate attention tensor shapes and probability invariants.

**Learn:** batch/head/sequence/head-dim dimensions; row probability sums.

**Expected state:** Tests cover shape errors and valid outputs.

**Validation:** pytest attention tests.

---

## Commit 015

**Exact commit message:** `compare explicit attention against PyTorch reference behavior`

**Purpose:** Compare explicit attention behavior with a trusted PyTorch reference path.

**Learn:** reference equivalence; causal semantics.

**Expected state:** Outputs are numerically close under controlled small cases.

**Validation:** pytest comparison.

**Documentation:** Avoid performance conclusions.

---

## Commit 016

**Exact commit message:** `add reusable tensor and seed helpers for experiments`

**Purpose:** Create deterministic tensor factories and seed helpers used by tests/benchmarks.

**Learn:** reproducibility; random seeds; fixture consistency.

**Expected state:** Experiments can regenerate identical inputs when requested.

**Validation:** Unit tests for seed/tensor helper behavior.

**Documentation:** Day 1 checkpoint in experiment_log.md; record CPU-only evidence and unresolved questions.

---

# Day 2 — Learning notebooks, extension infrastructure, and the first CUDA implementation

## Commit 017

**Exact commit message:** `add notebook lesson on softmax and numerical stability`

**Purpose:** Create an educational notebook section for tensors, softmax, and numerical stability.

**Learn:** notebook pedagogy; executable learning examples.

**Expected state:** Notebook runs conceptual CPU cells on Mac; CUDA cells are not required.

**Validation:** Notebook execution/syntax check where practical.

**Documentation:** Include TODO(student) reflections.

---

## Commit 018

**Exact commit message:** `add notebook lesson on transformer attention and causal masking`

**Purpose:** Add notebook lessons for Q/K/V, attention scores, scaling, and causal masking.

**Learn:** transformer attention intuition and tensor shapes.

**Expected state:** Small examples make attention matrices inspectable.

**Validation:** Notebook execution/syntax check.

**Documentation:** Include exercises/questions.

---

## Commit 019

**Exact commit message:** `document CPU reference methodology and learning notes`

**Purpose:** Document the CPU reference methodology and what has actually been established.

**Learn:** methods vs results; research traceability.

**Expected state:** No fabricated performance claims; CPU-only status is explicit.

**Validation:** Documentation review.

**Documentation:** Update methodology/experiment log.

---

## Commit 020

**Exact commit message:** `stabilize CPU reference test suite`

**Purpose:** Run and stabilize the complete CPU reference suite before CUDA comparison work expands.

**Learn:** regression testing; milestone checkpoints.

**Expected state:** All applicable CPU tests pass; unresolved issues are documented.

**Validation:** Full CPU pytest suite.

**Documentation:** Update the experiment log with the complete CPU regression result.

---

## Commit 021

**Exact commit message:** `add PyTorch C++ extension build infrastructure`

**Purpose:** Add build configuration for a PyTorch C++/CUDA extension without implementing kernel logic.

**Learn:** compilation; extension modules; source registration.

**Expected state:** Build infrastructure is present and macOS behavior remains graceful.

**Validation:** Config/static checks on Mac; actual CUDA build later on NVIDIA Linux.

**Documentation:** Document build prerequisites.

---

## Commit 022

**Exact commit message:** `add C++ bindings for fused causal softmax operator`

**Purpose:** Create the C++ binding/operator boundary for fused causal softmax.

**Learn:** Python-to-C++ call path; tensor validation; function signatures.

**Expected state:** C++ declarations/bindings exist without pretending CUDA implementation is complete.

**Validation:** Compile on CUDA environment when available; Mac package remains importable.

---

## Commit 023

**Exact commit message:** `add CUDA source skeleton and launch interface`

**Purpose:** Create the single CUDA source skeleton and launcher interface.

**Learn:** __global__; host launcher; kernel launch configuration.

**Expected state:** The file compiles structurally but contains only the planned minimal skeleton/TODO implementation state.

**Validation:** CUDA compile check on NVIDIA environment.

**Documentation:** Explain host vs device code.

---

## Commit 024

**Exact commit message:** `add CUDA availability guards and graceful Mac fallback`

**Purpose:** Prevent unsupported local CUDA compilation/import failures on Apple Silicon.

**Learn:** optional extension loading; capability guards.

**Expected state:** CPU/reference features still work without compiled CUDA extension.

**Validation:** Import and CPU tests on Mac; skip CUDA tests.

---

## Commit 025

**Exact commit message:** `add extension build and environment verification scripts`

**Purpose:** Add build/environment scripts for the NVIDIA execution machine.

**Learn:** nvcc; nvidia-smi; extension build; reproducible environment capture.

**Expected state:** Scripts report prerequisites and build the extension only when supported.

**Validation:** Run environment checks; build on NVIDIA Linux if available.

**Documentation:** Document commands.

---

## Commit 026

**Exact commit message:** `document Python C++ CUDA execution path`

**Purpose:** Document the complete Python -> PyTorch -> C++ -> CUDA -> GPU -> Tensor path.

**Learn:** compile-time vs runtime; bindings; launcher; device execution.

**Expected state:** A beginner can trace a function call end-to-end.

**Validation:** Documentation review.

**Documentation:** Update background/design journal.

---

## Commit 027

**Exact commit message:** `implement initial row-serial fused causal softmax kernel`

**Purpose:** Implement the first correct row-serial CUDA kernel: one GPU thread owns one softmax row.

**Learn:** global thread index; grid/block mapping; serial intra-row work.

**Expected state:** Kernel owns row mapping but later commits still add pieces of the math.

**Validation:** CUDA smoke compile/run with simple controlled input if implementation state permits.

**Documentation:** Document expected limitation: no intra-row parallelism.

---

## Commit 028

**Exact commit message:** `add stable maximum scan to CUDA kernel`

**Purpose:** Add stable maximum scanning to the row-owned CUDA thread.

**Learn:** serial max reduction; registers/thread-local state; numerical stability.

**Expected state:** Each active thread derives the allowed-row maximum correctly.

**Validation:** Compare intermediate/final behavior via correctness cases.

---

## Commit 029

**Exact commit message:** `add causal masking and scaling inside CUDA kernel`

**Purpose:** Apply attention scale and causal masking inside the CUDA kernel.

**Learn:** kernel fusion; query position; masked columns.

**Expected state:** Masked positions are excluded from max/sum and final probabilities.

**Validation:** CUDA masking tests.

**Documentation:** Document no intermediate mask/scale tensor.

---

## Commit 030

**Exact commit message:** `add exponential sum and normalization to CUDA kernel`

**Purpose:** Complete exp-sum and normalization in the row-serial kernel.

**Learn:** exp; normalization; stable softmax complete path.

**Expected state:** CUDA operator now returns full probabilities.

**Validation:** Compare CUDA output with PyTorch reference.

---

## Commit 031

**Exact commit message:** `add CUDA launch validation and error checks`

**Purpose:** Add robust input checks, kernel launch checks, and CUDA error handling.

**Learn:** launch failures; contiguous/dtype/device checks; defensive GPU programming.

**Expected state:** Bad inputs fail clearly; launch errors are surfaced.

**Validation:** Negative tests plus normal correctness suite.

---

## Commit 032

**Exact commit message:** `add CUDA versus PyTorch correctness tests`

**Purpose:** Create full CUDA-vs-PyTorch correctness tests.

**Learn:** reference testing across device boundary.

**Expected state:** Custom output matches reference over core shape set.

**Validation:** CUDA pytest suite.

**Documentation:** Day 2 checkpoint in experiment_log.md; distinguish Mac checks from NVIDIA-only evidence.

---

# Day 3 — CUDA robustness, baseline measurement, and shared-memory reduction foundations

## Commit 033

**Exact commit message:** `add CUDA numerical stress tests`

**Purpose:** Run numerical stress tests against the CUDA kernel.

**Learn:** precision differences; stability under extreme logits.

**Expected state:** No unexplained NaNs/Infs; errors are recorded honestly.

**Validation:** CUDA stress pytest.

---

## Commit 034

**Exact commit message:** `add CUDA odd sequence-length tests`

**Purpose:** Test odd sequence lengths against the row-serial CUDA path.

**Learn:** bounds; flattened row indexing; irregular sizes.

**Expected state:** 31/32/33 etc. work correctly.

**Validation:** CUDA edge-shape pytest.

---

## Commit 035

**Exact commit message:** `add benchmark configuration and shape registry`

**Purpose:** Centralize benchmark shapes and experimental parameters.

**Learn:** controlled experiments; config-driven benchmarking.

**Expected state:** Benchmark scripts share one authoritative shape registry.

**Validation:** Unit/static checks for config.

---

## Commit 036

**Exact commit message:** `add CUDA event timing utilities`

**Purpose:** Implement accurate CUDA event timing helpers.

**Learn:** asynchronous GPU execution; warmups; CUDA events; synchronization.

**Expected state:** Timing helper measures device execution rather than naive Python dispatch.

**Validation:** Self-check repeated timing on NVIDIA GPU.

**Documentation:** Document timing methodology.

---

## Commit 037

**Exact commit message:** `add PyTorch eager softmax benchmark baseline`

**Purpose:** Implement the PyTorch eager baseline benchmark for scale + causal mask + softmax.

**Learn:** fair baseline definition; timed region boundaries.

**Expected state:** Eager baseline produces CSV-ready measurements without custom kernel.

**Validation:** Benchmark smoke run on NVIDIA GPU.

---

## Commit 038

**Exact commit message:** `add custom CUDA softmax benchmark path`

**Purpose:** Add the custom CUDA benchmark path using identical shapes/input conventions.

**Learn:** fair comparison; identical workloads.

**Expected state:** Custom and eager paths can be measured by one harness.

**Validation:** Benchmark smoke run and correctness precheck.

---

## Commit 039

**Exact commit message:** `record hardware software and git metadata in benchmark CSV`

**Purpose:** Attach hardware/software/Git metadata to benchmark records.

**Learn:** reproducibility; commit traceability.

**Expected state:** Every measurement can be tied back to code and GPU environment.

**Validation:** Inspect generated CSV schema/content.

---

## Commit 040

**Exact commit message:** `collect and document initial CUDA baseline experiment`

**Purpose:** Run the first real NVIDIA GPU baseline experiment and document it without optimization.

**Learn:** baseline measurement; observation vs interpretation.

**Expected state:** Actual measured CSV exists; day checkpoint states hardware and limitations.

**Validation:** Run planned benchmark sizes; preserve raw results.

**Documentation:** Update experiment_log/results with real values only.

---

## Commit 041

**Exact commit message:** `document baseline bottleneck hypothesis from initial measurements`

**Purpose:** Use measured baseline behavior to state a bottleneck hypothesis before changing code.

**Learn:** evidence-driven optimization; serial intra-row limitation.

**Expected state:** Hypothesis is grounded in current mapping/measurements and clearly labeled.

**Validation:** Documentation-only review.

**Documentation:** Update design_journal.

---

## Commit 042

**Exact commit message:** `rewrite kernel mapping to one CUDA block per softmax row`

**Purpose:** Rewrite work mapping so one CUDA block owns one softmax row.

**Learn:** block-level cooperation; blockIdx.x as row; threadIdx.x within row.

**Expected state:** Mapping changes, while mathematical semantics remain unchanged.

**Validation:** Correctness smoke tests first.

**Documentation:** Do not add complete reduction optimization in same commit.

---

## Commit 043

**Exact commit message:** `distribute row elements with thread-strided access`

**Purpose:** Distribute row columns across threads in blockDim.x-sized strides.

**Learn:** intra-row parallelism; coalesced initial accesses.

**Expected state:** Each allowed column is covered exactly once by participating threads.

**Validation:** Correctness and bounds tests.

---

## Commit 044

**Exact commit message:** `add per-thread local maximum accumulation`

**Purpose:** Compute a thread-local partial maximum over each thread's assigned columns.

**Learn:** register-local partial reduction.

**Expected state:** Every thread owns a partial max ready for block combination.

**Validation:** Correctness integration tests.

---

## Commit 045

**Exact commit message:** `implement shared-memory maximum reduction`

**Purpose:** Implement a shared-memory tree reduction for the block maximum.

**Learn:** shared memory; parallel tree reduction.

**Expected state:** Block obtains one row maximum without warp shuffle.

**Validation:** Correctness tests, especially non-power-of-two sequence lengths.

---

## Commit 046

**Exact commit message:** `add synchronization for block maximum reduction`

**Purpose:** Add and justify synchronization required by the max reduction.

**Learn:** __syncthreads__; races; visibility between block threads.

**Expected state:** Reduction is race-free; comments explain each barrier.

**Validation:** Correctness suite and code review.

**Documentation:** Explain what breaks if each barrier is removed.

---

## Commit 047

**Exact commit message:** `add per-thread exponential partial sums`

**Purpose:** Compute per-thread exponentials and partial sums after the shared row max is known.

**Learn:** stable exponentials; per-thread partial sum.

**Expected state:** Each thread contributes a partial denominator.

**Validation:** Correctness suite.

---

## Commit 048

**Exact commit message:** `implement shared-memory sum reduction`

**Purpose:** Implement shared-memory block reduction for the softmax denominator.

**Learn:** sum reduction; synchronization.

**Expected state:** Block derives one denominator correctly.

**Validation:** Correctness suite.

**Documentation:** Day 3 checkpoint in experiment_log.md; record the current incomplete/complete reduction state honestly.

---

# Day 4 — Block-parallel validation, measured optimization, and warp-reduction foundations

## Commit 049

**Exact commit message:** `parallelize final probability normalization`

**Purpose:** Normalize probabilities in parallel across participating threads.

**Learn:** parallel writeback; masking.

**Expected state:** Allowed columns get normalized probabilities and masked columns remain zero.

**Validation:** Full correctness tests.

---

## Commit 050

**Exact commit message:** `validate block-parallel kernel against PyTorch`

**Purpose:** Validate the complete block-parallel kernel against PyTorch before measuring speed.

**Learn:** correctness gate before benchmarking.

**Expected state:** All planned core CUDA correctness tests pass.

**Validation:** Full CUDA correctness pytest.

**Documentation:** No optimization/benchmark conclusion yet.

---

## Commit 051

**Exact commit message:** `stress test block-parallel kernel on large magnitudes`

**Purpose:** Stress the block-parallel implementation with large-magnitude logits.

**Learn:** numerical stability under parallel reduction order.

**Expected state:** Behavior remains finite/close within justified tolerance.

**Validation:** CUDA stress suite.

---

## Commit 052

**Exact commit message:** `stress test block-parallel kernel on odd sequence lengths`

**Purpose:** Stress the block-parallel implementation on odd and irregular sequence lengths.

**Learn:** partial work; bounds; reduction assumptions.

**Expected state:** Irregular lengths remain correct.

**Validation:** CUDA edge suite.

---

## Commit 053

**Exact commit message:** `benchmark block-parallel kernel against baseline commit`

**Purpose:** Benchmark the new block-parallel implementation against the baseline commit.

**Learn:** before/after controlled experiment; Git commit comparison.

**Expected state:** Measured speed difference is recorded without changing kernel.

**Validation:** Run identical benchmark protocol; save raw CSV.

**Documentation:** Update experiment log with actual result.

---

## Commit 054

**Exact commit message:** `add throughput calculation and benchmark summary statistics`

**Purpose:** Add throughput and distribution summary metrics to benchmark analysis.

**Learn:** elements/second; median; quartiles.

**Expected state:** Summary statistics are computed from raw measurements.

**Validation:** Unit/check summary output.

---

## Commit 055

**Exact commit message:** `add latency and throughput plotting utilities`

**Purpose:** Create reusable latency and throughput plotting helpers.

**Learn:** data visualization; units; reproducible figures.

**Expected state:** Plots are generated from CSV only.

**Validation:** Generate smoke plots from available data.

---

## Commit 056

**Exact commit message:** `generate first optimization comparison figures`

**Purpose:** Generate the first evidence figures comparing historical baseline and block-parallel commit.

**Learn:** visual comparison; commit-aware results.

**Expected state:** Figures use actual stored measurements only.

**Validation:** Inspect generated figure files and data mapping.

---

## Commit 057

**Exact commit message:** `audit global memory access pattern for coalescing`

**Purpose:** Audit the kernel's global-memory access mapping for coalescing.

**Learn:** coalesced access; adjacent thread/address mapping.

**Expected state:** Document actual access pattern and any clear issue; no unsupported optimization claim.

**Validation:** Code/document review; optional profiler evidence if available.

**Documentation:** Update design journal.

---

## Commit 058

**Exact commit message:** `tighten arbitrary sequence-length boundary handling`

**Purpose:** Harden bounds and reduction behavior for arbitrary sequence lengths.

**Learn:** partial blocks/strides; safe reductions.

**Expected state:** Edge sizes remain correct without assuming power-of-two sequence length.

**Validation:** Full edge/correctness suite.

---

## Commit 059

**Exact commit message:** `document block reduction design and measured behavior`

**Purpose:** Document the block-parallel reduction design and measured behavior.

**Learn:** engineering narrative; hypothesis -> change -> measurement.

**Expected state:** Design journal accurately cites result files/commits.

**Validation:** Documentation review.

**Documentation:** No invented causes.

---

## Commit 060

**Exact commit message:** `stabilize block-parallel implementation`

**Purpose:** Run a full regression and stabilize the measured block-parallel implementation.

**Learn:** milestone stabilization.

**Expected state:** Block-parallel state is reproducible and documented.

**Validation:** Full applicable tests plus benchmark artifact checks.

**Documentation:** Record the stabilized block-parallel milestone without creating a duplicate implementation.

---

## Commit 061

**Exact commit message:** `add warp and lane helper utilities to CUDA code`

**Purpose:** Introduce helper functions/identifiers for warp and lane reasoning without yet replacing reductions.

**Learn:** warp=32; lane; warp ID.

**Expected state:** Helpers are clear and tested/compiled.

**Validation:** CUDA build/correctness regression.

**Documentation:** Document mental model.

---

## Commit 062

**Exact commit message:** `implement warp-level maximum reduction with shuffle operations`

**Purpose:** Implement warp-level maximum reduction using shuffle operations.

**Learn:** __shfl_down_sync; register exchange; warp reduction.

**Expected state:** Each warp can reduce its partial max correctly.

**Validation:** Correctness tests and targeted kernel checks.

---

## Commit 063

**Exact commit message:** `combine warp maxima through compact shared memory`

**Purpose:** Store one maximum result per warp and combine them through compact shared memory.

**Learn:** warp partials; small shared array; first-warp final reduction.

**Expected state:** Block row max remains correct while shared-memory footprint/strategy changes.

**Validation:** Full correctness suite.

---

## Commit 064

**Exact commit message:** `implement warp-level sum reduction with shuffle operations`

**Purpose:** Implement warp-level sum reduction for exponentials.

**Learn:** warp sum reduction; denominator computation.

**Expected state:** Each warp can reduce local denominator contributions.

**Validation:** Correctness suite.

**Documentation:** Day 4 checkpoint in experiment_log.md; state which warp-reduction stages are complete and what remains.

---

# Day 5 — Complete warp reductions, launch tuning, and stronger PyTorch baselines

## Commit 065

**Exact commit message:** `combine warp sums through compact shared memory`

**Purpose:** Combine warp-level sums through compact shared memory to form final denominator.

**Learn:** two-level reduction.

**Expected state:** Correct denominator available block-wide.

**Validation:** Full correctness suite.

---

## Commit 066

**Exact commit message:** `replace shared-memory block reductions with warp reductions`

**Purpose:** Replace the old shared-memory block reduction paths with warp-based reductions.

**Learn:** implementation simplification; reduced synchronization/shared intermediates.

**Expected state:** Only one active strategy remains in the single CUDA source file.

**Validation:** Full correctness and stress suites.

**Documentation:** Git history preserves prior state.

---

## Commit 067

**Exact commit message:** `validate warp-reduction kernel correctness`

**Purpose:** Perform complete correctness validation after the warp-reduction rewrite.

**Learn:** regression gate.

**Expected state:** All reference comparison tests pass.

**Validation:** Full CUDA pytest suite.

---

## Commit 068

**Exact commit message:** `stress test warp reductions on partial final warps`

**Purpose:** Stress cases where active data does not neatly fill warp-sized work.

**Learn:** partial final work; active lanes; shuffle correctness.

**Expected state:** Odd/irregular widths remain correct.

**Validation:** Targeted edge tests.

---

## Commit 069

**Exact commit message:** `verify fused scaling and causal masking remain in-kernel`

**Purpose:** Verify scaling and causal masking are still fused into the same kernel after reduction changes.

**Learn:** kernel fusion integrity; no accidental intermediates.

**Expected state:** Implementation still performs scale+mask+softmax in one operator/kernel path.

**Validation:** Code review plus correctness tests.

**Documentation:** Document fused operation boundary.

---

## Commit 070

**Exact commit message:** `benchmark warp-reduction optimization against prior commit`

**Purpose:** Benchmark the warp-reduction implementation against the prior measured block-reduction commit.

**Learn:** controlled optimization measurement.

**Expected state:** Actual before/after latency/throughput is stored.

**Validation:** Run identical benchmark protocol.

**Documentation:** Update design/experiment journals.

---

## Commit 071

**Exact commit message:** `add benchmark support for configurable block sizes`

**Purpose:** Parameterize benchmark/kernel launch block size for controlled tuning.

**Learn:** launch configuration; block size as independent variable.

**Expected state:** 128/256/512 can be evaluated without duplicating kernel source.

**Validation:** Build/correctness checks.

---

## Commit 072

**Exact commit message:** `benchmark 128-thread launch configuration`

**Purpose:** Measure 128-thread block configuration.

**Learn:** launch tuning measurement.

**Expected state:** Real results recorded; no selection yet.

**Validation:** Run launch-config benchmark with 128.

**Documentation:** Record raw CSV.

---

## Commit 073

**Exact commit message:** `benchmark 256-thread launch configuration`

**Purpose:** Measure 256-thread block configuration.

**Learn:** launch tuning measurement.

**Expected state:** Real results recorded under identical conditions.

**Validation:** Run launch-config benchmark with 256.

---

## Commit 074

**Exact commit message:** `benchmark 512-thread launch configuration`

**Purpose:** Measure 512-thread block configuration.

**Learn:** launch tuning measurement.

**Expected state:** Real results recorded under identical conditions.

**Validation:** Run launch-config benchmark with 512.

---

## Commit 075

**Exact commit message:** `select launch configuration from measured results`

**Purpose:** Select the default block size from measured data, not assumption.

**Learn:** data-driven tuning; tradeoffs across shapes.

**Expected state:** Chosen default and limitations are documented and implemented.

**Validation:** Correctness regression with chosen default; inspect launch results.

**Documentation:** Explain if no single configuration dominates.

---

## Commit 076

**Exact commit message:** `add torch compile softmax baseline`

**Purpose:** Add a torch.compile framework baseline for the same composed softmax operation.

**Learn:** compiled framework baseline; fusion/compiler comparison.

**Expected state:** Compiled baseline produces correct outputs and benchmark path.

**Validation:** Correctness precheck + benchmark smoke.

---

## Commit 077

**Exact commit message:** `separate compile warmup from steady-state measurements`

**Purpose:** Ensure compile/startup cost is excluded from steady-state timing.

**Learn:** warmup; compilation overhead vs inference latency.

**Expected state:** Benchmark timing reflects post-compile steady state.

**Validation:** Inspect timing workflow; repeated benchmark sanity check.

---

## Commit 078

**Exact commit message:** `compare custom CUDA against eager and compiled PyTorch`

**Purpose:** Run fair comparison among PyTorch eager, torch.compile, and custom CUDA.

**Learn:** strong baselines; fair workload comparison.

**Expected state:** Real framework comparison table/CSV exists.

**Validation:** Run benchmark matrix; preserve raw results.

**Documentation:** No cherry-picked claim.

---

## Commit 079

**Exact commit message:** `document launch tuning and framework comparison results`

**Purpose:** Document launch tuning and framework comparison results with evidence and limitations.

**Learn:** results communication.

**Expected state:** Documentation cites actual CSV/figures and separates observation from interpretation.

**Validation:** Documentation review.

---

## Commit 080

**Exact commit message:** `stabilize tuned fused kernel and Day 5 checkpoint`

**Purpose:** Run regression and create the Day 5 checkpoint for the tuned fused kernel.

**Learn:** stabilization; checkpoint.

**Expected state:** Correctness, benchmark artifacts, and documentation are coherent.

**Validation:** Full applicable tests.

**Documentation:** Day 5 checkpoint.

---

# Day 6 — Transformer integration, profiling, results, and research conclusions

## Commit 081

**Exact commit message:** `integrate custom fused softmax into explicit transformer attention`

**Purpose:** Use the custom fused softmax inside explicit transformer attention.

**Learn:** microkernel integration; QK^T -> custom softmax -> PV.

**Expected state:** Custom operator replaces only the intended attention softmax component.

**Validation:** Attention smoke/correctness tests.

---

## Commit 082

**Exact commit message:** `add end-to-end custom attention correctness tests`

**Purpose:** Add end-to-end custom attention correctness tests.

**Learn:** end-to-end numerical validation.

**Expected state:** Custom attention output matches trusted explicit reference within tolerance.

**Validation:** pytest attention CUDA suite.

---

## Commit 083

**Exact commit message:** `add PyTorch scaled dot-product attention production baseline`

**Purpose:** Add PyTorch scaled_dot_product_attention as a production full-attention baseline.

**Learn:** optimized framework baseline; apples-to-apples full attention.

**Expected state:** SDPA benchmark/reference path exists with causal behavior.

**Validation:** Output sanity/equivalence checks.

---

## Commit 084

**Exact commit message:** `validate custom attention against PyTorch SDPA outputs`

**Purpose:** Validate custom explicit attention against PyTorch SDPA outputs.

**Learn:** reference differences; tolerances; causal semantics.

**Expected state:** Outputs are appropriately close for tested configurations.

**Validation:** CUDA attention comparison tests.

---

## Commit 085

**Exact commit message:** `add end-to-end attention benchmark harness`

**Purpose:** Create end-to-end attention benchmark harness.

**Learn:** microbenchmark vs application benchmark; controlled tensor shapes.

**Expected state:** Explicit eager/custom/SDPA paths can be measured fairly.

**Validation:** Benchmark smoke run.

---

## Commit 086

**Exact commit message:** `benchmark custom attention across sequence lengths`

**Purpose:** Benchmark custom attention across planned sequence lengths.

**Learn:** end-to-end scaling behavior.

**Expected state:** Real attention results are stored with environment/Git metadata.

**Validation:** Run B=1,H=8,D=64,S planned values.

---

## Commit 087

**Exact commit message:** `compare kernel speedup with end-to-end attention speedup`

**Purpose:** Quantify and visualize the difference between kernel speedup and end-to-end attention speedup.

**Learn:** Amdahl's Law; bottleneck fractions.

**Expected state:** Comparison uses actual kernel and attention CSV data.

**Validation:** Generate/check summary and figure.

**Documentation:** Interpret cautiously.

---

## Commit 088

**Exact commit message:** `add PyTorch profiler instrumentation for softmax and attention`

**Purpose:** Instrument softmax and attention paths with PyTorch Profiler.

**Learn:** profiling; operator/kernel attribution; record_function.

**Expected state:** Profiler can capture representative runs and labels.

**Validation:** Run representative profiling session.

---

## Commit 089

**Exact commit message:** `export profiler traces and summarized CUDA timings`

**Purpose:** Export profiler traces and summarized CUDA timing tables.

**Learn:** trace artifacts; reproducibility.

**Expected state:** Profile outputs are saved without fabricated observations.

**Validation:** Check trace/table artifacts.

---

## Commit 090

**Exact commit message:** `add Nsight Compute profiling helper and documentation`

**Purpose:** Add optional Nsight Compute profiling helper and instructions.

**Learn:** kernel hardware profiling; ncu workflow; environment permissions.

**Expected state:** Helper works when available and fails gracefully otherwise.

**Validation:** Check ncu availability/script behavior.

**Documentation:** Do not claim Nsight results if unavailable.

---

## Commit 091

**Exact commit message:** `record measured profiling observations and open questions`

**Purpose:** Record only measured profiler observations and explicitly list open questions.

**Learn:** measurement vs interpretation; profiler evidence.

**Expected state:** Docs use MEASURED / INTERPRETATION / NEXT EXPERIMENT format.

**Validation:** Documentation review against actual artifacts.

---

## Commit 092

**Exact commit message:** `generate final latency throughput and speedup figures`

**Purpose:** Generate final latency, throughput, and historical optimization figures.

**Learn:** final reproducible visualization.

**Expected state:** Figures come entirely from stored CSVs.

**Validation:** Regenerate from clean analysis path.

---

## Commit 093

**Exact commit message:** `generate kernel versus attention speedup figure`

**Purpose:** Generate the kernel-speedup versus attention-speedup figure.

**Learn:** micro vs end-to-end visualization; Amdahl insight.

**Expected state:** Figure accurately maps measured baselines and units.

**Validation:** Verify underlying data values.

---

## Commit 094

**Exact commit message:** `complete reproducible results tables from benchmark CSV files`

**Purpose:** Generate reproducible result tables from benchmark CSVs.

**Learn:** table generation; traceable quantitative claims.

**Expected state:** Paper-ready tables are generated rather than manually typed.

**Validation:** Regenerate summaries and inspect values.

---

## Commit 095

**Exact commit message:** `write experimental results and discussion sections`

**Purpose:** Write results and discussion based only on real experiment artifacts.

**Learn:** scientific writing; observation vs interpretation.

**Expected state:** Every quantitative claim is traceable; missing evidence stays TODO.

**Validation:** Cross-check prose against CSV/figures.

**Documentation:** Update results.md/discussion.md.

---

## Commit 096

**Exact commit message:** `write limitations future work and research conclusions`

**Purpose:** Write limitations, future work, and conclusions without overselling.

**Learn:** scope; external validity; forward-only/FP32/one-GPU limitations.

**Expected state:** Conclusions answer hypotheses only to the extent supported.

**Validation:** Documentation review.

**Documentation:** Day 6 checkpoint in experiment_log.md; summarize measured attention/profiling evidence and its limitations.

---

# Day 7 — Publication, provenance, reproducibility, and final audit

## Commit 097

**Exact commit message:** `complete LaTeX-ready white paper source outline and references`

**Purpose:** Create LaTeX-ready white-paper source outline and reference placeholders grounded in the project.

**Learn:** paper structure; LaTeX organization; reproducible tables/figures.

**Expected state:** Paper source contains real project values only where available and explicit input placeholders otherwise.

**Validation:** Compile/check LaTeX if toolchain available.

**Documentation:** Do not invent citations/results.

---

## Commit 098

**Exact commit message:** `complete technical blog post and recruiter-focused README`

**Purpose:** Finish technical blog and recruiter-facing README from the measured story.

**Learn:** audience adaptation; concise project communication.

**Expected state:** README emphasizes problem, method, evidence, reproducibility; blog explains learning journey.

**Validation:** Cross-check claims against results.

---

## Commit 099

**Exact commit message:** `add NVIDIA interview notes and project defense questions`

**Purpose:** Create NVIDIA-style interview questions and defense notes tied to actual implementation decisions.

**Learn:** technical defense; CUDA/ML/performance follow-ups.

**Expected state:** Questions reference real code/experiments; answers do not hide weak spots.

**Validation:** Documentation review and optional mock interview.

---

## Commit 100

**Exact commit message:** `complete publication checkpoint before final audit work`

**Purpose:** Check that the paper, blog, README, and interview material are coherent before the final reproducibility audit.

**Learn:** milestone review; traceability; claim verification.

**Expected state:** Tests/artifacts/docs are coherent, claims are traceable, and remaining final-publication work is explicitly listed.

**Validation:** Run full applicable test suite; regenerate results; inspect Git status.

**Documentation:** Publication checkpoint; no new feature work.

---

## Commit 101

**Exact commit message:** `audit benchmark schema completeness across all result paths`

**Purpose:** Verify every benchmark producer emits the required traceability fields and identify any missing data without inventing it.

**Learn:** data provenance; schema validation; reproducible performance claims.

**Expected state:** A documented, automated schema audit reports complete fields or actionable gaps.

**Validation:** Run the audit against available CSV files and fixture data where applicable.

**Documentation:** Record missing measurements as TODO rather than filling them manually.

---

## Commit 102

**Exact commit message:** `add result provenance links from figures to source CSV files`

**Purpose:** Make each generated figure traceable to its source result files and Git commits.

**Learn:** provenance chains; reproducible visualization.

**Expected state:** Figure metadata or a manifest maps outputs to the inputs that produced them.

**Validation:** Regenerate a smoke figure and inspect its provenance entry.

---

## Commit 103

**Exact commit message:** `add CPU-only reproducibility verification workflow`

**Purpose:** Verify that setup, imports, reference tests, and analysis remain usable on Apple Silicon without CUDA.

**Learn:** portability boundaries; capability-based testing.

**Expected state:** A documented CPU-only workflow passes applicable checks and clearly skips GPU-only work.

**Validation:** Run the workflow locally; do not compile CUDA.

---

## Commit 104

**Exact commit message:** `add NVIDIA execution handoff checklist`

**Purpose:** Provide a precise handoff checklist for reproducing CUDA builds, tests, benchmarks, and profiling on Linux with NVIDIA hardware.

**Learn:** reproducible environment handoff; prerequisite verification.

**Expected state:** The checklist distinguishes required commands from optional profiler steps.

**Validation:** Static review; execute only environment-safe checks on the current platform.

---

## Commit 105

**Exact commit message:** `document complete kernel evolution through Git milestones`

**Purpose:** Connect the row-serial, block-parallel, warp-reduction, and tuned states to their specific historical commits.

**Learn:** using version control as an engineering narrative.

**Expected state:** Readers can navigate one source file's evolution without duplicate kernel files.

**Validation:** Check every cited commit/message against Git history or planned commit identifiers.

---

## Commit 106

**Exact commit message:** `add website-ready research journey narrative`

**Purpose:** Turn the technical progression into an accessible, evidence-conscious story for a personal website.

**Learn:** technical communication for a general engineering audience.

**Expected state:** Narrative separates the plan, measurements, and student reflections.

**Validation:** Cross-check all claims against available artifacts.

---

## Commit 107

**Exact commit message:** `add website-ready per-commit journal index`

**Purpose:** Publish concise narrative entries for every planned commit, allowing the website to show the learning path step by step.

**Learn:** transparent process documentation.

**Expected state:** All 112 entries are present with planned/completed status language.

**Validation:** Programmatically or manually confirm entries 001 through 112 are all present exactly once.

---

## Commit 108

**Exact commit message:** `cross-check website narrative against research artifacts`

**Purpose:** Audit website prose so it never turns hypotheses or plans into claimed measurements.

**Learn:** evidence-aware public communication.

**Expected state:** Every quantitative statement points to a reproducible result or remains TODO.

**Validation:** Claim-by-claim review against CSVs, profiler artifacts, and journals.

---

## Commit 109

**Exact commit message:** `add reproducibility commands to publication documentation`

**Purpose:** Collect verified setup, test, benchmark, plotting, and profiling commands in one reader-facing location.

**Learn:** executable documentation.

**Expected state:** Commands identify their platform requirements and expected artifacts.

**Validation:** Run applicable CPU commands and label unrun NVIDIA commands honestly.

---

## Commit 110

**Exact commit message:** `review educational comments in final CUDA implementation`

**Purpose:** Audit CUDA comments to ensure they explain decisions, synchronization, reductions, and memory behavior rather than restating syntax.

**Learn:** maintainable performance code; teaching through comments.

**Expected state:** Important concepts are introduced where they first matter, with no duplicate implementation files.

**Validation:** Code review plus CUDA correctness regression when an NVIDIA environment is available.

---

## Commit 111

**Exact commit message:** `perform final evidence and scope audit`

**Purpose:** Check that claims remain within the project scope and that all measurements, limitations, and future-work boundaries are honest.

**Learn:** research integrity; scope control.

**Expected state:** Unsupported claims are removed or marked TODO; forbidden scope additions remain future work only.

**Validation:** Documentation and artifact audit.

---

## Commit 112

**Exact commit message:** `finalize reproducibility audit and seven-day research checkpoint`

**Purpose:** Run the final applicable verification suite and publish the seven-day research checkpoint.

**Learn:** final research audit; reproducibility; defensible conclusions.

**Expected state:** Code, artifacts, journals, and public narrative agree; missing NVIDIA-only evidence is stated plainly.

**Validation:** Run all applicable tests; regenerate analysis artifacts; inspect Git status.

**Documentation:** Final Day 7 checkpoint; no new feature work.

---
