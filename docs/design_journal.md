# Design journal

## Optional Python/C++/CUDA boundary

### Problem

The project needs a native CUDA path without making CPU imports, documentation,
or tests depend on a CUDA compiler or extension binary.

### Existing evidence

The CPU package and 53 tests run on Apple Silicon. The environment checker
reports no CUDA build or execution capability. No NVIDIA build has run.

### Hypothesis

An explicitly enabled build plus a guarded runtime import will preserve the
Mac workflow while providing one clear extension path on NVIDIA Linux.

### Proposed change

Register `cuda_attention._C` behind `CUDA_ATTENTION_BUILD_CUDA=1`, bind one
`fused_causal_softmax(scores, scale)` function, and keep its launcher in the
single evolving CUDA translation unit.

### Implementation

`setup.py` owns build registration, `bindings.cpp` owns pybind11 exposure,
`common.cuh` owns the shared declaration, `fused_causal_softmax.cu` owns launcher
and device code, and `cuda_attention/operator.py` owns guarded Python dispatch.

### Correctness result

CPU imports and 53 CPU tests pass. Structural source checks pass. CUDA
correctness is not measured because no extension was compiled or run.

### Performance result

Not measured. No performance claim is supported.

### Interpretation

The layers now have explicit ownership, which should make later failures easier
to localize. Only an NVIDIA build can test whether the native boundary compiles
and links as intended.

### Next question

Can the row-serial kernel compile and match the PyTorch reference on the planned
CUDA correctness shapes?

### Git commit

Commit 026 — `document Python C++ CUDA execution path`

## Row-serial work decomposition

### Problem

A correctness-first kernel needs an unambiguous mapping from CUDA execution
coordinates to flattened softmax rows.

### Existing evidence

CPU tests define row semantics. No CUDA compilation or measurement exists.

### Hypothesis

One global CUDA thread per row will be straightforward to validate, although
serial column scans are expected to limit performance at larger sequence
lengths.

### Proposed change

Map `blockIdx.x * blockDim.x + threadIdx.x` to one row and guard threads whose
global index exceeds `rows`.

### Implementation

`csrc/fused_causal_softmax.cu` now contains the row mapping and a fixed
256-thread block constant. The public launcher remains disabled until all
softmax math exists in Commit 030.

### Correctness result

Static source checks and the CPU suite pass. CUDA behavior is not tested.

### Performance result

Not measured.

### Interpretation

The mapping makes ownership explicit but does not expose intra-row parallelism.
That limitation is a future optimization question, not a measured bottleneck.

### Next question

Can the owning thread compute a stable maximum over its allowed columns?

### Git commit

Commit 027 — `implement initial row-serial fused causal softmax kernel`

## Native input and launch failure contract

### Problem

The kernel assumes CUDA-resident, dense contiguous FP32 rows and a finite
positive scale. If those assumptions remain implicit, a caller can receive a
distant CUDA error or incorrect memory interpretation instead of a useful
message.

### Existing evidence

The Python reference already validates shape and scale. Source inspection shows
that the CUDA kernel indexes memory as contiguous FP32 values. No NVIDIA
compilation or execution evidence exists yet.

### Hypothesis

Checking assumptions before launch and checking CUDA's immediate launch status
will make failures local and diagnosable without adding work inside the kernel.

### Change

The pybind boundary now validates device, layout, dtype, dimensionality,
nonempty dimensions, contiguity, and scale. The launcher guards the input
device, uses that device's current PyTorch stream, bounds its one-dimensional
grid, and invokes `C10_CUDA_KERNEL_LAUNCH_CHECK()` after launch.

### Correctness result

Static source checks and the CPU suite pass. The checks have not executed in a
compiled extension because the local Apple Silicon host has no CUDA toolkit.

### Performance result

Not measured. Host-side validation is outside the kernel timing question.

### Interpretation

The operator contract is now explicit in code. A successful NVIDIA build and
negative-input tests are still needed before calling the error path validated.

### Next question

Does the compiled row-serial operator match the PyTorch reference on CUDA?

### Git commit

Commit 031 — `add CUDA launch validation and error checks`

## First CUDA-to-reference correctness gate

### Problem

Completing the kernel source does not demonstrate that it compiles or matches
the trusted PyTorch semantics across the Python/C++/CUDA boundary.

### Existing evidence

The CPU reference suite passes. The current Apple Silicon host has neither a
CUDA-enabled PyTorch build nor an NVIDIA device, so it cannot execute the
custom operator.

### Hypothesis

On a supported host, comparing identical CUDA-resident FP32 inputs at core
sequence lengths will expose mistakes in scaling, flattened-row masking,
normalization, output metadata, or launch handling.

### Change

`tests/test_cuda_operator.py` compares sequence lengths 32, 64, and 128 using
fixed `rtol=1e-5` and `atol=1e-6`. It also checks row sums, finite outputs,
exact causal zeros, shape, dtype, device, and selected invalid-input paths. A
single prerequisite rule makes every case skip visibly if CUDA or the compiled
extension is unavailable.

### Correctness result

On Apple Silicon, 53 CPU tests passed and all 10 CUDA-only cases skipped because
an NVIDIA CUDA device was unavailable. This validates collection and skip
behavior, not the CUDA implementation.

### Performance result

Not measured. Correctness tests are not benchmarks.

### Interpretation

The repository now contains a reproducible GPU correctness command, but the
core hypothesis remains open until the suite passes after an NVIDIA build.

### Next question

How does the CUDA path behave under extreme magnitudes and irregular sequence
lengths once NVIDIA execution is available?

### Git commit

Commit 032 — `add CUDA versus PyTorch correctness tests`

## Row-serial bottleneck hypothesis before block cooperation

### Problem

The first implementation assigns a complete row to one CUDA thread. Its three
allowed-column loops are serial inside that thread, so increasing sequence
length increases work that cannot be shared within the row.

### Existing evidence

Source inspection establishes the work mapping and serial loops. The initial
benchmark attempt exited before timing because no NVIDIA GPU was available;
there are no baseline latency, throughput, occupancy, or profiler measurements.

### Hypothesis

The row-serial mapping will underuse available parallelism for longer rows.
Giving a block ownership of one row and distributing columns across its threads
should reduce the serial work per participating thread, although reductions and
synchronization will add overhead.

### Proposed change

First map one block to one row without changing the mathematics. Then introduce
thread-strided columns, register-local partials, shared-memory combination, and
explicit synchronization as separate reviewable commits.

### Correctness result

Not applicable yet. The mapping change begins in Commit 042, and NVIDIA
correctness remains unverified.

### Performance result

Not measured. Commit 040 produced no timing data or CSV.

### Interpretation

This is a falsifiable prediction derived from the code structure, not a
diagnosed GPU bottleneck. The block design could lose at short rows if
coordination costs exceed the saved serial work.

### Next question

After both implementations have valid NVIDIA results under identical controls,
how does their latency crossover vary with sequence length?

### Git commit

Commit 041 — `document baseline bottleneck hypothesis from initial measurements`

## One-block-per-row ownership transition

### Problem

The global-thread mapping gives no natural group of threads that can cooperate
on one row's maximum and denominator reductions.

### Existing evidence

The kernel source contains three serial allowed-column loops. No GPU timing or
correctness measurement is available.

### Hypothesis

Making a block the unit of row ownership will provide a synchronization and
shared-memory scope for later intra-row reductions.

### Proposed change

Launch one block per row and use `blockIdx.x` as the row index. Keep thread 0 on
the existing serial mathematics in this commit so work decomposition changes
separately from reduction behavior.

### Implementation

The grid now contains `rows` blocks of 256 threads. Each block owns one row;
only `threadIdx.x == 0` is active until column distribution is introduced.

### Correctness result

CPU-safe tests pass and static inspection confirms the mapping. CUDA compilation
and execution are unavailable, so mathematical preservation is not measured.

### Performance result

Not measured. Most threads are deliberately idle in this transitional state.

### Interpretation

Block ownership is infrastructure for cooperation, not evidence of speedup.

### Next question

How can the block cover every allowed and masked column exactly once?

### Git commit

Commit 042 — `rewrite kernel mapping to one CUDA block per softmax row`

## Shared-memory maximum synchronization contract

### Problem

The maximum tree has producer/consumer dependencies between shared-memory
publication, successive reduction stages, and later scratch reuse.

### Existing evidence

Source inspection shows that each stage reads values written by other threads.
No CUDA execution or race-checking evidence exists.

### Hypothesis

Block-wide barriers at dependency boundaries make those reads ordered and
visible, provided every thread reaches every barrier.

### Proposed change

Document the publication and per-stage barriers, then add a handoff barrier
after all threads capture the row maximum and before denominator partials reuse
the shared array.

### Implementation

- Without the publication barrier, a thread can read a partner before that
  partner writes its local maximum.
- Without each tree-stage barrier, the next stride can consume an incomplete
  result from the previous stride.
- Without the handoff barrier, one thread can overwrite shared scratch before
  another has captured `shared_values[0]`.
- Returning or branching around a barrier is unsafe because all block threads
  must participate.

### Correctness result

Static dependency review and CPU-safe tests pass. CUDA behavior is unverified.

### Performance result

Not measured. Barrier cost remains an experimental question.

### Interpretation

Synchronization is part of the reduction algorithm's correctness, not an
optional performance annotation.

### Next question

Can the same shared array safely carry per-thread exponential sums after the
maximum handoff?

### Git commit

Commit 046 — `add synchronization for block maximum reduction`

## Shared-memory denominator reduction

### Problem

After the block maximum is known, every thread produces a partial exponential
sum. A serial scan of those partials leaves denominator combination on one
thread and does not complete the planned cooperative reduction foundation.

### Existing evidence

Source inspection confirms per-thread strided exponentials and partial sums.
No CUDA correctness or performance measurement exists.

### Hypothesis

The same synchronized halving tree used for maximum can combine denominator
partials with addition, using zero as the identity for threads without work.

### Proposed change

Reuse the shared array for sum partials after the maximum handoff, reduce
256 partials to `shared_values[0]`, and keep thread 0 normalization unchanged so
parallel writeback remains a separate experiment.

### Implementation

Every thread publishes `thread_exponential_sum`; each tree stage adds a partner
at the current stride and synchronizes before the next stage. All threads read
the final denominator, then only thread 0 divides the allowed probabilities.

### Correctness result

Static source checks, Python compilation, shell checks, and 68 CPU-safe tests
pass; 26 CUDA-only cases skip. The kernel has not compiled or executed locally.

### Performance result

Not measured. The initial baseline attempt and current block state have no CUDA
timing artifacts.

### Interpretation

Both mathematical reductions are structurally cooperative, but the whole
kernel is not yet block-parallel because normalization remains serial.

### Next question

Does parallel normalization complete a correct block-owned implementation on
the NVIDIA correctness suite?

### Git commit

Commit 048 — `implement shared-memory sum reduction`

## Global-memory coalescing audit

### Problem

Parallel column work is useful only if thread-to-address mapping avoids
unnecessary global-memory transactions. A claim of coalescing must distinguish
the address pattern visible in source from hardware transactions measured by a
profiler.

### Existing evidence

The kernel uses `column = threadIdx.x + k * blockDim.x` for scaling, stable
exponentiation, and normalization. Masked zeroing uses the same stride from the
first future column. No memory-sector or bandwidth metric exists.

### Hypothesis

Within each stride, active neighboring lanes access neighboring FP32 addresses,
which is favorable for coalesced loads and stores. Longer rows should provide
more fully active warp accesses than early causal rows.

### Proposed change

No source change is required for this audit. Trace the address formula for each
global-memory phase and record alignment, activity, and measurement caveats.

### Implementation

- Scaling reads `scores[row_offset + column]` and writes the corresponding
  output position. Consecutive active thread IDs produce consecutive columns.
- Exponentiation and normalization revisit those same consecutive positions.
- Masked zeroing starts at `query_position + 1 + threadIdx.x`, again giving
  consecutive addresses to consecutive active thread IDs.
- A later stride advances every thread by the full block width, so each warp
  begins another consecutive 32-value segment.
- For early causal queries, only a prefix of the first warp has allowed work;
  this preserves address adjacency but lowers lane utilization.
- If `row_offset` is not aligned to a memory-transaction boundary, an otherwise
  consecutive warp access may span additional sectors.

### Correctness result

No behavior changed. CPU-safe tests remain the applicable local regression;
CUDA correctness is still pending.

### Performance result

Not measured. There are no Nsight memory-sector, request, or bandwidth results.

### Interpretation

The mapping is coalescing-friendly by construction, but transaction efficiency
and achieved bandwidth cannot be concluded from source alone.

### Next question

Do Nsight Compute global-load/store efficiency and memory-sector metrics match
the predicted pattern across early and late causal rows?

### Git commit

Commit 057 — `audit global memory access pattern for coalescing`

## Block-parallel reduction milestone

### Problem

The block implementation spans several focused commits. Without one synthesis,
it is easy to confuse an implemented source property, a locally checked test
gate, and a measured NVIDIA result.

### Existing evidence

- Commit `2758618` changed ownership to one block per row.
- Commit `1a4ecbd` introduced thread-strided columns.
- Commit `77cd283` introduced register-local maxima.
- Commit `7c7991c` introduced the shared-memory maximum tree.
- Commit `3102417` documented and completed synchronization boundaries.
- Commit `49d783f` introduced per-thread stable exponential sums.
- Commit `02a4c51` introduced the shared-memory denominator tree.
- Commit `b64151f` parallelized normalization.
- Commits `d376f81`, `f467353`, `092a8c6`, and `0efe0f4` expanded prepared
  correctness coverage.
- Commit `8d86246` stopped the comparison because raw artifacts were absent.

### Hypothesis

Dividing row work across 256 threads should reduce serial work for sufficiently
long rows, while barriers, shared-memory traffic, and underfilled early causal
rows may offset the benefit for short rows.

### Proposed change

The completed shared-memory design is:

```text
one block -> one row
thread-strided scaled staging
register-local maximum partials
shared-memory maximum tree
thread-strided stable exponentials and local sums
shared-memory denominator tree
thread-strided normalization
```

### Implementation

The maximum tree uses negative infinity as its identity; the sum tree uses zero.
Both trees halve 256 partials until shared element zero holds the block result.
Every producer/consumer stage is separated by a block-wide barrier. Masked
positions are zeroed before reductions and never enter maximum or denominator
calculations.

### Correctness result

The current local suite reports 74 CPU-safe passes and 43 CUDA-related skips.
This confirms source-independent reference/tooling behavior and clean skip
semantics, not block-kernel correctness. No NVIDIA build has run.

### Performance result

Not measured. The comparison preflight rejected missing row-serial and block-
parallel raw CSVs, and no latency/throughput figure was generated.

### Interpretation

The source implements the intended block algorithm, but whether it is correct
or faster is unresolved. Coalescing-friendly address formulas and reduced
serial work are mechanisms to test, not results.

### Next question

After NVIDIA correctness and matched historical measurements exist, which
sequence lengths benefit and which costs dominate the crossover?

### Git commit

Commit 059 — `document block reduction design and measured behavior`

## Partial warp-reduction state at Day 4

### Problem

Replacing block-wide shared trees safely requires separating warp-local
register exchange from cross-warp communication.

### Existing evidence

The shared-tree milestone is source-complete but unverified on CUDA. Static
inspection confirms a fixed 256-thread block containing eight complete warps.

### Hypothesis

Shuffle reductions can reduce per-warp partials without shared-memory traffic or
block-wide synchronization at every intra-warp stage.

### Proposed change

Introduce warp helpers, reduce maximum and sum values within each warp, and
stage compact block combination separately for the two operations.

### Implementation

Maximum now uses five shuffle-down stages, one shared maximum per warp, and a
first-warp final shuffle reduction. Sum uses the same five warp-local stages,
but lane-zero results currently occupy every 32nd location in the old 256-entry
shared tree while other lanes write zero. This preserves denominator semantics
without duplicating warp sums; compact sum combination remains Commit 065.

### Correctness result

Static checks and CPU-safe regression pass. All custom CUDA cases skip locally,
so neither shuffle path is runtime-validated.

### Performance result

Not measured. No shared-memory, synchronization, latency, or occupancy metric
exists.

### Interpretation

The maximum path has completed the intended two-level structure. The sum path
has only completed its first level, so this checkpoint is intentionally partial.

### Next question

Can one sum per warp be combined through the same compact eight-slot bridge
without changing the fixed-tolerance output contract?

### Git commit

Commit 064 — `implement warp-level sum reduction with shuffle operations`

## Complete compact warp-reduction structure

### Problem

The Day 4 denominator path reduced values inside each warp but then padded the
eight warp sums back into a 256-entry shared-memory tree. That left two active
reduction strategies and retained shared storage and barriers that the warp
design was intended to avoid.

### Existing evidence

Static inspection establishes that the fixed 256-thread launch has eight full
warps. Local tests establish only that CPU behavior and CUDA skip guards remain
stable; the kernel has not compiled or executed on NVIDIA hardware.

### Hypothesis

A two-level shuffle reduction can produce the same maximum and denominator
while communicating only one value per warp through shared memory.

### Proposed change

Use an intra-warp shuffle reduction, publish lane-zero partials to an eight-slot
array, and have the first warp perform the final reduction. Remove the old
power-of-two tree assumption from the active source.

### Implementation

Both maximum and sum now follow the same hierarchy:

```text
thread-local partial
-> 32-lane shuffle reduction
-> one shared value per warp
-> first-warp shuffle reduction
-> one block-wide value
```

The identities are negative infinity for maximum and zero for addition. A
block barrier separates every cross-warp publish/consume boundary. The launcher
allocates eight floats of dynamic shared memory for the fixed configuration.

### Correctness result

The CPU-safe suite passes and CUDA-only tests skip on Apple Silicon. NVIDIA
compilation and output comparison remain unperformed.

### Performance result

Not measured. No claim is made about latency, shared-memory occupancy, or
synchronization savings until matched CUDA artifacts exist.

### Interpretation

The source now contains one reduction strategy and its invariants are explicit.
This is an implementation fact, not evidence that the strategy is faster.

### Next question

Does the completed warp hierarchy compile and match PyTorch across normal,
stress, and irregular-width cases on NVIDIA hardware?

### Git commit

Commit 066 — `replace shared-memory block reductions with warp reductions`

## Fusion-boundary audit after warp reductions

### Problem

Changing reduction mechanics can accidentally move work outside the custom
kernel, making a faster-looking kernel incomparable because it performs less
of the original scale-mask-softmax operation.

### Existing evidence

Source inspection shows one `__global__` function. It restores query position
with `row % sequence_length`, multiplies allowed scores by `scale`, writes
future columns to zero, computes stable exponentials, reduces their denominator,
and normalizes allowed probabilities. Python dispatch calls one extension
function and C++ dispatch calls one CUDA launcher.

### Hypothesis

Warp communication changes only how maximum and sum partials combine; it should
not change the fused operation boundary.

### Proposed change

Add a CPU-safe source-contract test that fails if the primary scale, causal
boundary, masked write, exponential, or single-kernel structure disappears.

### Implementation

The audit test checks structural markers in the one evolving CUDA source. The
existing device comparisons remain the numerical gate for the full fused
semantics.

### Correctness result

The source-contract test passes locally. Numerical CUDA validation remains
pending because device tests skip without NVIDIA hardware.

### Performance result

Not measured. Fusion integrity keeps future comparisons fair but does not imply
a speedup.

### Interpretation

The active source still performs the intended amount of work in one kernel.
Static inspection cannot establish generated code, runtime correctness, or
latency.

### Next question

How does this exact warp-reduction commit compare with the historical
shared-tree milestone under identical NVIDIA controls?

### Git commit

Commit 069 — `verify fused scaling and causal masking remain in-kernel`

## Launch tuning and framework comparison design

### Problem

A fixed block size is an assumption, and an eager-only baseline may make a
custom kernel look stronger than it is. Both choices need controlled evidence.

### Existing evidence

The source accepts 128, 256, and 512 threads without duplicating the kernel.
CPU-safe tests cover configuration validation, raw provenance, compile semantic
equivalence, selection math, and three-framework completeness. All NVIDIA
measurement attempts stopped at capability guards and created no artifacts.

### Hypothesis

Launch-size rankings may vary by row width. `torch.compile` may narrow the gap
to custom CUDA by optimizing the same PyTorch expression after startup.

### Proposed change

Measure all launch sizes on one kernel revision and GPU, select by median
per-shape relative latency, then measure eager/compiled/custom paths together
with compilation excluded from steady-state timing.

### Implementation

Block size flows through Python, C++, the runtime launch, and CSV metadata.
Launch selection refuses incomplete 128/256/512 results. Framework preflight
requires all three paths for every identical workload and one Git/GPU
environment.

### Correctness result

CPU-safe infrastructure tests pass. The compiled CPU fixture matches the eager
expression. CUDA correctness remains pending.

### Performance result

Not measured. The repository contains no launch or framework CSV.

### Interpretation

The experimental design is executable and guarded, but no block size or
implementation is a measured winner. The 256-thread default is provisional.

### Next question

After Colab correctness passes, which configurations win at each sequence
length and does the aggregate choice hide meaningful shape dependence?

### Git commit

Commit 079 — `document launch tuning and framework comparison results`

## CUDA 12.8 math-constant include compatibility

### Problem

The first real Colab build reached NVCC but failed because the active kernel
used `CUDART_INF_F` without directly including the CUDA header that defines it.
Relying on `cuda_runtime.h` to expose that constant transitively was not
portable to the Colab CUDA 12.8 compilation path.

### Existing evidence

On a Tesla T4 with compute capability 7.5, PyTorch 2.11.0+cu128, and NVCC 12.8,
the C++ binding compiled successfully. NVCC then reported
`identifier "CUDART_INF_F" is undefined` at the thread-local maximum identity.
The build returned code 1, so the notebook correctly blocked correctness and
performance sections.

### Hypothesis

Including `math_constants.h` explicitly will make the infinity identity visible
without changing any kernel operation, launch parameter, memory access, or
numerical tolerance.

### Proposed change

Add the defining header to the active CUDA source and make that dependency a
CPU-safe source contract. Because every configured historical implementation
uses the same constant, apply the identical header-only adjustment during those
temporary checkouts and preserve its exact diff as artifact metadata.

### Implementation

The active source now includes `math_constants.h`. The Colab historical stage
records the base commit, reason, exact Git diff, and whether the compatibility
include was applied; benchmark descriptions disclose the adjustment. The
temporary source is restored after each historical stage.

### Correctness result

Local static and notebook-integrity checks can verify the dependency and
workflow. A successful NVIDIA rebuild and the unchanged fixed-tolerance CUDA
correctness matrix remain required.

### Performance result

Not measured. A compilation repair has no performance result.

### Interpretation

The failed build exposed an include dependency, not a kernel-algorithm error.
Historical measurements remain attributable to their base commits only when
the compatibility adjustment is disclosed alongside them.

### Next question

Does the updated `main` revision compile and pass every correctness family on
the same T4 before any launch or framework timing begins?

### Git commit

Supplemental compatibility fix — hash recorded by Git history after commit.
