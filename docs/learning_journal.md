# Learning journal

Use one entry for each major concept. Generated project notes may describe the
concept and point to code or evidence, but the student's prior understanding,
personal learning, surprises, and remaining questions must stay as
`TODO(student)` until the student writes them.

## Concept entry template

### Concept

`TODO: Name the concept.`

#### What I thought before

`TODO(student): Describe your prior mental model in your own words.`

#### What I learned

`TODO(student): Explain what changed in your understanding.`

#### Why it matters

`TODO: Explain the correctness, performance, or engineering relevance.`

#### Mental model

`TODO: Give a compact analogy, diagram, or step-by-step model.`

#### Where it appears in code

`TODO: Link the relevant file, function, and Git commit.`

#### Experiment demonstrating it

`TODO: Name the controlled experiment or state that no experiment exists yet.`

#### Evidence/result

`TODO: Link measured output or explicitly write "Not measured yet."`

#### Explain it in my own words

`TODO(student): Explain this concept without copying the generated notes.`

#### Questions still open

`TODO(student): List what you still want to understand or test.`

## Entry checklist

- Separate facts, measurements, and interpretations.
- Link measurements to raw artifacts and the implementation Git commit.
- Do not turn a hypothesis into an observed result.
- Do not write first-person reflection on the student's behalf.

## Stable softmax

### Concept

Maximum subtraction for numerically stable row-wise softmax.

#### What I thought before

`TODO(student): Describe your prior mental model in your own words.`

#### What I learned

`TODO(student): Explain what changed in your understanding.`

#### Why it matters

Direct FP32 exponentiation can overflow for large positive logits. Subtracting
the row maximum bounds the largest exponential at one without changing the
mathematical probability.

#### Mental model

Shift every competitor by the same amount before comparing them; their relative
differences remain unchanged, but the numerical range becomes safer.

#### Where it appears in code

`cuda_attention/reference.py::stable_softmax`, introduced in Commit 006.

#### Experiment demonstrating it

`tests/test_numerical_stability.py` compares large inputs with PyTorch and
demonstrates overflow in a deliberately naïve exponential.

#### Evidence/result

The applicable CPU tests passed; no CUDA behavior has been measured.

#### Explain it in my own words

`TODO(student): Explain this concept without copying the generated notes.`

#### Questions still open

`TODO(student): List what you still want to understand or test.`

## Flattened causal rows

### Concept

Recovering query position after flattening attention score rows.

#### What I thought before

`TODO(student): Describe your prior mental model in your own words.`

#### What I learned

`TODO(student): Explain what changed in your understanding.`

#### Why it matters

The custom operator sees `[rows, sequence_length]`, not explicit batch, head,
and query axes. It must still exclude every future key correctly.

#### Mental model

Each group of `sequence_length` rows counts query positions from zero again, so
the remainder of `row_index / sequence_length` identifies the query.

#### Where it appears in code

`cuda_attention/reference.py::causal_allowed_mask`, introduced in Commit 007.

#### Experiment demonstrating it

`tests/test_causal_mask.py` checks first, middle, last, and wrapped rows by hand.

#### Evidence/result

CPU tests confirmed exact zeros in masked positions; CUDA remains unimplemented.

#### Explain it in my own words

`TODO(student): Explain this concept without copying the generated notes.`

#### Questions still open

`TODO(student): List what you still want to understand or test.`

## Thread-strided row access

### Concept

Distributing row columns across the threads of one CUDA block.

#### What I thought before

`TODO(student): Describe your prior mental model in your own words.`

#### What I learned

`TODO(student): Explain what changed in your understanding.`

#### Why it matters

A block cannot reduce a row in parallel until each thread owns a disjoint part
of that row. Striding by `blockDim.x` covers arbitrary widths without assuming
that the number of columns equals the block size.

#### Mental model

Thread `t` handles columns `t`, `t + blockDim.x`, `t + 2*blockDim.x`, and so on.
The first pass places neighboring threads on neighboring addresses.

#### Where it appears in code

`csrc/fused_causal_softmax.cu`, introduced in Commit 043.

#### Experiment demonstrating it

CUDA correctness cases for power-of-two and irregular widths are prepared but
have not run without an NVIDIA GPU.

#### Evidence/result

Static inspection confirms the disjoint indexing formula. Runtime correctness
and memory-transaction behavior are not measured.

#### Explain it in my own words

`TODO(student): Explain this concept without copying the generated notes.`

#### Questions still open

`TODO(student): List what you still want to understand or test.`

## Block synchronization

### Concept

Using `__syncthreads()` to create visibility and ordering among block threads.

#### What I thought before

`TODO(student): Describe your prior mental model in your own words.`

#### What I learned

`TODO(student): Explain what changed in your understanding.`

#### Why it matters

Shared memory is common storage, but writes do not become safely consumable in
the required order merely because all threads execute the same source code.
Every reduction stage depends on values produced by the preceding stage.

#### Mental model

A barrier is a checkpoint: no thread leaves until every block thread arrives,
and shared-memory work before the checkpoint is visible afterward.

#### Where it appears in code

`csrc/fused_causal_softmax.cu` around staging, partial publication, each maximum
tree stage, and the shared-scratch handoff in Commit 046.

#### Experiment demonstrating it

No NVIDIA race-checking or correctness experiment has run. The dependency can
be established from the producer/consumer relationships in the source.

#### Evidence/result

CPU-safe regression tests pass; CUDA synchronization behavior is unmeasured.

#### Explain it in my own words

`TODO(student): Explain why a barrier inside a branch taken by only some block
threads can deadlock or become undefined.`

#### Questions still open

`TODO(student): Which barriers can later disappear when warp shuffles replace
parts of the block-wide shared-memory tree?`

## Memory coalescing

### Concept

Combining neighboring lanes' global-memory requests into a small number of
hardware transactions when their addresses fall in nearby memory sectors.

#### What I thought before

`TODO(student): Describe your prior mental model in your own words.`

#### What I learned

`TODO(student): Explain what changed in your understanding.`

#### Why it matters

Global memory has high bandwidth, but scattered lane addresses can require more
transactions than a consecutive access pattern.

#### Mental model

A warp is a group ordering data from adjacent shelves: consecutive requests can
be served together, while scattered requests require more trips.

#### Where it appears in code

The thread-strided global-memory loops in `csrc/fused_causal_softmax.cu`.

#### Experiment demonstrating it

Source-level address audit only. A future Nsight Compute run must inspect actual
memory transactions and bandwidth.

#### Evidence/result

Neighboring active lanes use neighboring FP32 addresses. Hardware coalescing
efficiency is not measured.

#### Explain it in my own words

`TODO(student): Explain why adjacent addresses are favorable but do not prove
perfect transaction efficiency.`

#### Questions still open

`TODO(student): How do row alignment and partially active causal warps change
the number of sectors requested?`

## Warps and lanes

### Concept

A warp is CUDA's 32-thread hardware execution group; a lane is one thread's
position from 0 through 31 within that group.

#### What I thought before

`TODO(student): Describe your prior mental model in your own words.`

#### What I learned

`TODO(student): Explain what changed in your understanding.`

#### Why it matters

Warp shuffle instructions exchange register values among lanes without using a
full shared-memory array or block-wide barrier for each warp-local stage.

#### Mental model

The 256-thread block contains eight teams of 32. `lane_id()` is a seat number
inside a team; `warp_id()` selects the team.

#### Where it appears in code

`lane_id()` and `warp_id()` in `csrc/fused_causal_softmax.cu`, Commit 061.

#### Experiment demonstrating it

No CUDA execution yet. The helpers are introduced before they replace any
active reduction path.

#### Evidence/result

Static inspection confirms eight complete warps for the fixed 256-thread block.
Runtime behavior is unmeasured.

#### Explain it in my own words

`TODO(student): Explain the difference between a block, warp, and lane.`

#### Questions still open

`TODO(student): How should a shuffle reduction handle lanes that do not carry a
valid partial?`

## Partial data participation versus a partial hardware warp

### Concept

A launched 256-thread block contains eight complete 32-lane hardware warps.
An early causal row can nevertheless give only a prefix of lanes real column
data, which is partial *work participation*, not a partially launched warp.

### Why it matters

The shuffle helpers use a full-warp mask. Every lane must execute them, including
lanes with no assigned column, or the mask would promise participation that the
program does not provide.

### Mental model

Imagine 32 students required to remain in a reduction line. Students without a
number still stand in the line holding the identity: negative infinity for a
maximum or zero for a sum.

### Where it appears in this project

`thread_maximum` begins at negative infinity and
`thread_exponential_sum` begins at zero. The shuffle calls occur after the
thread-strided loops, so data ownership does not control shuffle participation.

### Experiment demonstrating it

The CUDA gate tests allowed causal prefixes immediately below, at, and above
32-column boundaries: 31/32/33, 63/64/65, and 95/96/97.

### Evidence/result

The test cases are collected but skip on Apple Silicon. Their numerical result
must be obtained on NVIDIA hardware.

### Explain it in my own words

`TODO(student): Explain why an idle data lane must still call the full-mask
shuffle helper.`

### Questions still open

- Would an active-mask implementation help sufficiently narrow rows, or add
  more control complexity than it saves?

## Compiled framework baseline

### Concept

`torch.compile` captures and compiles a PyTorch operation graph. It may combine
framework operations or reduce dispatch overhead while preserving PyTorch-level
semantics.

### Why it matters

Comparing only against eager PyTorch can exaggerate the apparent advantage of a
custom kernel. A compiled composition is a stronger baseline for the same
scale-mask-softmax workload.

### Mental model

Eager mode sends each operation separately through the framework. Compilation
first studies the whole recipe and then prepares an execution plan. That first
study is startup cost, while later calls represent steady-state use.

### Where it appears in this project

`compile_causal_softmax()` in `benchmarks/benchmark_softmax.py` wraps the same
function used by the eager benchmark.

### Experiment demonstrating it

A CPU-safe semantic test uses the eager compile backend. The research benchmark
will use the default CUDA compiler backend in Colab.

### Evidence/result

The CPU test establishes expression equivalence for its small case. No compiled
CUDA latency or generated-kernel result exists yet.

### Explain it in my own words

`TODO(student): Explain why torch.compile is a fairer competitor than eager
PyTorch alone.`

### Questions still open

- Does compilation fuse this particular expression on the assigned Colab GPU?
- How large is first-call compilation time relative to steady-state latency?

## Artifact provenance and evidence boundaries

### Concept

Provenance links a public claim to raw samples, derived summaries, environment,
Git revision, and generated output. An evidence boundary states what those
links can and cannot establish.

### Why it matters

A plausible figure can be stale, a correct summary can describe another code
revision, and a completed workflow can still lack a required raw artifact.
Performance research is defensible only when those relationships are explicit.

### Mental model

Treat a result as a chain of custody:

```text
measured code + environment
  -> indexed raw samples
  -> checked summary
  -> figure/table
  -> bounded public claim
```

Breaking or changing any link requires a new review; a SHA-256 detects byte
changes but cannot prove that the experimental design was unbiased.

### Where it appears in this project

The imported run directory contains source hashes and the executed notebook.
Schema auditing reconstructs statistics, figure provenance hashes direct
inputs, public-claim auditing checks headline ranges, and the scope audit keeps
the measured `ca87722a` revision separate from later documentation commits.

### Experiment demonstrating it

The Day 7 audit independently recomputed 84 benchmark-group summaries, linked
12 figure files to sources, and rejected stale missing-ZIP language after the
archive was imported.

### Evidence/result

All combined evidence/scope checks passed for the documented Commit 110 parent.
This establishes internal consistency, not cross-GPU replication.

### Explain it in my own words

`TODO(student): Explain why a figure hash is valuable but insufficient evidence
for a performance conclusion.`

### Questions still open

- How should multiple independent GPU sessions be aggregated without hiding
  between-session variance?
- Which provenance format would interoperate best with an external artifact
  repository or paper supplement?
