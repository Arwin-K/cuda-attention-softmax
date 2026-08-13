# AGENTS.md

## Project purpose

This repository is an educational machine-learning systems and GPU performance-engineering project.

It studies **one continuously evolving CUDA implementation** of fused causal scaled softmax used inside transformer attention.

The goal is not merely to obtain a fast kernel. The goal is to learn and document:

- transformer attention,
- numerically stable softmax,
- CUDA execution,
- GPU memory behavior,
- parallel reductions,
- synchronization,
- warp-level communication,
- benchmarking,
- profiling,
- experimental design,
- and the difference between microkernel performance and end-to-end ML performance.

The repository should look and read like a small research/engineering investigation performed by a Computer Engineering student learning CUDA.

---

## Primary research question

> How do GPU work decomposition, parallel reductions, warp-level communication, and launch configuration affect the performance of fused causal scaled softmax, and how much do those kernel-level optimizations translate into end-to-end transformer attention performance?

---

## One evolving implementation

There must be only **one primary CUDA implementation file**:

```text
csrc/fused_causal_softmax.cu
```

Do not create parallel source files named:

- `v1`
- `v2`
- `v3`
- `naive`
- `optimized`
- `final`

The implementation evolves through Git history.

Earlier implementation states are preserved by commits, benchmark CSVs, experiment logs, and Git hashes rather than duplicate source files.

---

## 112-commit constraint

This project is intentionally organized into **exactly 112 planned commits across 7 working days**: 16 commits per day.

The authoritative commit sequence is in:

```text
PROJECT_PLAN.md
```

Rules:

1. Do not skip planned commits.
2. Do not combine planned commits.
3. Do not create extra commits unless I explicitly approve them.
4. Use the exact commit message from `PROJECT_PLAN.md`.
5. Perform only the requested commit.
6. Stop after every commit.
7. Never proceed automatically to the next commit.
8. Keep every commit focused on one research, learning, testing, implementation, measurement, or documentation objective.
9. Where practical, leave the repository in a valid state after each commit.
10. Do not fabricate work just to satisfy the commit count.

---

## Per-commit workflow

Before changing code for a commit, explain:

1. what the commit accomplishes,
2. what problem it solves,
3. what concepts I need to understand,
4. which files are expected to change,
5. what behavior should change,
6. what behavior should remain unchanged,
7. how the change will be tested,
8. and, for performance work, the hypothesis being tested.

Then implement only that commit.

After implementation:

1. run every applicable test/check,
2. do not ignore failures,
3. update the relevant documentation,
4. commit using the exact planned commit message,
5. show:
   - `git status`
   - `git log -1 --oneline`
   - `git diff HEAD^ --stat`
6. summarize the important lines changed,
7. explain what I should understand,
8. give me three questions I should be able to answer,
9. then STOP.

---

## Platform constraints

Primary development machine:

```text
Apple Silicon macOS
```

CUDA execution environment:

```text
Linux + NVIDIA GPU + CUDA toolkit
```

Therefore:

- CPU/PyTorch reference code must work on macOS.
- The Python package must import successfully when CUDA is unavailable.
- CUDA-specific tests must skip cleanly without an NVIDIA GPU.
- CUDA compilation must not be attempted automatically on unsupported macOS environments.
- Results analysis and plotting must work without CUDA.
- CUDA benchmarks and profiling must only claim results produced on an actual NVIDIA GPU.
- Do not use Apple's MPS backend as a substitute for CUDA in this project.

---

## Repository structure

Maintain this structure:

```text
cuda-attention-softmax/
│
├── README.md
├── AGENTS.md
├── PROJECT_PLAN.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── setup.py
├── requirements.txt
├── requirements-dev.txt
│
├── cuda_attention/
│   ├── __init__.py
│   ├── reference.py
│   ├── attention.py
│   ├── operator.py
│   ├── benchmark.py
│   ├── environment.py
│   └── plotting.py
│
├── csrc/
│   ├── bindings.cpp
│   ├── common.cuh
│   └── fused_causal_softmax.cu
│
├── tests/
│   ├── test_reference.py
│   ├── test_softmax.py
│   ├── test_causal_mask.py
│   ├── test_numerical_stability.py
│   ├── test_edge_shapes.py
│   └── test_attention.py
│
├── benchmarks/
│   ├── benchmark_softmax.py
│   ├── benchmark_attention.py
│   ├── benchmark_launch_configs.py
│   ├── summarize_results.py
│   └── config.py
│
├── profiling/
│   ├── profile_pytorch.py
│   ├── profile_cuda.py
│   ├── run_ncu.sh
│   └── README.md
│
├── scripts/
│   ├── check_environment.py
│   ├── build_extension.sh
│   ├── run_tests.sh
│   ├── run_benchmarks.sh
│   └── generate_figures.py
│
├── notebooks/
│   ├── 01_learning.ipynb
│   ├── 02_gpu_experiments.ipynb
│   └── 03_results_analysis.ipynb
│
├── results/
│   ├── raw/
│   │   └── .gitkeep
│   └── summary/
│       └── .gitkeep
│
├── figures/
│   └── .gitkeep
│
└── docs/
    ├── research_question.md
    ├── background.md
    ├── methodology.md
    ├── design_journal.md
    ├── experiment_log.md
    ├── results.md
    ├── discussion.md
    ├── limitations.md
    ├── learning_journal.md
    ├── interview_notes.md
    ├── mini_paper.md
    └── blog_post.md
```

---

## ML operation

The project studies the softmax portion of causal scaled-dot-product attention:

\[
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right)V
\]

where `M` is the causal mask.

The custom CUDA operator accepts a flattened score tensor:

```text
[rows, sequence_length]
```

derived from:

```text
[batch, heads, sequence_length, sequence_length]
```

The query position is:

```text
query_position = row_index % sequence_length
```

Allowed columns:

```text
column <= query_position
```

Masked columns:

```text
column > query_position
```

The final output is a tensor of causal softmax probabilities with the same shape and dtype as the input.

---

## Stable softmax

Use numerically stable softmax:

```text
max_value = max(x)

exp_value = exp(x_i - max_value)

probability =
    exp_value / sum(exp_values)
```

The implementation and documentation must explain why subtracting the row maximum prevents dangerous exponential magnitudes without changing the mathematical softmax result.

---

## Main CUDA progression

The one implementation evolves conceptually through Git history.

Initial mapping:

```text
one CUDA thread -> one entire softmax row
```

Later:

```text
one CUDA block -> one softmax row
```

Later:

```text
thread-local partial reductions
-> warp-level reductions
-> compact shared-memory warp result array
-> final block result
```

Later work tunes launch configuration and integrates the kernel into transformer attention.

Never maintain multiple versions in source files.

---

## CUDA commenting style

CUDA comments should explain **why**, not merely restate syntax.

Bad:

```cpp
// increment col
col += blockDim.x;
```

Good:

```cpp
// Threads advance through the row in blockDim.x-sized strides.
// This exposes intra-row parallelism while allowing neighboring
// threads to begin with neighboring columns, which is favorable
// for coalesced global-memory access.
col += blockDim.x;
```

Explain important concepts where they first appear:

- host vs device,
- kernel,
- grid,
- block,
- thread,
- `threadIdx`,
- `blockIdx`,
- `blockDim`,
- warp,
- lane,
- global memory,
- shared memory,
- registers / thread-local partials,
- synchronization,
- reduction,
- warp shuffle,
- memory coalescing,
- kernel fusion,
- numerical stability.

Do not over-comment ordinary Python syntax.

---

## Correctness comes before performance

Before performance claims, compare the custom operator against PyTorch.

Required correctness sequence lengths:

```text
31
32
33
63
64
127
128
255
511
768
1023
```

Required input families:

- normal random values,
- random values multiplied by 10,
- random values multiplied by 100,
- random values multiplied by 1000,
- all zeros,
- all equal values,
- one dominant positive value,
- one dominant negative value.

Check:

- numerical closeness to PyTorch,
- correct causal masking,
- row sums approximately equal 1,
- no unexpected NaNs,
- no unexpected Infs,
- output shape,
- output dtype.

Never weaken tolerances merely to make broken code pass.

---

## Benchmark methodology

Required benchmark sequence lengths:

```text
128
255
512
768
1024
1536
2048
```

Primary experiment:

```text
dtype = FP32
batch_heads = 8
rows = batch_heads * sequence_length
columns = sequence_length
```

Every benchmark row should record:

```text
git_commit
implementation_description
sequence_length
rows
columns
dtype
warmups
iterations
median_us
p25_us
p75_us
elements_per_second
gpu_name
compute_capability
pytorch_version
cuda_version
timestamp
```

Benchmark rules:

- warm up first,
- use CUDA-aware timing,
- synchronize correctly,
- do not measure compilation as steady-state execution,
- keep allocations/mask construction outside the timed region where fairness requires it,
- use comparable inputs,
- save raw results,
- never hard-code measured values,
- never invent results when a CUDA GPU is unavailable.

---

## Framework baselines

Eventually compare:

- PyTorch eager composition,
- `torch.compile`,
- custom CUDA operator.

For complete attention also compare:

- `torch.nn.functional.scaled_dot_product_attention`.

Do not compare different amounts of work as if they were equivalent.

---

## Attention integration

The explicit custom path is:

```text
Q @ K^T
    ->
custom fused causal scaled-softmax CUDA operator
    ->
probabilities @ V
```

Benchmark separately:

1. custom softmax kernel latency,
2. complete attention latency.

The documentation must distinguish microkernel speedup from end-to-end attention speedup and discuss Amdahl's Law when supported by measured results.

---

## Profiling

Support:

1. PyTorch Profiler,
2. NVIDIA Nsight Compute when the environment permits it.

Profiler documentation must separate:

```text
MEASURED:
What the profiler actually reported.

INTERPRETATION:
What we think the measurement means.

NEXT EXPERIMENT:
How we would test the interpretation.
```

Never present an interpretation as a measured fact.

Do not claim Nsight was used if it was not actually available.

---

## Research integrity

Never fabricate:

- benchmark numbers,
- profiler results,
- GPU model,
- compute capability,
- CUDA version,
- speedups,
- throughput,
- numerical errors,
- failed/successful hypotheses,
- personal reflections.

Distinguish:

```text
HYPOTHESIS:
What we expect.

MEASUREMENT:
What the experiment produced.

INTERPRETATION:
Our explanation of the measurement.
```

Unknown results remain TODO.

Every performance claim must state its comparison baseline.

Good:

```text
The current kernel was X× faster than the row-serial baseline at S=1024.
```

Bad:

```text
The kernel was X× faster.
```

---

## Learning documentation

Maintain:

```text
docs/learning_journal.md
```

For major concepts record:

- Concept
- What I thought before
- What I learned
- Why it matters
- Mental model
- Where it appears in code
- Experiment demonstrating it
- Evidence/result
- `TODO(student): Explain this in your own words`
- Questions still open

Never invent first-person reflection for the student.

---

## Design journal

Maintain:

```text
docs/design_journal.md
```

For significant engineering changes use:

```text
Problem
Existing evidence
Hypothesis
Proposed change
Implementation
Correctness result
Performance result
Interpretation
Next question
Git commit
```

---

## Experiment log

Maintain:

```text
docs/experiment_log.md
```

For each actual experiment record:

```text
Date/time
Git commit
Hardware
Software
Research question
Hypothesis
Independent variable
Controlled variables
Metrics
Command/script
Raw result file
Observation
Interpretation
Limitations
Next experiment
```

At commits 16, 32, 48, 64, 80, 96, and 112, create a day checkpoint containing:

- what was implemented,
- what was actually measured,
- what I learned,
- what surprised me,
- unresolved questions,
- what the next day will investigate.

Student reflection sections must remain `TODO(student)` until I fill them in.

---

## Paper and blog

Eventually maintain:

```text
docs/mini_paper.md
docs/blog_post.md
```

The paper should contain:

1. Abstract
2. Introduction
3. Background
4. Research Question
5. Hypotheses
6. Methodology
7. CUDA Implementation
8. Experimental Setup
9. Results
10. Transformer Attention Experiment
11. Profiling Analysis
12. Discussion
13. Limitations
14. Future Work
15. Conclusion

Do not invent results.

The blog should explain the learning and optimization journey in a more accessible style.

---

## Scope limits

Do not automatically introduce:

- FlashAttention,
- Triton,
- CUTLASS,
- TensorRT,
- custom GEMM,
- custom backward/autograd,
- CUDA Graphs,
- distributed computing,
- multi-GPU,
- PTX hand optimization,
- SASS hand optimization.

These may appear only as future work unless explicitly approved.

---

## Definition of success

By the end of the project:

- the implementation is correct,
- benchmarks are reproducible,
- results are traceable to commits,
- the final CUDA code is understandable,
- experimental claims are evidence-based,
- documentation clearly distinguishes hypothesis from observation,
- and I can explain every significant line of `csrc/fused_causal_softmax.cu` in an interview without relying on generated notes.
