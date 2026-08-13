# Research question

## Primary question

> How do GPU work decomposition, parallel reductions, warp-level
> communication, and launch configuration affect the performance of fused
> causal scaled softmax, and how much do those kernel-level optimizations
> translate into end-to-end transformer attention performance?

This question has two levels. The first asks how CUDA implementation choices
change the softmax microkernel. The second asks whether those changes still
matter when softmax is surrounded by the two matrix multiplications in
attention. Keeping the levels separate prevents a fast microkernel from being
presented as proof of an equally large application-level improvement.

## System being studied

The target operation is the softmax portion of causal scaled-dot-product
attention:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d) + M)V
```

The custom operator will receive a two-dimensional score tensor with shape
`[rows, sequence_length]`. Each row corresponds to one query position from a
flattened `[batch, heads, sequence_length, sequence_length]` score tensor. The
causal rule is:

```text
query_position = row_index % sequence_length
allowed column = column <= query_position
masked column  = column > query_position
```

Only forward FP32 causal softmax and its use in explicit attention are in the
current experimental scope. The project does not replace the surrounding
matrix multiplications.

## Variables

### Independent variables

These are the implementation choices changed deliberately across Git commits:

- work decomposition: one thread per row versus one block per row;
- reduction method: serial scans, shared-memory block reductions, and
  warp-shuffle reductions;
- inter-warp communication through a compact shared-memory result array;
- launch configuration, including 128, 256, and 512 threads per block;
- implementation path: PyTorch eager, `torch.compile`, or the custom CUDA
  operator.

### Dependent variables

These are the outcomes used to evaluate a change:

- correctness relative to the PyTorch reference;
- causal-mask correctness and probability row sums;
- presence of unexpected NaNs or infinities;
- softmax median, 25th-percentile, and 75th-percentile latency;
- processed elements per second;
- complete attention latency;
- kernel-level speedup and end-to-end attention speedup, each with an explicit
  comparison baseline.

### Controlled variables

Performance comparisons will hold these conditions constant where applicable:

- FP32 inputs;
- `batch_heads = 8` and `rows = batch_heads * sequence_length` for the primary
  softmax experiment;
- the planned sequence-length registry;
- equivalent scale, causal mask, and softmax work across compared paths;
- input-generation method and random seed;
- warmup count, measured iteration count, and CUDA synchronization method;
- allocation and mask-construction boundaries around the timed region;
- GPU, software versions, and execution environment within a comparison.

Hardware and software metadata will be recorded rather than assumed to be
constant across unrelated experiment sessions.

## Initial hypotheses

These statements are predictions, not results. Each one remains unresolved
until a later commit records measurements produced on an NVIDIA GPU.

### H1 — Intra-row parallelism

**HYPOTHESIS:** Assigning one block to a row will reduce softmax latency relative
to the one-thread-per-row baseline for most of the larger planned sequence
lengths because multiple threads can process the row concurrently.

**FALSIFICATION:** The hypothesis is not supported if the block-parallel kernel
does not lower median latency for a majority of the planned sequence lengths at
or above 512 under the same benchmark protocol.

### H2 — Warp-level reductions

**HYPOTHESIS:** Warp-shuffle reductions with one compact shared result per warp
will reduce latency relative to full shared-memory block reductions for a
majority of the planned benchmark shapes.

**FALSIFICATION:** The hypothesis is not supported if the warp-reduction commit
does not improve median latency for a majority of shapes under controlled
before-and-after measurement.

### H3 — Launch configuration

**HYPOTHESIS:** Launch configuration will affect performance differently across
sequence lengths, so one thread-block size may not dominate every tested shape.

**FALSIFICATION:** The hypothesis is not supported if one of 128, 256, or 512
threads per block has the lowest median latency for every planned shape by more
than the observed timing variability.

### H4 — Framework baselines

**HYPOTHESIS:** The performance relationship among PyTorch eager,
`torch.compile`, and the custom CUDA operator will depend on sequence length;
the custom operator should not be assumed faster before measurement.

**FALSIFICATION:** This exploratory hypothesis is not supported if the ordering
of all three paths remains the same across every planned shape within measured
variability.

### H5 — Microkernel versus attention speedup

**HYPOTHESIS:** Any softmax-only speedup will translate into a smaller
end-to-end attention speedup because `QK^T` and the probability-times-`V`
matrix multiplication remain outside the custom kernel.

**FALSIFICATION:** The hypothesis is not supported if the measured end-to-end
speedup consistently matches or exceeds the softmax-only speedup under an
equivalent baseline and workload.

## Success criteria

The investigation succeeds when:

1. the reference and custom paths satisfy the planned correctness invariants;
2. each performance result is tied to raw data, a Git commit, and environment
   metadata;
3. comparisons measure equivalent work and name their baseline;
4. observed results are kept separate from interpretations;
5. kernel and complete-attention performance are reported separately; and
6. the evolution remains understandable through one CUDA source file and its
   Git history.

## Evidence status after Commit 003

No CUDA kernel, benchmark, profiler trace, GPU measurement, or speedup exists
yet. All hypotheses above are pending. The only verified platform observation
is that the current Apple Silicon environment has no installed PyTorch and no
available CUDA runtime, as reported by `scripts/check_environment.py`.
