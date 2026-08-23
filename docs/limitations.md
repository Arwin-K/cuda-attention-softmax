# Limitations

## Scope of the operator

The custom operator studies one forward-pass component: dense FP32 causal
scaled softmax over a contiguous two-dimensional score tensor. It does not
implement a backward kernel, autograd registration, dropout, arbitrary masks,
padding masks, sparse attention, FP16, BF16, or FP8. Therefore, it is an
educational forward operator rather than a drop-in training primitive.

The causal rule depends on `query_position = row_index % sequence_length`.
That contract matches flattened `[batch, heads, sequence, sequence]` attention
scores but does not encode batches with different valid lengths.

## Numerical scope

Correctness was measured in FP32 on one tested revision using `rtol=1e-5` and
`atol=1e-6`, including 88 structured cases. Passing those inputs does not prove
correct rounding for all floating-point values. Parallel reduction order differs
from PyTorch, and different architectures or compiler settings may change the
last few bits without changing the mathematical operation.

## Performance evidence

The preserved Tesla T4 notebook reports completed benchmark stages, but its raw
artifact ZIP is missing. Latency distributions, throughput, historical
speedups, framework speedups, and kernel-to-attention translation ratios cannot
be audited or reported from this checkout. The reported 128-thread selection is
also missing its per-shape source rows and remains specific to one run.

Colab is a managed, potentially shared environment. GPU clocks, thermal state,
background load, driver policy, and assigned GPU model are not fully controlled.
One T4 run cannot establish external validity across Ampere, Ada, Hopper, or
future GPUs.

## Baseline limitations

The isolated eager, compiled, and custom softmax paths are designed to perform
the same mathematical work. The explicit custom attention path, however, is not
an implementation-equivalent replacement for PyTorch SDPA: SDPA may select a
production fused backend spanning more of attention. Comparing their complete
outputs is valid for correctness; interpreting a latency difference requires
acknowledging the different optimization boundaries.

The explicit path materializes the full score and probability matrices, so its
memory footprint scales quadratically with sequence length. The study does not
compare against FlashAttention-style tiled algorithms that avoid full
materialization.

## Profiling limitations

PyTorch Profiler completed in the saved run, but its trace is not available in
this checkout. Nsight Compute failed at target import before any profiled kernel
launch, leaving occupancy, memory-transaction, instruction, and warp-efficiency
questions unanswered. The corrected target still needs an environment that
permits hardware performance counters.

## Future work

The immediate next steps are evidence recovery and replication:

1. recover the original artifact ZIP or rerun the corrected notebook;
2. verify its manifest, Git hash, schemas, sample counts, and correctness gate;
3. regenerate every figure and LaTeX table from those CSVs;
4. rerun Nsight with the corrected import path on an unrestricted NVIDIA host;
5. repeat the complete experiment on at least one newer GPU architecture; and
6. repeat sessions to estimate run-to-run variability, not only within-run
   CUDA-event quartiles.

Technical extensions should remain separate research questions: shape-aware
block-size dispatch, FP16/BF16 accumulation policy, vectorized loads where
alignment permits, backward/autograd support, variable-length masking, and
fusion strategies that reduce or avoid score-matrix materialization.

## Research conclusion

This project establishes a correct, inspectable path from serial row ownership
to cooperative block work and compact two-level warp reductions in one evolving
CUDA source file. On the preserved T4 revision, the extension compiled after a
portability fix and passed the fixed-tolerance CUDA correctness gates.

The performance part of the research question is only partially answered. The
notebook reports that launch tuning, softmax benchmarks, attention benchmarks,
and PyTorch profiling completed, but the missing raw artifacts prevent a
defensible statement about how much work decomposition, warp communication, or
launch size changed latency—or how much any kernel improvement reached complete
attention. The honest conclusion is therefore a verified implementation and a
reproducible measurement pipeline with quantitative findings still pending
artifact recovery or a controlled rerun.
