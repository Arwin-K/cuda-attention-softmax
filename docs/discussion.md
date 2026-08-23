# Discussion

## What changed performance most?

The final warp kernel's 3.22--8.92x advantage over row serial shows that a GPU
mapping can dominate performance even when the formula is unchanged. That
comparison changes both row ownership and reduction strategy, so it cannot
attribute the entire gain to warp shuffles. The 1.07--1.91x advantage over the
block/shared-tree history is the cleaner communication comparison because both
versions already distribute a row across one block.

The improvement varied with sequence length rather than increasing
monotonically. More columns expose more work to distribute, but also change
loop counts, active lanes, memory traffic, and reduction overhead. Historical
growth trends support the bottleneck hypothesis but do not prove the mechanism
without matched low-level profiling.

## Why did 128 threads win the aggregate rule?

On the T4, 128 threads minimized median latency for five shorter and middle
shapes, while 256 won both longest rows. With 128 threads, each thread performs
more strided iterations at long lengths; with more threads, the kernel pays for
additional warps and coordination. The result supports shape-dependent launch
tradeoffs. It does not justify calling 128 universally optimal, and the current
source default should not be silently changed from one architecture's study.

## Framework results

The custom kernel beat eager PyTorch for equivalent scale-mask-softmax work at
all lengths. Its relationship with `torch.compile` changed with shape and
reversed at 2048, supporting the decision to include a strong compiled baseline
instead of comparing only with eager framework operations. Startup compilation
was not included in the steady-state distribution, so the comparison answers a
steady-state question rather than first-call latency.

## Kernel speedup versus application speedup

The custom explicit path changes only softmax:

```text
QK^T -> custom fused causal softmax -> probabilities @ V
```

Its 1.54--2.31x improvement over explicit eager attention is meaningful, but
smaller than the softmax-only improvement at six of seven shapes. This is
consistent with Amdahl's-law reasoning: the matrix multiplications remain. The
provided Amdahl CSV is retained as a model, not relabeled as measurement,
because its predictions differ materially from observed attention speedups at
some shapes.

SDPA was faster than the explicit custom path everywhere. This does not
contradict the isolated softmax result. SDPA can optimize a broader scope,
including reducing score/probability materialization, while the custom exercise
intentionally leaves both matrix multiplications explicit. The result points
toward whole-attention IO optimization as future work.

## Profiling interpretation

The 16-byte dynamic shared allocation at 128 threads matches four floats: one
per warp. That supports, but does not independently prove, that the intended
compact reduction path was executed. Nsight's simultaneous roughly 48% DRAM
and 57% SM peak-throughput values are not enough to label the kernel solely
memory-bound or compute-bound. Matched profiler data from the shared tree would
better test whether fewer shared accesses and barriers explain its latency gap.

The PyTorch Profiler named-region totals are useful for locating the custom
kernel inside attention. They exceed or differ from steady-state medians because
profiling instrumentation and capture boundaries differ. Using them as a second
latency table would mix methodologies.

## Hypothesis assessment

- Intra-row parallelism is supported at the measured shapes, but causal
  attribution combines mapping and reduction changes.
- Warp reduction is supported relative to the block/shared-tree history at all
  seven shapes; low-level mechanism attribution remains open.
- Shape-dependent launch behavior is supported because both 128 and 256 win
  at least one shape.
- Framework ordering depends on shape: custom beats compiled PyTorch six times
  but loses slightly at 2048.
- Kernel speedup exceeds attention speedup in six of seven cases, so the
  application-translation hypothesis is partially rather than universally
  supported.

`TODO(student): In your own words, describe which result most changed your
mental model and why. Do not replace this prompt with an inferred reflection.`
