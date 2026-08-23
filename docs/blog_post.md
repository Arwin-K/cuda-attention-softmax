# From a Serial CUDA Kernel to Warp Reductions: Optimizing Transformer Softmax

Softmax looks like three lines of math: find a maximum, exponentiate, and divide
by a sum. On a GPU, those lines become a question about cooperation. Which
threads own a row? How do they agree on a maximum? Where do partial sums live?
And does making this one kernel faster actually make transformer attention much
faster?

I built this project to make those questions visible. There is only one CUDA
source file. It changes over Git history from one thread processing a whole row
to a block cooperating through warp shuffles. The history shows the learning
process while the final tree stays easy to navigate.

## The operation

The custom operator receives scores shaped `[rows, sequence_length]`. Those
rows originate in an attention score tensor shaped `[batch, heads, S, S]`.
After flattening, the query position is `row_index % sequence_length`; columns
after that position belong to the future and must get zero probability.

Stable softmax subtracts the largest allowed logit before exponentiation. A
large positive logit can otherwise make `exp(x)` overflow even when the final
probability is ordinary. Subtracting one constant from every logit leaves the
normalized distribution unchanged.

## Stage 1: one thread owns one row

The first kernel assigned one row to one CUDA thread. That thread scanned for
the maximum, scanned for the exponential sum, then normalized. The mapping is
easy to reason about, but a length-2048 row still asks one thread to do
thousands of operations while other threads cannot help with that row.

## Stage 2: one block cooperates on one row

The next design assigned a CUDA block to a row. Thread `t` processes columns
`t`, `t + blockDim.x`, and so on. Neighboring threads initially load neighboring
values, and every thread produces a local maximum and sum.

Shared-memory reduction trees combined those partials. This exposed intra-row
parallelism, but shared memory and block-wide barriers added communication.

## Stage 3: reduce inside warps first

Threads in a warp execute together and exchange register values using shuffle
instructions. The final kernel reduces inside each warp, stores only one value
per warp in a compact shared array, and lets the first warp finish the
block-wide reduction:

```text
thread-local partials
        -> one result per warp
        -> compact shared warp-result array
        -> one block-wide result
```

Synchronization is still necessary between writing the per-warp values and
reading them. Warp shuffles do not make independent warps automatically visible
to one another.

## What the T4 run measured

The full experiment ran on one NVIDIA Tesla T4 with FP32 data, 25 warmups, and
100 CUDA-event samples per implementation and length. Raw data, environment
metadata, figures, and profiler exports are checked in with the measured hash.

All 88 structured softmax cases passed against PyTorch, including odd lengths,
magnitudes up to 1000, equal values, and dominant logits. Maximum absolute
error was `3.5763e-7`, future probabilities were exactly zero, and no unexpected
NaNs or infinities appeared.

Across sequence lengths 128--2048:

- warp reduction was 3.22--8.92x faster than the row-serial history;
- warp reduction was 1.07--1.91x faster than the shared-tree history;
- custom softmax was 1.40--3.94x faster than equivalent PyTorch eager work;
- custom softmax beat steady-state `torch.compile` at six of seven shapes.

At length 2048, `torch.compile` was about 2.1% faster. That exception is useful:
a research report should show where its custom implementation does not win.

## There was no universally best block size

A 128-thread block won five shapes, while 256 threads won at 1536 and 2048.
The 512-thread choice won none. The aggregate rule selected 128, but the deeper
lesson is that more threads mean less loop work per thread and more coordination
and resource use. Launch tuning is a measurement problem.

## A fast kernel is not the whole application

The explicit attention experiment kept both matrix multiplications visible:

```text
Q @ K^T -> custom causal softmax -> probabilities @ V
```

Replacing eager softmax made this path 1.54--2.31x faster. Softmax-only speedup
was larger at six of seven shapes because the matrix multiplications did not
change. This is Amdahl's law in practice: untouched stages still set a limit.

PyTorch scaled-dot-product attention was faster than the explicit custom path
at every shape, by 1.25--2.53x. SDPA is a production baseline that can select a
fused attention backend and avoid intermediate work the explicit path retains.
The comparison clarifies what would need to be fused next.

## What profiling can and cannot say

For length 512, Nsight Compute recorded 24 registers per thread and 16 bytes of
dynamic shared memory for the 128-thread kernel, consistent with one value per
warp. It reported roughly 48% peak DRAM throughput and 57% peak SM throughput.
Those percentages do not prove one exclusive bottleneck. A matched profiler
comparison with the shared-tree history is a better next experiment.

Profiler captures also are not substitutes for steady-state medians. Profiling
asks where resources and time appear in a captured run; repeated CUDA-event
timing asks how latency is distributed under the benchmark protocol.

## The result I would defend

This project does not show that handwritten CUDA is universally faster than
PyTorch. It shows that, for one T4 workload, intra-row decomposition produced
the largest kernel gain, warp communication improved further on shared-memory
trees, and application benefit was constrained by work outside softmax.

Correctness gates, controlled baselines, raw artifacts, and explicit
limitations are the real research contribution. The next step is replication
on another architecture—not a stronger adjective in the README.
