# NVIDIA interview notes and project-defense questions

These notes are prompts for explaining the work in your own words. Memorizing a
speedup is less useful than being able to state the workload, baseline, reason,
and limitation behind it.

## Thirty-second project summary

I optimized the FP32 forward causal scaled-softmax stage of transformer
attention in CUDA. I kept one evolving source file and used Git commits to
preserve a row-serial baseline, a block-per-row shared-memory reduction, and a
warp-shuffle reduction. I validated against PyTorch, timed equivalent work with
CUDA events, tuned 128/256/512-thread launches, integrated the operator into
explicit attention, and compared with eager, `torch.compile`, and production
SDPA. On one Tesla T4, the final kernel was 3.22--8.92x faster than row serial
and 1.40--3.94x faster than eager softmax; explicit attention improved
1.54--2.31x, but SDPA remained faster at every tested shape.

## Architecture and CUDA fundamentals

### What is the mapping from transformer scores to the kernel?

Attention scores have shape `[batch, heads, S, S]`. The operator views the
first three dimensions as `rows = batch * heads * S` and keeps `S` columns. For
flattened row `r`, `r % S` recovers the query position; only columns at or
before it are allowed.

### Why use one block per row?

One thread per row leaves the maximum scan, sum, and normalization serial.
One block per row lets many threads process disjoint strided columns while
sharing the two row-wide reduction results. A grid with one block per row also
makes rows independent and removes inter-block coordination.

### What are lane and warp IDs?

For a 32-thread warp, `lane = threadIdx.x % 32` identifies a thread inside its
warp and `warp_id = threadIdx.x / 32` identifies the warp inside the block.
Lane-local register values can be exchanged with shuffle instructions.

### Why are warp shuffles useful here?

They combine values inside a warp without writing every intermediate value to
shared memory or executing a block-wide barrier at each reduction step. They
do not communicate across warps, so the design stores one partial per warp in
shared memory and uses the first warp for the final stage.

### Why is synchronization required?

Independent warps can reach the code at different times. After warp leaders
write partial results, `__syncthreads()` ensures all writes are visible before
the first warp reads them. Removing that barrier creates a race.

### What makes the memory access coalesced?

In a strided loop, neighboring threads first read neighboring columns. Their
addresses are contiguous, allowing the hardware to combine requests into fewer
memory transactions. Later iterations remain separated by `blockDim.x` per
thread but contiguous across neighboring lanes.

### Why subtract the maximum?

Softmax is invariant to adding or subtracting one constant from every logit.
Choosing the maximum makes every exponent argument non-positive, preventing
large positive values from overflowing `expf`. It is a correctness property,
not merely a speed trick.

### How does causal masking interact with the reduction?

Masked columns must not influence either maximum or denominator. Threads skip
columns greater than the query position in both reduction passes, and the final
write sets them to exact zero.

## Measurement and reasoning

### Why use CUDA events rather than Python time?

CUDA launches are asynchronous. Host wall-clock timing can stop before device
work finishes. Events are recorded on the CUDA stream and the end event is
synchronized, so elapsed time represents GPU work inside the selected boundary.

### Why warm up before measuring?

Warmups reduce contamination from lazy initialization, cache cold starts, and
`torch.compile` startup. Compile startup is explicitly separated from
steady-state timing rather than hidden in one distribution.

### Why report median and quartiles?

One latency sample can be disturbed by scheduling or runtime noise. The median
describes the center without being dominated by a few outliers; the 25th and
75th percentiles expose dispersion. They do not provide confidence across GPUs
or independent sessions.

### What did launch tuning show?

On the T4, 128 threads won five lengths and 256 won at 1536 and 2048; 512 won
none. The aggregate median-relative-latency rule selected 128. This supports
shape-dependent tradeoffs and does not establish a universal best block size.

### What is the clearest causal performance comparison?

Row serial versus block parallel changes both work decomposition and reduction,
so the large 3.22--8.92x result supports the combined change. Shared-tree versus
warp reduction keeps block-per-row decomposition and more closely isolates the
communication change; its 1.07--1.91x gain is smaller but more specific.

### Why did softmax speedup not fully translate to attention?

The custom path changes only softmax. `QK^T` and `probabilities @ V` still run.
Amdahl's law says unchanged work bounds total speedup. Measured softmax speedup
exceeded explicit-attention speedup for six of seven shapes.

### Why was SDPA faster?

The explicit custom path materializes scores and probabilities around a custom
softmax. PyTorch SDPA can select a production fused backend that reduces
intermediate memory traffic across more of attention. The benchmark is useful
as a production reference but does not isolate the softmax kernel alone.

### Is the final kernel memory-bound?

The Nsight run reported about 48% peak DRAM throughput and 57% peak SM
throughput. Those values alone cannot establish a unique bottleneck. I would
compare matched source-level and stall metrics across historical kernels and
repeat on another architecture before making that attribution.

## Evidence questions

### How do you know the output is correct?

The CUDA run passed 70 pytest device cases and 88 structured comparisons over
required odd lengths and eight input families. Fixed tolerances were
`rtol=1e-5`, `atol=1e-6`; maximum absolute error was `3.5763e-7`. Tests also
checked exact future zeros, row sums, dtype, shape, NaNs, and infinities.

### How can someone reproduce one number?

Start from the run directory. Its environment and manifest identify the clean
measured commit. Raw CSVs contain every timed sample; summaries derive medians
and quartiles; figures and prose point back to those sources. The output-free
notebook can rerun the workflow, while the executed notebook preserves the
original console evidence.

### What is the strongest limitation?

All performance conclusions come from one Tesla T4 session. They cover FP32
forward causal softmax and one attention head dimension, with no backward pass
or independent-session replication. The results explain this case study, not
every GPU or transformer workload.

## Project-defense questions to practice

1. Draw the data flow from `[B,H,S,S]` scores to flattened rows and back.
2. Walk through the maximum reduction for a 128-thread block.
3. Why does the first warp need an identity value for lanes beyond the number
   of warps?
4. What could break when the sequence length is 255 rather than 256?
5. Why must masked values be excluded before selecting the maximum?
6. What does `__shfl_down_sync` exchange, and what does its mask guarantee?
7. Which global reads and writes occur in each kernel pass?
8. Why can a 256-thread block beat 128 threads only at longer rows?
9. Which comparison isolates warp-reduction effects most closely, and why?
10. Why is a median speedup ratio not a hardware-independent constant?
11. Where are input allocation and mask creation relative to timing?
12. How does `torch.compile` warmup differ from ordinary CUDA warmup?
13. Why should profiler event totals not replace benchmark medians?
14. How would you test a claim that synchronization is the bottleneck?
15. Why is SDPA a fair application baseline but not a pure softmax baseline?
16. What additional work is required for backward and mixed precision?
17. How would you extend the kernel to arbitrary masks without corrupting the
    reduction identities?
18. Which conclusions would you expect to change on an H100, and why?
19. What does Git history preserve that extra `v1`/`v2` files would obscure?
20. If one raw CSV disagreed with a figure, which artifact should be trusted and
    what audit would you run?

## Common answer traps

- Do not say a block and warp are the same; a block contains one or more warps.
- Do not say shuffle instructions communicate across blocks or arbitrary warps.
- Do not say more threads always make a kernel faster.
- Do not report the custom explicit path as faster than SDPA; it was not.
- Do not call profiler throughput percentages definitive bottleneck proof.
- Do not generalize one T4 session to all NVIDIA architectures.
- Do not describe the Amdahl estimate as a measured attention result.
- Do not imply CPU or MPS execution validates CUDA compilation or performance.

## TODO(student)

Write your own two-minute explanation of the most surprising result, including
one measured fact, one interpretation, one limitation, and one next experiment.
Do not replace this prompt with an invented reflection.
