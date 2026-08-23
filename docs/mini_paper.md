# CUDA Optimization of Fused Causal Softmax for Transformer Attention

**Author:** Arwin Karir<br>
**Affiliation:** TODO(student): add institution, program, and contact information.

This Markdown manuscript mirrors `mini_paper.tex`. Every number below comes
from the preserved run at `results/runs/2026-08-23_tesla-t4_ca87722/`.

## 1. Abstract

This study investigates how CUDA work decomposition, block and warp
reductions, and launch configuration affect a fused FP32 causal scaled-softmax
operator and explicit transformer attention. A single evolving CUDA source was
validated against PyTorch, then benchmarked on an NVIDIA Tesla T4. All 88
structured softmax comparisons passed, with maximum absolute error
`3.5763e-7`; all seven attention shapes also passed. Across sequence lengths
128--2048, the final warp-reduction kernel was 3.22--8.92x faster than the
row-serial historical implementation, 1.07--1.91x faster than the shared-tree
block implementation, and 1.40--3.94x faster than equivalent PyTorch eager
softmax. Replacing eager softmax in explicit attention yielded 1.54--2.31x
end-to-end speedup, while PyTorch scaled-dot-product attention remained faster
than the explicit custom path at every tested shape. These results show both
the value of intra-row parallelism and the limit of optimizing one component of
a larger attention pipeline. Conclusions are restricted to one T4 session,
FP32 forward execution, and the tested shapes.

## 2. Introduction

Transformer attention computes scores, applies a causal scaled softmax, and
multiplies the probabilities by values. Softmax is conceptually small, but its
maximum and sum reductions require communication across a row. This makes it a
useful educational case for CUDA threads, blocks, warps, synchronization,
memory access, and numerical stability. Git history is the implementation
record: only `csrc/fused_causal_softmax.cu` is primary, and earlier designs are
recovered by commit rather than copied into versioned source files.

## 3. Background

For a row \(x\), stable softmax computes
\(p_i=\exp(x_i-m)/\sum_j\exp(x_j-m)\), where \(m=\max_j x_j\).
Subtracting the maximum prevents large positive logits from overflowing the
exponential without changing the normalized probability. Causal attention
also excludes columns after the query position. For flattened scores with
shape `[rows, S]`, this project uses `query_position = row_index % S` and
allows only `column <= query_position`.

A CUDA warp is a hardware execution group of 32 threads. Shuffle instructions
exchange register values within a warp, while a compact shared-memory array can
carry one partial result per warp to a final reduction. This design avoids a
full block-sized shared-memory reduction tree. It is related to, but much
narrower than, IO-aware full-attention algorithms such as FlashAttention.

## 4. Research Question

> How do GPU work decomposition, parallel reductions, warp-level
> communication, and launch configuration affect the performance of fused
> causal scaled softmax, and how much do those kernel-level optimizations
> translate into end-to-end transformer attention performance?

## 5. Hypotheses

The preregistered hypotheses predicted that intra-row block parallelism would
outperform a row-serial mapping at larger shapes; warp reductions would improve
on full shared-memory trees; the best block size would depend on sequence
length; framework ordering could vary with shape; and softmax speedup would
usually exceed explicit-attention speedup. Original falsification criteria are
preserved in `research_question.md`; status labels are evaluated only from the
imported evidence.

## 6. Methodology

The operator performs scaling, causal masking, maximum subtraction,
exponentiation, denominator reduction, and normalization. Historical kernels
were rebuilt from three Git commits: row serial (`8f07d762`), block/shared-tree
(`f9420de0`), and warp reduction (`a3736910`). The notebook applied a recorded
header-only CUDA 12.8 compatibility adjustment to historical checkouts; it did
not change their algorithms. Current framework and attention benchmarks use
the measured clean commit `ca87722a`.

CUDA events measured 100 iterations after 25 warmups. The primary softmax
workload used FP32, `batch_heads=8`, and `rows=8*S`; attention used batch 1,
eight heads, and head dimension 64. Medians and interquartile values were
computed from raw per-iteration samples. Inputs and allocations were outside
the timed interval. Correctness used fixed `rtol=1e-5` and `atol=1e-6`.

## 7. CUDA Implementation

The initial mapping assigned an entire row to one thread, leaving the row scan
serial. The next mapping assigned one block per row: each thread accumulated a
strided local maximum and exponential sum, then shared-memory reduction trees
combined partials. The final design first reduces within each warp using
shuffle operations, stores one value per warp in a small shared array, and lets
the first warp complete the block reduction. A second strided pass writes
normalized probabilities. All stages preserve stable maximum subtraction and
the causal predicate inside one kernel.

## 8. Experimental Setup

Measurements were collected on one Tesla T4 (compute capability 7.5) in
Google Colab with PyTorch 2.11.0+cu128, CUDA toolkit 12.8, Python 3.13.15, and
FP32 tensors. Sequence lengths were 128, 255, 512, 768, 1024, 1536, and 2048.
The run manifest records a clean worktree and commit
`ca87722a00ebf585cd788c67949e7c0b32dca788`. The experiment is therefore a
controlled case study, not a cross-architecture performance claim.

## 9. Results

All 88 structured softmax cases passed. They covered the required irregular
lengths and eight input families; all masked outputs were exactly zero and no
unexpected NaNs or infinities appeared. The maximum observed absolute error
was `3.5763e-7`, and the maximum probability-row sum error was `3.5763e-7`.

The final warp kernel improved over row serial at every measured length by
3.22--8.92x and over the shared-tree block kernel by 1.07--1.91x. Against
equivalent PyTorch eager scale-mask-softmax, it achieved 1.40--3.94x speedup.
It beat steady-state `torch.compile` in six of seven shapes; at sequence length
2048, `torch.compile` was about 2.1% faster.

Launch tuning did not produce one per-shape winner. The 128-thread block won
five shapes and 256 threads won at 1536 and 2048; 512 threads won none. The
notebook selected 128 by its median per-shape relative-latency rule. This is an
observed T4 policy, not a universal block-size recommendation.

## 10. Transformer Attention Experiment

The explicit custom path computes `Q @ K^T`, invokes the custom softmax, then
computes `probabilities @ V`. It was numerically close to explicit PyTorch for
all seven shapes, with maximum absolute output error `2.9802e-7`.

Replacing eager softmax produced 1.54--2.31x speedup for explicit attention.
Kernel speedup exceeded attention speedup in six of seven matched shapes,
consistent with the surrounding matrix multiplications limiting application
benefit. PyTorch's production scaled-dot-product attention was faster than the
explicit custom path at every length, by 1.25--2.53x. This is not a kernel-only
comparison: SDPA can choose a fused attention backend and therefore represents
a stronger production baseline.

## 11. Profiling Analysis

**MEASURED:** At sequence length 512, the PyTorch Profiler capture attributed
110.207 microseconds of CUDA time to the custom fused kernel. Named device
totals were 505.565 microseconds for custom explicit attention, 1095.195 for
explicit eager attention, and 408.222 for SDPA. Nsight Compute profiled four
launches of the target kernel (three warmups and one target), using 128 threads,
24 registers per thread, and 16 bytes of dynamic shared memory. Reported kernel
duration was 106.976--107.584 microseconds, DRAM throughput was 48.33--48.79%
of peak, and SM throughput was 56.72--57.00% of peak.

**INTERPRETATION:** The small shared-memory footprint is consistent with the
compact per-warp design. Throughput percentages alone do not prove that the
kernel is exclusively memory-bound or compute-bound. Profiler captures include
instrumentation conditions and are not substituted for 100-sample medians.

**NEXT EXPERIMENT:** Compare matched profiler metrics for the historical shared
tree and warp implementations, then repeat on a newer GPU architecture.

## 12. Discussion

The large row-serial-to-warp gains support exposing parallel work within each
row. The smaller, consistently positive shared-tree-to-warp gains isolate the
benefit of changing the reduction after work decomposition was already
parallel. Launch results show that occupancy, per-thread work, and coordination
costs interact with row length. Attention results demonstrate Amdahl's-law
reasoning: accelerating one stage cannot remove work in two matrix
multiplications. Measured and modeled attention values remain separate because
the simple model does not reproduce every observed latency, especially at
shorter shapes.

## 13. Limitations

Only FP32 forward execution, causal masking, one head dimension, one
`batch_heads` setting for softmax, and one Tesla T4 session were measured.
There is no backward kernel, mixed-precision study, dropout, arbitrary mask,
multi-GPU experiment, energy measurement, or statistical replication across
independent sessions. Historical builds used a documented compatibility include
for CUDA 12.8. Framework internals and Colab hardware/software can change.
Percentiles describe timing samples from this run; they are not confidence
intervals over environments.

## 14. Future Work

Repeat the complete matrix on at least one Ampere-or-newer GPU and across
independent sessions; add FP16/BF16 accumulation analysis and backward support;
profile matched historical kernels; test more batch/head/head-dimension
combinations; quantify compile startup separately; and compare with specialized
library softmax and fused-attention implementations under equivalent work.

## 15. Conclusion

On the measured T4 workload, changing work decomposition delivered the largest
kernel gain, warp communication improved further on shared-memory trees, and
block-size preference varied by shape. The custom operator substantially
outperformed equivalent eager softmax and improved explicit attention, but did
not outperform production SDPA. The central systems lesson is not that a custom
kernel is universally fastest: GPU mapping and communication choices matter
locally, while end-to-end value depends on how much of the application remains
outside the optimized component.

## References

The LaTeX source cites the Transformer paper, CUDA Programming Guide,
FlashAttention, Amdahl's original paper, PyTorch SDPA and profiler
documentation, and the Nsight Compute profiling guide. Bibliographic records
are stored in `references.bib`.

## Evidence map

- Raw softmax samples: `artifacts/benchmarks/raw/softmax_raw.csv`
- Historical samples: `artifacts/benchmarks/raw/historical_raw.csv`
- Launch samples: `artifacts/benchmarks/raw/launch_configuration_raw.csv`
- Correctness cases: `artifacts/correctness/correctness_results.csv`
- Attention samples: `artifacts/attention/attention_raw.csv`
- PyTorch profiler events: `artifacts/profiler/pytorch_profiler_events.csv`
- Nsight export: `artifacts/profiler/nsight/ncu_raw_export.csv`
- Environment and run state: `artifacts/environment/environment.json` and
  `artifacts/experiment_manifest.json`
