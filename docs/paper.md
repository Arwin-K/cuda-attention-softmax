# CUDA Optimization of Fused Causal Softmax for Transformer Attention

**Author:** Arwin Karir

This is the repository's primary, browser-readable research paper. Its
typesetting source is available as [`paper.tex`](paper.tex), with bibliography
records in [`references.bib`](references.bib). No compiled paper PDF is stored
in the repository. Every number below comes from the preserved run at
`results/runs/2026-08-23_tesla-t4_ca87722/`.

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
useful systems case for CUDA threads, blocks, warps, synchronization,
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

## 4. Research Questions

- **RQ1 — Correctness:** Does the operator preserve the intended causal
  scaled-softmax contract for the structured FP32 test domain?
- **RQ2 — Ablation:** How much do cooperative work decomposition and compact
  warp communication contribute to latency reduction?
- **RQ3 — Launch policy:** Is one threads-per-block choice optimal across the
  tested sequence lengths?
- **RQ4 — Framework comparison:** How does the fused operator compare with
  equivalent eager and compiled framework softmax paths?
- **RQ5 — Application translation:** How much isolated-softmax improvement
  survives in explicit attention, and how does that path compare with
  production SDPA?

## 5. Hypotheses

The hypotheses predicted that intra-row block parallelism would
outperform a row-serial mapping at larger shapes; warp reductions would improve
on full shared-memory trees; the best block size would depend on sequence
length; framework ordering could vary with shape; and softmax speedup would
usually exceed explicit-attention speedup. Their status was evaluated only
after importing the measured evidence.

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

The independent experimental unit for broader generalization is the session,
and this study has `n_session = 1`. The 100 event timings per configuration
describe conditional within-session variation; they are not 100 independent
hardware or session replications. Accordingly, the analysis uses medians,
interquartile ranges, geometric-mean ratios, and conditional rank-separation
effect sizes without population-level p-values or confidence intervals.

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

| Category | Configuration | Recorded value |
|---|---|---|
| Hardware | GPU | NVIDIA Tesla T4, compute capability 7.5, 15.6 GB |
| Software | Toolchain | PyTorch 2.11.0+cu128, CUDA 12.8, Python 3.13.15 |
| Softmax geometry | Matrix | `rows = 8S`, `columns = S`, FP32 |
| Attention geometry | Dimensions | batch 1, 8 heads, head dimension 64 |
| Sequence lengths | `S` | 128, 255, 512, 768, 1024, 1536, 2048 |
| Launch sweep | Threads per block | 128, 256, 512 |
| Timing | Protocol | 25 warmups, 100 CUDA-event observations |
| Provenance | Revision | `ca87722a00ebf585cd788c67949e7c0b32dca788` |

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

### 9.1 Historical-kernel ablation

Medians are in microseconds; ratios greater than one favor warp reduction.

| S | Row serial | Shared tree | Warp reduction | Row / warp | Shared / warp |
|---:|---:|---:|---:|---:|---:|
| 128 | 149.664 | 52.992 | 46.416 | 3.22x | 1.14x |
| 255 | 371.120 | 80.016 | 63.552 | 5.84x | 1.26x |
| 512 | 977.248 | 214.832 | 129.024 | 7.57x | 1.67x |
| 768 | 2101.248 | 315.632 | 235.552 | 8.92x | 1.34x |
| 1024 | 2032.896 | 594.016 | 311.728 | 6.52x | 1.91x |
| 1536 | 4595.920 | 733.184 | 668.560 | 6.87x | 1.10x |
| 2048 | 8827.104 | 1303.936 | 1220.928 | 7.23x | 1.07x |
| **Geometric mean** | — | — | — | **6.34x** | **1.32x** |

The row-to-warp comparison bundles work ownership, memory access, reduction,
and normalization changes. Shared tree versus warp reduction is the narrower
communication ablation.

### 9.2 Launch-configuration sweep

Medians are in microseconds; bold values are the lowest at each length.

| S | 128 threads | 256 threads | 512 threads | Best |
|---:|---:|---:|---:|---:|
| 128 | **24.528** | 38.816 | 51.504 | 128 |
| 255 | **38.912** | 59.584 | 110.432 | 128 |
| 512 | **112.256** | 112.912 | 250.464 | 128 |
| 768 | **166.048** | 223.216 | 372.992 | 128 |
| 1024 | **280.656** | 310.768 | 436.576 | 128 |
| 1536 | 710.656 | **653.824** | 1091.760 | 256 |
| 2048 | 1449.984 | **1179.760** | 1361.744 | 256 |

Launch tuning did not produce one per-shape winner. The 128-thread block won
five shapes and 256 threads won at 1536 and 2048; 512 threads won none. The
notebook selected 128 by its median per-shape relative-latency rule. This is an
observed T4 policy, not a universal block-size recommendation.

### 9.3 Framework softmax comparison

Each latency cell is `median (IQR)` in microseconds. `A12` is the conditional
probability that an observed custom timing is below an eager timing within the
captured session.

| S | Custom CUDA | PyTorch eager | `torch.compile` | Eager / custom | A12 |
|---:|---:|---:|---:|---:|---:|
| 128 | 38.752 (8.416) | 77.600 (11.672) | 225.664 (30.040) | 2.00x | 0.9724 |
| 255 | 58.544 (2.496) | 81.920 (10.392) | 143.456 (32.792) | 1.40x | 1.0000 |
| 512 | 131.040 (12.888) | 337.920 (4.016) | 197.392 (37.000) | 2.58x | 1.0000 |
| 768 | 239.440 (12.984) | 735.264 (4.984) | 344.352 (22.896) | 3.07x | 1.0000 |
| 1024 | 359.568 (6.304) | 1278.592 (7.256) | 465.424 (29.304) | 3.56x | 1.0000 |
| 1536 | 723.952 (104.120) | 2855.840 (20.072) | 1182.416 (21.016) | 3.94x | 1.0000 |
| 2048 | 1414.480 (34.168) | 5125.488 (116.352) | 1385.760 (279.384) | 3.62x | 1.0000 |
| **Geometric mean** | — | — | — | **2.73x** | — |

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

| S | Custom explicit (µs) | Eager explicit (µs) | SDPA (µs) | Eager / custom | Custom / SDPA | A12 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 129.008 | 198.544 | 84.000 | 1.54x | 1.54x | 0.9931 |
| 255 | 204.288 | 352.128 | 163.600 | 1.72x | 1.25x | 0.9983 |
| 512 | 323.584 | 747.888 | 245.808 | 2.31x | 1.32x | 1.0000 |
| 768 | 587.312 | 1071.088 | 384.864 | 1.82x | 1.53x | 1.0000 |
| 1024 | 1014.016 | 1820.672 | 543.744 | 1.80x | 1.86x | 1.0000 |
| 1536 | 2438.960 | 4115.824 | 1032.400 | 1.69x | 2.36x | 1.0000 |
| 2048 | 4198.768 | 7090.240 | 1660.576 | 1.69x | 2.53x | 1.0000 |
| **Geometric mean** | — | — | — | **1.78x** | **1.71x** | — |

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

The Amdahl diagnostic had mean absolute error 0.419x, root mean square error
0.499x, and mean absolute percentage error 22.7%. It underpredicted measured
speedup at the three shortest shapes and overpredicted at the three longest.
The residual trend reinforces that isolated component timings do not compose
mechanically into an application-level prediction.

## 13. Threats to Validity

### 13.1 Internal validity

Independent PyTorch compositions and structured edge cases reduce shared-bug
risk but do not prove correctness. Historical kernels required a documented,
header-only CUDA 12.8 compatibility change, so their algorithms are matched but
their build inputs are not byte-identical. One managed Colab session cannot
fully control clocks, thermals, allocator state, caching, contention, or
measurement order.

### 13.2 External validity

Only FP32 forward execution, causal masking, one head dimension, one
`batch_heads` setting for softmax, and one Tesla T4 session were measured.
There is no backward kernel, mixed-precision study, dropout, arbitrary mask,
multi-GPU experiment, or independent cross-session replication. Framework
internals and Colab hardware/software can change.

### 13.3 Construct validity

CUDA events measure device work rather than complete request wall-clock time.
The evaluation does not quantify energy, peak memory, compilation cost,
portability, or maintenance effort. Eager, compiled, custom, and SDPA paths
expose different fusion boundaries; each comparison answers a different
engineering question.

### 13.4 Conclusion validity

The 100 observations per configuration are repeated timings inside one session,
not independent deployments. Quartiles and conditional rank statistics are
descriptive, not confidence intervals or population-level significance tests.
Geometric means can conceal shape-specific reversals, so the per-shape tables
remain primary.

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

The complete structured bibliography is stored in
[`references.bib`](references.bib). Principal sources include:

1. Vaswani et al., [“Attention Is All You Need”](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html), 2017.
2. NVIDIA, [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html).
3. Harris, [“Optimizing Parallel Reduction in CUDA”](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf), 2007.
4. NVIDIA, [`cub::BlockReduce` documentation](https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockReduce.html).
5. Milakov and Gimelshein, [“Online Normalizer Calculation for Softmax”](https://arxiv.org/abs/1805.02867), 2018.
6. Blanchard, Higham, and Higham, [“Accurate Computation of the Log-Sum-Exp and Softmax Functions”](https://arxiv.org/abs/1909.03469), 2021.
7. Dao et al., [“FlashAttention”](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html), 2022.
8. Dao, [“FlashAttention-2”](https://openreview.net/forum?id=mZn2Xyh9Ec), 2024.
9. PyTorch, [`scaled_dot_product_attention` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
10. Amdahl, [“Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities”](https://doi.org/10.1145/1465482.1465560), 1967.
11. Mytkowicz et al., [“Producing Wrong Data Without Doing Anything Obviously Wrong!”](https://doi.org/10.1145/1508244.1508275), 2009.
12. NVIDIA, [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## Evidence map

Exact platform-labeled commands are collected in
[`reproducibility.md`](reproducibility.md). The paths below are relative to the
[preserved run](../results/runs/2026-08-23_tesla-t4_ca87722/).

- Raw softmax samples: `artifacts/benchmarks/raw/softmax_raw.csv`
- Historical samples: `artifacts/benchmarks/raw/historical_raw.csv`
- Launch samples: `artifacts/benchmarks/raw/launch_configuration_raw.csv`
- Correctness cases: `artifacts/correctness/correctness_results.csv`
- Attention samples: `artifacts/attention/attention_raw.csv`
- PyTorch profiler events: `artifacts/profiler/pytorch_profiler_events.csv`
- Nsight export: `artifacts/profiler/nsight/ncu_raw_export.csv`
- Environment and run state: `artifacts/environment/environment.json` and
  `artifacts/experiment_manifest.json`
