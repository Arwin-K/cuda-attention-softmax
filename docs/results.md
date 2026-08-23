# Results

## Evidence boundary

Results come from the imported run at
`results/runs/2026-08-23_tesla-t4_ca87722/`. Its manifest records a clean
`ca87722a00ebf585cd788c67949e7c0b32dca788` checkout, one Tesla T4, and one
Colab session. Raw per-iteration CSVs, summaries, environment metadata,
profiling exports, generated figures, and the exact executed notebook are all
preserved. These results do not apply automatically to later commits.

## Correctness

The device pytest stage reported 70 passes. The structured softmax matrix added
88 comparisons spanning required odd and non-power-of-two lengths and eight
input families. All 88 passed with fixed FP32 tolerances `rtol=1e-5` and
`atol=1e-6`. Across the raw correctness rows:

- maximum absolute error: `3.57627868652e-7`;
- maximum relative error: `4.69734317221e-7`;
- maximum probability-row sum error: `3.57627868652e-7`;
- all future masked probabilities exactly zero; and
- no unexpected NaNs or infinities.

All seven complete-attention comparisons also passed. Maximum absolute output
error versus explicit eager attention was `2.98023223877e-7`; SDPA versus the
same explicit reference reached `7.15255737305e-7`.

## Historical kernel evolution

The experiment rebuilt row-serial (`8f07d762`), block/shared-tree (`f9420de0`),
and warp-reduction (`a3736910`) commits. A header-only CUDA 12.8 compatibility
include was recorded for each historical build; algorithms were unchanged.

| Sequence length | Row serial / warp | Shared tree / warp |
|---:|---:|---:|
| 128 | 3.22x | 1.14x |
| 255 | 5.84x | 1.26x |
| 512 | 7.57x | 1.67x |
| 768 | 8.92x | 1.34x |
| 1024 | 6.52x | 1.91x |
| 1536 | 6.87x | 1.10x |
| 2048 | 7.23x | 1.07x |

The warp kernel was faster in every matched historical comparison. The larger
row-serial gap combines work-decomposition and reduction changes; the
shared-tree comparison more closely isolates reduction communication.

## Launch configuration

The 128-thread block had the lowest median latency at lengths 128, 255, 512,
768, and 1024. The 256-thread block won at 1536 and 2048; 512 threads won no
shape. The selection rule compared each configuration's latency relative to the
best at each shape, then took the median across shapes. It selected 128 threads
with a relative score of 1.000, versus 1.107 for 256 and 2.100 for 512.

## Framework softmax comparison

| S | Custom (us) | Eager (us) | `torch.compile` (us) | Custom vs eager |
|---:|---:|---:|---:|---:|
| 128 | 38.752 | 77.600 | 225.664 | 2.00x |
| 255 | 58.544 | 81.920 | 143.456 | 1.40x |
| 512 | 131.040 | 337.920 | 197.392 | 2.58x |
| 768 | 239.440 | 735.264 | 344.352 | 3.07x |
| 1024 | 359.568 | 1278.592 | 465.424 | 3.56x |
| 1536 | 723.952 | 2855.840 | 1182.416 | 3.94x |
| 2048 | 1414.480 | 5125.488 | 1385.760 | 3.62x |

The custom path beat eager at all seven shapes and steady-state
`torch.compile` at six. At 2048, compiled PyTorch was approximately 2.1% faster
than custom. Compilation startup was excluded from these steady-state values
and recorded separately as one compile warmup.

## Complete attention

| S | Custom explicit (us) | Eager explicit (us) | SDPA (us) | Custom vs eager |
|---:|---:|---:|---:|---:|
| 128 | 129.008 | 198.544 | 84.000 | 1.54x |
| 255 | 204.288 | 352.128 | 163.600 | 1.72x |
| 512 | 323.584 | 747.888 | 245.808 | 2.31x |
| 768 | 587.312 | 1071.088 | 384.864 | 1.82x |
| 1024 | 1014.016 | 1820.672 | 543.744 | 1.80x |
| 1536 | 2438.960 | 4115.824 | 1032.400 | 1.69x |
| 2048 | 4198.768 | 7090.240 | 1660.576 | 1.69x |

Kernel speedup exceeded explicit-attention speedup at six of seven matched
lengths. Production SDPA was faster than custom explicit attention everywhere;
custom/SDPA latency ranged from 1.25x to 2.53x.

## Profiling

### MEASURED

At length 512, the PyTorch Profiler capture reported 110.207 microseconds for
the fused custom kernel. Named device totals were 505.565 microseconds for
custom explicit attention, 1095.195 for explicit eager attention, and 408.222
for SDPA.

Nsight Compute 2025.1.1 successfully captured four target launches. The kernel
used a 4096-block grid, 128 threads per block, 24 registers per thread, zero
static and 16 bytes dynamic shared memory. Its reported durations were
106.976--107.584 microseconds, with 48.33--48.79% peak DRAM throughput and
56.72--57.00% peak SM throughput.

### INTERPRETATION

The compact dynamic shared-memory value agrees with storing one result per warp
for a four-warp block. The throughput percentages do not alone establish a
single memory or compute bottleneck. Profiler captures use different conditions
from the repeated benchmark and are not substituted for benchmark medians.

### NEXT EXPERIMENT

Profile matched shared-tree and warp kernels, then repeat timing and profiling
on another NVIDIA architecture and across independent sessions.
