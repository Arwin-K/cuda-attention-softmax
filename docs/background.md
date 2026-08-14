# Background

This chapter introduces the ideas needed to understand the implementation as it
evolves. It describes general behavior and planned designs, not measurements
from this project.

## Transformer attention

Attention lets each query vector select information from key/value vectors. For
one attention head:

```text
scores        = Q @ K^T
scaled_scores = scores / sqrt(head_dimension)
probabilities = causal_softmax(scaled_scores)
output        = probabilities @ V
```

If `Q`, `K`, and `V` have shape `[sequence_length, head_dimension]`, then
`Q @ K^T` has shape `[sequence_length, sequence_length]`. Row `i` contains the
scores produced by query `i`; column `j` corresponds to key `j`.

Dividing by `sqrt(head_dimension)` keeps dot-product magnitudes from growing
with vector width. Without scaling, large logits can push softmax toward very
sharp probabilities and small gradients.

### Causal masking

Autoregressive attention must not use future tokens. Query row `i` may attend
only to columns `j <= i`. Conceptually, masked scores receive negative infinity
before softmax, which makes their exponential and final probability zero.

The future custom operator receives flattened rows. If the original score
tensor is `[batch, heads, sequence_length, sequence_length]`, then:

```text
query_position = row_index % sequence_length
```

The modulo recovers which query row the flattened row represents.

## Softmax and numerical stability

For row values `x`, softmax is:

```text
softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
```

Directly evaluating `exp(x_i)` is unsafe when a logit is large because the
exponential can overflow. Stable softmax subtracts the row maximum `m`:

```text
m             = max_j(x_j)
unnormalized_i = exp(x_i - m)
softmax(x_i)  = unnormalized_i / sum_j(unnormalized_j)
```

This does not change the mathematical result because the same factor
`exp(-m)` appears in every numerator and in the denominator, where it cancels.
It does change the numerical range: the largest shifted logit is zero, so the
largest exponential is exactly one rather than a potentially enormous value.

Softmax requires two row-wide reductions—maximum and sum—followed by a
normalization pass. Those reductions are the main cooperation problem explored
by this project.

## CUDA host and device roles

The **host** is the CPU program that validates tensors, selects launch
parameters, and requests GPU work. The **device** is the NVIDIA GPU that runs
the kernel. A CUDA **kernel** is a device function launched across many GPU
threads.

The launch describes a hierarchy:

- a **grid** contains all blocks launched for the kernel;
- a **block** is a group of threads that can synchronize and share on-chip
  shared memory;
- a **thread** executes one logical instance of the kernel program.

CUDA exposes coordinates such as `blockIdx`, `threadIdx`, and `blockDim` so a
thread can determine which data it owns. A common one-dimensional global thread
index is:

```text
global_thread_index = blockIdx.x * blockDim.x + threadIdx.x
```

The initial kernel will use one global thread per row because that mapping is
easy to reason about. A later state will use one block per row so threads can
cooperate on the maximum, exponential sum, and normalization.

## GPU memory relevant to the kernel

### Global memory

Input scores and output probabilities live in device global memory. It has
large capacity but relatively high access cost. Neighboring threads should,
where practical, access neighboring addresses so hardware can combine their
requests into coalesced transactions.

### Registers and thread-local partials

A thread's running maximum or sum normally lives in a register. Registers are
fast and private to that thread, so a local partial must be communicated before
it can become a row-wide result.

### Shared memory

Shared memory is on-chip storage visible to every thread in a block. It can
hold thread or warp partial results, but access ordering matters. When one set
of threads writes values that another set reads, synchronization is needed to
avoid races.

## Synchronization and reductions

A **reduction** combines many values into one value, such as a maximum or sum.
For a block-parallel softmax, each thread first reduces the columns assigned to
it into a local partial. The block then combines those partials into the row
result.

`__syncthreads()` is a block-wide barrier. Threads in the block must reach it
before any proceed, and earlier shared-memory writes become visible afterward.
It is necessary between dependent shared-memory reduction stages; placing it
incorrectly can create races or deadlock.

## Warps and shuffle communication

Threads execute in hardware groups called **warps**. NVIDIA warps contain 32
lanes. A lane is a thread's position within its warp, and a warp ID identifies
which warp in the block contains that thread.

Warp shuffle instructions allow participating lanes to exchange register
values without first storing every value in shared memory. A tree-shaped
shuffle reduction can combine 32 partials in a small number of steps. A block
larger than one warp still needs a second level: each warp produces one result,
those results are stored in a compact shared array, and a final warp combines
them.

Warp execution does not remove every synchronization concern. The code must
still define which lanes participate, handle partial work safely, and
synchronize communication between different warps.

## Kernel fusion boundary

The planned CUDA kernel fuses attention scaling, causal masking, maximum
reduction, exponential sum, and probability normalization. Fusion avoids
materializing separate scaled-score and mask tensors. The surrounding `QK^T`
and probability-times-`V` matrix multiplications remain PyTorch operations and
must be included when measuring complete attention.

## Current evidence boundary

No CUDA implementation or GPU measurement exists after Commit 004. Statements
about coalescing, shared memory, reductions, and warp shuffles above are design
principles to test later, not explanations of observed project performance.

## Python-to-CUDA execution path

The custom operator crosses several boundaries. Each layer has a different
responsibility:

```text
Python caller
  -> cuda_attention.operator.fused_causal_softmax
  -> import optional cuda_attention._C extension
  -> pybind11 function in csrc/bindings.cpp
  -> host launcher in csrc/fused_causal_softmax.cu
  -> CUDA kernel launch
  -> GPU threads read scores and write probabilities
  -> PyTorch output tensor returns through C++ and Python
```

### Build time

`setup.py` describes a native module named `cuda_attention._C`. When
`CUDA_ATTENTION_BUILD_CUDA=1`, PyTorch's extension machinery compiles
`bindings.cpp` with a host C++ compiler, compiles the `.cu` translation unit
with `nvcc`, and links both objects into one Python-loadable library. Headers
such as `common.cuh` keep declarations consistent across translation units.

Compilation happens on the host and produces code that can later request GPU
work. It is not kernel execution, and a successful compile alone does not prove
correct results.

### Runtime

1. Python calls the guarded operator with a PyTorch tensor and scale.
2. The guard imports `_C`; an absent binary raises a specific optional-
   capability error rather than triggering a build.
3. Pybind11 converts Python arguments to the C++ function signature.
4. The binding calls `fused_causal_softmax_cuda` on the host.
5. The launcher validates inputs, allocates output, chooses grid/block sizes,
   and launches the `__global__` kernel.
6. CUDA schedules blocks and threads on the GPU. Device work is asynchronous
   with respect to the CPU unless an operation requires synchronization.
7. Launch/runtime errors must be surfaced before results are trusted.
8. The output remains a PyTorch tensor whose storage is owned and tracked by
   PyTorch.

### Host versus device code

The Python function, pybind11 binding, and launcher execute on the CPU. The
function marked `__global__` executes on the NVIDIA device. `blockIdx`,
`threadIdx`, and `blockDim` have meaning only in device code; Python does not
directly schedule individual GPU threads.

### Current implementation boundary

After Commit 026, the optional build path, binding signature, CUDA translation
unit, runtime guard, and handoff scripts exist. The launcher still fails
explicitly and the device skeleton performs no math. CUDA compilation and
execution have not been tested on this Apple Silicon host.
