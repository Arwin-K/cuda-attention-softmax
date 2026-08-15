#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/extension.h>

#include "common.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;

// A CUDA kernel is device code launched across a grid of thread blocks. The
// grid now contains one block for each softmax row, so blockIdx.x identifies
// row ownership and threadIdx.x identifies a worker within that row. This
// transitional commit keeps the mathematics on thread 0; later commits give
// the remaining block threads useful column and reduction work.
__global__ void fused_causal_softmax_kernel(
    const float* scores,
    float* probabilities,
    int64_t rows,
    int64_t sequence_length,
    float scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  // Flattening removes the explicit query dimension, but query positions repeat
  // every sequence_length rows. The modulo restores that position so future
  // keys can be excluded without allocating a separate mask tensor.
  const int64_t query_position = row % sequence_length;
  const int64_t row_offset = row * sequence_length;

  // Threads advance through the row in blockDim.x-sized strides. Every column
  // is owned by exactly one thread, and neighboring threads initially access
  // neighboring global-memory addresses, which is the pattern needed for
  // coalescing when the hardware combines their memory transactions.
  for (int64_t column = threadIdx.x; column <= query_position;
       column += blockDim.x) {
    probabilities[row_offset + column] = scores[row_offset + column] * scale;
  }
  for (int64_t column = query_position + 1 + threadIdx.x;
       column < sequence_length;
       column += blockDim.x) {
    probabilities[row_offset + column] = 0.0f;
  }

  // No thread may consume staged values until every peer has finished writing;
  // this barrier separates global-memory staging from local accumulation.
  __syncthreads();

  // A thread-local variable normally lives in a register, the GPU's fastest
  // per-thread storage. Each thread reduces only its strided subset, producing
  // one partial maximum that is ready for block-wide combination.
  float thread_maximum = -CUDART_INF_F;
  for (int64_t column = threadIdx.x; column <= query_position;
       column += blockDim.x) {
    thread_maximum =
        fmaxf(thread_maximum, probabilities[row_offset + column]);
  }

  // Dynamic shared memory is visible to every thread in this block. A tree
  // reduction halves the number of candidates at every stage until element 0
  // holds the maximum for the full row. Threads without assigned columns begin
  // at negative infinity, the identity value for maximum.
  extern __shared__ float shared_values[];
  shared_values[threadIdx.x] = thread_maximum;

  // Every partial must be visible before any thread reads its partner. Without
  // this barrier, early threads could reduce stale or uninitialized values.
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_values[threadIdx.x] = fmaxf(
          shared_values[threadIdx.x],
          shared_values[threadIdx.x + stride]);
    }

    // The next stage consumes values written by the current stage. A block-wide
    // barrier prevents those reads from racing ahead of their producers.
    __syncthreads();
  }

  // Every thread captures the completed maximum before shared_values is reused
  // for denominator partials in the next commit. This handoff barrier would be
  // unsafe after an early return because all block threads must participate.
  const float row_maximum = shared_values[0];
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }

  // Writing exponentials into the final output storage avoids allocating an
  // intermediate tensor. Maximum subtraction bounds the largest exponential
  // at one while preserving the mathematical softmax result.
  float exponential_sum = 0.0f;
  for (int64_t column = 0; column <= query_position; ++column) {
    const int64_t index = row_offset + column;
    const float shifted_value = probabilities[index] - row_maximum;
    const float exponential = expf(shifted_value);
    probabilities[index] = exponential;
    exponential_sum += exponential;
  }

  // Thread 0 still normalizes every allowed entry during this mapping-only
  // transition. The remaining threads begin cooperating in the next commits.
  for (int64_t column = 0; column <= query_position; ++column) {
    probabilities[row_offset + column] /= exponential_sum;
  }

  // Masked probabilities were already assigned exact zeros by their owners and
  // never entered either reduction.
}

}  // namespace

torch::Tensor fused_causal_softmax_cuda(
    const torch::Tensor& scores,
    double scale) {
  const int64_t rows = scores.size(0);
  const int64_t sequence_length = scores.size(1);
  TORCH_CHECK(
      rows <= static_cast<int64_t>(std::numeric_limits<int>::max()),
      "scores has too many rows for the one-dimensional CUDA grid");

  // A guard makes the input tensor's GPU current for this host thread. This is
  // important in multi-GPU programs: allocation, stream lookup, and launch must
  // all refer to the same device as scores.
  const c10::cuda::CUDAGuard device_guard(scores.device());
  auto probabilities = torch::empty_like(scores);

  const int blocks = static_cast<int>(rows);
  const size_t shared_memory_bytes =
      static_cast<size_t>(kThreadsPerBlock) * sizeof(float);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(scores.get_device());
  fused_causal_softmax_kernel<<<
      blocks,
      kThreadsPerBlock,
      shared_memory_bytes,
      stream>>>(
      scores.data_ptr<float>(),
      probabilities.data_ptr<float>(),
      rows,
      sequence_length,
      static_cast<float>(scale));

  // Kernel launches are asynchronous. Checking cudaGetLastError here catches
  // invalid launch configuration and immediate launch errors at this operator
  // boundary; later correctness tests synchronize when they consume outputs.
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return probabilities;
}
