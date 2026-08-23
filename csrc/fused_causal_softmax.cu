#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
// CUDART_INF_F is declared here. Including its defining header directly avoids
// depending on CUDA-version-specific transitive includes from cuda_runtime.h.
#include <math_constants.h>
#include <limits>
#include <torch/extension.h>

#include "common.cuh"

namespace {

constexpr int kWarpSize = 32;
// The full mask says all 32 lanes participate in each shuffle. The public
// operator accepts only block sizes divisible by 32, and no thread exits inside
// a valid row's reduction, so even lanes with no column data remain active with
// an identity value. A mask naming inactive lanes would make shuffle behavior
// invalid rather than merely waste work.
constexpr unsigned int kFullWarpMask = 0xffffffffu;

// A warp is the hardware group of 32 threads that executes instructions
// together. The lane identifies a thread inside its warp; the warp ID
// identifies which such group the thread belongs to inside this block.
__device__ __forceinline__ unsigned int lane_id() {
  return threadIdx.x % kWarpSize;
}

__device__ __forceinline__ unsigned int warp_id() {
  return threadIdx.x / kWarpSize;
}

// Shuffle instructions exchange register values directly among lanes in one
// warp. At each offset, lane 0 incorporates another half of the candidates;
// after offsets 16, 8, 4, 2, and 1 it owns the warp maximum. Broadcasting lane
// 0's result gives every lane a usable copy without shared memory.
__device__ __forceinline__ float warp_reduce_max(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(kFullWarpMask, value, offset));
  }
  return __shfl_sync(kFullWarpMask, value, 0);
}

// Addition uses the same register-exchange pattern. Lane 0 receives the sum of
// all 32 lane partials; other lanes need not hold the final value because only
// lane 0 publishes a warp denominator contribution.
__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

// A CUDA kernel is device code launched across a grid of thread blocks. The
// grid now contains one block for each softmax row, so blockIdx.x identifies
// row ownership and threadIdx.x identifies a worker within that row. Threads
// cooperate through strided column work, warp-local register reductions, and
// shared-memory bridges between warps. Unlike the historical one-thread-per-row
// mapping, this kernel does not need a global thread index such as
// blockIdx.x * blockDim.x + threadIdx.x: the block index alone owns the row.
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
  const int64_t allowed_columns = query_position + 1;
  const int64_t row_offset = row * sequence_length;
  const int64_t thread_column = static_cast<int64_t>(threadIdx.x);
  const int64_t column_stride = static_cast<int64_t>(blockDim.x);

  // Threads first fuse scaling into the output buffer, using it as scratch for
  // later exponentials and probabilities. This avoids allocating a separate
  // scaled-score or mask tensor while leaving the input scores unchanged.
  // Threads advance through the row in blockDim.x-sized strides. Every column
  // is owned by exactly one thread, and neighboring threads initially access
  // neighboring global-memory addresses, which is the pattern needed for
  // coalescing when the hardware combines their memory transactions.
  for (int64_t column = thread_column; column < allowed_columns;
       column += column_stride) {
    probabilities[row_offset + column] = scores[row_offset + column] * scale;
  }
  for (int64_t column = allowed_columns + thread_column;
       column < sequence_length;
       column += column_stride) {
    probabilities[row_offset + column] = 0.0f;
  }

  // Synchronization is a block-wide meeting point: no thread may pass it until
  // every peer arrives, and earlier writes become visible to those peers. No
  // thread may consume staged values until every peer has finished writing, so
  // this barrier separates global-memory staging from local accumulation.
  __syncthreads();

  // A thread-local variable normally lives in a register, the GPU's fastest
  // per-thread storage. Each thread reduces only its strided subset, producing
  // one partial maximum that is ready for block-wide combination.
  float thread_maximum = -CUDART_INF_F;
  for (int64_t column = thread_column; column < allowed_columns;
       column += column_stride) {
    thread_maximum =
        fmaxf(thread_maximum, probabilities[row_offset + column]);
  }

  // A reduction combines many partial values into one result using an
  // associative operation such as maximum or addition. The first reduction
  // level happens inside each warp through register
  // shuffles. Only lane 0 publishes, shrinking block communication from one
  // shared value per thread to one value per warp.
  const unsigned int lane = lane_id();
  const unsigned int warp = warp_id();
  const unsigned int warps_per_block = blockDim.x / kWarpSize;
  const float warp_maximum = warp_reduce_max(thread_maximum);
  // This dynamically sized shared-memory array is allocated once per block by
  // the launch configuration. Threads in this block can see it; other blocks
  // cannot. Only warps_per_block floats are needed because lane 0 alone
  // publishes each warp's partial result.
  extern __shared__ float shared_values[];
  if (lane == 0) {
    shared_values[warp] = warp_maximum;
  }
  __syncthreads();

  // The first warp performs the second reduction level. Its first
  // warps_per_block lanes load valid warp maxima; the remaining lanes
  // contribute negative infinity. Calling the helper from every lane in warp 0
  // keeps the full shuffle mask valid even when few lanes carry block data.
  if (warp == 0) {
    float block_maximum =
        lane < warps_per_block ? shared_values[lane] : -CUDART_INF_F;
    block_maximum = warp_reduce_max(block_maximum);
    if (lane == 0) {
      shared_values[0] = block_maximum;
    }
  }
  __syncthreads();

  // Every thread captures the completed maximum before the compact array is
  // reused for denominator partials. The handoff barrier prevents a fast warp
  // from overwriting element 0 before slower peers have loaded it.
  const float row_maximum = shared_values[0];
  __syncthreads();

  // Each thread converts its staged scaled values into stable exponentials and
  // accumulates one register-local denominator contribution. Maximum
  // subtraction gives numerical stability by bounding the largest exponential
  // at one and preventing overflow. It preserves softmax
  // because multiplying every numerator and the denominator by exp(-maximum)
  // cancels in the ratio even for large positive logits.
  float thread_exponential_sum = 0.0f;
  for (int64_t column = thread_column; column < allowed_columns;
       column += column_stride) {
    const int64_t index = row_offset + column;
    const float shifted_value = probabilities[index] - row_maximum;
    const float exponential = expf(shifted_value);
    probabilities[index] = exponential;
    thread_exponential_sum += exponential;
  }

  // Each warp first reduces its 32 register-local sums. Lane 0 publishes one
  // value, so the cross-warp bridge needs only one shared slot per warp rather
  // than one slot per thread.
  const float warp_exponential_sum = warp_reduce_sum(thread_exponential_sum);
  if (lane == 0) {
    shared_values[warp] = warp_exponential_sum;
  }
  __syncthreads();

  // Warp 0 performs the second level exactly as it did for the maximum. Lanes
  // beyond the number of warps contribute zero, the additive identity. Every
  // lane still executes each shuffle because the helper uses a full-warp mask.
  if (warp == 0) {
    float block_exponential_sum =
        lane < warps_per_block ? shared_values[lane] : 0.0f;
    block_exponential_sum = warp_reduce_sum(block_exponential_sum);
    if (lane == 0) {
      shared_values[0] = block_exponential_sum;
    }
  }
  __syncthreads();

  const float exponential_sum = shared_values[0];
  // The same strided ownership used for loads and exponentials now distributes
  // final writeback. Each allowed probability is divided exactly once, while
  // no thread touches the masked zeros staged before the reductions.
  for (int64_t column = thread_column; column < allowed_columns;
       column += column_stride) {
    probabilities[row_offset + column] /= exponential_sum;
  }

  // Masked probabilities were already assigned exact zeros by their owners and
  // never entered either reduction.
}

}  // namespace

torch::Tensor fused_causal_softmax_cuda(
    const torch::Tensor& scores,
    double scale,
    int64_t block_size) {
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
  const int threads = static_cast<int>(block_size);
  const int warps_per_block = threads / kWarpSize;
  // The third launch parameter requests one float of dynamic shared memory per
  // warp. For the measured 128-thread launch that is four floats, or 16 bytes;
  // the array is a communication bridge rather than storage for the whole row.
  const size_t shared_memory_bytes =
      static_cast<size_t>(warps_per_block) * sizeof(float);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(scores.get_device());
  // CUDA's <<<grid, block, dynamic_shared_bytes, stream>>> syntax creates one
  // block per row, the requested workers per block, one compact shared array
  // per block, and enqueues the work on PyTorch's current device stream.
  fused_causal_softmax_kernel<<<
      blocks,
      threads,
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
