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

// A CUDA kernel is device code launched across a grid of thread blocks. This
// first mapping assigns one entire softmax row to one global CUDA thread. It is
// intentionally simple: no threads cooperate within a row yet, which makes the
// correctness path easy to trace but leaves the row's column work serial.
__global__ void fused_causal_softmax_kernel(
    const float* scores,
    float* probabilities,
    int64_t rows,
    int64_t sequence_length,
    float scale) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  // Flattening removes the explicit query dimension, but query positions repeat
  // every sequence_length rows. The modulo restores that position so future
  // keys can be excluded without allocating a separate mask tensor.
  const int64_t query_position = row % sequence_length;
  const int64_t row_offset = row * sequence_length;

  // Scaling happens before both reductions, matching scaled dot-product
  // attention. Only causally allowed columns may influence the row maximum.
  float row_maximum = -CUDART_INF_F;
  for (int64_t column = 0; column <= query_position; ++column) {
    const float scaled_value = scores[row_offset + column] * scale;
    row_maximum = fmaxf(row_maximum, scaled_value);
  }

  // Writing exponentials into the final output storage avoids allocating an
  // intermediate tensor. Maximum subtraction bounds the largest exponential
  // at one while preserving the mathematical softmax result.
  float exponential_sum = 0.0f;
  for (int64_t column = 0; column <= query_position; ++column) {
    const int64_t index = row_offset + column;
    const float shifted_value = scores[index] * scale - row_maximum;
    const float exponential = expf(shifted_value);
    probabilities[index] = exponential;
    exponential_sum += exponential;
  }

  // The same owning thread normalizes every allowed entry. No block-level
  // synchronization is needed because no other thread reads or writes this row.
  for (int64_t column = 0; column <= query_position; ++column) {
    probabilities[row_offset + column] /= exponential_sum;
  }

  // Masked probabilities are exactly zero and never enter either reduction.
  for (int64_t column = query_position + 1; column < sequence_length; ++column) {
    probabilities[row_offset + column] = 0.0f;
  }
}

}  // namespace

torch::Tensor fused_causal_softmax_cuda(
    const torch::Tensor& scores,
    double scale) {
  const int64_t rows = scores.size(0);
  const int64_t sequence_length = scores.size(1);
  TORCH_CHECK(
      rows <= static_cast<int64_t>(std::numeric_limits<int>::max()) *
          kThreadsPerBlock,
      "scores has too many rows for the one-dimensional CUDA grid");

  // A guard makes the input tensor's GPU current for this host thread. This is
  // important in multi-GPU programs: allocation, stream lookup, and launch must
  // all refer to the same device as scores.
  const c10::cuda::CUDAGuard device_guard(scores.device());
  auto probabilities = torch::empty_like(scores);

  const int blocks = static_cast<int>(
      (rows + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(scores.get_device());
  fused_causal_softmax_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
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
