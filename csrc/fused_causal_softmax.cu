#include <cuda.h>
#include <cuda_runtime.h>
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

  // Masked probabilities are exactly zero. Allowed probabilities are written
  // after the exponential denominator is added in Commit 030.
  for (int64_t column = query_position + 1; column < sequence_length; ++column) {
    probabilities[row_offset + column] = 0.0f;
  }

  (void)row_maximum;
}

}  // namespace

torch::Tensor fused_causal_softmax_cuda(
    const torch::Tensor& scores,
    double scale) {
  // This host function will allocate output, select grid/block dimensions, and
  // launch the device kernel. Failing explicitly is safer than returning an
  // uninitialized tensor while the implementation is structurally incomplete.
  TORCH_CHECK(
      false,
      "row-serial CUDA math is incomplete until Commit 030");
  return torch::Tensor();
}
