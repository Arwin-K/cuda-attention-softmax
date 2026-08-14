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

  // Commit 027 establishes ownership only. Maximum, masking/scaling, and
  // normalization arrive in Commits 028-030 before the launcher is enabled.
  (void)scores;
  (void)probabilities;
  (void)sequence_length;
  (void)scale;
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
