#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "common.cuh"

namespace {

// A CUDA kernel is device code launched across a grid of thread blocks. The
// mapping from block/thread coordinates to softmax rows is deliberately left
// for Commit 027 so this commit establishes only the compilation boundary.
__global__ void fused_causal_softmax_kernel(
    const float* scores,
    float* probabilities,
    int64_t rows,
    int64_t sequence_length,
    float scale) {
  // TODO(Commit 027): assign one complete row to each global CUDA thread.
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
      "fused_causal_softmax CUDA launcher is not implemented until Commit 027");
}
