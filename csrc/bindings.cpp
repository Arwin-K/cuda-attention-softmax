#include <cmath>

#include <torch/extension.h>

#include "common.cuh"

torch::Tensor fused_causal_softmax(
    const torch::Tensor& scores,
    double scale) {
  // Validate the public contract before indexing sizes or launching device
  // code. TORCH_CHECK turns a violated assumption into a Python exception near
  // the caller instead of a later illegal access or misleading CUDA failure.
  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
  TORCH_CHECK(
      scores.layout() == at::kStrided,
      "scores must use the dense strided layout");
  TORCH_CHECK(
      scores.scalar_type() == at::kFloat,
      "scores must have dtype torch.float32");
  TORCH_CHECK(
      scores.dim() == 2,
      "scores must have shape [rows, sequence_length]");
  TORCH_CHECK(
      scores.size(0) > 0 && scores.size(1) > 0,
      "scores must contain at least one row and column");
  TORCH_CHECK(
      scores.is_contiguous(),
      "scores must be contiguous so row offsets match physical memory");
  TORCH_CHECK(
      std::isfinite(scale) && scale > 0.0,
      "scale must be finite and positive");

  return fused_causal_softmax_cuda(scores, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "fused_causal_softmax",
      &fused_causal_softmax,
      "Fused causal scaled softmax over [rows, sequence_length] CUDA scores",
      pybind11::arg("scores"),
      pybind11::arg("scale"));
}
