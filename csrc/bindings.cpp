#include <torch/extension.h>

#include "common.cuh"

torch::Tensor fused_causal_softmax(
    const torch::Tensor& scores,
    double scale) {
  // Detailed device, dtype, layout, shape, and scale checks are introduced in
  // Commit 031. This commit defines only the stable public function boundary.
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
