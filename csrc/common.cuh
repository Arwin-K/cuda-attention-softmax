#pragma once

#include <torch/extension.h>

// Host-callable interface implemented by fused_causal_softmax.cu and consumed
// by bindings.cpp. A shared declaration keeps both translation units aligned.
torch::Tensor fused_causal_softmax_cuda(
    const torch::Tensor& scores,
    double scale);
