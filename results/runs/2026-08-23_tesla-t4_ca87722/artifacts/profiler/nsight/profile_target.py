import math
import sys
import torch
sys.path.insert(0, '/content/cuda-attention-softmax')
from cuda_attention.operator import fused_causal_softmax
scores = torch.randn(4096, 512, device="cuda", dtype=torch.float32)
scale = 1.0 / math.sqrt(64)
for _ in range(3):
    fused_causal_softmax(scores, scale, block_size=128)
torch.cuda.synchronize()
fused_causal_softmax(scores, scale, block_size=128)
torch.cuda.synchronize()
