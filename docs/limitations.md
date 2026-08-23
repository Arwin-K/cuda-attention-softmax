# Limitations

## Operator and numerical scope

The operator implements contiguous FP32 forward causal scaled softmax. It does
not provide backward/autograd support, dropout, arbitrary or padding masks,
sparse attention, variable valid lengths, or FP16/BF16/FP8 policies. Parallel
reduction order differs from PyTorch; fixed-tolerance success over 88 structured
cases is strong tested evidence, not a proof for every floating-point input.

The flattened causal contract assumes scores derived from
`[batch, heads, sequence, sequence]`, with query position `row % sequence`.
Other layouts require an explicit contract change.

## Experimental scope

All quantitative performance findings come from one Tesla T4 in one Google
Colab session. The experiment covers seven sequence lengths, FP32,
`batch_heads=8` for isolated softmax, and batch 1, eight heads, head dimension
64 for attention. It does not establish run-to-run variability or transfer to
Ampere, Ada, Hopper, or other architectures.

Colab is managed infrastructure. Clock state, background activity, driver
policy, and future software images are not fully controlled. The raw quartiles
describe within-session CUDA-event samples; they are not confidence intervals
over machines or sessions.

Historical kernels required an explicitly recorded header include so their old
source would compile with CUDA 12.8. The change was compatibility-only, but it
means the rebuild is not byte-identical to the historical commit. Each adapted
source hash and patch description is retained in metadata.

## Baseline scope

Eager, compiled, and custom isolated-softmax paths perform equivalent scaling,
causal masking, and softmax work. The complete explicit custom path is not
implementation-equivalent to production SDPA: SDPA can select a backend fused
across a larger portion of attention. Output comparison is mathematically
valid; performance interpretation must name the different fusion boundary.

The explicit paths materialize quadratic score and probability tensors. The
study does not implement FlashAttention-style tiling that avoids those
intermediates. `torch.compile` startup is excluded from steady-state timing and
must be evaluated separately for short-lived workloads.

## Profiling scope

PyTorch Profiler and Nsight each captured one configured length rather than the
full shape registry. Four Nsight launches, including three warmups, are too few
to characterize run-to-run latency. Peak DRAM and SM throughput percentages do
not uniquely identify a bottleneck; the project lacks matched hardware metrics
for row-serial and shared-tree versions.

## Future work

1. Repeat the complete run in multiple independent sessions.
2. Repeat on at least one Ampere-or-newer GPU with the same raw schema.
3. Profile shared-tree and warp kernels under matched conditions.
4. Study shape-aware launch dispatch rather than one global default.
5. Add FP16/BF16 accumulation analysis and fixed error criteria.
6. Add backward/autograd support as a separate correctness project.
7. Vary batch, head count, head dimension, and masking forms.
8. Compare with specialized library softmax and full-attention kernels under
   explicitly equivalent boundaries.

## Evidence-bounded conclusion

The measured case supports the value of block-per-row work and compact warp
reductions, demonstrates a shape-dependent launch tradeoff, and shows partial
translation from kernel speedup to explicit attention. It does not support a
claim that the custom operator is universally faster, that 128 threads is
universally optimal, or that the kernel has one proven hardware bottleneck.
Production SDPA's consistent advantage is a central result, not an exception to
hide.
