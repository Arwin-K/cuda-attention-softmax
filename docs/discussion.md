# Discussion

## What the available evidence establishes

The strongest current research result is correctness, not speed. A Tesla T4
compiled and imported the extension after an explicit-header portability fix,
and the tested revision passed both its CUDA pytest suite and all 88 structured
FP32 comparisons without changing the numerical tolerances. This supports the
claim that causal indexing, maximum subtraction, two reductions, normalization,
and masked-zero writes work together for the tested cases.

The failed first build is also informative. Depending on a transitive include
made a CUDA version/toolchain change visible before numerical testing. Explicit
dependencies are part of reproducibility, even when they do not alter the
algorithm.

## Launch configuration

The executed notebook reported a 128-thread selection, which differs from the
source's provisional 256-thread default. That is a useful prompt to inspect the
raw per-shape tradeoffs, but not enough to claim that 128 is universally best.
The missing CSV prevents checking whether the aggregate choice hides lengths
that favor 256 or 512, and the result comes from one T4 environment.

## Kernel speedup versus application speedup

The central application hypothesis remains unresolved. The custom integration
changes only the middle operation:

```text
QK^T -> custom fused causal softmax -> PV
```

Even a large softmax improvement cannot accelerate the unchanged matrix
multiplications. Amdahl's Law predicts that complete-attention speedup is
limited by the fraction of eager attention originally spent in the replaceable
softmax stage. PyTorch SDPA is a stronger production baseline because it may
optimize the complete operation, not merely substitute the middle stage.

This is an interpretation framework, not a reported finding. The raw softmax
and attention CSVs are necessary to calculate both median ratios and the
translation ratio for each sequence length.

## Profiling interpretation

The PyTorch Profiler completion marker supports no claim about which operator
dominated because the trace is unavailable. The Nsight attempt supports only
the fact that package import failed before kernel launch. It would be incorrect
to infer memory bandwidth, occupancy, divergence, or reduction efficiency from
either fact.

The corrected follow-up should first recover named PyTorch regions and then use
Nsight basic metrics to challenge a specific explanation. For example, an
observed short-row penalty could motivate a hypothesis about coordination or
inactive lanes, but hardware metrics and a controlled block-size comparison
would be needed to test it.

## Threats to interpretation

- The retained execution evidence comes from one Tesla T4 and one software
  environment.
- The raw result ZIP is missing, preventing distribution and provenance audits.
- The executed revision predates the latest documentation and harness commits.
- Forward-only FP32 results do not establish behavior for FP16/BF16, backward
  propagation, noncausal masks, or production training workloads.
- Explicit custom attention is not equivalent implementation work to fused
  production SDPA, even though outputs can be mathematically compared.

`TODO(student): After importing and plotting the real data, describe one result
that contradicted or refined your original hypothesis in your own words.`
