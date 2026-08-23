# Kernel evolution through Git milestones

The project has one primary CUDA implementation:
`csrc/fused_causal_softmax.cu`. Earlier algorithms are commits, not parallel
`v1`, `v2`, or renamed source files. This keeps the current tree uncluttered and
makes the transition itself reviewable with `git diff`.

## Stage 0 — Extension boundary

| Planned commit | Git hash | What entered the source |
|---:|---|---|
| 023 | `c96a49b9` | Kernel/launcher skeleton and one-grid launch interface |
| 027 | `e309f1f9` | One global CUDA thread owns one complete row |
| 028 | `e31e6387` | Serial stable maximum scan |
| 029 | `d29490c4` | In-kernel scaling and causal predicate |
| 030 | `27eb32b2` | Serial exponential sum and normalization |
| 031 | `a59d5f12` | Launch/input validation and CUDA error checks |

The global thread index selected a row. For that row, one thread performed all
three scans. This was intentionally simple and correct before it was fast.

The measured historical row-serial endpoint is commit `8f07d762`, after
benchmark metadata support was present. Its artifact description also records
a header-only CUDA 12.8 compatibility include used during the Day 7 rebuild.

## Stage 1 — Cooperative block-per-row work

| Planned commit | Git hash | What changed |
|---:|---|---|
| 042 | `27586189` | Grid mapping becomes one block per softmax row |
| 043 | `1a4ecbd4` | Threads traverse columns separated by `blockDim.x` |
| 044 | `77cd283e` | Each thread accumulates a register-local maximum |
| 045 | `7c7991c8` | Shared-memory maximum reduction tree |
| 046 | `31024173` | Synchronization protects reduction dependencies |
| 047 | `49d783fb` | Per-thread exponential partial sums |
| 048 | `02a4c519` | Shared-memory sum reduction tree |
| 049 | `b64151f6` | Threads normalize output columns in parallel |
| 058 | `0efe0f45` | Arbitrary-length boundary handling tightened |

This stage changes work decomposition. Neighboring threads initially read
neighboring columns, then each thread advances by the block width. Local
partials reduce the number of shared values, but every tree level still needs
shared-memory communication and synchronization.

The measured block/shared-tree endpoint is checkpoint commit `f9420de0`. It is
the fairest historical baseline for asking what changes when the reduction
mechanism improves but block-per-row decomposition remains.

## Stage 2 — Warp-local communication

| Planned commit | Git hash | What changed |
|---:|---|---|
| 061 | `4b5880a9` | Explicit warp and lane helpers |
| 062 | `d763b22d` | Maximum reduction uses warp shuffles |
| 063 | `f77ddcc6` | One maximum per warp enters compact shared memory |
| 064 | `4b2bfd49` | Sum reduction uses warp shuffles |
| 065 | `8b10a574` | One sum per warp enters compact shared memory |
| 066 | `885c07b0` | Full block trees are replaced by two-level reductions |
| 069 | `fd36682e` | Scaling and causal masking verified to remain fused |

Within each warp, shuffle-down operations exchange register values without a
shared write/barrier/read at every step. Warp leaders write one partial apiece;
after a block barrier, the first warp finishes the inter-warp reduction. At 128
threads this needs four floats, matching the 16-byte dynamic shared allocation
reported by Nsight.

The measured warp historical endpoint is `a3736910`. Across the supplied T4
matrix it was 3.22--8.92x faster than the row-serial endpoint and 1.07--1.91x
faster than the block/shared-tree endpoint.

## Stage 3 — Launch and integration boundary

| Milestone | Git hash | Role |
|---|---|---|
| Configurable 128/256/512 blocks | `9ac83c64` | One source accepts controlled launch choices |
| CUDA 12.8 explicit constant header | `9841d5fd` | Portability fix after the first T4 build failed |
| Measured clean workflow | `ca87722a` | Source and Day 6 harness revision recorded by the run |

The current source retains its documented default rather than silently changing
algorithm policy from one T4 study. The Colab experiment explicitly launched
128 threads after its aggregate tuning rule; 256 threads still won the two
longest individual shapes.

## Reconstructing any transition

Inspect a historical source without creating a duplicate:

```bash
git show e309f1f9:csrc/fused_causal_softmax.cu
```

Compare work decomposition:

```bash
git diff e309f1f9 27586189 -- csrc/fused_causal_softmax.cu
```

Compare block trees with the final warp transition:

```bash
git diff f9420de0 a3736910 -- csrc/fused_causal_softmax.cu csrc/common.cuh
```

Verify the milestone messages, object availability, and one-source invariant:

```bash
python3 scripts/audit_kernel_history.py
```

## What the history establishes

Git shows what source changed and the imported benchmark shows what was
measured. Neither alone proves why performance changed. The closest causal
comparison is the block/shared-tree endpoint versus the warp endpoint because
both assign one block to a row; matched profiler data would still strengthen
the explanation about communication and synchronization.
