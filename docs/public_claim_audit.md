# Public-claim audit

Commit 108 generated
`results/runs/2026-08-23_tesla-t4_ca87722/public_claim_audit.json` directly
from correctness, historical, framework, launch, attention, Amdahl, PyTorch
Profiler, and Nsight artifacts.

The audit confirms that README, blog, paper, and website journey use the derived
display ranges for:

- row-serial to warp speedup: 3.22--8.92x;
- shared-tree to warp speedup: 1.07--1.91x;
- custom to eager softmax speedup: 1.40--3.94x; and
- custom to eager explicit-attention speedup: 1.54--2.31x.

It also derives, rather than manually enters, 88 correctness passes, six of
seven custom wins over steady-state `torch.compile`, seven of seven SDPA wins
over custom explicit attention, six of seven cases where kernel speedup exceeds
attention speedup, the launch selector, and the profiler values.

The check is intentionally narrow: phrase presence proves that selected public
numbers agree with the current artifact registry. It does not determine whether
every qualitative interpretation is scientifically justified. The final scope
audit must still review baselines, single-session limits, causal language, and
unmeasured future work.

Run:

```bash
python3 scripts/audit_public_claims.py \
  results/runs/2026-08-23_tesla-t4_ca87722/artifacts \
  --output results/runs/2026-08-23_tesla-t4_ca87722/public_claim_audit.json
```
