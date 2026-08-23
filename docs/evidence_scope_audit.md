# Evidence and scope audit — Commit 111

The machine-readable report is
`results/runs/2026-08-23_tesla-t4_ca87722/evidence_scope_audit.json`.
It audits the Commit 110 parent before this report is committed, so the recorded
repository head is deliberately `19b2d9d`; the measured GPU revision remains
`ca87722a`.

The combined gate checks:

- raw schemas, sample counts, current summaries, and profiler fields;
- figure/source hashes and generated public claims;
- one tracked `.cu` file and no forbidden version-like filenames;
- the exact first 14 Day 7 commit messages;
- website entries 001--112 with accurate completed/planned status;
- an output-free reusable notebook and separately preserved executed notebook;
- the supplied executed-notebook SHA-256;
- absence of stale “raw ZIP missing” claims in public documents;
- separation of the measured commit from later audit/documentation revisions;
- explicit one-T4, forward-FP32, backward/mixed-precision, and SDPA limits; and
- separation of the Amdahl model from measured attention results.

Passing this gate means the checked-in evidence is internally consistent and
the publication scope is explicit. It does not create cross-architecture
replication, backward support, mixed-precision evidence, or a new GPU run.

Run before the checkpoint:

```bash
python3 scripts/audit_evidence_scope.py \
  --output results/runs/2026-08-23_tesla-t4_ca87722/evidence_scope_audit.json
```
