# Results

No benchmark results have been measured yet.

The first row-serial baseline was attempted on 2026-08-15, but the Apple
Silicon host had no CUDA device or CUDA-enabled PyTorch runtime. The benchmark
exited before timing and created no CSV. This is an unavailable experiment, not
a zero-latency result. The intended NVIDIA command is documented in the
experiment log and must be run before any baseline or speedup claim.

The Day 4 block comparison and first figure-generation attempt also stopped
without artifacts: neither the historical row-serial CSV nor the current block-
parallel CSV exists. The generator intentionally refuses to create latency or
throughput figures from missing or empty summaries.

## Block-reduction evidence status

| Claim | Current status | Required evidence |
|---|---|---|
| One block owns each row | Established by source inspection | CUDA build still required |
| Threads cover columns in strides | Established by source inspection | GPU correctness suite |
| Max and sum use shared trees | Established by source inspection | GPU correctness suite |
| Accesses are coalesced efficiently | Hypothesis from address mapping | Nsight memory metrics |
| Block design is faster than row serial | Unmeasured | Matched commit-tagged CSVs |
| Speedup varies with sequence length | Untested hypothesis | Full benchmark registry |
