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

## CUDA implementation evidence status

| Claim | Current status | Required evidence |
|---|---|---|
| One block owns each row | Established by source inspection | CUDA build still required |
| Threads cover columns in strides | Established by source inspection | GPU correctness suite |
| Max and sum use two-level warp reductions | Established by source inspection | GPU correctness suite |
| Cross-warp communication uses compact shared storage | Established by source inspection | CUDA build and profiler evidence |
| Accesses are coalesced efficiently | Hypothesis from address mapping | Nsight memory metrics |
| Block design is faster than row serial | Unmeasured | Matched commit-tagged CSVs |
| Speedup varies with sequence length | Untested hypothesis | Full benchmark registry |
## Launch-configuration selection status

The kernel accepts 128, 256, and 512 threads per block, but no NVIDIA launch
CSV exists for any configuration. Therefore, **no launch configuration has been
selected from measurements**. The code retains 256 only as the pre-tuning,
provisional default.

Once all three artifacts exist from one Git revision and one GPU environment,
`benchmarks/benchmark_launch_configs.py --select-from ...` assigns equal
importance to each required sequence length by dividing each latency by the
best latency at that same length. It selects the lowest median relative latency
and reports per-shape wins. This policy is implemented and tested on explicitly
synthetic unit-test fixtures; those fixtures are not project results.

Expected raw artifacts:

| Artifact | Status | Intended comparison |
|---|---|---|
| `results/raw/launch_128.csv` | Missing | 128 threads across all required shapes |
| `results/raw/launch_256.csv` | Missing | 256 threads under identical controls |
| `results/raw/launch_512.csv` | Missing | 512 threads under identical controls |
| `results/summary/launch_selection.json` | Missing by design | Data-driven default and shape tradeoffs |

## Framework-comparison status

The eager, `torch.compile`, and custom paths implement the same scale, causal
mask, and softmax workload. Compilation receives one explicitly untimed startup
call before steady-state warmups and samples. The custom path records its block
size; framework baselines leave that field empty because it is not their launch
control.

`results/raw/framework_comparison.csv` and its summary are absent. Consequently,
there is no evidence that the custom kernel beats either framework path, no
framework speedup, and no supported crossover claim. The comparison preflight
is tested with synthetic fixtures and will reject a raw artifact unless every
workload contains all three implementations from one Git/GPU environment.

| Question | Evidence now | Evidence still needed |
|---|---|---|
| Does compiled output match the expression? | Small CPU semantic test | CUDA correctness precheck per benchmark shape |
| Which launch size is best? | No measurement | Three complete launch CSVs from one session |
| Is custom faster than eager? | No measurement | Complete three-way raw/summary CSV |
| Is custom faster than `torch.compile`? | No measurement | Same complete three-way artifact |
| Does the ranking change with sequence length? | Untested hypothesis | Per-shape median and quartile comparison |
