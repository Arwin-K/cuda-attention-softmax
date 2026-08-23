# Publication checkpoint — Commit 100

## Scope reviewed

The README, blog, paper, results, discussion, limitations, interview notes, and
imported T4 evidence were checked together before the final reproducibility
work. This checkpoint changes documentation only.

## Claim traceability

| Public claim | Direct evidence |
|---|---|
| 88/88 structured cases passed | `correctness/correctness_results.csv` and `correctness_summary.json` |
| Historical kernel speedups | `benchmarks/raw/historical_raw.csv` and `historical_summary.csv` |
| Custom/eager/compile results | `benchmarks/raw/softmax_raw.csv` and `softmax_summary.csv` |
| 128-thread aggregate selection | `launch_configuration_raw.csv` and `launch_selection.json` |
| Explicit attention and SDPA results | `attention/attention_raw.csv` and `attention_summary.csv` |
| Kernel-versus-attention translation | `attention/amdahl_analysis.csv` plus matched summaries |
| PyTorch profile values | `profiler/pytorch_profiler_events.csv` |
| Nsight resource and throughput values | `profiler/nsight/ncu_raw_export.csv` and `.ncu-rep` |
| GPU/software/run revision | `environment/environment.json` and `experiment_manifest.json` |

Every path above is relative to
`results/runs/2026-08-23_tesla-t4_ca87722/artifacts/`.

## Coherence decisions

- Use `ca87722a` as the measured revision; later documentation commits do not
  become retroactive benchmark revisions.
- Treat 128 threads as the experiment selector's T4 choice, not a universal
  source default.
- Keep the Amdahl table as model output distinct from observed attention data.
- Keep profiler captures separate from 100-sample benchmark distributions.
- Report that custom explicit attention lost to SDPA at every measured shape.
- Preserve `TODO(student)` prompts instead of inventing personal reflection or
  affiliation details.

## Remaining Day 7 audit work

Commits 101--112 must automate schema checks, add file-level provenance,
exercise CPU-only portability, document the NVIDIA handoff and Git evolution,
prepare website material, audit public claims, publish exact reproducibility
commands, review CUDA comments without changing behavior, and produce final
evidence and seven-day checkpoints.
