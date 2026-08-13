# Experiment log

This log records experiments that were actually run. Planned work belongs in
`PROJECT_PLAN.md`; an unavailable GPU or missing result remains explicit rather
than being replaced with an estimate.

Use these labels consistently:

```text
HYPOTHESIS: What we predict before collecting data.
MEASUREMENT: What the command or profiler actually reported.
INTERPRETATION: A possible explanation of the measurement.
NEXT EXPERIMENT: A controlled way to test that explanation.
```

## Experiment entry template

### Experiment: `TODO: short descriptive name`

- **Date/time:** `TODO: ISO 8601 timestamp with timezone`
- **Git commit:** `TODO: full or unambiguous commit hash`
- **Hardware:** `TODO: CPU/GPU model and relevant capability, or unavailable`
- **Software:** `TODO: OS, Python, PyTorch, CUDA, compiler, and driver versions`
- **Research question:** `TODO: the specific question this run addresses`
- **HYPOTHESIS:** `TODO: prediction written before the run`
- **Independent variable:** `TODO: the one factor deliberately changed`
- **Controlled variables:** `TODO: shapes, dtype, inputs, timing, and environment`
- **Metrics:** `TODO: correctness values, latency, throughput, or profiler metrics`
- **Command/script:** `TODO: exact reproducible command`
- **Raw result file:** `TODO: repository-relative artifact path, or none`
- **MEASUREMENT:** `TODO: direct observation from the run`
- **INTERPRETATION:** `TODO: evidence-bounded explanation`
- **Limitations:** `TODO: threats to validity and missing evidence`
- **NEXT EXPERIMENT:** `TODO: follow-up that could confirm or reject the interpretation`
- **Student reflection:** `TODO(student): What surprised you or changed your understanding?`

## Day checkpoint template

Use this template at commits 016, 032, 048, 064, 080, 096, and 112.

### Day `TODO` checkpoint — Commit `TODO`

- **What was implemented:** `TODO: code, tests, tooling, and documentation`
- **What was actually measured:** `TODO: measured evidence, or explicitly none`
- **What I learned:** `TODO(student): Write this in your own words.`
- **What surprised me:** `TODO(student): Write this in your own words.`
- **Unresolved questions:** `TODO: technical and experimental questions`
- **Next day:** `TODO: what the next working day will investigate`

## Current evidence status

No performance or profiling experiment has been recorded yet.
