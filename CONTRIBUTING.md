# Contributing

Thank you for improving this research artifact. Contributions should keep the
CUDA implementation inspectable, the numerical contract explicit, and every
performance claim traceable to preserved evidence.

## Development workflow

1. Create a focused branch from `main`.
2. Install the development environment with `python -m pip install -e ".[dev]"`.
3. Run `./scripts/run_tests.sh` for code changes.
4. Run `./scripts/verify_cpu_reproducibility.sh` for experiment, notebook, or
   artifact-schema changes.
5. Run `./scripts/build_paper.sh` for changes to `docs/paper.tex` or its
   bibliography. The generated `docs/paper.pdf` is local and must not be
   committed.

## Research-evidence requirements

- Do not combine benchmark rows from different commits, GPUs, drivers, CUDA
  toolkits, or framework versions into one matched comparison.
- Preserve raw observations, environment metadata, source revision, warmup and
  iteration counts, and the exact command used to collect new measurements.
- Label interpretation separately from measured output. Avoid causal or
  cross-device claims that the experiment does not support.
- Add new experiment output under a new provenance-bearing directory in
  `results/runs/`; do not overwrite the preserved Tesla T4 run.
- Keep correctness criteria fixed before interpreting performance.

## Pull requests

Describe the problem, the implemented change, the commands used for
verification, and any new evidence or limitations. Keep unrelated refactors out
of research-result changes so reviewers can audit the causal link between code,
measurements, tables, and prose.
