# Imported experiment source

This directory preserves the complete experiment handoff supplied for Day 7.
It is evidence from an NVIDIA run, not output produced on the local Mac.

- Experiment archive: `cuda_softmax_research_artifacts (1).zip`
- Archive SHA-256: `2a00bd500313521797b9121b52e1b662638663618e34eb8c0cc82b79233ad1bb`
- Executed notebook: `04_colab_research_experiments.ipynb`
- Notebook SHA-256: `97badbb4428449fca0e07e3f6a893611b09428ddebf4ccffd6114dbbf4043770`
- Measured Git commit: `ca87722a00ebf585cd788c67949e7c0b32dca788`
- Recorded GPU: NVIDIA Tesla T4, compute capability 7.5
- Recorded run date: 2026-08-23 UTC

`artifacts/` is the extracted archive without modification. The exact executed
notebook is preserved as `executed_notebook.ipynb`. The tracked workflow
notebook in `notebooks/` remains output-free so that notebook validation and a
fresh Colab run are deterministic.

Quantitative claims must be derived from the raw CSVs or profiler exports under
`artifacts/`, not from notebook display text alone. Later audit commits add
machine-readable schema and figure-to-source checksums. See
`../schema_audit.json` and `../figure_provenance.json` from inside the artifact
directory, or the two files beside `artifacts/` from the run root.
