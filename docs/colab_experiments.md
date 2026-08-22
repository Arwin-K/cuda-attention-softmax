# Google Colab experiment notebook

The single Colab execution notebook is:

```text
notebooks/04_colab_research_experiments.ipynb
```

It is generated deterministically by `scripts/generate_colab_notebook.py` so
large notebook JSON changes remain reviewable. The checked-in notebook contains
no executed outputs or example measurements.

## Before opening Colab

Push the supplemental branch:

```bash
git switch day-five-pt-2
git status
git push -u origin day-five-pt-2
```

These are user-authorized supplemental commits. They do not renumber or replace
the planned Commit 001–112 research roadmap.

## Exact Colab procedure

1. Open the notebook from GitHub using the README badge.
2. Select **Runtime → Change runtime type → NVIDIA GPU**.
3. Review the configuration cell. Keep the repository URL and
   `day-five-pt-2` branch unless deliberately testing another revision.
4. Use a new `/content/cuda_softmax_artifacts` directory and leave
   `ARTIFACT_POLICY="ERROR"` for the first run.
5. Run the notebook from the top in order.
6. Require GPU validation, extension build/import, pytest, and structured CUDA
   correctness to pass before accepting benchmark data.
7. Let historical benchmarking verify its configured commit objects and commit
   subjects. Do not replace a failed historical build with a different commit
   without documenting that change.
8. Run all three launch configurations in the same Colab session. The notebook
   selects a size only after the 128/256/512 matrices are complete.
9. Run eager, `torch.compile`, and custom softmax comparisons, followed by
   explicit eager/custom/SDPA attention comparisons.
10. Preserve PyTorch Profiler outputs. Nsight Compute may be unavailable because
    Colab can restrict hardware counters; the notebook records this explicitly.
11. Inspect `validation_report.md`. Resolve every critical `FAIL` before using a
    number in the paper.
12. Download `cuda_softmax_research_artifacts.zip` and keep a second copy. The
    optional Drive stage is useful because `/content` disappears with the
    runtime.
13. Upload the ZIP with the paper source for evidence-backed writing.

## Recovering after disconnection

If the runtime disconnects, the assigned GPU or software image may change.
Restore the artifact directory from Drive, set `ARTIFACT_POLICY="REUSE"`, and
rerun environment/source/build cells. The notebook refuses reuse when the GPU,
compute capability, Python, PyTorch, CUDA, or Git commit differs. If they differ,
start a new artifact directory and rerun the complete matched comparison series.

## What the notebook does not claim

Creating and validating the notebook does not establish CUDA compilation,
correctness, latency, throughput, speedup, profiler metrics, or hypothesis
outcomes. Those become evidence only when the notebook actually runs on an
NVIDIA runtime and the resulting ZIP passes its validation audit.
