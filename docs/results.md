# Results

## Evidence boundary

The executed Colab notebook is preserved in Git commit `0e319a5`; the current
checked-in notebook is regenerated without outputs so it remains reviewable and
reproducible. The executed output establishes environment, build, correctness,
and stage-completion facts. Its downloadable raw-artifact ZIP has not been
imported into this checkout, so no latency, quartile, throughput, speedup, or
profiler-event number is reported here.

## Experimental environment

The successful remote run used an NVIDIA Tesla T4 with compute capability 7.5,
PyTorch 2.11.0+cu128, CUDA 12.8, and Python 3.13.15. It cloned Git revision
`d1b3fd38b28075c7fbfcff2b03cde4a2a6b02f1d`. These facts describe that run;
they do not automatically apply to later source commits.

## Compilation and correctness

The first NVCC attempt exposed an undefined `CUDART_INF_F`, traced to a missing
explicit CUDA runtime constants header. After the header fix, the executed
notebook reported the extension build/import ready.

The same T4 notebook reported:

- 70 CUDA pytest cases passed;
- 88 of 88 structured correctness cases passed at the fixed FP32 tolerances
  `rtol=1e-5` and `atol=1e-6`;
- no tolerance weakening was used; and
- the notebook's CUDA correctness gate was `PASS` before benchmark stages ran.

This is evidence that the tested warp-reduction operator produced acceptable
outputs on that revision and environment. It is not a correctness result for
every historical or later commit.

## Launch configuration

The notebook output reports completion of the 128/256/512 launch experiment and
selection of 128 threads under its configured rule. The launch raw CSV and
selection JSON are absent from this checkout, so per-shape timings, normalized
scores, and wins cannot be audited here. The source therefore retains 256 as
its provisional default rather than converting an unimported result into a
code-level tuning claim.

## Softmax performance

The saved output reports the softmax benchmark stage `COMPLETE`, including its
historical workflow. However, neither the raw CUDA-event samples nor their
summary CSV is present. The following quantities remain unavailable:

| Question | Status | Required artifact |
|---|---|---|
| Custom median latency by sequence length | Unavailable here | Softmax raw CSV |
| Eager and compiled baseline latency | Unavailable here | Framework raw CSV |
| Elements per second | Unavailable here | Derived softmax summary |
| Row-serial/block/warp improvement | Unavailable here | Matched historical CSVs |
| Custom versus eager/compiled speedup | Unavailable here | Complete framework summary |

Stage completion proves the workflow reached its end; it does not reconstruct
the sample distribution.

## Complete attention

The executed output reports complete-attention benchmarking `COMPLETE` and the
creation of raw/summary/correctness artifacts for explicit eager, custom CUDA,
and PyTorch SDPA paths. Those files are inside the unavailable ZIP. Therefore,
the project cannot yet state whether custom attention is faster, how it compares
with SDPA, or how much isolated softmax speedup reaches the full operation.

## Profiling

PyTorch Profiler reported `COMPLETE`, but its trace and event table are absent.
Nsight Compute 2025.1.1 was installed; its target failed to import
`cuda_attention` before launching the kernel. Consequently, there is no Nsight
occupancy, memory, instruction, or launch metric. Commit 090 corrects the target
import path for the next attempt.

## Reproducible outputs still pending

After the raw ZIP is imported and its manifest is checked, repository commands
can regenerate:

- softmax latency, throughput, and eager-relative speedup figures;
- the isolated-kernel versus complete-attention speedup figure; and
- LaTeX softmax, attention, and translation tables.

Until then, the repository intentionally contains no generated result figure or
table beyond `.gitkeep` placeholders.
