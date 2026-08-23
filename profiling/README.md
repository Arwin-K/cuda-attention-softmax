# Profiling

Profiling complements the CUDA-event benchmarks; it does not replace them.
PyTorch Profiler locates time within the framework execution path, while
NVIDIA Nsight Compute can collect kernel launch, resource, and hardware-counter
evidence when the host permits access.

## PyTorch Profiler

From the repository root on Linux/NVIDIA, after building the extension:

```bash
python profiling/profile_pytorch.py \
  --output-dir results/raw/pytorch_profiler \
  --sequence-length 512 --warmups 5 --repeats 10
```

The command writes a Chrome trace, a region-summary CSV, and JSON provenance.
It refuses existing artifacts. Open the trace in a compatible trace viewer and
use the CSV for reviewable totals. The trace is the source of truth for child
operators and kernels.

## NVIDIA Nsight Compute

Run the narrow kernel capture with:

```bash
sh profiling/run_ncu.sh results/raw/nsight
```

The helper selects Nsight's version-compatible `basic` metric set, filters for
`fused_causal_softmax_kernel`, exports both `.ncu-rep` and raw CSV forms, and
refuses to overwrite them. Its explicit `PYTHONPATH` ensures the target imports
the checked-out `cuda_attention` package even when `ncu` changes launch context.

Nsight Compute is optional. Managed services such as Colab may install `ncu`
but deny access to hardware performance counters. Treat a missing tool,
permission error, or empty report as unavailable evidence—not as a property of
the kernel. Preserve the command log and follow this reporting structure:

```text
MEASURED:
Only facts directly present in the report.

INTERPRETATION:
A causal hypothesis explaining those facts.

NEXT EXPERIMENT:
A controlled test capable of challenging the hypothesis.
```

Neither profiler workflow runs on Apple Silicon, and MPS is not a substitute
for CUDA.
