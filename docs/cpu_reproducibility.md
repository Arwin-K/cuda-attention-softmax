# CPU-only reproducibility

Apple Silicon is the supported local environment for learning, reference
correctness, documentation, notebook validation, and result analysis. It is not
a CUDA execution environment, and MPS is never used as a replacement.

From the repository root, create/install the development environment as
documented in `README.md`, then run:

```bash
./scripts/verify_cpu_reproducibility.sh
```

The workflow deliberately sets `CUDA_VISIBLE_DEVICES` to an empty value and:

1. records platform and optional dependency capabilities;
2. imports the package without building a native extension;
3. confirms the checked-in Colab notebook matches its deterministic generator;
4. runs all CPU, documentation, static-CUDA-contract, and capability-gated
   tests;
5. independently audits the imported raw/summary result schemas; and
6. verifies current figure and source hashes against the provenance manifest.

Expected GPU-only tests report `skipped`; that is a visible platform boundary,
not a test failure. The workflow must not call `build_extension.sh`, run a CUDA
benchmark, claim a GPU latency, or redirect to MPS.

## Local verification record

On 2026-08-23, the workflow passed on Darwin arm64 with Python 3.11.15 and
PyTorch 2.13.0 CPU. Environment detection reported no NVCC, no `nvidia-smi`, no
PyTorch CUDA build, and no CUDA device. The notebook was current, package import
passed, the suite reported 128 passed and 69 GPU-only skips, and both imported-
result audits passed. No CUDA compiler, benchmark, profiler, or MPS backend was
invoked.
