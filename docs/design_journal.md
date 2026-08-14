# Design journal

## Optional Python/C++/CUDA boundary

### Problem

The project needs a native CUDA path without making CPU imports, documentation,
or tests depend on a CUDA compiler or extension binary.

### Existing evidence

The CPU package and 53 tests run on Apple Silicon. The environment checker
reports no CUDA build or execution capability. No NVIDIA build has run.

### Hypothesis

An explicitly enabled build plus a guarded runtime import will preserve the
Mac workflow while providing one clear extension path on NVIDIA Linux.

### Proposed change

Register `cuda_attention._C` behind `CUDA_ATTENTION_BUILD_CUDA=1`, bind one
`fused_causal_softmax(scores, scale)` function, and keep its launcher in the
single evolving CUDA translation unit.

### Implementation

`setup.py` owns build registration, `bindings.cpp` owns pybind11 exposure,
`common.cuh` owns the shared declaration, `fused_causal_softmax.cu` owns launcher
and device code, and `cuda_attention/operator.py` owns guarded Python dispatch.

### Correctness result

CPU imports and 53 CPU tests pass. Structural source checks pass. CUDA
correctness is not measured because no extension was compiled or run.

### Performance result

Not measured. No performance claim is supported.

### Interpretation

The layers now have explicit ownership, which should make later failures easier
to localize. Only an NVIDIA build can test whether the native boundary compiles
and links as intended.

### Next question

Can the row-serial kernel compile and match the PyTorch reference on the planned
CUDA correctness shapes?

### Git commit

Commit 026 — `document Python C++ CUDA execution path`

## Row-serial work decomposition

### Problem

A correctness-first kernel needs an unambiguous mapping from CUDA execution
coordinates to flattened softmax rows.

### Existing evidence

CPU tests define row semantics. No CUDA compilation or measurement exists.

### Hypothesis

One global CUDA thread per row will be straightforward to validate, although
serial column scans are expected to limit performance at larger sequence
lengths.

### Proposed change

Map `blockIdx.x * blockDim.x + threadIdx.x` to one row and guard threads whose
global index exceeds `rows`.

### Implementation

`csrc/fused_causal_softmax.cu` now contains the row mapping and a fixed
256-thread block constant. The public launcher remains disabled until all
softmax math exists in Commit 030.

### Correctness result

Static source checks and the CPU suite pass. CUDA behavior is not tested.

### Performance result

Not measured.

### Interpretation

The mapping makes ownership explicit but does not expose intra-row parallelism.
That limitation is a future optimization question, not a measured bottleneck.

### Next question

Can the owning thread compute a stable maximum over its allowed columns?

### Git commit

Commit 027 — `implement initial row-serial fused causal softmax kernel`
