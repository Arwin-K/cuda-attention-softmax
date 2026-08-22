# Methodology

## Study structure

The project changes one primary CUDA implementation through Git history. Each
implementation change is validated for correctness before it can be measured,
and each measurement is tied to the code commit and execution environment that
produced it. CPU reference work establishes semantics; it does not predict GPU
speed.

## Operation contract

The reference softmax operator accepts floating-point scores shaped
`[rows, sequence_length]`, applies a positive scale, excludes columns after the
flattened row's query position, and returns probabilities of the same shape,
dtype, and device.

```text
query_position = row_index % sequence_length
allowed         = column <= query_position
```

Stable softmax subtracts the maximum over allowed values before exponentiation.
Complete explicit attention uses `[batch, heads, sequence_length,
head_dimension]` Q/K/V tensors and computes `QK^T -> probabilities -> PV`.

## Trusted comparisons

Reference functions are compared with independent PyTorch operations rather
than only with helpers that share their implementation:

- `stable_softmax` is compared with `torch.softmax`;
- causal probabilities are compared with an independently constructed lower-
  triangular mask and `torch.softmax`;
- explicit attention output is compared with separately composed PyTorch
  `QK^T`, mask, softmax, and `PV` operations.

The custom CUDA operator will later be compared with these CPU/PyTorch
semantics. Passing CPU tests is not evidence that CUDA code compiles or runs.

## Correctness dimensions

Applicable tests check:

- numerical closeness to PyTorch with fixed tolerances;
- exact zero probability at masked future positions;
- probability rows approximately summing to one;
- no unexpected NaNs or infinities;
- output shape, dtype, and device;
- valid handling of first, middle, last, and wrapped query rows;
- clear rejection of invalid ranks, shapes, dtypes, dimensions, and scales.

FP32 stress cases use `rtol=1e-5` and `atol=1e-6`. Tolerances are part of the
test contract and must not be weakened merely to accept an incorrect result.

## Inputs and boundary cases

The CPU suite covers normal random inputs, magnitudes of 10, 100, and 1000,
zeros, equal values, one dominant positive value, and one dominant negative
value. Required sequence lengths are:

```text
31, 32, 33, 63, 64, 127, 128, 255, 511, 768, 1023
```

Values adjacent to warp and power-of-two boundaries are included before CUDA
exists so later GPU code cannot silently narrow the valid shape contract.

## Reproducibility

Test data uses explicit seeds. Reusable factories use local PyTorch generators,
so unrelated global random draws do not alter generated score or Q/K/V tensors.
The primary future softmax experiment uses FP32, `batch_heads=8`, and
`rows=batch_heads*sequence_length` unless an experiment explicitly states a
different controlled configuration.

## CPU validation workflow

Run from the repository root with the project virtual environment:

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pytest -q tests
```

Notebook validation parses the JSON and executes code cells in order with a
shared namespace. This checks the educational code without committing generated
cell outputs.

## Platform boundary

Apple Silicon macOS supports documentation, notebooks, PyTorch CPU references,
tests, and later result analysis. It must not compile the CUDA extension or use
MPS as a CUDA substitute. CUDA compilation, custom-operator correctness,
benchmarking, and NVIDIA profiling require Linux, an NVIDIA GPU, and a CUDA
toolkit.

## CUDA timing boundary

GPU kernels launch asynchronously, so Python wall-clock time around a function
call can measure dispatch rather than completed device work. The benchmark
utility records CUDA events immediately before and after the operation on the
current stream, synchronizes the ending event, and converts CUDA's millisecond
duration to microseconds. Inputs are prepared outside this interval, and
untimed warmups precede every sample set. This method is available only with an
NVIDIA CUDA runtime; it never substitutes CPU or MPS timings.

## Evidence after Commit 019

On 2026-08-14, the CPU suite reported 47 passing tests in 1.02 seconds and both
educational notebook code paths executed successfully. The environment report
identified Darwin arm64, Python 3.11.15, PyTorch 2.13.0, and no CUDA device or
PyTorch CUDA build. Test-run duration is validation metadata, not a performance
benchmark.
## Compilation startup versus steady-state latency

The `torch.compile` baseline uses three ordered phases for each shape:

1. one explicit untimed compile/startup call,
2. the same numerical comparison against the trusted causal reference,
3. ordinary untimed CUDA warmups followed by CUDA-event samples.

The raw CSV records `compile_warmups=1` for the compiled path and zero for eager
and custom paths. Compilation time is intentionally outside the primary
steady-state latency metric; if startup cost is later reported, it must be a
separate metric rather than mixed into the distribution. Input and causal-mask
allocation remain outside every timed region.
