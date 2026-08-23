# CUDA softmax experiment artifacts

## Runtime environment

- Git commit: `ca87722a00ebf585cd788c67949e7c0b32dca788`
- Git dirty: `False`
- GPU: `Tesla T4`
- Compute capability: `7.5`
- PyTorch: `2.11.0+cu128`
- PyTorch CUDA: `12.8`
- Python: `3.13.15`

## Build

The notebook invoked `bash scripts/build_extension.sh` from the repository root.
See `build/build_log.txt` and `build/build_metadata.json` for the complete result.

## Methodology

Inputs and masks were allocated outside timed regions. Each operation received
25 untimed warmups and 100 CUDA-event samples. The compiled
baseline used an additional untimed first invocation. CUDA synchronization made
device completion part of every sample. Correctness used rtol=1e-05, atol=1e-06.

## Output files

See `experiment_manifest.json` for the machine-readable inventory and stage
status. Raw samples are retained rather than replaced by summary values.

## Known limitations

Google Colab is a shared cloud environment. GPU model, clocks, contention,
runtime lifetime, Nsight availability, and performance-counter permissions are
not guaranteed. Results from different GPU environments must not be combined.

## Reproduction

Open `notebooks/04_colab_research_experiments.ipynb`, choose an NVIDIA GPU,
configure the same repository revision and controls, use a new artifact
directory, and run cells in order. Resolve validation failures before quoting
results.
