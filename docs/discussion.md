# Discussion

No performance interpretation is currently possible because the NVIDIA result
artifacts are absent. The following are hypotheses to test, not findings:

- 128 threads may reduce coordination overhead for short rows but increase
  per-thread strided work for long rows.
- 512 threads may increase long-row parallelism but add inactive lanes and
  cross-warp partials, particularly for early causal rows.
- 256 threads may be a compromise rather than a universal winner.
- `torch.compile` may reduce the gap between eager composition and the custom
  kernel by compiling the whole expression, but the generated CUDA work must be
  observed rather than assumed.

The launch selector uses the median of per-shape relative latencies so one large
workload does not dominate merely because its absolute time is larger. Final
discussion must still report per-shape results: an aggregate winner can hide a
configuration that is preferable for a specific deployment length.

`TODO(student): After collecting Colab results, describe one observation that
contradicted or refined your original launch-tuning prediction.`
