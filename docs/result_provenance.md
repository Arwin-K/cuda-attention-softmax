# Result provenance

Each publication figure is linked to exact input bytes in
`results/runs/2026-08-23_tesla-t4_ca87722/figure_provenance.json`. A record
contains:

- the PNG or PDF path and SHA-256;
- each direct raw/summary CSV path and SHA-256;
- every Git commit appearing in those source rows; and
- the SHA-256 of the run's experiment manifest.

The hashes answer whether a checked-in image still corresponds to the source
files reviewed for publication. They do not prove that the original GPU
measurement was unbiased; environment metadata, benchmark methodology, and
replication remain separate requirements.

Regenerate the manifest from the repository root:

```bash
python3 scripts/generate_result_provenance.py \
  results/runs/2026-08-23_tesla-t4_ca87722/artifacts \
  --output results/runs/2026-08-23_tesla-t4_ca87722/figure_provenance.json
```

Then run `python3 -m pytest -q tests/test_result_provenance.py`. Any changed
figure or source requires an intentional regenerated manifest and review.
