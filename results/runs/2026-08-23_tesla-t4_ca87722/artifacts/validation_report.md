# Validation report

## CUDA environment: PASS

Tesla T4

## CUDA extension build: PASS

{'timestamp': '2026-08-23T19:09:24.282153+00:00', 'git_commit': 'ca87722a00ebf585cd788c67949e7c0b32dca788', 'command': ['bash', '/content/cuda-attention-softmax/scripts/build_extension.sh'], 'return_code': 0, 'duration_seconds': 112.10187813200002, 'dependency_install_return_code': 0, 'extension_import_ready': True, 'extension_import_error': None}

## Custom operator import: PASS

cuda_attention._C

## CUDA correctness: PASS

{'timestamp': '2026-08-23T19:09:40.806268+00:00', 'git_commit': 'ca87722a00ebf585cd788c67949e7c0b32dca788', 'pytest_return_code': 0, 'case_count': 88, 'passed_cases': 88, 'failed_cases': 0, 'rtol': 1e-05, 'atol': 1e-06, 'correctness_ready_for_benchmark': True}

## No unexpected NaNs/Infs: PASS

correctness CSV

## Masked probabilities exactly zero: PASS

correctness CSV

## Softmax row sums: PASS

threshold=1.1000000000000001e-05

## Softmax raw schema: PASS

{'git_commit', 'pytorch_version', 'cuda_version', 'implementation', 'latency_us', 'compile_warmups', 'scale', 'dtype', 'implementation_description', 'columns', 'compute_capability', 'sequence_length', 'iterations', 'warmups', 'timestamp', 'sample_index', 'gpu_name', 'launch_block_size', 'rows'}

## Softmax raw samples: PASS

rows=2100

## Git hashes in benchmark rows: PASS

ca87722a00ebf585cd788c67949e7c0b32dca788

## Summary median/quartiles: PASS

recomputed from raw

## Throughput formula: PASS

rows*columns/median_seconds

## Matched softmax speedups: PASS

eager median / candidate median

## Environment metadata: PASS

/content/cuda_softmax_artifacts/environment/environment.json

## Figures from measured data: PASS

generated_files=12; missing figures have text reports

## Git working tree provenance: PASS

clean
