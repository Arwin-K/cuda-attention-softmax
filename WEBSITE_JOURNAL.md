# CUDA Attention Softmax — Website Journal

This is a publication-ready **planned** journal for the 112-commit investigation. Each entry states the question I am taking into the commit, not a claim that it has already succeeded. After completing a commit, I will replace its planned wording only with evidence from its diff, tests, measurements, and experiment log. Personal reflections remain my own words.

## Day 1 — A trustworthy CPU foundation

001. Completed — `scaffold research-oriented CUDA project`: I created the repository scaffold, verified its required paths and placeholder Python syntax, and made no ML, CUDA, benchmark, or profiler claim.
002. Completed — `add environment detection and platform-aware imports`: I added import-safe host/PyTorch/CUDA capability detection, verified this Apple Silicon Mac reports no PyTorch and no CUDA, and documented that MPS is not a CUDA substitute.
003. Completed — `document research question and initial hypotheses`: I defined the implementation, performance, and control variables; wrote five falsifiable predictions; and recorded that no CUDA performance evidence exists yet.
004. Completed — `add CUDA and transformer background notes`: I documented attention shapes, causal masking, stable softmax, CUDA execution and memory, block synchronization, reductions, warps, and the planned fusion boundary without claiming measurements.
005. Completed — `add learning journal and experiment log templates`: I added reusable concept, experiment, and day-checkpoint templates that separate hypotheses, measurements, interpretations, and next experiments while reserving personal reflection for `TODO(student)`.
006. Completed — `implement stable softmax reference in PyTorch`: I implemented explicit maximum subtraction, exponentiation, denominator reduction, and normalization; checked small and large logits against `torch.softmax`; and preserved shape and dtype.
007. Completed — `add causal mask construction to reference operator`: I constructed an explicitly named boolean allowed-position mask using `row_index % sequence_length` and verified first, middle, last, and wrapped query rows by hand.
008. Completed — `combine scaling masking and softmax reference path`: I composed scaling, causal `-inf` masking, and manual stable softmax in the correct order, then checked row sums and exact future-position zeros against PyTorch.
009. Completed — `add basic reference softmax correctness tests`: I added CPU regression tests for uniform values, seeded FP32/FP64 inputs, alternate reduction dimensions, probability invariants, and rejected invalid rows.
010. Completed — `add causal masking correctness tests`: I tested first, middle, last, and wrapped query rows; confirmed future probabilities are exactly zero; and confirmed allowed rows remain normalized.
011. Completed — `add numerical stability stress tests`: I tested random logits scaled by 10, 100, and 1000 plus zeros, equal logits, and dominant values; all stable results stayed finite and matched PyTorch while naïve exponentials overflowed in the controlled example.
012. Completed — `add odd and non-power-of-two shape tests`: I validated every planned correctness width from 31 through 1023, including values adjacent to warp and power-of-two boundaries, for equivalence, causal zeros, row sums, finiteness, shape, and dtype.
013. Completed — `implement explicit scaled dot-product attention reference`: I implemented and inspected `QK^T`, flattening, `1/sqrt(d)` causal stable softmax, reshaping, and `PV`, returning both output and probabilities so the future custom-kernel boundary stays visible.
014. Completed — `add attention shape and probability validation tests`: I tested several batch/head/sequence/head-dimension layouts, output and probability contracts, normalization, causal zeros, finiteness, and clear failures for invalid ranks, shapes, and dtypes.
015. Completed — `compare explicit attention against PyTorch reference behavior`: I compared probabilities and outputs against an independent PyTorch mask/`torch.softmax` composition across shapes and dtypes, and verified a future-value change cannot affect the first causal output.
016. Completed — `add reusable tensor and seed helpers for experiments`: I added validated Python/PyTorch seeding and local-generator score/Q/K/V factories, proved same-seed repeatability and different-seed variation, ran the complete CPU suite, and recorded an evidence-bounded Day 1 checkpoint.

## Day 2 — Learning material and the Python-to-GPU boundary

017. Completed — `add notebook lesson on softmax and numerical stability`: I added an executable CPU notebook that exposes tensor shapes, naïve FP32 exponential overflow, maximum subtraction, stable-softmax equivalence, assertions, exercises, and untouched `TODO(student)` reflection prompts.
018. Completed — `add notebook lesson on transformer attention and causal masking`: I added an executable `[1,1,3,2]` CPU lesson exposing Q/K/V, `QK^T`, scaling, lower-triangular masking, probabilities, `PV`, reference equivalence, exercises, and student-owned explanations.
019. Completed — `document CPU reference methodology and learning notes`: I documented the operation contract, independent PyTorch oracles, correctness axes, fixed tolerances, stress shapes, reproducibility, platform boundary, two concept entries, and an actual 47-test/two-notebook CPU validation without treating duration as performance.
020. Completed — `stabilize CPU reference test suite`: I added a source-checkout test runner and automated optional-environment tests, ran the complete CPU suite, and recorded CUDA absence as an expected Mac capability boundary rather than a project failure.
021. Completed — `add PyTorch C++ extension build infrastructure`: I registered the future `cuda_attention._C` extension and its single bindings/CUDA source pair behind an explicit build flag, while verifying ordinary package metadata and CPU workflows do not initialize CUDA tooling.
022. Completed — `add C++ bindings for fused causal softmax operator`: I defined the pybind11 `fused_causal_softmax(scores, scale)` boundary and its C++-to-CUDA launcher declaration, while leaving device math and full validation to their planned commits.
023. Completed — `add CUDA source skeleton and launch interface`: I connected the shared host declaration to the single `.cu` translation unit, introduced an educational `__global__` device skeleton, and made its unimplemented launcher fail explicitly instead of returning false results.
024. Completed — `add CUDA availability guards and graceful Mac fallback`: I added discoverability checks and an actionable `CudaExtensionUnavailableError`, verified CPU imports/tests remain usable without `_C`, and prohibited silent compilation or MPS substitution.
025. Completed — `add extension build and environment verification scripts`: I added command/toolkit readiness fields, a strict `--require-cuda` mode, and an opt-in Linux build script that reports and skips on macOS instead of invoking a CUDA compiler.
026. Completed — `document Python C++ CUDA execution path`: I mapped build time and runtime from guarded Python dispatch through pybind11 and the host launcher to asynchronous GPU threads and a PyTorch-owned output, while marking compilation, correctness, and performance as unverified.
027. Completed — `implement initial row-serial fused causal softmax kernel`: I mapped one global CUDA thread to one flattened row with an out-of-range guard, documented serial intra-row work as an unmeasured limitation, and kept the public launcher disabled until the math is complete.
028. Completed — `add stable maximum scan to CUDA kernel`: I added a serial `fmaxf` reduction from negative infinity into a thread-local row maximum, explaining why a register-local maximum requires no synchronization in the one-thread-per-row baseline.
029. Completed — `add causal masking and scaling inside CUDA kernel`: I recovered query position with row modulo, scaled values inside the allowed-column maximum scan, excluded future columns from reductions, and assigned exact zero to masked output slots without intermediate tensors.
030. Completed — `add exponential sum and normalization to CUDA kernel`: I wrote stable exponentials into output storage, accumulated the allowed denominator, normalized in place, preserved causal zeros, and enabled the current-stream launcher while marking CUDA compile/run evidence unavailable.
031. Completed — `add CUDA launch validation and error checks`: I made the native contract explicit for CUDA device, dense contiguous FP32 layout, nonempty 2D shape, and finite positive scale; guarded the input device, used its current PyTorch stream, bounded the grid, and checked immediate launch errors, while noting that these paths still require NVIDIA execution.
032. Completed — `add CUDA versus PyTorch correctness tests`: I added fixed-tolerance CUDA comparisons at core power-of-two widths, probability and causal invariants, and selected negative-contract cases. On Apple Silicon all ten CUDA cases skipped for the explicit NVIDIA-device reason while 53 CPU tests passed; actual kernel correctness remains unverified until this suite runs with the compiled extension on NVIDIA Linux.

## Day 3 — CUDA robustness, measurement, and block-reduction foundations

033. Completed — `add CUDA numerical stress tests`: I added fixed-tolerance CUDA comparisons for random logits scaled by 10, 100, and 1000 plus zeros, equal values, and dominant positive/negative cases. All seven new cases skip on this Mac because no NVIDIA CUDA device is present, so numerical GPU behavior remains unmeasured.
034. Completed — `add CUDA odd sequence-length tests`: I added CUDA reference comparisons for 31, 33, 63, 127, 255, 511, 768, and 1023 columns, complementing the existing 32/64/128 cases. They collect and skip cleanly without NVIDIA hardware; arbitrary-width kernel correctness is still pending a real CUDA run.
035. Completed — `add benchmark configuration and shape registry`: I centralized the seven required sequence lengths, FP32 dtype, eight batch-head groups, seed, warmups, iterations, and derived row/column counts in a validated immutable configuration shared by future benchmark paths.
036. Completed — `add CUDA event timing utilities`: I added warmup-aware per-iteration CUDA event timing in microseconds, synchronizing each ending event so samples represent completed stream work rather than Python dispatch. CPU-safe tests cover validation and no-fallback behavior; the CUDA timing smoke test skips locally.
037. Completed — `add PyTorch eager softmax benchmark baseline`: I defined eager PyTorch as scale plus a prebuilt flattened causal mask plus row-wise softmax, kept input/mask construction outside CUDA-event timing, and added a CPU semantic comparison with the trusted reference. No latency was collected locally.
038. Completed — `add custom CUDA softmax benchmark path`: I added the fused operator beside eager PyTorch under the same shape, seed, scale, warmup, and timing controls, with an untimed PyTorch correctness precheck. An unavailable extension now fails explicitly instead of changing the workload or backend.
039. Completed — `record hardware software and git metadata in benchmark CSV`: I added raw-sample CSV output whose rows carry the exact Git hash, implementation description, workload, timing controls, sample, GPU identity/capability, PyTorch/CUDA versions, and UTC timestamp. CPU tests verify schema and provenance preservation without inventing a GPU run.
040. Completed with unavailable evidence — `collect and document initial CUDA baseline experiment`: I attempted the row-serial benchmark, but the Apple Silicon host failed the explicit CUDA prerequisite and produced no CSV. I recorded the command, exit status, planned controls, and NVIDIA handoff without inventing latency or speedup.
041. Completed with limited evidence — `document baseline bottleneck hypothesis from initial measurements`: because the baseline attempt produced no GPU timing, I framed serial intra-row work as a falsifiable source-derived hypothesis rather than a measured bottleneck, including reduction overhead and short-row crossover as possible counterevidence.
042. Completed — `rewrite kernel mapping to one CUDA block per softmax row`: I changed row ownership from a global thread index to `blockIdx.x` and launched one 256-thread block per row. Thread 0 temporarily retains the serial math, making this a collaboration-scope transition rather than a measured optimization.
043. Completed — `distribute row elements with thread-strided access`: I assigned allowed and masked columns in `blockDim.x` strides, so every column has one owner and neighboring threads begin at neighboring addresses. Thread 0 still finishes the softmax after a staging barrier; runtime coalescing and correctness remain unmeasured.
044. Completed — `add per-thread local maximum accumulation`: I gave every block thread a register-local maximum over its strided allowed columns, stored the 256 partials in dynamic shared memory, and temporarily let thread 0 combine them serially. The parallel shared-memory tree comes next.
045. Completed — `implement shared-memory maximum reduction`: I replaced thread 0's serial partial scan with a 256-to-1 shared-memory tree reduction. Negative-infinity identity values let threads without allowed columns participate safely; the required barrier reasoning is documented in the next commit.
046. Completed — `add synchronization for block maximum reduction`: I documented the publication and per-stage barriers, then added a handoff barrier so every thread captures the row maximum before shared scratch is reused. The journal explains the race or participation failure caused by removing each boundary.
047. Completed — `add per-thread exponential partial sums`: after the block maximum handoff, each thread now computes stable exponentials for its strided columns and accumulates a register-local denominator partial. Thread 0 temporarily combines partials and normalizes serially, preserving a focused next step.
048. Completed — `implement shared-memory sum reduction`: I replaced the serial denominator scan with a synchronized 256-to-1 shared-memory addition tree. Maximum and sum reductions now use block cooperation, while thread 0 intentionally retains final normalization until Commit 049; CUDA runtime evidence remains unavailable.

## Day 4 — Validating block parallelism and introducing warp communication

049. Completed — `parallelize final probability normalization`: I removed thread 0's serial division loop and gave every thread strided ownership of its final allowed probabilities. Masked entries remain exact zeros and stay outside both reductions; runtime CUDA correctness is still pending NVIDIA execution.
050. Completed with unavailable CUDA evidence — `validate block-parallel kernel against PyTorch`: I strengthened every reusable GPU comparison with shape, dtype, device, non-negativity, exact masking, finiteness, row-sum, and fixed-tolerance checks. All CUDA cases skipped locally, so the validation gate is prepared but not passed.
051. Completed with pending GPU execution — `stress test block-parallel kernel on large magnitudes`: I expanded random ×10/×100/×1000 cases across widths 31, 128, and 511 and moved structured extremes to width 127, exercising idle threads and multi-stride work without changing the fixed tolerances. The cases skip locally.
052. Completed with pending GPU execution — `stress test block-parallel kernel on odd sequence lengths`: I extended the CUDA edge matrix to tiny widths and 255/257, 511/513, 768, and 1023/1025 boundaries, covering idle threads and partial strided passes around the 256-thread block size. All cases collect but skip locally.
053. Completed with missing artifacts — `benchmark block-parallel kernel against baseline commit`: I added a preflight requiring two nonempty, schema-valid, different-commit CSVs with identical workloads and environments. The real comparison stopped at the absent row-serial artifact, so no statistic or speedup was produced.
054. Completed — `add throughput calculation and benchmark summary statistics`: I added raw-sample aggregation for median, interpolated p25/p75, and `rows × columns / median_seconds`, preserving all Git/hardware/software provenance in the summary CSV. Synthetic fixture tests verify the math; no project measurement was created.
055. Completed — `add latency and throughput plotting utilities`: I added CPU-safe summary loading, commit-specific series construction, and lazy noninteractive Matplotlib rendering for median latency and elements per second. Empty or malformed data fails visibly, and no figure is generated without measurements.
056. Completed with no generated artifact — `generate first optimization comparison figures`: I wired one measured summary CSV to commit-aware latency and throughput outputs and tested the mapping without rendering synthetic results. The project command fails on the missing summary, so no misleading figure was committed.
057. Completed — `audit global memory access pattern for coalescing`: I traced every global load/store and confirmed that active neighboring lanes address neighboring FP32 columns on each stride. I also recorded partial-warp and row-alignment caveats and left transaction efficiency unmeasured pending Nsight.
058. Completed — `tighten arbitrary sequence-length boundary handling`: I made the reduction's power-of-two block assumption a compile-time invariant while expressing row work with exclusive 64-bit bounds independent of sequence width. New cases cover partial and wrapped flattened-query cycles; they await CUDA execution.
059. Completed — `document block reduction design and measured behavior`: I linked every block-mapping/reduction commit into one design narrative and added a claim-status table. Source structure is established; CUDA correctness, coalescing efficiency, latency, throughput, and crossover behavior remain explicitly unmeasured.
060. Completed — `stabilize block-parallel implementation`: I audited the complete block source, local regression, imports, syntax, and artifact directories. The repository is reproducible on Mac with 74 passes and 43 expected skips, but no CUDA CSV, figure, correctness result, or speedup exists.
061. Completed — `add warp and lane helper utilities to CUDA code`: I defined the 32-thread warp, eight warps per 256-thread block, compile-time divisibility, and device helpers for lane and warp IDs. The active shared-memory reductions are unchanged, so this is vocabulary and infrastructure only.
062. Completed — `implement warp-level maximum reduction with shuffle operations`: I added synchronized register shuffles at offsets 16/8/4/2/1, followed by a lane-0 broadcast. Each lane temporarily publishes its warp maximum into the existing full shared tree, preserving a staged transition before compact cross-warp combination.
063. Completed — `combine warp maxima through compact shared memory`: lane 0 of each of eight warps now publishes one maximum; the first warp loads those eight values, fills unused lanes with negative infinity, performs a second shuffle reduction, and broadcasts the block maximum through shared slot zero.
064. Completed — `implement warp-level sum reduction with shuffle operations`: each warp now reduces its register-local denominator contributions with shuffle-down operations; lane 0 publishes the sum while other lanes publish zero into the still-padded shared tree. The Day 4 checkpoint records compact maximum as complete and compact sum as pending.

## Day 5 — Completing warp reductions and tuning the fused kernel

065. Completed — `combine warp sums through compact shared memory`: lane zero now publishes one denominator partial per warp into eight shared slots, and the first warp combines them with a second shuffle reduction. Static inspection and the CPU-safe suite pass; NVIDIA compilation and numerical validation remain pending.
066. Completed — `replace shared-memory block reductions with warp reductions`: the active source now uses the same two-level shuffle-plus-compact-shared-memory hierarchy for maximum and sum. I removed the obsolete shared-tree power-of-two assumption and documented that correctness and speed still require NVIDIA evidence.
067. Completed — `validate warp-reduction kernel correctness`: I made the required shape set an explicit test contract and reran the regression suite. The coverage check passes locally, while every device comparison skips without NVIDIA; this records readiness, not CUDA correctness.
068. Completed — `stress test warp reductions on partial final warps`: I added targeted causal-prefix cases around 32-column boundaries and documented why lanes with no data still execute full-mask shuffles using identity values. The cases collect and skip locally; Colab must produce their CUDA results.
069. Completed — `verify fused scaling and causal masking remain in-kernel`: a CPU-safe source contract now checks that one CUDA kernel still owns query-position recovery, scaling, causal exclusion, stable exponentiation, and normalization. This guards equal work for later timing without claiming runtime evidence.
070. Completed — `benchmark warp-reduction optimization against prior commit`: I identified the stabilized shared-tree revision and attempted the controlled custom benchmark. The Mac guard refused execution and created no CSV, so the warp-versus-block result remains a Colab experiment rather than a claimed measurement.
071. Completed — `add benchmark support for configurable block sizes`: 128, 256, and 512 threads now flow through the Python API, C++ validation, one runtime-configurable CUDA kernel, benchmark registry, and raw CSV provenance. The unchanged 256 default is explicitly provisional until NVIDIA measurements exist.
072. Completed — `benchmark 128-thread launch configuration`: I attempted the full-registry 128-thread run, but the platform guard found no NVIDIA CUDA runtime and wrote no CSV. Its short-row-overhead versus long-row-work hypothesis remains untested until Colab execution.
073. Completed — `benchmark 256-thread launch configuration`: the controlled 256-thread command was attempted and correctly refused on the non-CUDA Mac. No CSV exists, so 256 remains only the provisional default rather than a measured optimum.
074. Completed — `benchmark 512-thread launch configuration`: the 512-thread experiment also stopped at the CUDA guard and produced no artifact. Its added-parallelism versus extra-warps hypothesis remains open, preserving an honest three-configuration tuning gap for Colab.
075. Completed — `select launch configuration from measured results`: no selection was possible because the 128/256/512 CUDA artifacts do not exist. I added a complete-data selector based on median per-shape relative latency and left 256 explicitly provisional; synthetic tests validate the policy, not kernel performance.
076. Completed — `add torch compile softmax baseline`: the benchmark now offers a torch.compile path over the exact eager scale-mask-softmax expression, and a CPU-safe compiled semantic check passes. CUDA compilation behavior and latency remain unmeasured.
077. Completed — `separate compile warmup from steady-state measurements`: each compiled shape now has an explicit untimed first call, a correctness check, ordinary CUDA warmups, and only then event-timed samples. Raw provenance distinguishes one compile warmup from zero on eager/custom paths.
078. Completed — `compare custom CUDA against eager and compiled PyTorch`: one benchmark mode now emits all three paths, and analysis rejects any workload missing eager, compiled, or custom samples. The local CUDA guard produced no CSV, so this commit establishes comparison integrity rather than a winner.
079. Completed — `document launch tuning and framework comparison results`: I documented the expected artifact map, guarded comparison methodology, hypotheses, and evidence gaps. No launch size or framework path is called faster; every quantitative conclusion remains pending Colab CSVs.
080. Completed — `stabilize tuned fused kernel and Day 5 checkpoint`: the final local audit reports 92 passes and 63 explicit GPU-only skips, including all 128/256/512 configurations on irregular widths. Python/shell/static checks and graceful Mac build behavior remain intact; the kernel is configurable, not measured-tuned.

## Day 6 — Does the microkernel change attention?

081. Completed — I placed the custom softmax only in the intended middle of explicit attention: QKᵀ, softmax, then PV. A CPU-safe boundary test confirms the flattening, scale, launch setting, and unchanged matrix multiplications; NVIDIA execution remains governed by the extension and correctness gates.
082. Completed — I added end-to-end CUDA tests so a correct-looking softmax kernel cannot hide an incorrect reshape or `PV` result. They compare outputs and probabilities on 31/33/64-token shapes and enforce causal zeros, normalization, finiteness, shape, dtype, and device at the fixed FP32 tolerances.
083. Completed — I added PyTorch SDPA as a production-oriented full-attention baseline with causal semantics and zero dropout. CPU tests compare it against the transparent explicit reference so later speed comparisons do not trade away mathematical equivalence.
084. Completed — I added direct custom-versus-SDPA CUDA comparisons on 31/33/64-token inputs, keeping the fixed FP32 tolerances. The tests recognize that reduction order may differ while still rejecting causal, indexing, shape, dtype, device, or non-finite errors.
085. Completed — I built a CUDA-event harness that measures whole explicit-eager, custom-softmax, and SDPA attention paths on identical preallocated Q/K/V tensors. It performs correctness checks before timing and stores every raw sample with B/H/S/D, Git, GPU, and software provenance.
086. Completed with evidence pending import — The T4 Colab workflow reported complete attention benchmarks across the planned lengths and created raw/summary artifacts. The ZIP is not present in this checkout, so I recorded the controlled execution but made no latency or speedup claim; the repository harness now checkpoints every completed case.
087. Completed as guarded analysis — I added a matched-provenance comparison for kernel-level and end-to-end attention speedups, including a translation ratio. It refuses missing paths or mixed Git/GPU/software environments; formulas pass fixture tests, while project ratios remain pending the raw Colab ZIP.
088. Completed — I instrumented representative isolated-softmax and complete-attention paths with named PyTorch Profiler regions. The paths share preallocated inputs, warmups stay outside capture, CUDA synchronization brackets the trace, and CPU tests verify the region contract without pretending to provide GPU timings.
089. Completed — I made profiling durable by exporting a Chrome trace, a stable CSV of CPU and CUDA region totals, and JSON run metadata. Every row carries workload and Git/GPU/software provenance, and the command refuses to overwrite a prior capture; fixture event values test the exporter but are not research results.
090. Completed — I added a one-launch Nsight Compute target, a guarded `basic`-metrics shell workflow, and reporting guidance. The helper exports report and CSV forms, filters the fused kernel, refuses overwrites, and explicitly supplies the repository on `PYTHONPATH`, fixing the import failure observed in the first Colab attempt without claiming hardware counters were captured.
091. Completed — I recorded the actual T4 profiling evidence using `MEASURED / INTERPRETATION / NEXT EXPERIMENT`: PyTorch Profiler reported completion, while Nsight Compute 2025.1.1 failed at a missing package import before any kernel launched. Because the artifact ZIP is absent, I preserved the open questions and made no timing, occupancy, or bottleneck claim.
092. Completed as evidence-gated generation — I added final latency, throughput, and eager-relative speedup plots with commit-aware labels and matched GPU/software provenance. The generator rejects missing, empty, nonpositive, unmatched, or pre-existing evidence; synthetic fixtures validate formulas, while no project figure is created until the Colab CSVs are imported.
093. Completed as evidence-gated generation — I added a matched comparison figure with isolated fused-softmax speedup and complete explicit-attention speedup on the same axes. It requires one Git/GPU/software environment and one positive row per sequence length; fixture ratios validate the plot contract, but the project figure remains absent pending raw Colab artifacts.
094. Completed as evidence-gated generation — I added three directly includable LaTeX tables for softmax, complete attention, and speedup translation. Generation requires all planned paths and lengths from one Git/GPU/software environment, derives values from CSVs, escapes LaTeX safely, and refuses overwrites; only fixture tables were created during tests.
095. Completed — I rewrote results and discussion around the preserved T4 evidence: successful corrected build, 70 CUDA pytest passes, 88/88 structured cases at fixed tolerances, completed benchmark/profiler stages, and the reported 128-thread selection. Because the raw ZIP is missing, all latency, throughput, speedup, detailed launch, and profiler conclusions remain explicitly unavailable.
096. Completed — I documented forward-only FP32/operator scope, one-T4 and managed-Colab external-validity limits, missing performance/profiler artifacts, baseline boundaries, and concrete replication and extension work. The conclusion answers only what the evidence supports: CUDA correctness is established for the tested revision; quantitative optimization and application-speedup findings remain pending raw-artifact recovery or rerun.

## Day 7 — Making the work auditable and shareable

097. I completed a LaTeX-ready paper and reference library whose claims point to the imported T4 evidence, while preserving explicit limitations and student-fillable author details.
098. I translated the measured evolution into an educational blog and recruiter-facing README, including the negative SDPA comparison and single-T4 scope instead of turning hypotheses into marketing claims.
099. I prepared NVIDIA interview and defense notes tied to the implemented mapping, reductions, measurements, negative results, limitations, and reproducibility evidence.
100. I reconciled the paper, blog, README, results, discussion, limitations, and interview notes with the recovered T4 archive, then recorded each public claim's direct evidence and remaining audit work.
101. I added a standard-library audit that verifies every benchmark path's schema, provenance, sample indices, shape matrix, implementation matrix, profiler fields, and raw-to-summary statistics without repairing missing data.
102. I linked all 12 PNG/PDF figures to SHA-256 hashes of their exact source CSVs, output bytes, experiment manifest, and measured Git commits in a deterministic provenance manifest.
103. I added and ran one CPU-only verification command for imports, deterministic notebooks, the full CPU/static suite, schema auditing, and provenance; it hides CUDA deliberately and never compiles or benchmarks the kernel.
104. I wrote a stop-aware NVIDIA handoff that separates exact T4 replication from a latest-commit rerun and covers plain-URL checkout, clean build, fixed-tolerance correctness, benchmarking, profiling, unique artifacts, hashing, and return validation.
105. I documented the kernel's evolution through Git milestones, keeping one implementation file and a readable history.
106. I shaped the research journey into a website-ready narrative for readers outside the repository.
107. I completed this per-commit journal index so each of the 112 steps has a public learning context.
108. I audited public-facing prose against artifacts so plans and interpretations cannot masquerade as measurements.
109. I collected reproducibility commands with clear platform requirements and expected outputs.
110. I reviewed CUDA comments to ensure they teach the reasoning behind reductions, synchronization, and memory access.
111. I performed a final evidence and scope audit, retaining TODOs wherever proof is absent.
112. I finished with a seven-day checkpoint that verifies coherence among code, tests, results, and the public narrative.

## Supplemental Day 5 Colab handoff

These unnumbered commits were explicitly requested after the 112-commit roadmap.
They package the planned NVIDIA work into one reproducible Colab notebook without
changing the numbering or meaning of Commits 001–112.

- Completed — `add Colab experiment notebook foundation`: I added one
  deterministic, reviewable notebook with configuration, NVIDIA runtime checks,
  source acquisition, Git provenance, build logging, and artifact safeguards.
- Completed — `add reproducible CUDA experiment workflows`: I connected the
  notebook to the repository's CUDA tests and benchmark entry points, added the
  structured correctness gate, launch-size comparison, framework comparison,
  and verified historical-revision workflow.
- Completed — `add attention benchmarks and profiler capture`: I added explicit
  eager/custom/SDPA attention experiments, kernel-versus-attention analysis,
  PyTorch Profiler exports, and an optional Nsight Compute attempt with honest
  unavailable markers.
- Completed — `add paper artifact generation and experiment packaging`: I added
  evidence-derived figures, CSV and LaTeX tables, hypothesis status summaries,
  artifact validation, a manifest, optional Drive backup, and a downloadable
  experiment ZIP.
- Completed — `document and validate Colab research workflow`: I documented the
  run and recovery procedure and added CPU-safe structural tests that reject
  stale notebook generation, saved outputs, embedded secrets, missing stages,
  and absent evidence-integrity guards.
- Completed — `fix CUDA 12.8 infinity constant include`: The first T4 build
  reached NVCC and exposed an undefined `CUDART_INF_F`. I added its defining
  header, guarded the dependency with a static test, and made historical builds
  record the same header-only compatibility adjustment instead of hiding it.

The first NVIDIA compilation attempt failed before a kernel launched. CUDA
correctness, performance, profiler observations, and hypothesis outcomes remain
pending a successful rebuild and correctness gate in the generated notebook.

## Completion template

For each completed entry, append only evidence you can support:

- **What changed:** files and the intent of the diff.
- **What I checked:** exact commands and pass/skip/fail outcomes.
- **What I observed:** links to CSVs, traces, or figures where relevant.
- **What I learned:** `TODO(student)` in my own words.
- **What remains:** limitations, failures, and the next question.
