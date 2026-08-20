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
052. I tested whether partial work assignment remains correct for odd widths.
053. I measured the block-parallel design against the recorded row-serial baseline under the same protocol.
054. I added median, quartiles, and throughput so performance means more than one timing number.
055. I built plotting utilities that derive figures from stored CSV data.
056. I generated the first historical comparison figure only from real baseline and block-parallel measurements.
057. I audited whether the thread-to-column mapping supports coalesced global-memory access, without overstating its effect.
058. I hardened the implementation for arbitrary sequence lengths rather than relying on power-of-two assumptions.
059. I documented the block-reduction design as a chain from evidence to hypothesis to measured outcome.
060. I stabilized the block-parallel state as a reproducible Git milestone.
061. I introduced the vocabulary and helpers for reasoning about warps, lanes, and warp IDs.
062. I reduced maxima within each warp using shuffle instructions and register exchange.
063. I combined one maximum per warp through a compact shared-memory bridge.
064. I applied the same warp-level idea to the softmax denominator and recorded the fourth-day checkpoint.

## Day 5 — Completing warp reductions and tuning the fused kernel

065. I combined warp sums into the final denominator with only a small shared array.
066. I removed the old full shared-memory reduction path, letting Git history preserve it while the source stays singular.
067. I re-ran full correctness validation after the warp-reduction rewrite.
068. I tested partial final warps, where lane participation is easy to get subtly wrong.
069. I verified that scaling and causal masking are still fused inside the kernel after the reduction changes.
070. I measured the warp-reduction kernel against the prior block-reduction milestone.
071. I made block size a controlled experimental variable rather than a hidden launch constant.
072. I measured the 128-thread configuration under the shared protocol.
073. I measured the 256-thread configuration under the shared protocol.
074. I measured the 512-thread configuration under the shared protocol.
075. I selected the default launch configuration from results, documenting any shape-dependent tradeoff.
076. I added torch.compile as a stronger framework baseline for the same operation.
077. I separated compilation warmup from steady-state timing so startup cost does not distort latency.
078. I compared eager PyTorch, compiled PyTorch, and the custom CUDA route on equal work.
079. I documented the launch-tuning and framework results with their source artifacts and limitations.
080. I stabilized the tuned kernel and captured the fifth-day checkpoint.

## Day 6 — Does the microkernel change attention?

081. I placed the custom softmax only in the intended middle of explicit attention: QKᵀ, softmax, then PV.
082. I added end-to-end tests so a correct-looking kernel cannot hide an incorrect attention result.
083. I added PyTorch SDPA as a production-oriented full-attention baseline.
084. I checked custom attention against SDPA, paying attention to causal semantics and justified tolerances.
085. I built a harness that measures whole attention paths fairly, not just the softmax microkernel.
086. I benchmarked custom attention across sequence lengths only on actual NVIDIA hardware.
087. I compared kernel-level and end-to-end speedups to test the Amdahl's Law lesson in this system.
088. I instrumented representative softmax and attention runs with PyTorch Profiler.
089. I exported profile traces and summary timings as inspectable artifacts.
090. I added an optional Nsight Compute workflow without claiming it ran where it was unavailable.
091. I recorded profiler observations in measured, interpretation, and next-experiment sections.
092. I generated final latency, throughput, and historical speedup figures from stored measurements.
093. I generated a figure that directly contrasts kernel speedup with attention speedup.
094. I generated paper-ready tables from CSVs rather than transcribing numbers by hand.
095. I wrote results and discussion only to the extent the experiment artifacts support them.
096. I wrote limitations, future work, and conclusions that keep the project's scope honest, then recorded the sixth-day checkpoint.

## Day 7 — Making the work auditable and shareable

097. I assembled a LaTeX-ready paper outline whose figures, tables, and references remain traceable.
098. I translated the work into a technical blog and recruiter-facing README without turning hypotheses into marketing claims.
099. I prepared interview and project-defense questions tied to real implementation choices.
100. I checked the publication material for coherence before the final reproducibility pass.
101. I audited benchmark schemas so every performance record either has its required provenance or a visible gap.
102. I linked figures back to source CSV files and commits, making visual claims reproducible.
103. I verified the CPU-only workflow on Apple Silicon, including what must deliberately skip without CUDA.
104. I wrote an NVIDIA handoff checklist for rebuilding, testing, benchmarking, and profiling remotely.
105. I documented the kernel's evolution through Git milestones, keeping one implementation file and a readable history.
106. I shaped the research journey into a website-ready narrative for readers outside the repository.
107. I completed this per-commit journal index so each of the 112 steps has a public learning context.
108. I audited public-facing prose against artifacts so plans and interpretations cannot masquerade as measurements.
109. I collected reproducibility commands with clear platform requirements and expected outputs.
110. I reviewed CUDA comments to ensure they teach the reasoning behind reductions, synchronization, and memory access.
111. I performed a final evidence and scope audit, retaining TODOs wherever proof is absent.
112. I finished with a seven-day checkpoint that verifies coherence among code, tests, results, and the public narrative.

## Completion template

For each completed entry, append only evidence you can support:

- **What changed:** files and the intent of the diff.
- **What I checked:** exact commands and pass/skip/fail outcomes.
- **What I observed:** links to CSVs, traces, or figures where relevant.
- **What I learned:** `TODO(student)` in my own words.
- **What remains:** limitations, failures, and the next question.
