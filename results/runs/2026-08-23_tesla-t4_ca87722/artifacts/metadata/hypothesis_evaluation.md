# Hypothesis evaluation

## H1

HYPOTHESIS:
One-thread-per-row scales poorly as row work remains serial.

MEASURED:
Measured endpoint latency growth: row-serial=58.9795x, block=24.6063x.

INTERPRETATION:
Growth comparison can support the observed trend but cannot alone prove the causal mechanism.

STATUS:
PARTIALLY SUPPORTED

NEXT EXPERIMENT:
Use profiler instruction/memory evidence on matched early and late lengths.

## H2

HYPOTHESIS:
One-block-per-row reduces latency at sufficiently long rows despite coordination overhead.

MEASURED:
Measured long-shape speedups vs row serial: [3.4222916364595997, 6.268440406978871, 6.7695838294536]

INTERPRETATION:
Status describes only tested lengths and this GPU.

STATUS:
SUPPORTED

NEXT EXPERIMENT:
Repeat on another NVIDIA architecture and inspect short-row crossover.

## H3

HYPOTHESIS:
Warp reductions reduce latency relative to full shared-memory trees.

MEASURED:
Measured block/warp speedups by matched shape: [1.1416753212356638, 1.259063425248291, 1.6650545800767649, 1.3399674094030685, 1.9055587380935264, 1.0966614542914637, 1.0679876212258357]

INTERPRETATION:
Latency shows outcome; Nsight data is needed to attribute it to communication or synchronization.

STATUS:
SUPPORTED

NEXT EXPERIMENT:
Compare barrier/shared-memory metrics if Nsight permissions allow.

## H4

HYPOTHESIS:
The best block size depends on sequence length and resource tradeoffs.

MEASURED:
Per-shape winning block sizes: [128, 256].

INTERPRETATION:
A single observed winner does not establish universality beyond the tested matrix.

STATUS:
SUPPORTED

NEXT EXPERIMENT:
Repeat launch tuning on a second GPU architecture.

## H5

HYPOTHESIS:
Softmax-kernel speedup exceeds complete-attention speedup.

MEASURED:
Kernel speedup exceeded attention speedup for 6/7 matched shapes.

INTERPRETATION:
Unchanged matrix multiplications limit application-level benefit; measured results remain distinct from the Amdahl model.

STATUS:
PARTIALLY SUPPORTED

NEXT EXPERIMENT:
Profile the fraction of attention time attributable to softmax by shape.
