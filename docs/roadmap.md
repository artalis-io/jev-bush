# Jev Bush roadmap

Jev Bush is already accuracy-competitive with OpenJev and broadly comparable
to Jev on the public typed-decisions benchmark. The next research question is
how much faster its deliberately small CPU implementation can become without
turning into a general inference framework.

The established throughput configuration uses the evaluated `-ffast-math`
build and two concurrent 32-thread workers on a 64-core Threadripper 9980X.
Each worker averages 5.49 seconds per five-decision row, or 0.911 decisions per
second, with a p95 row latency of 8.66 seconds. Together they sustain about
1.82 decisions per second. OpenJev on an RTX PRO 6000 Blackwell averages 54.2
milliseconds per one-read row.

That is approximately a 101x single-row latency gap and, after accounting for
the two CPU workers, a 51x aggregate-throughput gap. The comparison is useful,
but not perfectly controlled: the CPU number is a two-worker throughput run,
not the best 48-thread single-request configuration, and the GPU has much more
memory bandwidth plus native FP4 tensor hardware.

## What is already cached

- `eval` loads the model and tokenizer once and keeps them alive for the input
  stream. Repeatedly invoking `decide` as a new process should not be used for
  throughput measurements.
- Encoder prefill runs once per request. Its K/V is reused by every diffusion
  read, so explicit samples and automatic rereads do not repeat the prefill.
- The operating system page cache keeps memory-mapped model pages warm after
  initial access.

Candidate projection is not a useful optimization target. It averages 0.026
milliseconds per decision, about 0.13 milliseconds per five-decision row and
0.0013% of total row time. Transformer execution dominates.

## 1. Measure before changing kernels

Add timing for each layer and major operation:

- prompt embedding and causal prefill;
- attention projections and attention itself;
- dense feed-forward blocks;
- router and expert selection;
- expert gate/up/down projections;
- answer-canvas decoding;
- allocation, copying, and OpenMP synchronization.

Record hardware counters where available: sustained memory bandwidth, vector
utilization, cache misses, instructions per cycle, and time spent at OpenMP
barriers. Report cold and warm runs separately.

The first NVFP4 `JB_PROFILE` wall-clock measurements establish two prompt-
dependent profiles on the Threadripper 9980X with 32 threads. A warm 219-token
request spends 62.7% of total inference time in MoE experts, 23.5% in
attention, and 7.8% in the dense FFN. A 974-token one-read evaluation row
spends 47.9% in experts, 34.4% in attention, and 10.2% in the dense FFN.
Routing is below 1% in both. Optimize and benchmark both prompt sizes: short-
request results alone substantially understate the importance of attention.

Within the NVFP4 expert path, gate/up/down projections plus input and hidden
QDQ account for 54.6% of total short-request time and 37.3% of the long row.
This is enough headroom to justify one exact integer-coded E2M1/VNNI prototype.
Its go/no-go result must use uninstrumented end-to-end latency because detailed
profiling adds about 0.8% overhead on the warm short request.

The expanded profile accounts for all but 0.03 ms of a 7.40-second, 974-token,
five-decision row. Experts consume 48.5%, attention 34.0%, dense FFN 10.1%,
and remaining FFN work 5.6%. Q/K/V projections alone consume 18.6% of the
complete row. Leave the AVX-512 arithmetic primitives unchanged until a later
profile moves the limit.

Physical-core scaling on the same row peaks at 32 threads:

| threads | row seconds | decisions/s | prefill tokens/s |
|---:|---:|---:|---:|
| 8 | 13.73 | 0.364 | 74.0 |
| 16 | 9.44 | 0.530 | 107.9 |
| 24 | 7.96 | 0.628 | 128.1 |
| 32 | **7.36** | **0.679** | **138.5** |
| 40 | 8.26 | 0.606 | 123.4 |
| 48 | 8.74 | 0.572 | 116.2 |
| 56 | 9.06 | 0.552 | 112.0 |
| 64 | 9.10 | 0.550 | 111.3 |

Using both SMT threads (`128` threads with `OMP_PLACES=threads`) takes 84.22
seconds, 11.4 times slower than 32 physical threads. Published runs must set
thread count and placement explicitly.

Every optimization must preserve byte-identical probabilities unless a change
is explicitly presented and validated as a numerical experiment. Fast-math is
the first such experiment: it changed distributions, matched OpenJev automatic-
read accuracy at 66.80%, and improved throughput, while slightly worsening log
loss, Brier score, and score MAE. Benchmark strict and fast-math builds
separately, and separate single-request latency from multi-worker throughput.

## 2. Batch predicates over one state

The existing single-canvas path already demonstrates useful natural batching.
With one fixed state and compact boolean predicates, 32 physical threads scale
from 0.245 decisions/s at one predicate to 2.39 decisions/s at sixteen, a
9.75x throughput gain for 16x the decisions. Total latency grows from 4.08 to
6.70 seconds. A realistic long-rubric schema peaks at 0.66 decisions/s with
eight predicates and begins falling at sixteen because prompt prefill grows.

The former 64-token answer canvas fit at most 17 compact predicates. The
multi-canvas path now preserves the single canvas for requests through sixteen
decisions, then groups at most eight decisions per canvas. Every canvas shares
one causal prompt K/V; matrix and FFN execution is flattened across canvases,
while attention restricts each canvas to the shared prefix and its own answer
tokens.

At 32 predicates (four canvases), total latency is 10.95 seconds and throughput
is 2.92 decisions/s. At 64 predicates (eight canvases), total latency is 16.68
seconds, including 14.52 seconds of shared prefill and 2.17 seconds of decode,
for 3.84 decisions/s. Batched decode is 18.7% faster than sequential canvases
at 32 predicates and 20.1% faster at 64. Both runs produce byte-identical
probabilities to the isolated sequential-canvas reference; the established
eight-row OpenJev regression is also byte-identical to the prior engine.

The 64-predicate prompt contains 1,248 system/predicate tokens, 525 state
tokens, and about 14 chat-boundary tokens. An experimental compact rendering
removed 155 tokens and reduced latency by 10.4%, but failed the quality gate:
on 120 decisions accuracy fell from 68.33% to 60.00%, Brier worsened from
0.249 to 0.299, ECE from 0.178 to 0.286, and argmax agreement with the default
prompt was only 77.5%. The compact prompt was removed.

The same request enters 16,327 OpenMP regions, but an isolated 32-thread region
launch benchmark puts their fixed floor near 100 ms, below 0.6% of request
time; persistent teams would retain synchronization barriers. Likewise 19,071
profiled allocations consume only 43 ms. Persistent OpenMP and reusable scratch
storage are therefore rejected until a future profile shows a larger ceiling.

## 3. Cache shared prompt-prefix K/V

Implemented for strict builds as a deliberately small one-entry cache. `eval`
tokenizes the prefix through the start of the user turn, verifies an exact
schema string and token-id match, and retains immutable per-layer K/V. A miss
runs ordinary monolithic prefill and copies its exact prefix K/V; a hit performs
causal suffix prefill at the original absolute positions. `decide` remains
uncached.

The prompt places the system text, questions, and criteria before the user
state. Rows in the same workflow reuse that prefix while changing only the
state. Jev Bush currently recomputes the complete prefix for every row and
frees its K/V after the request.

The initial implementation intentionally does not add an LRU, eviction policy,
persistent cache, hashes, or arbitrary prefix matching. Its key is:

- model and tokenizer identity;
- the tokenized prefix through the start of the user turn;
- inference settings that affect hidden states.

On a hit, only the unique state suffix is processed causally using the cached
per-layer prefix K/V. A schema change replaces the sole entry.

The attention code currently treats cache reuse as answer-canvas decoding.
Split those concepts into explicit modes:

1. causal prefill from position zero;
2. causal suffix prefill with existing K/V;
3. bidirectional answer-canvas decoding with immutable encoder K/V.

FP32 K/V is large: the model's mixed attention layout requires roughly 440 KB
per cached prefix token. A 500-token entry is therefore about 220 MB. Bound the
cache by bytes. The one-entry design bounds growth without a cache framework.

An eight-row repeated-schema regression is byte-identical to eight independent
monolithic strict-build runs, including every printed probability. Its 380-token
prefix reduced warm prefill to 102--124 evaluated tokens. Warm total latency
fell from 5.77--6.10 seconds to 2.19--2.36 seconds on the 32-thread Threadripper
9980X, a 2.5--2.7x improvement. The cold row remains monolithic and pays a
small copy/allocation cost.

A 256-row repeated-document run isolates amortization while holding the schema,
document, five decisions, and four-read policy fixed. All 256 answer objects
are byte-identical. Times include the cold cache construction row:

| documents/schema | ms/document | documents/s | predicates/s | prefill tokens evaluated |
|---:|---:|---:|---:|---:|
| 1 | 6291 | 0.159 | 0.795 | 486 |
| 2 | 4292 | 0.233 | 1.165 | 592 |
| 4 | 3250 | 0.308 | 1.538 | 804 |
| 8 | 2717 | 0.368 | 1.840 | 1228 |
| 16 | 2442 | 0.410 | 2.048 | 2076 |
| 64 | 2257 | 0.443 | 2.215 | 7164 |
| 256 | 2214 | 0.452 | 2.259 | 27516 |

The cold row took 6.291 seconds, including 4.919 seconds of prefill. Across
the 255 warm hits, mean total latency was 2.198 seconds and mean suffix prefill
was 0.912 seconds. Cache construction is therefore amortized quickly and the
steady-state rate is about 2.28 predicates/s for this five-decision workload.

Fast-math prefix reuse is disabled. Testing found that prompt splitting changes
temporary-buffer shape and therefore fast-math rounding; later NVFP4 activation
rounding amplified the initially tiny difference. Strict compilation produced
byte-identical cached and monolithic results. Approximate fast-math caching
would violate the execution-only optimization contract. (Superseded: fast-math
builds now reuse the prefix as an approximation of the same kind as fast math;
strict caching remains exact. See "A fast CUDA mode" below.)

The attainable speedup is limited by the answer-canvas work that remains. Use
the measured prefill fraction and shared-prefix fraction to predict the ceiling
before implementation:

```text
new time = old time - cached prefill time
speedup  = old time / new time
```

## 4. Microbatch independent rows

Implemented as an opt-in strict-build experiment. `JB_MICROBATCH=1..16`
groups consecutive exact-prefix hits with compatible schemas. Suffixes are
padded only in the execution matrix; attention uses independent document
boundaries, absolute positions, and useful lengths. Answer canvases likewise
remain isolated. Dense projections operate over the flattened batch and MoE
routing buckets selected tokens by expert across every document.

Measure:

- rows and decisions per second;
- mean and p95 latency per row;
- peak private memory and total resident memory;
- scaling from one to two processes;
- interaction with prefix-cache hits.

Five repeated batches at each size on the 32-thread Threadripper 9980X gave:

| batch | batch latency ms | p95 ms | docs/s | predicates/s | scaling vs B=1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2200 | -- | 0.455 | 2.273 | 1.000x |
| 2 | 4057 | 4084 | 0.493 | 2.465 | 1.085x |
| 4 | 7992 | 8108 | **0.501** | **2.503** | **1.101x** |
| 8 | 16128 | 16164 | 0.496 | 2.480 | 1.091x |

All repeated-document outputs were byte-identical to B=1. A separate B=4
run over eight different OpenJev documents was also 8/8 byte-identical to the
sequential strict reference. B=4 is the measured sweet spot, but the 10.1%
gain decisively rejects the hoped-for multi-x improvement and does not cross
the 10 predicates/s research target.

A final Linux PMU sweep closes this optimization hypothesis more directly.
The counters below are steady-state interval means from the strict AVX-512
build on the 64-core Threadripper 9980X, using the same 486-token cached-prefix
workload. Values reported by `perf` are scaled for counter multiplexing. The
DRAM figure is an estimate of 64 bytes per
`ls_any_fills_from_sys.dram_io_all` event, not a memory-controller byte count:

| batch | IPC | retired FP | retired MAC FP | cache miss ratio | estimated DRAM fills |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.607 | 738 GFLOP/s | 681 GFLOP/s | 3.37% | 8.70 GB/s |
| 4 | 1.391 | 787 GFLOP/s | 735 GFLOP/s | 5.29% | 5.31 GB/s |

B=4 increases retired floating-point work per second while estimated DRAM
fills fall and IPC declines. That is strong evidence that execution/compute,
not sustained DRAM bandwidth, is the remaining limit. Package-energy counters
were unavailable on this machine, so this is deliberately not presented as a
power-efficiency result.

The reason is visible in the timings. B=4 suffix prefill took 3.72 seconds
versus about 4 x 0.91 seconds sequential, while decode took 4.24 seconds versus
about 4 x 1.25 seconds. Existing kernels already process token matrices and
the AVX-512 NVFP4 path is compute-bound; adding documents supplies little new
weight-traffic amortization. Keep microbatching opt-in rather than paying its
latency and memory cost by default.

## 5. Reuse request-independent work

Cache or precompute the inexpensive control-plane work only after measuring
wall time separately from the current inference timer, which begins after
tokenization and template construction:

- tokenized system prompts for repeated question schemas;
- answer-template tokens, label slots, and candidate token ids;
- parsed and validated question metadata;
- reusable activation and scratch buffers sized to the largest seen request.

Exact-result and semantic memoization are out of scope: neither accelerates the
unique public evaluation rows, and both add state unrelated to model execution.

## 6. Improve the CPU kernels

Once profiles identify the expensive operations:

- preserve native packed NVFP4 weights; test only bounded metadata or scale
  preprocessing whose memory cost and end-to-end benefit are measured;
- tile over several output rows and several tokens to reuse activation and
  weight loads;
- extend the existing packed-activation and two-output-row NVFP4 tiles only
  where profiles demonstrate additional reuse;
- replace repeated inner OpenMP regions with a persistent parallel region or
  lightweight worker pool;
- retain large scratch buffers instead of allocating them for every layer and
  expert.

Keep the scalar kernels as the correctness reference. Validate optimized
primitives against them and against the existing PyTorch checks.

### Machines without AVX-512

AVX2-only x86 now has dedicated BF16, NVFP4, RMS-normalization, and attention-
dot kernels, and little-endian AArch64 has the NEON tier described below. The
earlier portable baseline on a Ryzen 9 5950X (Zen 3, 32 threads,
`-O3 -march=native -ffast-math -fopenmp`, 35 GB/s measured streaming read):

| Kernel | Tokens | GFLOP/s | Weight GB/s |
|---|---:|---:|---:|
| BF16 matmul, 4096 x 2816 | 1 | 26 | 26 |
| | 8 | 129 | 16 |
| | 64 | 85 | 1.3 |
| | 256 | 74 | 0.3 |
| NVFP4 experts, 128 x 704 x 2816 | 1 | 9 | 2.5 |
| | 4 | 16 | 1.1 |
| | 16 | 24 | 0.4 |
| | 64 | 28 | 0.1 |

Single-token BF16 already runs near memory bandwidth, though the 23 MB matrix
partly fits the 64 MB L3, so that figure may be cache-assisted. Every multi-token case is
compute-bound far below the CPU's roughly 1.9 TFLOP/s FP32 FMA peak, and the
NVFP4 expert path, the largest share of model time in the profiles above,
streams weights at under a tenth of the available bandwidth even for one token.
Strict output is deterministic for each selected backend. AVX2 uses its own
FP32 reduction order and is therefore not promised to be byte-identical to the
reference; the NEON and strict AVX-512 tiers are.

### AVX-512: exact row-lane execution

Strict AVX-512 now transforms matrices once at load. BF16 is stored as
`[row-tile][column][16 rows]`; NVFP4 keeps eight packed bytes per column for
16 rows and expands E4M3 block scales to 16 FP32 lane scales. A vector lane
therefore owns one output row, and separate multiply and add instructions
reproduce the scalar accumulation order. Fast-math keeps the original compact
weights and horizontal-reduction kernels.

Randomized transformed-kernel fixtures are bit-identical to the scalar BF16
and NVFP4 references. On one real 486-token row, all five answer objects also
matched the scalar build exactly (zero total variation and zero maximum
probability difference). On the 9980X at 32 threads, strict transformed BF16
reached 3.55 TFLOP/s at 256 tokens and NVFP4 reached 2.26 TFLOP/s at 64 tokens.
The complete row initially took 3.16 s internally versus 3.07 s for the previous
non-exact strict AVX-512 path, while scalar took 37.94 s. Transposing E4M3 scale
bytes instead of expanding every scale to 16 FP32 lanes reduced peak RSS from
41.4 GB to 37.2 GB. The kernel decodes those bytes with AVX-512 integer bit
construction. It remained bit-identical, cost about 12% at one token and less
than 2% at 16--64 tokens in the isolated NVFP4 benchmark, and took 3.23 s for
the complete measured row. The smaller exact image is retained.

### NEON: bit-identical to the reference

The NEON tier puts one output row in each vector lane, and each lane accumulates
in the reference kernels' column order with a separate multiply and add. Strict
NEON builds are therefore bit-identical to the reference kernels, a property the
self-test checks for every kernel. BF16 and NVFP4 matrices and RMS scaling are
vectorized; NVFP4 activation quantization uses the shared parallel quantizer.
Attention dots put one key in each double lane, eight keys at a time, so each
key's sequential double sum keeps its order. `-DJB_SCALAR` builds the reference
on any CPU.

Measured on an NVIDIA DGX Spark (GB10, 10 Cortex-X925 + 10 Cortex-A725 cores,
20 threads, gcc 13.3, `-O3 -march=native -std=c11 -fopenmp`) with the NVFP4
checkpoint at revision `ec4ff3d`, over 40 dataset rows (every tenth row, 10 per
workflow, 200 decisions, one read): all 200 answer objects are byte-identical
between the NEON and `-DJB_SCALAR` builds. NEON took 660 s against 3,325 s for
the reference (16.3 against 81.6 s per row after the first), 5.0x faster. After
the NVFP4 decoding work below, NEON takes 626 s against 3,322 s (15.4 against
81.5 s per row after the first), 5.3x faster, still byte-identical on all 200.
With the batched attention dots below, NEON takes 608 s, 5.5x faster than the
reference, again byte-identical on all 200. With the fused expert kernel and
dynamic scheduling below, NEON takes 275 s against 3,353 s for the reference,
12.2x faster, byte-identical on all 200; the reference's own answers are also
unchanged.

Fast-math is not a parity configuration on ARM either: against the strict
reference, fast-math NEON agrees on 93.5% of argmaxes (mean total variation
0.075) and fast-math reference on 92.0% (0.072). It is also slower on this
workload, because it disables prefix reuse. On ARM the strict NEON build is both
exact and the fastest.

A profiled row spends about half its time in NVFP4 expert matrices, 26% in
attention (10% projections, 9% dots and softmax), 9% in the dense feed-forward
layer, and 12% in GELU and activation quantization. NVFP4 decoding therefore
reads each 16-weight block for four rows with one table lookup per lane group,
from a table of signed E2M1 floats that keeps -0.0; decodes the four rows'
E4M3 scales with vector integer operations; and loads each block's activations
once, multiplying by lane. The per-row order of operations is unchanged. On one
X925 core, single-token NVFP4 rose from 9.6 to 10.8 GFLOP/s; on 20 threads from
56 to 68. End to end, the 40 rows went from 654 to 626 s at the same binding.
Without fused multiply-adds, which bit-identity rules out, each weight pair
costs five vector operations, so multi-token NVFP4 runs at about 70% of what
exact arithmetic allows.

Attention scores go through a batched `dots` kernel operation, which takes one
query and a run of keys. On NEON, eight keys share the vector registers, one
per double lane; a product of two floats is exact in double, so each lane's sum
matches the reference bit for bit. The softmax also keeps each exponential
instead of computing it twice, which changes no value on any backend. Over
eight profiled rows, attention scoring went from 6.8 to 5.8 s, and the 40 rows
went from 626 to 608 s. Scoring is only about 8% of a row, so the projections
around it, not the dots, dominate attention.

Fused multiply-adds for fast-math NEON builds, including summing each NVFP4
block before scaling it, were measured and rejected: the 40 rows went from 915
to 901 s (1.5%), NVFP4 matrices improved about 4%, and GCC's fast-math code was
already close. The strict/fast split was not worth that. CI still builds and
self-tests fast-math NEON.

Each expert's gate and up products and its GELU run as one `nvfp4_gated`
kernel operation. The reference and x86 backends keep the three steps; NEON
computes gate and up for the same four rows and applies GELU in the same
parallel loop, so GELU no longer runs serially for experts with few tokens.
Every value is the one the separate steps produce. The 40 rows went from 608 to
591 s.

Scheduling mattered far more. The kernels' row blocks, the attention and
per-token loops, and the activation quantizer's blocks were split evenly
across threads, so on GB10 every such region waited for the A725 cores, whose
SIMD is much weaker than the X925's. Dynamic scheduling lets the fast cores take
more of the work; iterations are independent, so results are unchanged. Over
eight profiled rows the total fell from 74 to 33 s: expert products from 37 to
14 s, activation quantization from 6.1 to 1.5 s, attention from 20 to 12 s, and
the dense layer from 7.1 to 3.6 s. The 40 rows went from 591 to 275 s. Every
loop dynamic with chunk 1 was slower (37 s over eight rows), because
per-element loops pay for it; those stay static. The AVX2 and AVX-512 kernels
keep static schedules until someone measures dynamic ones on x86.

### CUDA: exact NVFP4 experts

`-DJB_CUDA` adds an accelerator for the NVFP4 expert matrices, still inside
`jb.c`: the CUDA driver and NVRTC are loaded with `dlopen`, and the kernel is
compiled from a source string at first use with `--fmad=false`, so builds need
neither `nvcc` nor CUDA headers. Expert weights are uploaded at model load, all
or none. Each layer's routed tokens are gathered by expert, and gate, up and
down each run as one grouped launch: a block computes 32 rows for up to eight
tokens of one expert from weights, scales and activations staged in shared
memory, and each thread sums one row and token in the reference's order. GELU,
activation quantization, attention and the dense layer stay on the CPU, whose
`tanhf` and `expf` the reference uses.

On the DGX Spark the 40 rows take 196 s against 275 s for NEON alone and 3,353 s
for the reference, 17.1x faster, byte-identical on all 200 answers. Over eight
profiled rows, experts fell from 15.6 to 5.0 s and the total from 32.5 to 22.0
s. Two earlier designs were slower than the CPU and dropped: one launch per
expert (about 92,000 launches over eight rows, each too small to fill the GPU,
with OpenMP threads sleeping between them), and a grouped launch in which every
thread read its weight row from memory (experts 13.7 s). Attention, now 54% of
a row, is the next cost; it stays on the CPU because exact GPU `expf` would have
to reproduce glibc's.

The kernel now gives each thread one row and every token of its tile, one
register sum per token, so each weight is decoded once per tile; weights are
staged with 8-byte coalesced loads in blocks of 128 rows. GELU between the
products takes dynamic chunks. Over eight profiled rows, experts fell from 5.3
to 3.6 s, and GPU kernels account for about 1.5 s of that at roughly 45% of
memory bandwidth; the rest is copies, GELU, activation quantization and the
serial gather. Block rows from 64 to 256 and tiles of 4 to 16 tokens were
within 10% of each other. Parallelizing the gather and scatter was slower (0.47
to 0.68 s). The 40 rows went from 196 to 186 s, byte-identical on all 200.

Kernel throughput against the reference on the same machine (`--bench-kernels`,
20 threads, `OMP_PROC_BIND=close`): BF16 2.3x at one token, 3.5x at 64, and
11x at 256; NVFP4 1.9x at one token and 4.5x at 64. This microbenchmark is
sensitive to thread placement on GB10, whose firmware reports a fast and a slow
core as one "core": `OMP_PLACES=cores` made single-token NVFP4 about five times
slower there. End-to-end rows are not: all 40 took within 1% under either
placement, for both builds.

Already completed: packed activation swizzling, two-output-row NVFP4 expert
tiles, and routed-token grouping by expert. BF16 dot-product instructions were
2.8% slower and introduced measurable drift; expert-parallel scheduling was
about 12% slower. Keep both rejected experiments out of the hot path unless a
new profile changes their economics.

### Portable expf and tanhf

The CUDA path keeps GELU and softmax on the CPU because the reference used the
C library's `expf` and `tanhf`, which no GPU reproduces and which also differ
between C libraries. Strict builds now use `dg_expf` and `dg_tanhf`, Cephes
forms built from IEEE additions, multiplications and divisions in a fixed
order: within 1 ulp and 1.33 ulp of double `exp` and `tanh` over 2.7 billion
inputs. GCC, Clang and MSVC on x86-64 and GCC on AArch64 produce the same bits,
and the self-test checks a hash of them. Fast-math builds keep the C library's
functions: fast math reassociates the range reduction and flushes subnormals,
and those builds make no bit-identity claim.

The change moves the reference itself. On the 40 parity rows the new reference
changed 6.5% of argmaxes against the old one, as much as fast math does; NEON
and CUDA match the new reference on all 200 answers. On all 2,000 decisions it
scores 65.95% accuracy and log loss 1.4296, against 65.85% and 1.4382 with the
C library's functions on the same code and machine: the difference is
rounding noise that this model amplifies, not quality. The next step is exact
GELU and softmax on the GPU with the same functions.

### CUDA layers, phase 1

The accelerator now runs whole layers, not only experts. The accelerator
table gained device memory and one operation per layer step (BF16 product,
RMS norm with an optional factor or residual, GELU, NVFP4 quantization, row
gather, and the MoE tail with the layer scalar), and `dg_layer_accel` issues
them in `dg_layer`'s order; the CUDA block holds only kernels and memory.
Each session owns its device activations, so concurrent sessions share
nothing but the uploaded weights. Two steps stay on the host: attention's
middle, whose Q, K and V come down and whose output goes back up, and the
router's top-k and softmax, which use the C library's double `exp`.

On the 40 parity rows the layers take 84 s, against 204 s with GPU experts
alone, 284 s on NEON and 3,318 s for the reference: 40x the reference, and
byte-identical to it on all 200 answers. GPU kernel time over two rows is
about 1.4 s: BF16 products 0.54 s, NVFP4 products 0.50 s, RMS norms 0.26 s,
the MoE tail 0.09 s. Two fixes found by profiling each launch got the BF16
products there. Staging loops with run-time trip counts had serialized every
global load; fixed trip counts cut a launch from 1.34 to 0.57 ms. Padding
shared memory against bank conflicts, by contrast, changed nothing.

The RMS norms are bound by GB10's FP64 latency. The exact sum of squares is a
chain of dependent double additions, about 80 ns each; unrolling it or
fitting more tokens per SM did not help, and more tokens made it slower. Over
eight rows the host's share is now attention's middle, about 7 s of 13;
running it on the GPU, with the K/V cache in device memory, is phase 2.

### CUDA attention, phase 2

Attention now runs on the GPU too: Q/K/V head norms with the RMS kernel,
rotary embedding from tables the host computes with the C library's `cosf`
and `sinf`, and one block per token and head for the rest. Each thread
scores one key with a double sum in index order, as `dg_dot_ref`; a tree
finds the maximum, which order cannot change; `dg_expf` runs per key; one
thread sums the denominator in key order; and each thread sums one output
value over the keys in order. The K/V cache stays on the host, and each
call uploads its distinct caches' rows, once per layer; K and V come back
only for a K/V output. A self-test checks the rotary embedding and
attention against the reference's loops bit for bit, with segments sharing
a cache and one without, in canvas and causal modes.

Profiling the first version found model load dominating a short run: the
upload made 23,576 device allocations, which took 12.9 s, and freeing them
another 3.6 s. Every weight now goes into one allocation, placed in two
passes, and one row with model load went from 23 to 11.5 s.

With other jobs on the machine, phase 1 took 111 s for the 40 parity rows
and phase 2 72 to 81 s, back to back, both byte-identical on all 200
answers. Over eight profiled rows host-side time fell from 16.0 to 10.0 s.
What remains per layer is synchronization: the router's logits come down,
and each attention call uploads the cached rows. A device-resident K/V cache
and the router on the device are the next steps.

### FP64 throughput and idle time

A timeline of two rows on a quiet machine showed the GPU busy 83% of its
window, so synchronization could win at most 17%; the kernels were the rest,
and the exact FP64 sums a large part of them. GB10's FP64 turned out to be
bound by throughput, not latency: computing a row's squares in parallel,
each exact in double, and leaving only the additions to one thread cut an
RMS launch from 318 to 141 us. The MoE tail's two norms do the same, and
attention converts its query head to double once instead of once per key.
GPU time for two rows fell from 2.07 to 1.72 s, with every result unchanged.

The activations now stay on the device for a whole forward pass, and
attention uploads its rotary tables, segment tables and cached rows before
queueing the projections. The remaining idle time, about 18% of the window,
is in three places: the router's round trip (top-k and grouping on the host,
166 ms over two rows), prefill's K/V output (downloads and the host's cache
copy, 158 ms), and the host's work between forward passes. The first needs
the router, its double `exp`, and the expert grouping on the device; the
second a device-resident K/V cache.

The 40 parity rows take 51.5 s, against 60.9 s before, back to back on a
quiet machine: 64x the reference, byte-identical on all 200 answers. The
BF16 and NVFP4 products are now 60% of GPU time.

### Matrix products and attention scores

Over all 40 rows, not two, attention was the largest GPU cost: each row's
suffix prefill runs 250 to 670 tokens against 400 to 900 keys, and the exact
scores are FP64-bound. Three changes, each checked by kernel time on eight
rows rather than by end-to-end time on a shared machine:

- Scores read each layer's keys transposed once into a [kv head, dimension,
  key] scratch, so a warp's threads, which own consecutive keys, read
  consecutive addresses; and each product `(double)q * k`, exact because two
  24-bit significands fit in 48 bits, is built with integer operations,
  leaving only the addition on GB10's FP64 pipe. Attention fell from 1.34 to
  0.83 s over eight rows. Double keys, with an FP64 multiply, reached only
  0.97 s. A standalone test matched the integer product to the FP64 multiply
  on 16.7 million random pairs and every edge case, including zeros,
  subnormals, infinities and NaNs.
- BF16 products with a multiple of 64 columns, which every model matrix has,
  prefetch the next chunk into registers while computing the current one,
  and choose between 128 rows by 32 tokens a block and, when that leaves the
  GPU short of blocks, 32 rows by 32 tokens with four warps splitting the
  tokens, so weights are still read once. 32-row blocks of 8 tokens, tried
  first, re-read the weights for every tile and were slower. BF16 fell from
  1.53 to 1.27 s.
- Prefetching in the NVFP4 kernel made it slower, 1.33 to 2.40 s; with
  sixteen weight pairs, scales and activations held per thread it likely
  lost occupancy. Integer products for the RMS and MoE-tail squares changed
  nothing: those are bound by their serial additions.

GPU time over eight rows fell from 4.84 to 4.08 s, and the 40 parity rows
from 52.4 to 45.4 s back to back: 73x the reference, byte-identical on all
200 answers. The NVFP4 expert products, at 1.33 s, are now the largest cost.

### NVFP4 expert products

The NVFP4 kernel turned out to be bound by issued instructions, not by
memory: tiles of 16 or 32 tokens, which read each expert's weights fewer
times, were slower than tiles of 8 (1.69 and 1.73 s against 1.33 s over
eight rows). Decode routes about two tokens to each expert, and every thread
ran all eight token slots, the empty ones predicated off but still issued.
A tile's count is the same for every thread of its block, so the block now
branches, without divergence, to a loop of 1, 2, 4 or 8 slots with no
predicates; slots past the count read stale activations and are never
stored. NVFP4 time fell from 1.33 to 1.08 s, GPU time from 4.07 to 3.81 s,
and the 40 parity rows from 45.3 to 43.9 s back to back, byte-identical on
all 200 answers: 76x the reference. The selftest gained a two-token shape.

### Routing on the GPU and a deferred K/V output

Two waits per layer remained: the router's logits came down for the host's
top-k, softmax and grouping, and a prefill's K and V came down for the K/V
output. Both are gone.

The router's softmax used the C library's double `exp`. Strict builds now use
`dg_exp`, fdlibm's `__ieee754_exp`, within 1 ulp; GCC, Clang and MSVC on
x86-64 and GCC on AArch64 produce the same bits, and the router rounds its
weights to float, so the reference did not move: the 40 parity rows were
byte-identical to the previous reference. A routing kernel runs
`dg_route_token` with a block per token, each thread computing one expert's
exponential and one thread the top-k and the ordered sums; a grouping kernel,
with a thread per expert, writes the routed rows in the host loop's order and
the NVFP4 tiles, whose count stays in device memory. Each layer's expert
matrix tables are built once, inside the weights allocation, and the NVFP4
kernel launches for the most tiles the rows can make. Selftests check routing
and grouping against the reference's loops, ties included.

A prefill's K/V output now copies each layer's K and V to device staging; the
cached rows, host memory already, are copied at once. After the last layer
one copy brings the staging to page-locked host memory and the host copies
each layer's rows into the cache. Copying straight from staging into the
freshly allocated cache was slower than the per-layer downloads it replaced
(1.7 s of copies over eight rows): pageable destinations go through the
driver's bounce buffers.

Over eight rows the kernel window fell from 5.04 to 4.26 s and idle time from
1.2 to 0.4 s; the 40 parity rows take 41.3 s against 43.9 s back to back,
byte-identical on all 200 answers: 80x the reference. The K/V cache itself
still lives on the host, and each attention call uploads its cached rows.

### Where the exact GPU path stands

Measured over eight rows on a quiet DGX Spark, the GPU is busy 90% of the
kernel window (3.85 of 4.26 s), so what remains is the kernels, about a
second each for attention, the BF16 products and the NVFP4 products.
Rejected or deferred, so nobody retries them blind:

- A device-resident K/V cache. The cache uploads moved 3.07 GB over eight
  rows but took 0.058 s of GPU time, and the host spent 0.068 s in them while
  the GPU waited: GB10's shared memory makes them cheap. At about 2% it is
  not worth keeping device copies of the prefix cache in step with the
  host's across rows, sessions and error unwinding.
- Paired `float2` reads of the staged activations in the BF16 and NVFP4
  kernels: no change, as the compiler already merged the loads.
- Packed FP32 arithmetic (`__fmul2_rn`, `__fadd2_rn`). GB10, compute
  capability 12.1, has no packed FP32 instructions: the intrinsics compile
  to scalar operations, and their multiply and add were fused into an FMA
  even with `--fmad=false`, differing from the separate operations on 2.2%
  of 16.7 million random inputs. It cannot be used while results must be
  exact.

What bounds the kernels: removing the arithmetic from the largest BF16
launch cut it from 1,970 to 385 us, so about 80% of the BF16 (and, by the
same structure, NVFP4) time is FP32 multiplies and adds, issued at roughly
half the instruction rate; fused multiply-adds would halve it but round
differently. Attention is bound by the FP64 additions of the exact score
sums, whose count the reference's order fixes; the other FP64 work is
already off that pipe.

For comparison, the full benchmark of 400 rows takes 490 s on the Spark's
exact GPU path, model load included, against about 1,100 s on the 64-core
Threadripper's AVX-512 fast-math build with two 32-thread workers. (A
correction: the Spark ran automatic reads, four for 366 rows, and the
Threadripper one read a row, so the Spark did more work.) Against
the GPU's own capability the exact path is roughly an order of magnitude
slower than a non-exact engine: it gives up tensor cores, split sums and
FP32 scores to stay bit-identical to the reference. Large further gains
would need a fast accelerator mode, as fast math is for the CPU, that
gives up that identity.

### A fast CUDA mode

Fast-math builds (`-ffast-math -DJB_CUDA`) now give up bit-identity on the
GPU as they do on the CPU. On compute capability 8.0 and later:

- attention runs as one fused pass with an online softmax, its products on
  tensor cores with each operand split into a TF32 high and low part. Split
  BF16 operands (about 16 bits) were tried first and rejected: their score
  errors reached 1e-4, where 3xTF32 stays at FP32 level;
- BF16 projections run on tensor cores with each activation split into three
  BF16 parts, which makes every product exact; decode-sized launches split
  the columns across warps, as 22 blocks left most of the 48 SMs idle;
- norms sum their squares across the block, still in double.

On compute capability 12.x the expert activations stay packed as NVFP4 and
the expert products run on the block-scaled FP4 tensor cores
(`mma.sync ... kind::mxf4nvf4.block_scale`, target `compute_121a`), whose
E2M1 and E4M3 products are exact.

Kernel time over eight rows on the same DGX Spark, both fast-math builds
without prefix reuse:

| kernels | exact | fast |
|---|---:|---:|
| BF16 products | 2.99 s | 0.97 s |
| NVFP4 experts | 2.56 s | 1.26 s |
| attention | 1.64 s | 0.21 s |
| norms and MoE tail | 1.23 s | 0.27 s |
| total | 8.75 s | 2.89 s |

Fast-math builds had prefix reuse disabled, which made them prefill 2.5 times
the tokens of strict builds (249,902 against 99,521 over the benchmark). They
now reuse the prefix as an approximation: on the 40 parity rows cached and
monolithic fast CUDA results agree on 96.5% of argmaxes (mean total variation
0.035), less than fast math's own distance from strict. Over the full
benchmark, the fast CUDA build takes 265 s against 776 s with the exact
kernels and no prefix reuse, and 490 s for the strict build, at 66.80%
accuracy and 1.4358 log loss.

A correction to the first version of this section, which said the FP4
kernel read expert weights at about 320 GB/s, at the memory limit. That
figure assumed each 64-token canvas touches all 128 experts; it touches
fewer. Timed alone with all 128 experts, the kernel streamed weights at only
about 115 GB/s, whether an expert had 4 tokens or 60. Its scattered 4-byte
loads, one step in flight per warp, were the limit, not re-reading an
expert's weights for each tile of eight tokens: reading them once per run of
tiles changed nothing, and neither did issuing a run's loads before its
MMAs.

Streaming each block's weights, 64 rows by 256 columns, and its run's
activations through shared memory with coalesced 16-byte loads, the next
stage's loads in flight while this one is multiplied, fixed it. Packed
activation rows are padded to 16 bytes for those loads.

| gate and up, 128 experts | before | staged |
|---|---:|---:|
| 4 tokens an expert | 1,252 us | 685 us |
| 30 tokens an expert | 1,331 us | 796 us |
| 60 tokens an expert | 1,447 us | 805 us |

That is about 200 GB/s of GB10's rated 273 GB/s. The down projection gains
less as tokens grow, which is consistent with its larger output of 2,816
floats a token, though that is not measured. Over eight rows with prefix
reuse, FP4 kernel time falls from 0.90 to 0.54 s, and the full benchmark
from 265 to 226 s, with all 2,000 answers unchanged. Decoding is now the
largest phase: 111 s of the benchmark, against 73 s of prefill.

### Batched reads

Profiling eight four-read rows by forward pass showed decoding kept the GPU
94% busy, so it was not waiting on the host: each row ran four decode passes
of 32 tokens, one per read, and each streamed nearly all the weights again.
Reads differ only in their random answer-slot tokens, and only the first
automatic read's entropy decides whether the others run, so the others now
run together as extra canvases of one decode. Strict output is byte-identical
to reading one at a time (all 200 parity answers, and all 40 answers of the
eight four-read rows). Over those eight rows, fast CUDA decoding falls from
2.42 to 1.47 s. Over the full benchmark with automatic reads, the fast CUDA
build takes 181 s instead of 226 s, decoding 69 s instead of 111 s, and the
strict CUDA build 439 s instead of 490 s.

### The entropy check's vocabulary logits

After batched reads, candidate scoring was 32 s of the fast CUDA build's
181 s, all on the CPU, and most of it the first automatic read's entropy
check: every answer slot's logits over the whole 262,144-token vocabulary,
softcapped and scanned for the top 20 on one thread. The accelerator now
holds the tied embedding (1.48 GB more device memory) and computes those
logits with its BF16 product, 7.3 ms a row on the DGX Spark and bit for bit
the reference's in strict builds. On the host, the softcap and the softmax
terms run in parallel, one value each, while the maximum, the top 20 and the
sums keep their order; the top-20 scan skips logits below its smallest
entry, which removed most of what remained. Strict output is byte-identical,
on the GPU and on NEON alone. Over the full benchmark the fast CUDA build
takes 163 s instead of 181 s, its candidate scoring 14 s instead of 32 s.

### Microbatching on the GPU

`JB_MICROBATCH` runs consecutive requests that share a schema through one
forward pass, streaming the weights once for all of them. The fast CUDA
build, full benchmark, DGX Spark:

| reads | batch | seconds | rows batched | accuracy | log loss | argmax vs unbatched |
|---|---:|---:|---:|---:|---:|---:|
| one | 1 | 115 | 0 | 66.65% | 1.5191 | |
| one | 4 | 83 | 384 | 66.05% | 1.5361 | 94.9% |
| one | 8 | 82 | 344 | 66.15% | 1.5405 | 96.1% |
| one | 16 | 108 | 80 | 66.40% | 1.5309 | 99.1% |
| automatic | 1 | 163 | 0 | 66.70% | 1.4398 | |
| automatic | 4 | 129 | 384 | 66.60% | 1.4510 | 97.0% |
| automatic | 8 | 120 | 344 | 66.40% | 1.4596 | 97.35% |
| automatic | 16 | 152 | 80 | 66.55% | 1.4396 | 99.25% |

Batches of 16 mostly exceed the 4 GiB bound on copied prefix K/V and run
sequentially. The fast kernels sum in an order that depends on how many
tokens they get, and NVFP4 activation rounding amplifies the difference, so
fast answers depend on the batch; the slightly worse scores of every batched
run may be noise, as the runs overlap heavily, but they all point the same
way. Strict CUDA with a batch of 4 stays byte-identical on the 40 parity
rows, and gains no time there, as the exact kernels are bound by arithmetic.

## 7. Make benchmark comparisons auditable

Before publishing a faster number:

- default the evaluation runner to one process; require explicit concurrency;
- store a run manifest beside resumable row outputs and reject changes to the
  dataset, executable, model, samples, or thread settings;
- fetch the declared immutable dataset revision rather than only checking the
  latest revision against a saved hash;
- distinguish billed input tokens from tokens actually processed by prefill;
- record compiler identity, flags, benchmark date, affinity, memory channels,
  model revision, and cache state;
- compare OpenJev with prefix caching both enabled and disabled when possible.

Report at least four CPU modes:

1. cold schema and cold prefix cache;
2. warm repeated schema;
3. lowest single-request latency;
4. best sustained throughput under a fixed memory budget.

## 8. Concluded Bonsai experiment

The separate Ternary Bonsai 2 27B backend experiment was removed from the
production tree. It established tokenizer compatibility and close numerical
agreement with Prism, and native AVX-512 VNNI plus OpenMP reduced a one-token
forward from 15.25 s scalar to 0.37 s on the 9980X. That did not make the model
a good Jev Bush backend: causal execution required independent predicate
continuations, a representative five-decision OpenJev row took 206.35 s, and
its smoke result was only 3/5 argmax agreement with poor calibration. Further
cross-token tiling was compute-bound and did not materially improve throughput.

The result is retained as negative evidence, not supported code. Jev Bush
therefore returns to one implementation file and one model family:
DiffusionGemma. Future CPU work targets portable AVX2 kernels while preserving
the scalar reference and existing AVX-512 behavior.

That AVX2 tier is now established. A forced `-mavx2 -mfma -mno-avx512f`
build selects independent kernels for NVFP4, BF16 matrices, RMS normalization,
and attention dots; forced scalar and native AVX-512 builds still compile and
pass the same oracle-backed self-test. Token tiling reuses each BF16 vector or
decoded NVFP4 block across several activation rows. On one 9980X thread, BF16
now reaches 17.7 GFLOP/s at one token and 53.6 GFLOP/s at 256 tokens; NVFP4
reaches 6.93 GFLOP/s at one token and 19.72 GFLOP/s at 64 tokens. The original
untiled multi-token results were 24.5 and 7.15 GFLOP/s respectively.

On the repeated-schema OpenJev row, warm strict AVX2 latency scales from 18.51
seconds at 8 threads to 4.51 seconds at 64 before token tiling. Tiling reduces
the 64-thread result to 2.94--3.00 seconds while preserving byte-identical AVX2
answers, a further 34% reduction. The unchanged AVX-512 path reaches 2.18
seconds at 48 threads, so AVX2 is now within 1.35x on this workload. A warm
64-thread profile attributes 2.28 of 2.94 seconds to MoE experts; GELU and
activation quantization total 0.91 seconds and are the clearest remaining AVX2
targets. These figures are from the 9980X with physical-core placement.

Those activation targets are now implemented without approximating either
operation. Routed GELU retains the existing `tanhf` expression and distributes
large independent activation arrays over the OpenMP team. NVFP4 QDQ retains
the scalar E2M1 rounding and scale calculation byte for byte, while distributing
independent 16-value quantization blocks. An attempted fully vectorized E2M1
path was rejected because small intermediate differences changed final
probabilities.

At 64 physical threads the warm non-profiled row now takes 2.32--2.35 seconds,
down from 2.94--3.00 seconds after matrix tiling and 4.51 seconds before it.
In the instrumented run, GELU fell from 602 to 164 ms, input QDQ from 97 to
4 ms, and hidden QDQ from 209 to 84 ms. The full answer object remains byte-
identical to the pre-optimization strict AVX2 path. At that stage this put AVX2
within about 7% of the earlier 2.18-second AVX-512 result, before remeasuring
both backends with the shared GELU change. The two ISA paths retain their own
deterministic FP reduction order.

The next AVX2 matrix pass removes the last scalar NVFP4 weight decoder from the
hot loop. `vpshufb` maps packed E2M1 nibbles to signed integer magnitudes, which
are expanded and interleaved as FP32 entirely in registers before the existing
FMA accumulation. The retained topology is one output row by four tokens: a
two-row/two-token experiment fell from about 29 to 19 GFLOP/s because AVX2 has
only 16 vector registers. NVFP4 now reaches 10.2 GFLOP/s at one token and about
29 GFLOP/s for multi-token work, versus 6.9 and 20 before register decoding.

BF16 has only one accumulator per output row and therefore benefits from a
two-row/four-token tile. It reaches 86.4 GFLOP/s at 256 tokens, up from 53.1,
while reusing each activation load across two rows. Combined warm latency is
1.98--1.99 seconds at 64 physical threads, down from 2.29 seconds immediately
before this matrix pass. The refreshed AVX-512 baseline is 1.82 seconds, leaving
AVX2 about 9% behind on this row. All retained changes preserve byte-identical
strict AVX2 answer objects; the two ISA tiers still have distinct reduction
orders.


## Expected outcome

Prefix caching should reduce repeated-schema latency, and microbatching plus
better kernels should improve sustained throughput. Neither can erase the
hardware difference: the GPU has substantially more memory bandwidth and
native low-precision tensor acceleration.

The goal is not GPU parity. The useful result is a measured account of how far
a transparent, dependency-free, model-specific CPU engine can close the gap,
which optimizations matter, and where the remaining hardware boundary lies.

## Non-goals

- Metal, Vulkan, WebGPU, or ROCm backends (the README explains why CUDA is
  the exception);
- server or HTTP mode;
- generic GGUF or arbitrary-model loading;
- model abstraction layers or plugin systems;
- approximate caches that change strict decision semantics;
- optimizing candidate projection while transformer execution dominates.

The project remains one model, one hypothesis, one C inference engine, and a
reproducible evaluation path.

## 9. CPU-native quantization gate

The first CPU-native quantization experiment is implemented in
`--bench-kernels` rather than wired into inference. It converts the existing
NVFP4 expert weights to symmetric linear INT4 in 32-value blocks, quantizes
activations to blockwise INT8, and executes an AVX2 INT16/INT32 dot-product
kernel. At model expert shapes it reaches 33.7 GFLOP/s at one token and 53.2
GFLOP/s at 64 tokens, versus 10.2 and 29.1 for the strict NVFP4 kernel.

That speed does not pass the numerical gate. Requantizing weights that are
already E2M1 produces 14.46% relative matrix-output RMSE and a 14.18 maximum
absolute error on the deterministic fixture. Building a roughly model-sized
`.jbc` cache or running OpenJev with this representation is therefore rejected:
the primitive mismatch is already substantially beyond an acceptable inference
perturbation.

A second kernel preserves the original packed E2M1 weight codes and activation
codes exactly, using integer dot products with the original block scales. It
matches strict NVFP4 to 7.16e-7 relative RMSE, with differences attributable to
accumulation order, but reaches only 29.9 GFLOP/s at 64 tokens and is slower at
one token. The current register-decoded NVFP4 kernel is already near the useful
AVX2 ceiling for this representation.

The result keeps NVFP4 plus BF16 as canonical CPU execution formats for this
model. A persistent transformed cache is deferred until a representation shows
both meaningful kernel speedup and acceptable primitive accuracy; storage
compactness alone is not sufficient.

## 10. v0 freeze

The v0 execution architecture is frozen after the exact AVX-512 image and its
compact-scale pass. Scalar defines the arithmetic; strict NEON, AVX-512 and
CUDA reproduce it; AVX2 and fast builds are deterministic performance tiers.
The CLI and `jb.h` expose the same typed decision engine, sessions own reusable
K/V and scratch state, and the model is immutable and shareable.

Release validation consists of hosted compiler, sanitizer, fuzz, hardening and
kernel jobs plus the manual model-backed golden check on the benchmark rigs.
The golden manifest pins the request, tokenizer, configuration and model-shard
hashes; its typed-library run must reproduce the JSON API probabilities. The
40-row release gate additionally covers cached, monolithic, microbatched,
sequential-canvas and, when supplied, strict CUDA execution. Every published
measurement carries the manifest emitted by `tools/capture_run_manifest.py`.
Future arithmetic or layout changes need a profile demonstrating a dominant
cost, bitwise strict fixtures, the model golden hashes, and refreshed benchmark
numbers. New models, generic formats, servers, additional GPU APIs and broad
framework abstractions remain outside v0 rather than accumulating behind the
frozen interface. v0 promises source compatibility when application and
library use the same header; a stable cross-release binary ABI and runtime CPU
multiversioning are post-v0 library work, not claims made by this freeze.
