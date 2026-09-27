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

The current 64-token answer canvas fits at most 17 compact predicates; 18 are
rejected. The next implementation milestone is one shared causal prefill
followed by a microbatch of independent answer canvases. Benchmark canvas
batches of 1, 2, 4, 8, and 16 and require semantic equality with the equivalent
independently evaluated predicates.

## 3. Cache shared prompt-prefix K/V

First measure the exact shared-token prefix and potential hit rate for every
benchmark row. Implement this only if the measured prefill fraction and shared
prefixes predict a material end-to-end gain.

The prompt places the system text, questions, and criteria before the user
state. Rows in the same workflow reuse that prefix while changing only the
state. Jev Bush currently recomputes the complete prefix for every row and
frees its K/V after the request.

Add a small exact-match LRU cache keyed by:

- model and tokenizer identity;
- the tokenized prefix through the start of the user turn;
- inference settings that affect hidden states.

On a hit, process only the unique state suffix causally, using the cached
per-layer prefix K/V. Start with a one-entry cache, then test whether one entry
per recurring question schema is worth the memory.

The attention code currently treats cache reuse as answer-canvas decoding.
Split those concepts into explicit modes:

1. causal prefill from position zero;
2. causal suffix prefill with existing K/V;
3. bidirectional answer-canvas decoding with immutable encoder K/V.

FP32 K/V is large: the model's mixed attention layout requires roughly 440 KB
per cached prefix token. A 500-token entry is therefore about 220 MB. Bound the
cache by bytes, expose its hit rate, and do not allow unbounded schema growth.

The attainable speedup is limited by the answer-canvas work that remains. Use
the measured prefill fraction and shared-prefix fraction to predict the ceiling
before implementation:

```text
new time = old time - cached prefill time
speedup  = old time / new time
```

## 4. Microbatch independent rows

This is the highest-priority throughput experiment.

Two processes currently stream the same weights independently. Add a small
in-process microbatch so each weight tile can serve activations from several
requests before leaving cache. Begin with batches of two and four and preserve
each request's independent prompt length, seed, canvas, and output.

Measure:

- rows and decisions per second;
- mean and p95 latency per row;
- peak private memory and total resident memory;
- scaling from one to two processes;
- interaction with prefix-cache hits.

Batching is allowed to improve throughput while leaving latency unchanged or
slightly worse. Do not present amortized decision time as request latency.

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

Only the AVX-512 build has vectorized kernels; everything else, including
Apple Silicon, Graviton, and AVX2-only x86, runs the portable reference
kernels. `jb --bench-kernels` on a Ryzen 9 5950X (Zen 3, AVX2, 32 threads,
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
AVX2 and NEON versions of the BF16 and NVFP4 kernels are the main opportunity.
To keep strict builds byte-identical across ISAs, each should reproduce the
AVX-512 kernels' 16-lane accumulation and reduction order rather than choose
its own.

Already completed: packed activation swizzling, two-output-row NVFP4 expert
tiles, and routed-token grouping by expert. BF16 dot-product instructions were
2.8% slower and introduced measurable drift; expert-parallel scheduling was
about 12% slower. Keep both rejected experiments out of the hot path unless a
new profile changes their economics.

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

## Expected outcome

Prefix caching should reduce repeated-schema latency, and microbatching plus
better kernels should improve sustained throughput. Neither can erase the
hardware difference: the GPU has substantially more memory bandwidth and
native low-precision tensor acceleration.

The goal is not GPU parity. The useful result is a measured account of how far
a transparent, dependency-free, model-specific CPU engine can close the gap,
which optimizations matter, and where the remaining hardware boundary lies.

## Non-goals

- CUDA, Metal, Vulkan, or WebGPU backends;
- server or HTTP mode;
- generic GGUF or arbitrary-model loading;
- model abstraction layers or plugin systems;
- approximate caches that change decision semantics;
- optimizing candidate projection while transformer execution dominates.

The project remains one model, one hypothesis, one C inference engine, and a
reproducible evaluation path.
