# Jev Bush roadmap

Jev Bush is already accuracy-competitive with OpenJev and broadly comparable
to Jev on the public typed-decisions benchmark. The next research question is
how much faster its deliberately small CPU implementation can become without
turning into a general inference framework.

The established throughput configuration uses two concurrent 32-thread workers
on a 64-core Threadripper 9980X. Each worker averages 10.11 seconds per
five-decision row, or 0.495 decisions per second, with a p95 row latency of
15.03 seconds. Together they sustain about 0.99 decisions per second. OpenJev
on an RTX PRO 6000 Blackwell averages 54.2 milliseconds per one-read row.

That is approximately a 187x single-row latency gap and, after accounting for
the two CPU workers, a 93x aggregate-throughput gap. The comparison is useful,
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

Every optimization must preserve byte-identical probabilities unless a change
is explicitly presented as a numerical experiment. Benchmark single-request
latency and multi-worker throughput separately.

## 2. Cache shared prompt-prefix K/V

This is the highest-priority latency experiment.

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

## 3. Microbatch independent rows

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

## 4. Reuse request-independent work

Cache or precompute the inexpensive control-plane work only after measuring it:

- tokenized system prompts for repeated question schemas;
- answer-template tokens, label slots, and candidate token ids;
- parsed and validated question metadata;
- reusable activation and scratch buffers sized to the largest seen request.

Add optional exact-result memoization keyed by the canonical request, model
revision, and read settings. Identical retries are deterministic and may return
immediately. Keep this cache bounded and report hits; it does not improve a
benchmark containing unique states.

Do not use approximate or semantic caching in correctness evaluations.

## 5. Improve the CPU kernels

Once profiles identify the expensive operations:

- prepack BF16 and NVFP4 weights into layouts consumed directly by the
  AVX-512 kernels;
- tile over several output rows and several tokens to reuse activation and
  weight loads;
- decode block scales and packed FP4 values once per tile rather than once per
  scalar use;
- investigate BF16 dot-product instructions where expansion cost and memory
  traffic make them beneficial;
- group routed tokens by expert once, then process active experts in parallel;
- replace repeated inner OpenMP regions with a persistent parallel region or
  lightweight worker pool;
- retain large scratch buffers instead of allocating them for every layer and
  expert.

Keep the scalar kernels as the correctness reference. Validate optimized
primitives against them and against the existing PyTorch checks.

## 6. Make benchmark comparisons auditable

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
