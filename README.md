# Jev Bush

![Jev Bush robot mascot](jev-bush-mascot.png)

> Probabilistic decisions for DiffusionGemma: on any CPU, and bit for bit the
> same on CUDA.
>
> Please clap.

Jev Bush evaluates bounded OpenJev decisions directly from DiffusionGemma's
answer-slot logits. It never generates or parses free-form text. The engine is
one C11 file, [`jb.c`](jb.c), with no runtime dependency beyond the C
standard library; OpenMP is optional.

## What this experiment establishes

Jev Bush tests whether bounded probabilistic decisions can be extracted
directly from a diffusion language model without autoregressive generation or
structured-output parsing, and whether the resulting distributions remain
comparable to the reference GPU implementation. On the pinned public benchmark,
its strongest run reaches 67.40% accuracy, 5.3 percentage points behind Jev
1.13.0's published 72.70% on the same 2,000 decisions. That establishes
comparable argmax decision quality on this benchmark, not model equivalence.

```mermaid
flowchart LR
    R[OpenJev request] --> T[tokenize / template] --> P[encoder prefill]
    P --> K[cached K/V] --> C[diffusion answer canvas]
    C --> L[candidate logits] --> S[softmax] --> O[typed probabilities]
```

## Project goals

- Be a small, readable, educational implementation of direct probabilistic
  decisions on a real diffusion language model.
- Run inference on CPUs, with the portable scalar build as the reference that
  defines every result. One optional accelerator, CUDA, reproduces that
  reference exactly in strict builds and, like the CPU, trades exactness for
  speed in fast-math builds; other GPU backends (Metal, Vulkan, WebGPU, ROCm)
  remain out of scope. Why CUDA is the exception is explained below.
- Make the CPU path correct and fast through model-specific data layouts,
  vectorization, and OpenMP without turning `jb.c` into a generic framework.
- Remain compatible with OpenJev's bounded decision semantics and public
  evaluation data.

### Why CUDA is different

Jev Bush began CPU-only, and a GPU backend was ruled out because it would have
defeated the point: a second implementation to read, a toolkit to install, and
results that differ from the code a reader studies. The CUDA path was admitted
only because it avoids all three.

- **It computes the same bits** in strict builds. It is not an alternative
  numerical method.
  The kernels are compiled with `--fmad=false`, IEEE division and square
  root, and no flush to zero. Every value is computed in the reference's order
  of operations, with the same portable `expf`, `tanhf` and `exp`. The strict
  scalar build stays the oracle. The self-test checks each GPU operation
  against its bits, and on the 40 parity rows all 200 answers are
  byte-identical. Reading the CPU code still tells you exactly what the GPU
  computes.
- **It needs nothing to build.** It needs no `nvcc`, no CUDA headers, and no
  link-time dependency. The driver and NVRTC are loaded with `dlopen` when a
  model loads, and the kernels compile from source embedded in `jb.c`.
  Default builds do not contain it at all. A `-DJB_CUDA` build without a
  driver, NVRTC or a device, or whose kernels fail to compile, runs on the
  CPU kernels.
- **It stays out of the way.** All CUDA code sits in one `#if
  defined(JB_CUDA)` block behind a small operations table, like the SIMD
  backends. The engine calls the table in the reference's order and has no
  other CUDA conditionals.
- **It is narrow.** It covers one vendor and one model's transformer layers.
  Tokenization, the prefix cache, the K/V cache, candidate scoring and
  everything else stay on the CPU. It is not the start of a GPU framework.

CUDA can make that guarantee because NVRTC exposes the controls exactness
needs: no contraction into FMAs, correctly rounded division and square root,
and denormals kept. Another GPU API would have to prove the same before it
could join, which is why the others remain out of scope.

Exactness has a price. The exact kernels give up tensor cores, split sums and
FP32 attention scores, so the GPU runs roughly an order of magnitude below a
non-exact engine on the same hardware. Even so, the full 400-row benchmark
takes 490 s on a DGX Spark's exact GPU path, against about 1,100 s for the
64-core Threadripper's AVX-512 fast-math build.

Fast-math builds give up bit-identity on the GPU as they do on the CPU, and
use the tensor cores (see [Build](#build)). The strict build stays the oracle:
fast kernels are checked against it within FP32 rounding, and the fast mode's
quality is judged on the benchmark, where its full 400-row run takes 226 s on
the same DGX Spark at 66.80% accuracy, against the strict build's 65.95%.

For production GPGPU inference, use
[OpenJev](https://github.com/razorback16/openjev) or
[SemIf](https://github.com/TheoLeeCJ/SemIf).

## Build

Linux, optimized for the current CPU with exact repeated-schema prefix reuse:

```sh
cc -O3 -march=native -std=c11 -Wall -Wextra -pedantic \
  -fopenmp jb.c -lm -o jb
```

`eval` keeps one immutable, exact-match schema-prefix K/V entry. The first row
for a schema is a cache miss and uses ordinary monolithic prefill; later exact token-prefix matches evaluate only the document-dependent
suffix. `decide` is always monolithic. Output reports `prefix_cache` as
`off`, `miss`, or `hit`, `cache_tokens`, and the number of tokens actually
evaluated as `usage.prefill_tokens`.

Strict arithmetic defines canonical Jev Bush semantics: reproducible output
must not depend on batching or temporary-buffer shape. It does not depend on
the platform either: strict builds compute `expf`, `tanhf` and the router's
double `exp` with their own portable implementations instead of the C
library's, so every strict build and the CUDA accelerator produce the same
bits, which the self-test checks with recorded hashes.

Linux, approximate highest-throughput experiment:

```sh
cc -O3 -march=native -ffast-math -std=c11 -Wall -Wextra -pedantic \
  -fopenmp jb.c -lm -o jb
```

This is the evaluated high-throughput build. `-ffast-math` changes floating-
point reduction and transcendental behavior; its full accuracy and calibration
results are reported below. Output records `"math":"fast"`; omit the flag for
strict IEEE behavior and `"math":"strict"` output. Guards against NaN and
infinity use bit-level tests, so they still hold under `-ffast-math`.

Prefix reuse is exact in strict builds: cached and monolithic execution are
byte-identical. Fast-math builds reuse the prefix too, as an approximation of
the same kind as fast math itself. Splitting the prompt changes temporary-buffer
shape and therefore fast-math rounding, and NVFP4 activation rounding can
amplify that into different probabilities, so a fast build's answer can depend
on whether the cache held the schema. On the 40 parity rows, cached and
monolithic results agree on 96.5% of argmaxes (mean total variation 0.035) for
the fast CUDA build and on 98% (0.026) for fast-math NEON, less than fast
math's own distance from the strict build. Use the strict build where results must not depend on cache
state.

Output also records `"kernels"`: `"avx512"` when the build targets AVX-512F
and AVX-512DQ, `"avx2"` when it targets AVX2 and FMA without AVX-512, and
otherwise `"scalar"`. Dispatch is compile-time and ordered AVX-512, AVX2,
scalar. `"threads"` records the OpenMP thread count the process ran with.

To build specifically for an AVX2/FMA machine while keeping the binary free of
AVX-512 instructions:

```sh
cc -O3 -mavx2 -mfma -mno-avx512f -std=c11 -Wall -Wextra -pedantic \
  -fopenmp jb.c -lm -o jb
```

The AVX2 tier covers NVFP4 expert projections, BF16 matrices, RMS normalization,
and attention dot products. The scalar reference and AVX-512 implementations
remain separately compiled and checked by the same self-test.

On little-endian AArch64 (Apple Silicon, Graviton, NVIDIA Grace and GB10) the
portable build selects the NEON tier automatically. It covers NVFP4 and BF16
matrices and RMS normalization; each vector lane computes one output row in the
reference's accumulation order, so strict NEON builds are bit-identical to the
reference kernels, which the self-test checks. Attention dots put one key in
each double lane, which keeps each key's sequential double sum in order.

On Linux, `-DJB_CUDA` adds an optional CUDA accelerator for the transformer
layers of the NVFP4 checkpoint. It needs neither `nvcc` nor CUDA headers: the
driver and NVRTC are loaded at run time and the kernels are compiled from
source on first use.

```sh
cc -O3 -march=native -std=c11 -Wall -Wextra -pedantic -fopenmp -DJB_CUDA \
  jb.c -lm -ldl -lpthread -o jb
```

The expert matrices and the layers' BF16 tensors are placed in one device
allocation when the model loads. Each layer then runs on the GPU from its
input norm to its layer scalar: the projections, Q/K norms, rotary embedding,
attention scores, softmax and value sums, the dense feed-forward layer, the
norms, GELU, activation quantization and the experts, grouped by expert. The
router's top-k and softmax and the grouping of routed tokens by expert run on
the GPU too, so routing no longer round-trips through the host. The CPU computes the rotary
tables and keeps the K/V cache: each attention call uploads its cached rows,
and a prefill's new rows come back once, after the last layer. The
kernels compile with `--fmad=false` and IEEE division and square root, and
compute every value in the reference's order, using the same portable `expf`,
`tanhf` and `exp`, so results are bit-identical to the reference. The self-test
checks each GPU operation against the reference's bits and reports
`"accelerator":"cuda"`. Without a driver, NVRTC or a device, or with
`JB_CUDA=0` in the environment, or when the device cannot hold the weights,
the build runs on the CPU kernels alone. So it does when the kernels fail to
compile or load; the library stays silent, and the CLI says why on stderr,
with NVRTC's log.

Fast-math builds keep the same layout but give up bit-identity on the GPU as
they do on the CPU:

```sh
cc -O3 -march=native -ffast-math -std=c11 -Wall -Wextra -pedantic -fopenmp \
  -DJB_CUDA jb.c -lm -ldl -lpthread -o jb
```

On GPUs with BF16 tensor cores (compute capability 8.0 or later):

- **Attention** runs as one fused pass over the keys with an online softmax.
  Its score and value products run on tensor cores, each FP32 operand split
  into a TF32 high and low part (three MMAs a product), so they stay about
  as accurate as FP32 products, but scores sum in FP32, not double.
- **BF16 projections** run on tensor cores with each activation split into
  three BF16 parts that hold all of its bits, so every product is exact and
  only the FP32 sums round. Decode-sized launches split the columns across
  warps to keep the GPU busy.
- **Norms** sum their squares across a block's threads instead of in order,
  still in double.

On compute capability 12.x, which has block-scaled FP4 tensor cores, expert
activations stay packed as NVFP4 (E2M1 values, E4M3 block scales and a global
scale), and the expert products run on those tensor cores. Their E2M1 and E4M3
products are exact, so only the FP32 sums round. That needs NVRTC's
architecture-specific target; when NVRTC rejects it, the other fast kernels
still compile. The self-test checks each fast operation against the reference
within FP32 rounding bounds, and strict builds compile and run exactly the
same kernels as before.

Over eight rows on a DGX Spark without prefix reuse, GPU kernel time falls
from 8.75 s with the exact kernels to 2.89 s. The FP4 expert kernel streams
weights and activations through shared memory with coalesced 16-byte loads,
which is why packed activation rows are padded to 16 bytes; it reads expert
weights at about 200 GB/s of GB10's rated 273 GB/s. With the fast-math prefix
reuse described above, the full 400-row benchmark takes 226 s, against 776 s
for a fast-math build with the exact kernels and no prefix reuse, and 490 s
for the strict build. Decoding the answer canvases is now the largest
phase.

Loading the driver and NVRTC with `dlopen` means the libraries found on the
usual search path (`LD_LIBRARY_PATH`, `ld.so.conf`) are the code that runs, as
with any shared library. Their entry points sit in one function-pointer table
that is filled once, under `pthread_once`, and only read afterwards; every
other dispatch table in `jb.c` is `static const`.

### C library API

[`jb.h`](jb.h) exposes opaque model and session handles, typed
decisions, typed batches, and JSON compatibility calls. Build the engine without
its CLI entry point and link it into an application:

```sh
cc -O3 -march=native -std=c11 -Wall -Wextra -pedantic -fopenmp \
  -DJB_NO_MAIN -c jb.c -o jb.o
ar rcs libjb.a jb.o
cc -O3 -march=native -std=c11 -Wall -Wextra -pedantic -fopenmp \
  examples/library.c libjb.a -lm -o jb-library-example
```

The normal `jb.c` build still contains the CLI, but the CLI is itself a client
of `jb_model_load()`, `jb_session_create_json()`, and the session JSON decision
calls. There is no separate privileged inference route.

A `jb_model` is immutable after loading and may be shared by multiple sessions.
A `jb_session` owns one worker's schema-prefix cache, inference workspace, K/V
storage, and microbatch controls. A session is deliberately single-threaded;
create one session per concurrent worker. Inputs are borrowed for the duration
of a call. Typed results use one contiguous library allocation and are released
with `jb_result_free()`. JSON buffers and JSON batch arrays are released with
`jb_free()`.

Every call returns a `jb_status`. `jb_session_last_error()` describes a
session's most recent decide call from whichever thread reads it, so it stays
correct when a session moves between threads. `jb_last_error()` describes the
calling thread's most recent failure, which covers calls without a session:
model loading and session creation.

Typed sessions copy their decision schema at creation. `jb_session_decide()`
accepts a JSON state value plus an optional stable identifier. The equivalent
`jb_session_decide_json()` accepts the complete OpenJev request shape and
returns its deterministic machine-readable result. Batch variants preserve
input order and use the existing cross-document microbatch path when the exact
schema prefix permits it. See [`examples/library.c`](examples/library.c)
for a complete typed example.

Internally, model execution uses one small `DGKernelOps` table for BF16 GEMM,
NVFP4 quantization/GEMM, RMS normalization, and attention dots. Scalar, AVX2,
AVX-512, and NEON differ only behind that boundary; inference, validation, benchmarks,
and JSON output contain no ISA dispatch branches. Selection remains compile-time
so the portable build does not require runtime CPU detection.

Weights stay memory-mapped. Each evaluation worker owns one checked,
64-byte-aligned, grow-only inference workspace. Prefill, cached suffixes,
repeated canvas reads, candidate scoring, and document microbatches reset and
reuse it; only a later request with a larger high-water requirement reallocates
it. A second grow-only worker slab holds all per-request K/V layers and
documents contiguously. A third, much smaller slab reuses the microbatch token,
descriptor, and attention-control arrays. Every operation is constrained to an
exact plan derived from its token and batch shape. The immutable schema-prefix
cache owns separate exact metadata and K/V slabs because it outlives worker
resets. This makes the ownership boundary explicit without copying model
weights or hiding persistent state in a general-purpose allocator. `JB_PROFILE`
reports `hot_alloc_calls`; after initial worker growth, an exact-prefix cache
hit should report zero.

For Linux release builds, use the hardened command below. These flags are part
of the release contract rather than an optional deployment tweak; they do not
change numerical results:

```sh
cc -O3 -march=native -ffast-math -std=c11 -Wall -Wextra -pedantic \
  -fstack-protector-strong -D_FORTIFY_SOURCE=3 -fPIE -pie \
  -Wl,-z,relro,-z,now,-z,noexecstack -fopenmp jb.c -lm -o jb
```

CI verifies that this produces PIE with stack protection, FORTIFY, full RELRO,
immediate binding, and a non-executable stack. Platform-specific development
commands below remain intentionally minimal and portable.

Portable Linux or macOS:

```sh
cc -O3 -std=c11 -Wall -Wextra -pedantic jb.c -lm -o jb
```

macOS with Homebrew LLVM/OpenMP:

```sh
$(brew --prefix llvm)/bin/clang -O3 -std=c11 -Wall -Wextra -pedantic \
  -fopenmp jb.c -lm -o jb
```

Windows with MSYS2/MinGW-w64:

```sh
gcc -O3 -std=c11 -Wall -Wextra -pedantic -fopenmp jb.c -lm -o jb.exe
```

Windows with MSVC, which implements OpenMP 2.0:

```bat
cl /O2 /std:c11 /W4 /openmp /arch:AVX2 jb.c
```

Any of these with `-DJB_SCALAR` forces the portable reference kernels whatever
the CPU supports. That build is the oracle the vector backends are checked
against.

Run the dependency-free smoke test without downloading a model:

```console
$ ./jb --selftest
{"selftest":"ok","kernels":"avx2"}
```

The selftest also checks every compute kernel, both the portable reference and
the one selected for the build, against a double-precision oracle on random
data, and reports which set it tested: `scalar`, `avx2`, or `avx512`. CI runs
it natively on x86-64 (scalar and AVX2) and ARM64, and under Intel's Software
Development Emulator for the AVX-512 kernels. `jb --bench-kernels` reports
kernel throughput at model shapes next to a memory-read baseline, for comparing
builds and machines.

`jb --check-request REQUEST.json` validates a request (JSON, fields, questions,
and prompt construction) without a model. CI runs it under AddressSanitizer
and UndefinedBehaviorSanitizer against the malformed requests in
[`tools/check_requests.sh`](tools/check_requests.sh).

[`fuzz/fuzz_json.c`](fuzz/fuzz_json.c) is a libFuzzer target for the JSON
reader and request validation. Besides sanitizer findings, it checks that
canonical JSON output re-parses to identical bytes. CI fuzzes it for two
minutes per push:

```sh
clang -g -O1 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
  fuzz/fuzz_json.c -lm -o fuzz_json
mkdir -p corpus && ./fuzz_json -max_len=16384 -dict=fuzz/json.dict corpus fuzz/corpus
```

For operation-level profiling, add `-DJB_PROFILE`. Each request then emits one
timing line to standard error for attention, dense FFN, routing, MoE experts,
expert input/hidden QDQ, gate/up/down projections, expert activation and
miscellaneous expert work, remaining FFN work, prompt composition, OpenMP
region count, total and hot-path allocation counts, allocation time, workspace
capacity, and active K/V size. Normal JSONL on standard output is unchanged.

Generate one-state/many-predicate scaling requests from any OpenJev row with:

```sh
python3 tools/make_predicate_benchmark.py request.jsonl predicates.jsonl --compact
./jb MODEL_DIR eval predicates.jsonl
```

Without `--compact`, the source row's first full predicate and rubric are
cloned. Compact mode isolates execution scaling from prompt-schema length.

## Use

```sh
./jb MODEL_DIR decide request.json
./jb MODEL_DIR eval requests.jsonl
```

Strict `eval` can microbatch consecutive exact-prefix hits in the same schema:

```sh
OMP_NUM_THREADS=32 JB_MICROBATCH=4 ./jb MODEL_DIR eval requests.jsonl
```

`JB_MICROBATCH` accepts `1..16` and defaults to `1`. Incompatible rows fall
back to sequential execution, as do groups whose K/V would exceed 4 GiB: each
document holds its own copy of the shared prefix K/V, about 440 KiB per token.
Build with `-DJB_MICROBATCH_KV_LIMIT=BYTES` to change that bound. Attention remains isolated per document and per
answer canvas, while dense projections, routing, and expert buckets span the
batch. Output records the actual `microbatch` size. Strict microbatched
results are byte-identical to sequential ones; fast-math ones may differ as
cached ones do.

[`examples/request.json`](examples/request.json) is a small request spanning
all three decision types. Its response has the structure shown in
[`examples/response-shape.json`](examples/response-shape.json); probabilities
and timings depend on the checkpoint and read policy.

`MODEL_DIR` is either the public `google/diffusion-gemma-26b-it` BF16
checkpoint or NVIDIA's NVFP4 variant. It must contain `config.json`,
`model.safetensors.index.json`, `tokenizer.json`, and all referenced
safetensor shards.

The input is OpenJev's System One request shape:

```json
{
  "id": "case-1",
  "state": "A customer cannot sign in.",
  "questions": {
    "urgent": {
      "type": "noul",
      "instructions": "This needs action today.",
      "criteria": {"true": "Yes.", "false": "No."}
    },
    "route": {
      "type": "choice",
      "instructions": "Who owns it?",
      "criteria": {"billing": "Billing.", "technical": "Technical."}
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is it?",
      "criteria": ["low", "medium", "high"]
    }
  }
}
```

Requests are rejected if they contain duplicate object keys, `\u0000`,
nesting deeper than 256 levels, or a prompt longer than the 4096-token context.

Request text is not a trust boundary. As in OpenJev, `state`, instructions and
criteria are placed in the chat template and tokenized as one string, so
special-token spellings such as `<turn|>` in request text become control
tokens. Callers that pass third-party text should strip or escape them first.

## Decision semantics

Jev Bush reproduces OpenJev's bounded read:

1. Give each decision a one-token label in a fixed answer template.
2. Replace only those label slots with seeded vocabulary tokens.
3. Prefill the immutable prompt once, then run one bidirectional decoder read
   over the answer canvas using cached encoder K/V.
4. Project answer slots through the tied LM head and apply the checkpoint's
   `30*tanh(logit/30)` softcap.
5. Normalize only the declared candidate logits:
   `p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))`.

Up to sixteen decisions retain the original single answer canvas. Larger
requests are split into groups of at most eight and decoded as one microbatch:
all canvases share the prompt K/V, while attention masks keep their answer
tokens isolated from one another. Matrix and expert kernels see the flattened
canvas batch and therefore reuse each streamed weight tile.

Long choices use one-token labels rather than autoregressive string
likelihoods. Boolean values map to `yes/no`, ordinal scores to decimal
labels, and choices to stable alphabetic labels. Output contains every
candidate's probability and no generated text.

SHA-256 over Python-compatible canonical JSON seeds a Python-compatible
MT19937 canvas. Identical requests are deterministic. With no `samples`
field, OpenJev's entropy rule performs either one or four reads; `samples: N`
requests exactly `N` reads. Only `steps: 1` is supported.

## Exact model path

The code implements only DiffusionGemma's text path:

- memory-mapped BF16 or mixed BF16/NVFP4 safetensors;
- 30 Gemma 4 layers with mixed sliding/full attention, partial RoPE, GQA,
  dense FFN, and top-8-of-128 MoE;
- native packed E2M1 experts with E4M3 block scales for NVIDIA NVFP4;
- FP32 activations, portable scalar kernels, optional AVX-512 kernels, and
  OpenMP over independent work;
- cached causal encoder K/V and bidirectional answer-canvas attention;
- tied LM head and candidate-only normalization.

Unknown dtypes, tensor layouts, request shapes, and step counts are rejected.
There is no generic model framework, GGUF path, server, HTTP transport, image
tower, CUDA backend, or text generator.

## Reproduce the benchmark

The four scripts in `tools/` use only Python's standard library. Fetch the
pinned 400-row, 2,000-decision public set, run one deterministic read per
request, then report accuracy/calibration and timing:

```sh
python3 tools/fetch_openjev.py typed-decisions.jsonl

OMP_NUM_THREADS=32 OMP_PROC_BIND=spread OMP_PLACES=cores \
  python3 tools/run_openjev_eval.py ./jb MODEL_DIR typed-decisions.jsonl \
  predictions.jsonl --jobs 2 --samples 1

python3 tools/score_openjev.py typed-decisions.jsonl predictions.jsonl
python3 tools/summarize_eval.py predictions.jsonl
```

Give concurrent workers disjoint CPU sets when possible; otherwise separate
OpenMP processes may bind to the same cores. On the measured 64-core
Threadripper 9980X, 40--48 threads performed similarly for single-request
latency; the two-worker benchmark used 32 threads per worker. Always set
`OMP_NUM_THREADS`: without it, OpenMP uses every hardware thread (128 on that
machine), and performance is dramatically worse. Check `"threads"` in the
output to confirm the setting took effect.

## Established results

The dataset hash, implementation commits, hardware, quality results, agreement,
and timing measurements are also recorded in machine-readable form in
[`benchmarks/established-results.json`](benchmarks/established-results.json).

The trusted reference is OpenJev commit `91d5005` with patched vLLM commit
`9bbf7418`:

| engine and policy | accuracy | log loss | Brier | ECE | score MAE |
|---|---:|---:|---:|---:|---:|
| Jev Bush BF16, one read | **67.40%** | 1.6134 | 0.3350 | 0.2523 | 0.4394 |
| OpenJev NVFP4, one read | 66.60% | 1.4747 | 0.3205 | 0.2446 | 0.4460 |
| OpenJev NVFP4, automatic reads | 66.80% | **1.3940** | **0.3107** | **0.2352** | 0.4404 |
| Jev Bush NVFP4, strict, one read | 66.10% | 1.5476 | 0.3268 | 0.2540 | 0.4421 |
| Jev Bush NVFP4, fast-math, one read | 66.80% | 1.5777 | 0.3272 | 0.2454 | 0.4441 |
| Jev Bush NVFP4, strict, portable `expf`/`tanhf`, one read | 65.95% | 1.4296 | 0.3168 | 0.2442 | 0.4366 |
| Jev Bush NVFP4, fast-math CUDA, tensor cores, one read | 66.80% | 1.4358 | 0.3167 | 0.2402 | 0.4353 |

The strict portable row is the current strict build, whose output is the same
on every platform, and the last row the fast CUDA build with prefix reuse on a
DGX Spark; the rows above them were recorded with the C library's `expf` and
`tanhf` at earlier commits. On the same code and machine, switching to the
portable functions changed 4.6% of argmaxes and moved accuracy from 65.85% to
65.95% and log loss from 1.4382 to 1.4296: this model's decisions shift with
any one-ulp change, as between fast-math and strict builds, and the numbers
here differ by that noise, not by quality.

### Why this matters

The result is striking: a compact repository centered on one C file, with no
runtime dependencies and inference that needs no GPU, lands just 5.3 accuracy points
behind the flagship model of
[a startup that raised $40 million](https://www.theregister.com/2026/09/16/typesafe_ai_debuts_model_for_machines/)
on the same 2,000 decisions. The benchmark's
[published Jev 1.13.0 result](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/blob/main/README.md#baseline-results)
is 72.70%, compared with Jev Bush BF16 at 67.40%.

The caveat matters: Jev Bush does not reproduce Jev's training, calibration,
latency, generality, or production service. It does not establish matching
distributional fidelity either: Jev's model and training are undisclosed, and
no raw Jev predictions were compared by this repository's scorer. Even with
those limits, the result suggests that a meaningful part of the decision-model
advantage comes from the interface and inference method: bound the answer
space, read candidate logits directly, avoid autoregressive prose and JSON,
and share the expensive context computation.

Jev Bush NVFP4 and one-read OpenJev agree on 92.70% of 2,000 argmaxes; mean
candidate-distribution total variation is 0.0765. The packed NVFP4 primitive
was independently checked against PyTorch: activation QDQ matched exactly and
checked expert projections differed by at most `2.4e-7`.

The recommended fast-math build with two concurrent 32-thread workers averaged
5.49 s per row (p95 8.66 s), 121.88 prefill tokens/s per worker, and 0.911
decisions/s per worker. Candidate projection averaged 0.025 ms per decision;
transformer execution dominates. The strict build scored better log loss and
Brier, while fast-math improved accuracy, ECE, and throughput. Resident memory
is about 14.8 GB per process.

OpenJev on an RTX PRO 6000 Blackwell averaged 54.2 ms per one-read row. That
GPU comparison is context, not a target: Jev Bush is an educational
implementation whose optional CUDA path is bit-identical to the CPU reference
in strict builds, and whose fast-math CUDA build trades that for speed as the
CPU's fast-math build does.

Current limits are one request per process, 4,096 prompt tokens, 64 tokens per
answer canvas, 4,096 total batched canvas tokens, and one denoising step. The
hypothesis is deliberately narrow:
bounded typed decisions should read candidate logits instead of paying for
autoregressive JSON generation.

## License

[0BSD](LICENSE): use, copy, modify, and distribute Jev Bush for any purpose,
with or without fee and without an attribution requirement.
