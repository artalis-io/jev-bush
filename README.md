# Jev Bush

![Jev Bush robot mascot](jev-bush-mascot.png)

> CPU-first probabilistic decisions for DiffusionGemma.
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
- Run inference on CPUs. GPU backends are deliberately out of scope: adding
  CUDA, Metal, Vulkan, WebGPU, or another GPGPU path would defeat the point of
  this project.
- Make the CPU path correct and fast through model-specific data layouts,
  vectorization, and OpenMP without turning `jb.c` into a generic framework.
- Remain compatible with OpenJev's bounded decision semantics and public
  evaluation data.

For production GPGPU inference, use
[OpenJev](https://github.com/razorback16/openjev) or
[SemIf](https://github.com/TheoLeeCJ/SemIf).

## Build

Linux, optimized for the current CPU:

```sh
cc -O3 -march=native -ffast-math -std=c11 -Wall -Wextra -pedantic \
  -fopenmp jb.c -lm -o jb
```

This is the evaluated high-throughput build. `-ffast-math` changes floating-
point reduction and transcendental behavior; its full accuracy and calibration
results are reported below. Output records `"math":"fast"`; omit the flag for
strict IEEE behavior and `"math":"strict"` output.

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

Run the dependency-free smoke test without downloading a model:

```console
$ ./jb --selftest
{"selftest":"ok"}
```

For operation-level profiling, add `-DJB_PROFILE`. Each request then emits one
timing line to standard error for attention, dense FFN, routing, MoE experts,
expert input/hidden QDQ, gate/up/down projections, expert activation and
miscellaneous expert work, and remaining FFN work. Normal JSONL on standard
output is unchanged.

## Use

```sh
./jb MODEL_DIR decide request.json
./jb MODEL_DIR eval requests.jsonl
```

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
latency; the two-worker benchmark used 32 threads per worker.

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

### Why this matters

The result is striking: a compact repository centered on one C file, with no
runtime dependencies and CPU-only inference, lands just 5.3 accuracy points
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
GPU comparison is context, not a target backend: Jev Bush is intentionally a
CPU educational implementation.

Current limits are batch size one per process, 4,096 prompt tokens, a 64-token
answer canvas, and one denoising step. The hypothesis is deliberately narrow:
bounded typed decisions should read candidate logits instead of paying for
autoregressive JSON generation.

## License

[0BSD](LICENSE): use, copy, modify, and distribute Jev Bush for any purpose,
with or without fee and without an attribution requirement.
