#!/usr/bin/env python3
"""Check that strict Jev Bush output does not depend on how a request runs.

Strict builds promise byte-identical answers however the engine executes a
request. The self-test cannot check that without the model, so this runs a
strict binary on the first rows of a request file several ways and compares
every answer object as printed:

  - eval, where rows after the first of a schema hit the prefix cache;
  - eval with JB_MICROBATCH=4, which runs consecutive hits as one batch;
  - decide on each row alone, which always prefills the whole prompt;
  - with --sequential, eval by a second strict binary built with
    -DJB_CANVAS_SEQUENTIAL, which decodes every canvas and every read on its
    own instead of batching them.

Usage: tools/check_strict_invariance.py JB MODEL_DIR REQUESTS.jsonl
           [--rows N] [--sequential JB_SEQUENTIAL]
Exits 0 when every run matches the first, 1 otherwise. Set OMP_NUM_THREADS
as for any run.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run(cmd, env=None):
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode:
        sys.exit(f"{' '.join(cmd)} failed ({result.returncode}):\n{result.stderr}")
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def answers(rows, what):
    out = []
    for row in rows:
        if row.get("math") != "strict":
            sys.exit(f"{what}: output records math {row.get('math')!r}; use a strict build")
        out.append((row.get("id"), json.dumps(row["answers"], sort_keys=True)))
    return out


def compare(name, reference, candidate):
    if len(reference) != len(candidate):
        print(f"{name}: {len(candidate)} rows, expected {len(reference)}")
        return False
    differing = [rid for (rid, a), (_, b) in zip(reference, candidate) if a != b]
    if differing:
        print(f"{name}: {len(differing)} of {len(reference)} rows differ: {differing[:5]}")
        return False
    print(f"{name}: all {len(reference)} rows identical")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("jb")
    parser.add_argument("model")
    parser.add_argument("requests")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--sequential", help="a strict jb built with -DJB_CANVAS_SEQUENTIAL")
    ns = parser.parse_args()
    lines = [line for line in Path(ns.requests).read_text(encoding="utf-8").splitlines()
             if line.strip()][:ns.rows]
    if not lines:
        sys.exit("no requests")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "sample.jsonl"
        sample.write_text("\n".join(lines) + "\n", encoding="utf-8")
        env = dict(os.environ)
        env.pop("JB_MICROBATCH", None)
        base = answers(run([ns.jb, ns.model, "eval", str(sample)], env), "eval")
        print(f"eval: {len(base)} rows")
        batched = dict(env, JB_MICROBATCH="4")
        ok &= compare("microbatch 4", base,
                      answers(run([ns.jb, ns.model, "eval", str(sample)], batched), "microbatch"))
        alone = []
        for i, line in enumerate(lines):
            request = Path(tmp) / f"{i}.json"
            request.write_text(line, encoding="utf-8")
            alone += answers(run([ns.jb, ns.model, "decide", str(request)], env), "decide")
        ok &= compare("decide, monolithic prefill", base, alone)
        if ns.sequential:
            ok &= compare("sequential canvases and reads", base,
                          answers(run([ns.sequential, ns.model, "eval", str(sample)], env),
                                  "sequential"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
