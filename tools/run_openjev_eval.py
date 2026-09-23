#!/usr/bin/env python3
"""Resumable process-parallel runner for Jev Bush JSONL evaluations."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def write_result(path, result):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    os.replace(temporary, path)


def run_batch(args):
    worker, tasks, jb, model, rows, samples = args
    requests = []
    for _, line in tasks:
        value = json.loads(line)
        if samples is not None:
            value["samples"] = samples
        requests.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    proc = subprocess.run(
        [jb, model, "eval", "-"],
        input="\n".join(requests) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode:
        (rows / f"worker-{worker:02d}.log").write_text(proc.stderr, encoding="utf-8")
        raise RuntimeError(f"worker {worker} failed with exit code {proc.returncode}")
    results = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    if len(results) != len(tasks):
        raise RuntimeError(f"worker {worker} returned {len(results)} of {len(tasks)} rows")
    for (index, _), result in zip(tasks, results):
        write_result(rows / f"{index:06d}.json", result)
    return worker, len(tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jb")
    parser.add_argument("model")
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--samples", type=int, choices=range(1, 33))
    ns = parser.parse_args()
    lines = [line for line in Path(ns.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty dataset")
    if ns.jobs < 1:
        raise ValueError("--jobs must be positive")
    rows = Path(str(ns.output) + ".rows")
    rows.mkdir(parents=True, exist_ok=True)
    pending = []
    done = 0
    for index, line in enumerate(lines):
        final = rows / f"{index:06d}.json"
        if final.exists():
            json.loads(final.read_text(encoding="utf-8"))
            done += 1
        else:
            pending.append((index, line))
    jobs = min(ns.jobs, len(pending))
    batches = [[] for _ in range(jobs)]
    for index, task in enumerate(pending):
        batches[index % jobs].append(task)
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_batch, (i, batch, ns.jb, ns.model, rows, ns.samples))
                       for i, batch in enumerate(batches)]
            try:
                for future in concurrent.futures.as_completed(futures):
                    worker, count = future.result()
                    done += count
                    print(f"[{done}/{len(lines)}] worker {worker}: {count} rows",
                          file=sys.stderr, flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    output = Path(ns.output)
    with output.open("w", encoding="utf-8") as stream:
        for i in range(len(lines)):
            data = json.loads((rows / f"{i:06d}.json").read_text(encoding="utf-8"))
            json.dump(data, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")


if __name__ == "__main__":
    main()
