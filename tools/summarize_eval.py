#!/usr/bin/env python3
"""Aggregate Jev Bush timing and throughput fields from prediction JSONL."""

import argparse
import json
import math
from pathlib import Path
import statistics


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    at = (len(values) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] if lo == hi else values[lo] * (hi - at) + values[hi] * (at - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    ns = parser.parse_args()
    rows = [json.loads(line) for line in Path(ns.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("no prediction rows")
    decisions = sum(len(row["answers"]) for row in rows)
    tokens = sum(int(row["usage"]["input_tokens"]) for row in rows)
    total_s = [float(row["timing_ms"]["total"]) / 1000.0 for row in rows]
    prefill_s = sum(float(row["timing_ms"]["prefill"]) for row in rows) / 1000.0
    candidate_s = sum(float(row["timing_ms"]["candidates"]) for row in rows) / 1000.0
    worker_s = sum(total_s)
    result = {
        "rows": len(rows), "decisions": decisions, "input_tokens": tokens,
        "worker_seconds": worker_s,
        "prefill_tokens_per_second_per_worker": tokens / prefill_s,
        "decisions_per_second_per_worker": decisions / worker_s,
        "row_latency_seconds_mean": statistics.mean(total_s),
        "row_latency_seconds_p50": percentile(total_s, 0.50),
        "row_latency_seconds_p95": percentile(total_s, 0.95),
        "decision_latency_seconds_amortized": worker_s / decisions,
        "candidate_seconds_total": candidate_s,
        "candidate_milliseconds_per_decision": candidate_s * 1000.0 / decisions,
        "candidate_fraction": candidate_s / worker_s,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
