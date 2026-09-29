#!/usr/bin/env python3
"""Compare two Jev Bush result files row by row, for backend parity checks.

Rows are compared in file order. For every decision the tool reports whether
the whole answer object is byte-identical as printed, and compares the
candidate probability distributions: argmax agreement, total variation, and
the largest absolute probability difference. Timing and other per-run fields
are ignored.

Usage: tools/compare_answers.py REFERENCE.jsonl CANDIDATE.jsonl
Exits 0 when every answer object is byte-identical, 1 otherwise.
"""
import json
import sys


def distribution(answer):
    probabilities = answer.get("probabilities", {})
    return {str(k): float(v) for k, v in probabilities.items()}


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    with open(sys.argv[1], encoding="utf-8") as a, open(sys.argv[2], encoding="utf-8") as b:
        left = [json.loads(line) for line in a if line.strip()]
        right = [json.loads(line) for line in b if line.strip()]
    if len(left) != len(right):
        raise SystemExit(f"row count differs: {len(left)} vs {len(right)}")
    decisions = identical = argmax_equal = 0
    total_variation = max_abs = 0.0
    worst = None
    for row, (x, y) in enumerate(zip(left, right)):
        if x.get("id") != y.get("id"):
            raise SystemExit(f"row {row}: ids differ: {x.get('id')!r} vs {y.get('id')!r}")
        if list(x["answers"]) != list(y["answers"]):
            raise SystemExit(f"row {row}: decision keys differ")
        for key in x["answers"]:
            p, q = x["answers"][key], y["answers"][key]
            decisions += 1
            identical += json.dumps(p, sort_keys=True) == json.dumps(q, sort_keys=True)
            dp, dq = distribution(p), distribution(q)
            if set(dp) != set(dq):
                raise SystemExit(f"row {row} {key}: candidate sets differ")
            argmax_equal += max(dp, key=dp.get) == max(dq, key=dq.get)
            tv = sum(abs(dp[c] - dq[c]) for c in dp) / 2
            total_variation += tv
            diff = max(abs(dp[c] - dq[c]) for c in dp)
            if diff > max_abs:
                max_abs, worst = diff, f"row {row} ({x.get('id')}) {key}"
    summary = {
        "rows": len(left),
        "decisions": decisions,
        "identical_answer_objects": identical,
        "argmax_agreement": argmax_equal / decisions if decisions else None,
        "mean_total_variation": total_variation / decisions if decisions else None,
        "max_abs_probability_difference": max_abs,
        "largest_difference_at": worst,
    }
    print(json.dumps(summary, indent=2))
    return 0 if identical == decisions else 1


if __name__ == "__main__":
    sys.exit(main())
