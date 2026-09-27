#!/usr/bin/env python3
"""Build one-state/many-predicate requests from an existing OpenJev row."""

import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", help="JSON or first row of a JSONL file")
    parser.add_argument("output")
    parser.add_argument("--counts", default="1,2,4,8,16,32,64")
    parser.add_argument("--compact", action="store_true",
                        help="use short synthetic boolean predicates")
    ns = parser.parse_args()

    line = next(x for x in Path(ns.request).read_text(encoding="utf-8").splitlines() if x.strip())
    source = json.loads(line)
    questions = source["questions"]
    if isinstance(questions, str):
        questions = json.loads(questions)
    if not isinstance(questions, dict) or not questions:
        raise ValueError("source request has no questions")
    prototype = next(iter(questions.values()))
    counts = [int(x) for x in ns.counts.split(",")]
    if not counts or any(x < 1 or x > 1024 for x in counts):
        raise ValueError("counts must be between 1 and 1024")

    with Path(ns.output).open("w", encoding="utf-8") as out:
        for count in counts:
            generated = {}
            for i in range(count):
                generated[f"q{i + 1}"] = (
                    {"type": "noul", "instructions": f"Does condition {i + 1} apply?"}
                    if ns.compact else copy.deepcopy(prototype)
                )
            request = {
                "id": f"predicates-{count}",
                "state": source["state"],
                "questions": generated,
                "samples": 1,
            }
            out.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
