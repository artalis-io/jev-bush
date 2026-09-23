#!/usr/bin/env python3
"""Fetch the public typed-decisions all/test split as evaluator-ready JSONL."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

REVISION = "ea9306458d6e9563628369a3d1e72e362fb381d2"
SHA256 = "78ed4c464597e1334315d1b2b432d2911fa158733658c1c6500d1e5fe9dad8b7"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--dataset", default="LocalLLaMA/typed-decisions")
    parser.add_argument("--expected-sha256", default=SHA256)
    parser.add_argument("--config", default="all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--page", type=int, default=100)
    ns = parser.parse_args()
    rows = []
    total = None
    while total is None or len(rows) < total:
        query = urllib.parse.urlencode({"dataset": ns.dataset, "config": ns.config,
                                        "split": ns.split, "offset": len(rows),
                                        "length": ns.page})
        with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + query) as response:
            page = json.load(response)
        total = int(page["num_rows_total"])
        chunk = page["rows"]
        if not chunk:
            raise RuntimeError(f"dataset server stopped at row {len(rows)} of {total}")
        for item in chunk:
            if int(item["row_idx"]) != len(rows):
                raise RuntimeError("non-contiguous dataset row indices")
            rows.append(item["row"])
    target = Path(ns.output)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            json.dump(row, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if ns.expected_sha256 and digest != ns.expected_sha256:
        raise RuntimeError(f"dataset SHA-256 mismatch: expected {ns.expected_sha256}, got {digest}")
    temporary.replace(target)
    print(json.dumps({"dataset": ns.dataset, "validated_revision": REVISION,
                      "config": ns.config,
                      "split": ns.split, "rows": len(rows),
                      "decisions": sum(int(x["n_questions"]) for x in rows),
                      "sha256": digest}, separators=(",", ":")))


if __name__ == "__main__":
    main()
