#!/usr/bin/env python3
"""Run the checked-in request and require the canonical strict model result."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


EXPECTED = {
    "answers": "c1c71b45ed93789f35bc2621bae9f3c06abd7ea89f7e1483500a9cc2d28391b2",
    "usage": "eda840cf5b7aabc1065507c1c5bc64581504b4c489f7d8fa87355e6f970e3318",
}


def digest(value):
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Check a strict Jev Bush build against the model-backed golden result."
    )
    parser.add_argument("jb", help="strict Jev Bush executable")
    parser.add_argument("model", help="DiffusionGemma model directory")
    parser.add_argument(
        "--request",
        default=str(root / "examples" / "request.json"),
        help="request fixture (default: examples/request.json)",
    )
    args = parser.parse_args()

    run = subprocess.run(
        [args.jb, args.model, "decide", args.request],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    result = json.loads(run.stdout)
    if result.get("math") != "strict":
        raise SystemExit("golden check requires a strict build")
    failed = False
    for field, expected in EXPECTED.items():
        actual = digest(result.get(field))
        state = "ok" if actual == expected else "FAIL"
        print(f"{state:4} {field}: {actual}")
        failed |= actual != expected
    if failed:
        raise SystemExit("model-backed golden result changed")
    print(
        "ok   backend: "
        + (result.get("accelerator") or result.get("kernels", "unknown"))
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
