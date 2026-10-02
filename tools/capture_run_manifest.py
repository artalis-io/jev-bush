#!/usr/bin/env python3
"""Print reproducibility metadata for a Jev Bush benchmark or accuracy run."""

import argparse, hashlib, json, os, pathlib, platform, subprocess


def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=pathlib.Path)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--compiler-flags", default="")
    args = parser.parse_args()
    files = ["config.json", "tokenizer.json", "model-00001-of-00002.safetensors",
             "model-00002-of-00002.safetensors"]
    result = {
        "schema_version": 1,
        "jev_bush_commit": command("git", "rev-parse", "HEAD"),
        "jev_bush_dirty": bool(command("git", "status", "--porcelain")),
        "executable": {"path": str(args.executable), "sha256": sha256(args.executable)},
        "model": {"path": str(args.model), "files": {
            name: sha256(args.model / name) for name in files}},
        "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
        "compiler": command("cc", "--version"),
        "compiler_flags": args.compiler_flags,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": command("sh", "-c", "lscpu 2>/dev/null || sysctl -a 2>/dev/null"),
        "environment": {name: os.environ.get(name) for name in
                        ["OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "JB_CUDA",
                         "JB_MICROBATCH"]},
        "nvidia_smi": command("nvidia-smi", "-q"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
