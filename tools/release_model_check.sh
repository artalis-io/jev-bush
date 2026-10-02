#!/bin/sh
set -eu

if [ "$#" -lt 5 ]; then
    echo "usage: $0 JB JB_SEQUENTIAL LIBJB MODEL DATASET.jsonl [CUDA_JB]" >&2
    exit 2
fi

jb=$1
sequential=$2
library=$3
model=$4
dataset=$5

python3 tools/check_model_golden.py "$jb" "$model" --library "$library"
python3 tools/check_strict_invariance.py "$jb" "$model" "$dataset" \
    --rows 40 --sequential "$sequential"

if [ "$#" -ge 6 ]; then
    cuda=$6
    tmp=${TMPDIR:-/tmp}/jb-cuda-parity-$$.jsonl
    trap 'rm -f "$tmp" "$tmp.cpu" "$tmp.cuda"' EXIT HUP INT TERM
    head -n 40 "$dataset" > "$tmp"
    "$jb" "$model" eval "$tmp" > "$tmp.cpu"
    "$cuda" "$model" eval "$tmp" > "$tmp.cuda"
    python3 tools/compare_answers.py "$tmp.cpu" "$tmp.cuda"
    rm -f "$tmp.cpu" "$tmp.cuda"
fi
