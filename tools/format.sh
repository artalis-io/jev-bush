#!/bin/sh
set -eu

mode=${1:-write}
if [ -n "${CLANG_FORMAT:-}" ]; then
    formatter=$CLANG_FORMAT
elif command -v clang-format-18 >/dev/null 2>&1; then
    formatter=clang-format-18
elif command -v clang-format >/dev/null 2>&1; then
    formatter=clang-format
else
    echo "clang-format 18 or newer is required" >&2
    exit 1
fi

version=$($formatter --version 2>/dev/null || true)
major=$(printf '%s\n' "$version" | sed -n 's/.*version \([0-9][0-9]*\).*/\1/p')
if [ -z "$major" ] || [ "$major" -lt 18 ]; then
    echo "clang-format 18 or newer is required (found: ${version:-unknown})" >&2
    exit 1
fi

set -- jb.h src/jb.c src/*.inc examples/*.c test/*.c fuzz/*.c
if [ "$mode" = --check ]; then
    "$formatter" --dry-run --Werror "$@"
elif [ "$mode" = write ]; then
    "$formatter" -i "$@"
    python3 tools/amalgamate.py
else
    echo "usage: $0 [write|--check]" >&2
    exit 2
fi
