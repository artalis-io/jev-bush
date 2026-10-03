#!/bin/sh
set -eu

tools/format.sh --check
python3 tools/amalgamate.py --check
python3 tools/check_error_boundaries.py jb.c jb.h
python3 tools/check_parallel_regions.py jb.c
python3 -m py_compile tools/*.py

for script in tools/*.sh .githooks/*; do
    sh -n "$script"
done

${CC:-cc} -std=c11 -Wall -Wextra -pedantic -Wshadow -Wformat=2 -Werror \
    -fsyntax-only jb.c
git diff --check
