#!/usr/bin/env python3
"""Check that nothing inside an OpenMP parallel region can fail or allocate.

die() inside a parallel region exits the host process from a worker thread
or longjmps out of the region, and an allocation there escapes the error
frame's cleanup. jb.c aborts on both at run time; this finds them before
they run. It builds jb.c's call graph, marks every function that can reach
die() or an allocator, and reports any parallel region that calls one.

Calls through function pointers (the kernel dispatch table) are not
followed, so dispatched kernels are checked as their own functions.
Usage: tools/check_parallel_regions.py [jb.c]
"""
import re
import sys

FAILS = {"die", "die2", "xmalloc", "xcalloc", "xrealloc", "xstrdup", "jb_allocate",
         "jb_try_allocate", "jb_arena_alloc", "jb_arena_init", "jb_size_mul",
         "jb_size_add", "jb_align_up"}
KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "defined", "_Pragma"}


def strip(source):
    """Blank comments and string/char literals, keeping offsets and lines."""
    out = list(source)
    i, n = 0, len(source)
    while i < n:
        if source.startswith("/*", i):
            j = source.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif source.startswith("//", i):
            j = source.find("\n", i)
            j = n if j < 0 else j
        elif source[i] in "\"'":
            quote, j = source[i], i + 1
            while j < n and source[j] != quote:
                j += 2 if source[j] == "\\" else 1
            j += 1
        else:
            i += 1
            continue
        for k in range(i, min(j, n)):
            if out[k] != "\n":
                out[k] = " "
        i = j
    return "".join(out)


def match(text, start, open_char, close_char):
    """Index just past the bracket that closes the one at start."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_char:
            depth += 1
        elif text[i] == close_char:
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError(f"unbalanced {open_char} at offset {start}")


def calls(text):
    return {name for name in re.findall(r"\b([A-Za-z_]\w*)\s*\(", text) if name not in KEYWORDS}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "jb.c"
    text = strip(open(path, encoding="utf-8").read())
    functions = {}
    for m in re.finditer(r"^[A-Za-z_][\w \t\*]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{", text, re.M):
        body_start = m.end() - 1
        functions[m.group(1)] = (m.start(), body_start, match(text, body_start, "{", "}"))

    graph = {name: calls(text[b:e]) for name, (_, b, e) in functions.items()}
    reach = {name: [name] for name in FAILS}
    changed = True
    while changed:
        changed = False
        for name, callees in graph.items():
            if name in reach:
                continue
            for callee in sorted(callees):
                if callee in reach:
                    reach[name] = [name] + reach[callee]
                    changed = True
                    break

    findings = 0
    for m in re.finditer(r"#\s*pragma\s+omp\s+parallel\b[^\n]*", text):
        i = m.end()
        # The governed statement follows any preprocessor lines, such as the
        # #endif that closes an #ifdef _OPENMP around the pragma.
        while True:
            while text[i].isspace():
                i += 1
            if text[i] != "#":
                break
            i = text.index("\n", i)
        if text.startswith("for", i):
            header = text.index("(", i)
            i = match(text, header, "(", ")")
            while text[i].isspace():
                i += 1
            end = match(text, i, "{", "}") if text[i] == "{" else text.index(";", i) + 1
            region = text[header:end]
        elif text[i] == "{":
            region = text[i:match(text, i, "{", "}")]
        else:
            region = text[i:text.index(";", i) + 1]
        line = text.count("\n", 0, m.start()) + 1
        owner = max((f for f, (s, _, e) in functions.items() if s <= m.start() < e),
                    key=lambda f: functions[f][0], default="?")
        for callee in sorted(calls(region)):
            if callee in reach:
                findings += 1
                chain = " -> ".join(reach[callee])
                print(f"{path}:{line}: parallel region in {owner}() can fail or allocate: {chain}")
    regions = len(re.findall(r"#\s*pragma\s+omp\s+parallel\b", text))
    if findings:
        return 1
    print(f"{regions} parallel regions checked; none can fail or allocate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
