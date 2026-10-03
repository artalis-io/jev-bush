#!/usr/bin/env python3
"""Verify that failure-capable call paths cross an error-frame boundary."""

import re
import sys

from check_parallel_regions import calls, match, strip


DANGEROUS = {"die", "die2", "die_status", "die2_status", "jb_out_of_memory"}
NOFAIL_PUBLIC = {
    "jb_api_version",
    "jb_free",
    "jb_last_error",
    "jb_model_free",
    "jb_model_retain",
    "jb_result_free",
    "jb_results_free",
    "jb_session_free",
    "jb_session_last_error",
    "jb_status_string",
    "jb_version",
}


def functions(text):
    result = {}
    pattern = r"^[A-Za-z_][\w \t\*]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
    for found in re.finditer(pattern, text, re.M):
        body = found.end() - 1
        result[found.group(1)] = text[body : match(text, body, "{", "}")]
    return result


def path_to(graph, root, targets, barriers=frozenset()):
    pending = [(root, [root])]
    seen = set()
    while pending:
        name, path = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in targets:
            return path
        if name in barriers and name != root:
            continue
        pending.extend((callee, path + [callee]) for callee in graph.get(name, ()))
    return None


def main():
    source_path = sys.argv[1] if len(sys.argv) > 1 else "jb.c"
    header_path = sys.argv[2] if len(sys.argv) > 2 else "jb.h"
    source = strip(open(source_path, encoding="utf-8").read())
    bodies = functions(source)
    graph = {name: calls(body) for name, body in bodies.items()}
    framed = {name for name, body in bodies.items() if "setjmp" in calls(body)}
    public = set(re.findall(r"\bJB_API\s+[^(;]+?\b(jb_[A-Za-z_]\w*)\s*\(",
                            open(header_path, encoding="utf-8").read()))

    findings = []
    for root in sorted(NOFAIL_PUBLIC):
        path = path_to(graph, root, DANGEROUS)
        if path:
            findings.append(f"non-failing public function reaches failure: {' -> '.join(path)}")
    for root in sorted(public - NOFAIL_PUBLIC):
        path = path_to(graph, root, DANGEROUS, framed)
        if path and root not in framed:
            findings.append(f"public failure bypasses an error frame: {' -> '.join(path)}")
    path = path_to(graph, "main", DANGEROUS, framed)
    if path and "main" not in framed:
        findings.append(f"CLI failure bypasses its root frame: {' -> '.join(path)}")

    expected_frames = {
        "dg_test_nonfinite_activation",
        "jb_model_load_impl",
        "jb_session_create",
        "jb_session_create_json",
        "jb_session_decide_batch_call",
        "jb_session_decide_call",
        "jb_session_decide_json_batch_call",
        "jb_session_decide_json_call",
        "main",
    }
    missing = expected_frames - framed
    if missing:
        findings.append("missing required error frames: " + ", ".join(sorted(missing)))
    unexpected = framed - expected_frames
    if unexpected:
        findings.append("unreviewed error frames: " + ", ".join(sorted(unexpected)))
    if findings:
        print("\n".join(f"{source_path}: {finding}" for finding in findings))
        return 1
    print(f"{len(public)} public functions and {len(framed)} error frames checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
