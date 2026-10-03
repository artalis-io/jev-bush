#!/usr/bin/env python3
"""Stress independent sessions sharing a model and verify deterministic answers."""

import argparse
import concurrent.futures
import ctypes
import json
import pathlib
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=pathlib.Path)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("request", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.iterations < 1:
        raise SystemExit("workers and iterations must be positive")

    lib = ctypes.CDLL(str(args.library.resolve()))
    pointer = ctypes.c_void_p
    lib.jb_model_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(pointer)]
    lib.jb_session_create_json.argtypes = [pointer, ctypes.c_char_p, ctypes.c_size_t,
                                           ctypes.POINTER(pointer)]
    lib.jb_session_decide_json.argtypes = [pointer, ctypes.c_char_p, ctypes.c_size_t,
                                           ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_size_t)]
    lib.jb_free.argtypes = [pointer]
    lib.jb_session_free.argtypes = [pointer]
    lib.jb_model_free.argtypes = [pointer]

    request = args.request.read_bytes()
    model = pointer()
    if lib.jb_model_load(str(args.model).encode(), ctypes.byref(model)):
        raise SystemExit("model load failed")
    sessions = []
    for _ in range(args.workers):
        session = pointer()
        if lib.jb_session_create_json(model, None, 0, ctypes.byref(session)):
            raise SystemExit("session creation failed")
        sessions.append(session)
    shared_session = pointer()
    if lib.jb_session_create_json(model, None, 0, ctypes.byref(shared_session)):
        raise SystemExit("shared session creation failed")
    # Sessions retain the model; releasing the caller's reference must be safe.
    lib.jb_model_free(model)

    def decide(session):
        output, length = pointer(), ctypes.c_size_t()
        status = lib.jb_session_decide_json(session, request, len(request), ctypes.byref(output),
                                            ctypes.byref(length))
        if status:
            raise RuntimeError(f"decision failed with status {status}")
        value = ctypes.string_at(output, length.value)
        lib.jb_free(output)
        return json.loads(value)["answers"]

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(lambda one=session: [decide(one)
                                       for _ in range(args.iterations)])
                       for session in sessions]
            answers = [answer for future in futures for answer in future.result()]
        reference = answers[0]
        if any(answer != reference for answer in answers[1:]):
            raise SystemExit("concurrent sessions produced different answers")

        barrier = threading.Barrier(2)

        def contend():
            output, length = pointer(), ctypes.c_size_t()
            barrier.wait()
            status = lib.jb_session_decide_json(shared_session, request, len(request),
                                                ctypes.byref(output), ctypes.byref(length))
            value = ctypes.string_at(output, length.value) if output.value else None
            lib.jb_free(output)
            return status, value

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            contended = [future.result() for future in
                         [executor.submit(contend), executor.submit(contend)]]
        if sorted(status for status, _ in contended) != [0, 1]:
            raise SystemExit(f"same-session contention returned {[x[0] for x in contended]}")
        if sum(value is not None for _, value in contended) != 1:
            raise SystemExit("same-session rejection did not clear its output")
    finally:
        for session in sessions:
            lib.jb_session_free(session)
        lib.jb_session_free(shared_session)
    print(f"concurrency: {len(answers)} calls across {args.workers} sessions identical; "
          "same-session contention rejected")


if __name__ == "__main__":
    main()
