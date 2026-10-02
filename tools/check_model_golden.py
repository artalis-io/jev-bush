#!/usr/bin/env python3
"""Verify the exact model fixture, JSON result, and typed C API result."""

import argparse
import ctypes
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


class String(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char_p), ("length", ctypes.c_size_t)]


class Candidate(ctypes.Structure):
    _fields_ = [("id", String), ("description", String)]


class Question(ctypes.Structure):
    _fields_ = [
        ("id", String),
        ("predicate", String),
        ("type", ctypes.c_int),
        ("candidates", ctypes.POINTER(Candidate)),
        ("candidate_count", ctypes.c_size_t),
    ]


class Schema(ctypes.Structure):
    _fields_ = [("questions", ctypes.POINTER(Question)), ("question_count", ctypes.c_size_t)]


class Input(ctypes.Structure):
    _fields_ = [("id", String), ("state_json", String), ("samples", ctypes.c_uint32)]


class Probability(ctypes.Structure):
    _fields_ = [("candidate", String), ("probability", ctypes.c_double)]


class Answer(ctypes.Structure):
    _fields_ = [
        ("id", String),
        ("type", ctypes.c_int),
        ("probabilities", ctypes.POINTER(Probability)),
        ("probability_count", ctypes.c_size_t),
        ("selected_candidate", ctypes.c_size_t),
        ("expected_score", ctypes.c_double),
        ("confidence", ctypes.c_double),
    ]


class Result(ctypes.Structure):
    _fields_ = [
        ("answers", ctypes.POINTER(Answer)),
        ("answer_count", ctypes.c_size_t),
        ("input_tokens", ctypes.c_uint32),
        ("prefill_tokens", ctypes.c_uint32),
        ("total_ms", ctypes.c_double),
        ("prefill_ms", ctypes.c_double),
        ("candidate_ms", ctypes.c_double),
    ]


def span(text, keep):
    raw = text.encode()
    keep.append(raw)
    return String(raw, len(raw))


def typed_answers(library, model_path, request):
    lib, void = ctypes.CDLL(str(library)), ctypes.c_void_p
    lib.jb_api_version.restype = ctypes.c_uint32
    lib.jb_model_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(void)]
    lib.jb_session_create.argtypes = [void, ctypes.POINTER(Schema), ctypes.POINTER(void)]
    lib.jb_session_decide.argtypes = [
        void,
        ctypes.POINTER(Input),
        ctypes.POINTER(ctypes.POINTER(Result)),
    ]
    lib.jb_result_free.argtypes = [ctypes.POINTER(Result)]
    lib.jb_session_free.argtypes, lib.jb_model_free.argtypes = [void], [void]
    if lib.jb_api_version() != 1:
        raise SystemExit("typed library has an incompatible API version")
    keep, questions = [], []
    kinds = {"noul": 0, "choice": 1, "score": 2}
    for identifier, value in request["questions"].items():
        criteria = value["criteria"]
        pairs = (
            list(criteria.items())
            if isinstance(criteria, dict)
            else [("", item) for item in criteria]
        )
        candidates = (Candidate * len(pairs))()
        keep.append(candidates)
        for index, (candidate_id, description) in enumerate(pairs):
            candidates[index] = Candidate(span(candidate_id, keep), span(description, keep))
        questions.append(
            Question(
                span(identifier, keep),
                span(value["instructions"], keep),
                kinds[value["type"]],
                candidates,
                len(pairs),
            )
        )
    question_array = (Question * len(questions))(*questions)
    keep.append(question_array)
    schema = Schema(question_array, len(questions))
    state = json.dumps(request["state"], ensure_ascii=False, separators=(",", ":"))
    item = Input(span(request.get("id", ""), keep), span(state, keep), request.get("samples", 0))
    model, session, result = void(), void(), ctypes.POINTER(Result)()
    status = lib.jb_model_load(str(model_path).encode(), ctypes.byref(model))
    if status == 0:
        status = lib.jb_session_create(model, ctypes.byref(schema), ctypes.byref(session))
    if status == 0:
        status = lib.jb_session_decide(session, ctypes.byref(item), ctypes.byref(result))
    if status:
        lib.jb_session_free(session)
        lib.jb_model_free(model)
        raise SystemExit(f"typed C API failed with status {status}")
    output = {}
    for index in range(result.contents.answer_count):
        answer = result.contents.answers[index]
        key = ctypes.string_at(answer.id.data, answer.id.length).decode()
        output[key] = {}
        for candidate_index in range(answer.probability_count):
            probability = answer.probabilities[candidate_index]
            name = ctypes.string_at(
                probability.candidate.data, probability.candidate.length
            ).decode()
            output[key][name] = probability.probability
    lib.jb_result_free(result)
    lib.jb_session_free(session)
    lib.jb_model_free(model)
    return output


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jb")
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--library", type=pathlib.Path)
    parser.add_argument("--request", type=pathlib.Path, default=root / "examples/request.json")
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=root / "benchmarks/golden-fixture.json",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    fixtures = [(args.request, manifest["request"]["sha256"])]
    fixtures += [
        (args.model / name, expected)
        for name, expected in manifest["model"]["files"].items()
    ]
    for path, expected in fixtures:
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"fixture hash mismatch: {path}: {actual}, expected {expected}")
        print(f"ok   fixture: {path.name} {actual}")
    run = subprocess.run(
        [args.jb, str(args.model), "decide", str(args.request)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    result = json.loads(run.stdout)
    if result.get("math") != "strict":
        raise SystemExit("golden check requires a strict build")
    for field, expected in manifest["result"].items():
        actual = digest(result.get(field))
        if actual != expected:
            raise SystemExit(f"golden {field} changed: {actual}, expected {expected}")
        print(f"ok   {field}: {actual}")
    request = json.loads(args.request.read_text())
    if args.library:
        typed = typed_answers(args.library.resolve(), args.model, request)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            library = pathlib.Path(tmp) / "libjb.so"
            subprocess.run(
                [
                    "cc", "-O3", "-march=native", "-std=c11", "-fopenmp",
                    "-fPIC", "-shared", "-DJB_NO_MAIN", str(root / "jb.c"),
                    "-lm", "-o", library,
                ],
                check=True,
            )
            typed = typed_answers(library, args.model, request)
    expected = {key: value["probabilities"] for key, value in result["answers"].items()}
    if typed != expected:
        raise SystemExit("typed C API probabilities differ from the JSON API")
    print("ok   typed C API probabilities equal the JSON API")
    print(f"ok   executable: {sha256(pathlib.Path(args.jb))}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
