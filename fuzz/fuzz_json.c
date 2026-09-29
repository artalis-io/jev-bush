/*
 * libFuzzer target for Jev Bush's input boundaries.
 *
 * Build: clang -g -O1 -fsanitize=fuzzer,address,undefined \
 *        -fno-sanitize-recover=all fuzz/fuzz_json.c -lm -o fuzz_json
 * Run:   ./fuzz_json -dict=fuzz/json.dict corpus fuzz/corpus
 *
 * The first byte selects what an input exercises:
 *   0x01  typed API: fields separated by 0x1f build a jb_schema and a
 *         jb_input, which go through jb_schema_json and jb_typed_request;
 *   0x02  jb_result_parse on the rest of the input;
 *   else  the whole input is a JSON request: parsing, canonicalization,
 *         request and question validation, and prompt construction.
 *
 * jb.c is included unchanged, and each input runs under an error frame as
 * the library's entry points do, so die() unwinds instead of exiting.
 * Rejection is fine. Memory errors, undefined behavior, broken invariants,
 * and any allocation still live after an input (a leak, including on the
 * rejection path) abort.
 */
#define _POSIX_C_SOURCE 200809L
#if defined(__MINGW32__) && !defined(__USE_MINGW_SETJMP_NON_SEH)
#define __USE_MINGW_SETJMP_NON_SEH
#endif
/* Every system header jb.c uses, included before the macros below so they
 * rename calls inside jb.c only, never declarations in system headers. */
#include <ctype.h>
#include <float.h>
#include <locale.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* Count live heap blocks so a leak after any input can be detected. */
static long fuzz_live;
static int fuzz_must_succeed;

static void *fuzz_malloc(size_t n) {
    void *p = malloc(n);
    fuzz_live += p != NULL;
    return p;
}

static void *fuzz_calloc(size_t n, size_t z) {
    void *p = calloc(n, z);
    fuzz_live += p != NULL;
    return p;
}

static void *fuzz_realloc(void *p, size_t n) {
    void *q = realloc(p, n);
    fuzz_live += q && !p;
    return q;
}

static void fuzz_free(void *p) {
    fuzz_live -= p != NULL;
    free(p);
}

/* jb.c only exits when die() runs without an error frame, and every input
 * here runs under one. */
static void fuzz_exit(int code) {
    (void)code;
    abort();
}

static int fuzz_fprintf(FILE *f, const char *fmt, ...) {
    (void)f;
    (void)fmt;
    return 0;
}

#define malloc fuzz_malloc
#define calloc fuzz_calloc
#define realloc fuzz_realloc
#define free fuzz_free
#define exit fuzz_exit
#define fprintf fuzz_fprintf
#define main jb_main
#include "../jb.c"
#undef malloc
#undef calloc
#undef realloc
#undef free
#undef exit
#undef fprintf
#undef main

/* Canonical output must itself parse, canonicalize to the same bytes, and
 * be pure ASCII when ASCII escaping is requested. */
static void fuzz_canonical(const char *j, JTok *t, int nt, int ascii) {
    char *c1 = dg_json_canonical(j, t, nt, 0, 1, ascii);
    fuzz_must_succeed = 1;
    int n2;
    JTok *t2 = json_tokens(c1, strlen(c1), &n2);
    char *c2 = dg_json_canonical(c1, t2, n2, 0, 1, ascii);
    fuzz_must_succeed = 0;
    if (strcmp(c1, c2))
        abort();
    if (ascii)
        for (const char *p = c1; *p; p++)
            if ((unsigned char)*p > 127)
                abort();
    jb_release(c2);
    jb_release(t2);
    jb_release(c1);
}

static void fuzz_json_request(const char *j, size_t n) {
    int nt;
    JTok *t = json_tokens(j, n, &nt);
    fuzz_canonical(j, t, nt, 1);
    fuzz_canonical(j, t, nt, 0);
    jb_release(dg_text_of(j, t, nt, 0));
    jb_release(t);

    DGRequest rq;
    dg_request_parse(&rq, j, n);
    (void)dg_request_seed(&rq);
    char *labels[128];
    for (int i = 0; i < 128; i++) {
        char l[3];
        dg_choice_candidate(i, l);
        labels[i] = xstrdup(l);
    }
    int nq;
    DecisionWork *w = dg_questions(&rq, labels, &nq);
    char *sys = dg_system_prompt(rq.qj, rq.qt, rq.qnt, w, nq);
    int *pick = xcalloc((size_t)nq, sizeof *pick);
    for (int x = 0; x < nq; x++)
        pick[x] = w[x].nc - 1;
    jb_release(dg_answer_template(w, nq, pick));
    jb_release(pick);
    jb_release(sys);
    dg_questions_free(w, nq);
    for (int i = 0; i < 128; i++)
        jb_release(labels[i]);
    dg_request_free(&rq);
}

/* Next 0x1f-separated field as a counted string; it may contain NUL. */
static jb_string fuzz_field(const char **p, const char *end) {
    jb_string s = {*p, 0};
    while (*p < end && **p != 0x1f)
        (*p)++;
    s.length = (size_t)(*p - s.data);
    if (*p < end)
        (*p)++;
    return s;
}

static unsigned fuzz_byte(const char **p, const char *end) {
    jb_string s = fuzz_field(p, end);
    return s.length ? (unsigned char)s.data[0] : 0;
}

static void fuzz_typed(const char *j, size_t n) {
    const char *p = j, *end = j + n;
    jb_candidate candidates[4][6];
    jb_question questions[4];
    size_t nq = fuzz_byte(&p, end) % 5;
    for (size_t q = 0; q < nq; q++) {
        questions[q].id = fuzz_field(&p, end);
        questions[q].predicate = fuzz_field(&p, end);
        /* Includes one value outside the enum. */
        questions[q].type = (jb_decision_type)(fuzz_byte(&p, end) % 4);
        size_t count = fuzz_byte(&p, end) % 7;
        for (size_t c = 0; c < count && c < 6; c++) {
            candidates[q][c].id = fuzz_field(&p, end);
            candidates[q][c].description = fuzz_field(&p, end);
        }
        questions[q].candidates = count ? candidates[q] : NULL;
        questions[q].candidate_count = count;
    }
    jb_schema schema = {nq ? questions : NULL, nq};
    size_t schema_length = 0;
    char *schema_json = jb_schema_json(&schema, &schema_length);
    if (!schema_json)
        return;
    /* A schema the builder accepted must be well-formed JSON. */
    fuzz_must_succeed = 1;
    int nt;
    jb_release(json_tokens(schema_json, schema_length, &nt));
    fuzz_must_succeed = 0;

    jb_session session;
    memset(&session, 0, sizeof session);
    session.questions_json = schema_json;
    session.questions_length = schema_length;
    jb_input input;
    input.id = fuzz_field(&p, end);
    input.state_json = fuzz_field(&p, end);
    input.samples = fuzz_byte(&p, end) % 40;
    size_t request_length = 0;
    char *request = jb_typed_request(&session, &input, &request_length);
    fuzz_must_succeed = 1;
    jb_release(json_tokens(request, request_length, &nt));
    fuzz_must_succeed = 0;
    jb_release(request);
    jb_release(schema_json);
}

static void fuzz_result(const char *j, size_t n) {
    jb_result_free(jb_result_parse(j, n));
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    /* Exact-size copy, not NUL-terminated, as eval rows are. */
    char *j = malloc(size ? size : 1);
    if (!j)
        return 0;
    if (size)
        memcpy(j, data, size);
    long baseline = fuzz_live;
    fuzz_must_succeed = 0;
    JBErrorFrame frame;
    jb_frame_enter(&frame, JB_ERROR_REQUEST);
    if (setjmp(frame.jump)) {
        jb_frame_fail(&frame);
        if (fuzz_must_succeed)
            abort();
    } else {
        if (size && data[0] == 0x01)
            fuzz_typed(j + 1, size - 1);
        else if (size && data[0] == 0x02)
            fuzz_result(j + 1, size - 1);
        else
            fuzz_json_request(j, size);
        jb_frame_leave(&frame);
    }
    if (fuzz_live != baseline)
        abort();
    free(j);
    return 0;
}
