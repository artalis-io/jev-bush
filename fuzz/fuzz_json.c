/*
 * libFuzzer target for Jev Bush's JSON reader and request validation.
 *
 * Build: clang -g -O1 -fsanitize=fuzzer,address,undefined \
 *        -fno-sanitize-recover=all fuzz/fuzz_json.c -lm -o fuzz_json
 * Run:   ./fuzz_json -dict=fuzz/json.dict corpus fuzz/corpus
 *
 * jb.c is included unchanged. Its die() ends in exit(), which libFuzzer
 * reports as a crash, so exit is redirected to a longjmp back to the fuzz
 * loop, and allocations are tracked so rejected inputs do not read as leaks.
 * Rejection is fine; memory errors, undefined behavior, and broken
 * invariants are not.
 */
#define _POSIX_C_SOURCE 200809L
#if defined(__MINGW32__) && !defined(__USE_MINGW_SETJMP_NON_SEH)
#define __USE_MINGW_SETJMP_NON_SEH
#endif
/* Every system header jb.c uses, included before the macros below so they
 * rename calls inside jb.c only, never declarations in system headers. */
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__AVX512F__)
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

typedef union FuzzHdr {
    struct {
        union FuzzHdr *prev, *next;
    } l;
    max_align_t align;
} FuzzHdr;
static FuzzHdr fuzz_live = {{&fuzz_live, &fuzz_live}};
static jmp_buf fuzz_jmp;
static int fuzz_must_succeed;

static void fuzz_link(FuzzHdr *h) {
    h->l.prev = &fuzz_live;
    h->l.next = fuzz_live.l.next;
    fuzz_live.l.next->l.prev = h;
    fuzz_live.l.next = h;
}
static void fuzz_unlink(FuzzHdr *h) {
    h->l.prev->l.next = h->l.next;
    h->l.next->l.prev = h->l.prev;
}
static void *fuzz_malloc(size_t n) {
    if (n > SIZE_MAX - sizeof(FuzzHdr))
        return NULL;
    FuzzHdr *h = malloc(sizeof *h + n);
    if (!h)
        return NULL;
    fuzz_link(h);
    return h + 1;
}
static void *fuzz_calloc(size_t n, size_t z) {
    if (z && n > SIZE_MAX / z)
        return NULL;
    void *p = fuzz_malloc(n * z);
    if (p)
        memset(p, 0, n * z);
    return p;
}
static void *fuzz_realloc(void *p, size_t n) {
    if (!p)
        return fuzz_malloc(n);
    if (n > SIZE_MAX - sizeof(FuzzHdr))
        return NULL;
    FuzzHdr *h = (FuzzHdr *)p - 1;
    fuzz_unlink(h);
    FuzzHdr *g = realloc(h, sizeof *g + n);
    if (!g) {
        fuzz_link(h);
        return NULL;
    }
    fuzz_link(g);
    return g + 1;
}
static void fuzz_free(void *p) {
    if (!p)
        return;
    FuzzHdr *h = (FuzzHdr *)p - 1;
    fuzz_unlink(h);
    free(h);
}
static void fuzz_free_all(void) {
    while (fuzz_live.l.next != &fuzz_live) {
        FuzzHdr *h = fuzz_live.l.next;
        fuzz_unlink(h);
        free(h);
    }
}
static void fuzz_exit(int code) {
    (void)code;
    if (fuzz_must_succeed)
        abort();
    longjmp(fuzz_jmp, 1);
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
    fuzz_free(c2);
    fuzz_free(t2);
    fuzz_free(c1);
}

static void fuzz_one(const char *j, size_t n) {
    int nt;
    JTok *t = json_tokens(j, n, &nt);
    fuzz_canonical(j, t, nt, 1);
    fuzz_canonical(j, t, nt, 0);
    fuzz_free(dg_text_of(j, t, nt, 0));
    fuzz_free(t);

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
    int *pick = fuzz_calloc((size_t)nq, sizeof *pick);
    for (int x = 0; x < nq; x++)
        pick[x] = w[x].nc - 1;
    fuzz_free(dg_answer_template(w, nq, pick));
    fuzz_free(pick);
    fuzz_free(sys);
    dg_questions_free(w, nq);
    for (int i = 0; i < 128; i++)
        fuzz_free(labels[i]);
    dg_request_free(&rq);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    /* Exact-size copy, not NUL-terminated, as eval rows are. */
    char *j = malloc(size ? size : 1);
    if (!j)
        return 0;
    if (size)
        memcpy(j, data, size);
    fuzz_must_succeed = 0;
    if (!setjmp(fuzz_jmp))
        fuzz_one(j, size);
    fuzz_free_all();
    free(j);
    return 0;
}
