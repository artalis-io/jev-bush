/*
 * Jev Bush -- bounded decisions with DiffusionGemma, on CPUs and, bit for
 * bit the same, on CUDA.
 *
 * Build: cc -O3 -march=native -ffast-math -std=c11 -Wall -Wextra
 *        -pedantic -fopenmp jb.c -lm -o jb
 * The engine reads the exact public DiffusionGemma safetensors layout directly.
 */
#define _POSIX_C_SOURCE 200809L
#if defined(__MINGW32__) && !defined(__USE_MINGW_SETJMP_NON_SEH)
/* MinGW's default setjmp makes longjmp unwind through SEH, which crashes in
 * optimized builds; errors here only need a plain register restore. */
#define __USE_MINGW_SETJMP_NON_SEH
#endif
#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "jb.h"

/* Evaluate floating-point expressions as written, never fusing a multiply
 * and an add. GCC does this in ISO C mode (-std=c11); clang fuses within a
 * statement unless told not to. The reference kernels' rounding, which the
 * NEON tier reproduces bit for bit, depends on it. Explicit FMA intrinsics
 * are unaffected. */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

/* GCC and clang build x86 ISA variants as function-level multiversions, so a
 * single baseline binary safely selects scalar, AVX2 or AVX-512 at runtime.
 * MSVC emits the variants enabled by /arch and still checks the host before
 * selecting one. -DJB_SCALAR remains the portable oracle. */
#if defined(JB_SCALAR)
#define JB_TARGET_AVX2
#define JB_TARGET_AVX512
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define JB_HAVE_AVX2 1
#define JB_HAVE_AVX512 1
#define JB_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define JB_TARGET_AVX512 __attribute__((target("avx2,fma,avx512f,avx512dq,avx512bw")))
#include <immintrin.h>
#elif defined(_MSC_VER) && defined(_M_X64) && defined(__AVX2__)
#define JB_HAVE_AVX2 1
#define JB_TARGET_AVX2
#define JB_TARGET_AVX512
#include <intrin.h>
#include <immintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON) && !defined(__AARCH64EB__)
/* Little-endian only: the kernels load BF16 bytes in little-endian order. */
#define JB_NEON 1
#include <arm_neon.h>
#define JB_TARGET_AVX2
#define JB_TARGET_AVX512
#else
#define JB_TARGET_AVX2
#define JB_TARGET_AVX512
#endif
#ifdef _OPENMP
/* Parallel loops declare a signed index before the loop, each in its own
 * block: MSVC implements OpenMP 2.0, which requires that form. */
#include <omp.h>
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

/* Lets GCC and clang check the arguments of printf-style helpers. MinGW's
 * C runtime takes the C99 formats, which plain printf there would not. */
#if defined(__MINGW32__)
#define JB_PRINTF(f, a) __attribute__((format(gnu_printf, f, a)))
#elif defined(__GNUC__) || defined(__clang__)
#define JB_PRINTF(f, a) __attribute__((format(printf, f, a)))
#else
#define JB_PRINTF(f, a)
#endif

#define JB_VERSION "0.3.0"
#define JB_MAX_CTX 4096
#define JB_MAX_CAND 255
#define JB_MAX_JSON (64u * 1024u * 1024u)
#define JB_MAX_DEPTH 256
/* Each microbatch document holds its own copy of the shared prefix K/V.
 * Groups whose K/V would exceed this many bytes run sequentially. */
#ifndef JB_MICROBATCH_KV_LIMIT
#define JB_MICROBATCH_KV_LIMIT (4ull << 30)
#endif

#define DG_H 2816
#define DG_L 30
#define DG_HEADS 16
#define DG_VOCAB 262144
#define DG_DENSE 2112
#define DG_MOE 704
#define DG_TOPK 8
#define DG_EXPERTS 128
#define DG_EXPERT_PAIR (2 * DG_MOE)
#define DG_FULL_EVERY 6
#define DG_FULL_HEAD_DIM 512
#define DG_LOCAL_HEAD_DIM 256
#define DG_FULL_KV_HEADS 2
#define DG_LOCAL_KV_HEADS 8
#define DG_FULL_ROPE_DIM 64
#define DG_LOCAL_WINDOW 1024
#define DG_CHOICE_LABELS 128
#define DG_MAX_QUESTIONS 1024
/* The automatic reread test scores every answer slot in one pass, which is
 * limited to 128 slots; explicit samples have no such limit. */
#define DG_AUTO_READ_MAX_QUESTIONS 128
#define DG_MAX_SCORE 10
#define DG_CANVAS_TOKENS 64
#define DG_CANVAS_ALIGN 16
#define DG_CANVAS_GROUP 8
#define DG_CANVAS_MASK_TOKEN 106

#if defined(__FAST_MATH__)
#define JB_MATH_MODE "fast"
#define JB_STRICT_MATH 0
#else
#define JB_MATH_MODE "strict"
#define JB_STRICT_MATH 1
#endif

#ifdef JB_PROFILE
typedef struct {
    uint64_t layers, attention, attention_qkv, attention_prepare;
    uint64_t attention_kv, attention_core, attention_output;
    uint64_t dense, router, experts, ff_other, embedding, final_norm, kv_free;
    uint64_t moe_input_qdq, moe_gated;
    uint64_t moe_hidden_qdq, moe_down;
    uint64_t prompt_tokens, system_tokens, state_tokens;
    uint64_t omp_regions, alloc_calls, hot_alloc_calls, alloc_ns;
    int hot;
} JBProfile;

/* A request's counters live in its session's workspace. This points at the
 * running request's counters, or is NULL (kernel self-tests, benchmarks),
 * for code that has no session in scope: allocators and kernels. */
static _Thread_local JBProfile *jb_profile_active;
#define JB_TICK(name) uint64_t name = now_ns()
#define JB_TO(field, name)                                                                         \
    (jb_profile_active ? (void)(jb_profile_active->field += now_ns() - (name)) : (void)0)
#define JB_OMP() (jb_profile_active ? (void)jb_profile_active->omp_regions++ : (void)0)
#else
#define JB_TICK(name)
#define JB_TO(field, name)
#define JB_OMP()
#endif

typedef struct {
    int id;
    char *s;
    uint32_t n;
} Vocab;

typedef struct {
    uint32_t a, b, rank;
} Merge;

typedef struct {
    uint8_t *map;
    uint64_t size;
    int mapped;
#if defined(_WIN32)
    HANDLE hf, hm;
#endif
} FileMap;

typedef struct {
    int *v;
    uint32_t n, cap;
} Tokens;

typedef struct {
    void *allocation;
    unsigned char *base;
    size_t size, used, peak;
} JBArena;

static uint64_t now_ns(void);

/* Error handling and allocation ownership.
 *
 * A public API call enters an error frame, and die() longjmps back to it.
 * Every allocation made on a thread while it has a frame is linked into that
 * thread's tracked list, so a failed call releases everything it allocated
 * since entering its frame, however deep the failure. When the outermost
 * call succeeds, its allocations stop being tracked: results then belong to
 * the caller and session state to the session. Storage installed into a
 * long-lived object partway through a call (session arenas, the prefix
 * cache) is detached with jb_untrack() so a later failure cannot free it.
 * Tracked allocations are released only by the thread that made them; OpenMP
 * workers allocate untracked. Every fallible public entry crosses a frame,
 * the CLI installs one root frame, and destructors are non-failing. CI checks
 * those call-graph invariants as well as every OpenMP region. Reaching die()
 * without a frame therefore means an internal call-contract violation. */
/* max_align_t is missing from MSVC's C mode; align the header for the
 * strictest standard types instead so every allocation stays aligned. */
typedef union {
    long double ld;
    long long ll;
    double d;
    void *p;
    void (*f)(void);
} JBMaxAlign;

typedef union JBAllocation {
    struct {
        union JBAllocation *previous, *next;
        jb_allocator allocator;
        size_t size;
        uint64_t sequence;
        int tracked;
    } link;

    JBMaxAlign align;
} JBAllocation;

typedef struct JBErrorFrame {
    jmp_buf jump;
    struct JBErrorFrame *previous;
    uint64_t mark;
    jb_status previous_status;
} JBErrorFrame;

static _Thread_local JBErrorFrame *jb_error_frame;
static _Thread_local jb_status jb_error_status;
static _Thread_local char jb_error_message[256];
static _Thread_local JBAllocation *jb_tracked;
static _Thread_local uint64_t jb_allocation_sequence;
static _Thread_local const jb_allocator *jb_active_allocator;

static void *jb_system_allocate(void *context, size_t size) {
    (void)context;
    return malloc(size);
}

static void *jb_system_reallocate(void *context, void *allocation, size_t size) {
    (void)context;
    return realloc(allocation, size);
}

static void jb_system_release(void *context, void *allocation) {
    (void)context;
    free(allocation);
}

static const jb_allocator jb_system_allocator = {sizeof(jb_allocator), NULL, jb_system_allocate,
                                                 jb_system_reallocate, jb_system_release};

static const jb_allocator *jb_allocator_current(void) {
    return jb_active_allocator ? jb_active_allocator : &jb_system_allocator;
}

static int jb_in_parallel(void) {
#ifdef _OPENMP
    return omp_in_parallel();
#else
    return 0;
#endif
}

/* die() must never run inside an OpenMP parallel region: workers have no
 * error frame, and a longjmp out of the region is undefined. Reaching it
 * there is an engine bug; stop loudly. */
static void jb_forbid_parallel(const char *a, const char *b) {
    if (jb_in_parallel()) {
        fprintf(stderr, "jb: internal error: failure inside a parallel region: %s%s%s\n", a,
                b ? ": " : "", b ? b : "");
        abort();
    }
}

static void die(const char *s) {
    jb_forbid_parallel(s, NULL);
    if (jb_error_frame) {
        snprintf(jb_error_message, sizeof jb_error_message, "%s", s);
        longjmp(jb_error_frame->jump, 1);
    }
    fprintf(stderr, "jb: %s\n", s);
    exit(2);
}

static void die2(const char *a, const char *b) {
    jb_forbid_parallel(a, b);
    if (jb_error_frame) {
        snprintf(jb_error_message, sizeof jb_error_message, "%s: %s", a, b);
        longjmp(jb_error_frame->jump, 1);
    }
    fprintf(stderr, "jb: %s: %s\n", a, b);
    exit(2);
}

static void die_status(jb_status status, const char *message) {
    if (jb_error_frame)
        jb_error_status = status;
    die(message);
}

static void die2_status(jb_status status, const char *a, const char *b) {
    if (jb_error_frame)
        jb_error_status = status;
    die2(a, b);
}

#ifndef JB_NO_MAIN
static void flush_output(void) {
    if (fflush(stdout) == EOF || ferror(stdout))
        die("cannot write output");
}
#endif

static void jb_link(JBAllocation *a) {
    a->link.previous = NULL;
    a->link.next = jb_tracked;
    if (jb_tracked)
        jb_tracked->link.previous = a;
    jb_tracked = a;
}

static void jb_unlink(JBAllocation *a) {
    if (a->link.previous)
        a->link.previous->link.next = a->link.next;
    else
        jb_tracked = a->link.next;
    if (a->link.next)
        a->link.next->link.previous = a->link.previous;
}

static void jb_out_of_memory(void) {
    if (jb_error_frame)
        jb_error_status = JB_ERROR_OUT_OF_MEMORY;
    die("out of memory");
}

/* Returns NULL on failure, for callers that must release something else
 * (an open file) before reporting out of memory. */
static void *jb_try_allocate(size_t n, int clear) {
    if (n > SIZE_MAX - sizeof(JBAllocation))
        return NULL;
#ifdef JB_PROFILE
    uint64_t start = now_ns();
#endif
    const jb_allocator *allocator = jb_allocator_current();
    JBAllocation *a = allocator->allocate(allocator->context, sizeof *a + n);
#ifdef JB_PROFILE
    if (jb_profile_active) {
        jb_profile_active->alloc_ns += now_ns() - start;
        jb_profile_active->alloc_calls++;
        jb_profile_active->hot_alloc_calls += (uint64_t)jb_profile_active->hot;
    }
#endif
    if (!a)
        return NULL;
    if (clear)
        memset(a, 0, sizeof *a + n);
    a->link.allocator = *allocator;
    a->link.size = n;
    a->link.sequence = ++jb_allocation_sequence;
#ifndef NDEBUG
    /* Allocations inside parallel regions stay untracked, so a later failure
     * of the call could not release them. None exist; keep it that way. */
    if (jb_error_frame && jb_in_parallel()) {
        fprintf(stderr, "jb: internal error: allocation inside a parallel region\n");
        abort();
    }
#endif
    a->link.tracked = jb_error_frame && !jb_in_parallel();
    a->link.previous = a->link.next = NULL;
    if (a->link.tracked)
        jb_link(a);
    return a + 1;
}

static void *jb_allocate(size_t n, int clear) {
    void *p = jb_try_allocate(n, clear);
    if (!p)
        jb_out_of_memory();
    return p;
}

static void *xmalloc(size_t n) {
    return jb_allocate(n ? n : 1, 0);
}

static void *xcalloc(size_t n, size_t z) {
    if (!n)
        n = 1;
    if (z && n > SIZE_MAX / z)
        jb_out_of_memory();
    return jb_allocate(n * z, 1);
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *z = xmalloc(n);
    memcpy(z, s, n);
    return z;
}

static void *xrealloc(void *p, size_t n) {
    if (!p)
        return xmalloc(n);
    if (!n)
        n = 1;
    if (n > SIZE_MAX - sizeof(JBAllocation))
        jb_out_of_memory();
    JBAllocation *a = (JBAllocation *)p - 1;
    int tracked = a->link.tracked;
    if (tracked)
        jb_unlink(a);
#ifdef JB_PROFILE
    uint64_t start = now_ns();
#endif
    JBAllocation *b = a->link.allocator.reallocate(a->link.allocator.context, a, sizeof *a + n);
#ifdef JB_PROFILE
    if (jb_profile_active) {
        jb_profile_active->alloc_ns += now_ns() - start;
        jb_profile_active->alloc_calls++;
        jb_profile_active->hot_alloc_calls += (uint64_t)jb_profile_active->hot;
    }
#endif
    if (!b) {
        if (tracked)
            jb_link(a);
        jb_out_of_memory();
    }
    if (tracked)
        jb_link(b);
    b->link.size = n;
    return b + 1;
}

static void jb_release(void *p) {
    if (!p)
        return;
    JBAllocation *a = (JBAllocation *)p - 1;
    if (a->link.tracked)
        jb_unlink(a);
    a->link.allocator.release(a->link.allocator.context, a);
}

/* Detach an allocation that a long-lived object now owns. */
static void jb_untrack(void *p) {
    if (!p)
        return;
    JBAllocation *a = (JBAllocation *)p - 1;
    if (a->link.tracked) {
        jb_unlink(a);
        a->link.tracked = 0;
        a->link.previous = a->link.next = NULL;
    }
}

/* Fields are set before setjmp and not modified afterwards, and the status
 * lives in thread-local storage, so all of them are valid after a longjmp. */
static void jb_frame_enter(JBErrorFrame *f, jb_status status) {
    f->previous = jb_error_frame;
    f->previous_status = jb_error_status;
    f->mark = jb_allocation_sequence;
    jb_error_status = status;
    jb_error_frame = f;
}

/* Successful return: the outermost frame hands every allocation it made to
 * its owner (caller or session). */
static void jb_frame_leave(JBErrorFrame *f) {
    jb_error_frame = f->previous;
    jb_error_status = f->previous_status;
    if (!jb_error_frame)
        while (jb_tracked) {
            JBAllocation *a = jb_tracked;
            jb_tracked = a->link.next;
            a->link.tracked = 0;
            a->link.previous = a->link.next = NULL;
        }
}

/* Failed return, including an inner call's failure: release everything
 * allocated since the frame was entered. */
static void jb_frame_abandon(JBErrorFrame *f) {
    for (JBAllocation *a = jb_tracked, *next; a; a = next) {
        next = a->link.next;
        if (a->link.sequence > f->mark) {
            jb_unlink(a);
            a->link.allocator.release(a->link.allocator.context, a);
        }
    }
    jb_frame_leave(f);
}

/* Called after die() longjmps to the frame. */
static jb_status jb_frame_fail(JBErrorFrame *f) {
    jb_status status = jb_error_status;
    jb_frame_abandon(f);
    return status;
}

static size_t jb_size_add(size_t a, size_t b) {
    if (b > SIZE_MAX - a)
        die("allocation size overflow");
    return a + b;
}

static size_t jb_size_mul(size_t a, size_t b) {
    if (a && b > SIZE_MAX / a)
        die("allocation size overflow");
    return a * b;
}

static size_t jb_align_up(size_t n, size_t alignment) {
    if (!alignment || (alignment & (alignment - 1)))
        die_status(JB_ERROR_INTERNAL, "invalid arena alignment");
    return jb_size_add(n, alignment - 1) & ~(alignment - 1);
}

static void jb_arena_init(JBArena *a, size_t size) {
    const size_t alignment = 64;
    size_t allocation_size = jb_size_add(size ? size : 1, alignment - 1);
    a->allocation = xmalloc(allocation_size);
    uintptr_t p = (uintptr_t)a->allocation;
    a->base = (unsigned char *)((p + alignment - 1) & ~(uintptr_t)(alignment - 1));
    a->size = size;
    a->used = 0;
    a->peak = 0;
}

static void *jb_arena_alloc(JBArena *a, size_t count, size_t element_size, int clear) {
    const size_t alignment = 64;
    size_t start = jb_align_up(a->used, alignment);
    size_t bytes = jb_size_mul(count, element_size);
    size_t end = jb_size_add(start, bytes);
    if (end > a->size)
        die_status(JB_ERROR_INTERNAL, "inference workspace plan is too small");
    void *p = a->base + start;
    a->used = end;
    if (end > a->peak)
        a->peak = end;
    if (clear)
        memset(p, 0, bytes);
    return p;
}

static size_t jb_arena_mark(const JBArena *a) {
    return a->used;
}

static void jb_arena_reset(JBArena *a, size_t mark) {
    if (mark > a->used)
        die_status(JB_ERROR_INTERNAL, "invalid arena reset");
    a->used = mark;
}

static void jb_arena_free(JBArena *a) {
    jb_release(a->allocation);
    memset(a, 0, sizeof *a);
}

/* Reuse a grow-only arena for a new plan. Growth allocates the replacement
 * before releasing the old block, so a failed allocation leaves the arena
 * and its recorded capacity intact for later requests. */
static void jb_arena_reserve(JBArena *a, size_t *capacity, size_t required) {
    if (required > *capacity) {
        JBArena grown;
        jb_arena_init(&grown, required);
        jb_untrack(grown.allocation);
        if (a->allocation)
            jb_arena_free(a);
        *a = grown;
        *capacity = required;
    } else {
        a->size = required;
        a->used = 0;
        a->peak = 0;
    }
}

static void jb_workspace_plan(size_t *used, size_t count, size_t element_size);

/* Bit-level finiteness tests: -ffast-math lets the compiler assume NaN and
 * infinity never occur, which can fold isfinite() and NaN comparisons away.
 * Clang goes further and marks floating-point arguments as never NaN or
 * infinite, which folds even a bit test on the argument. A volatile round
 * trip hides the value, so the bits tested are the bits actually passed. */
static int jb_finitef(float x) {
    volatile float hidden = x;
    float value = hidden;
    uint32_t u;
    memcpy(&u, &value, sizeof u);
    return (u & 0x7f800000u) != 0x7f800000u;
}

static int jb_finite(double x) {
    volatile double hidden = x;
    double value = hidden;
    uint64_t u;
    memcpy(&u, &value, sizeof u);
    return (u & 0x7ff0000000000000ull) != 0x7ff0000000000000ull;
}

/* Threads OpenMP will use for parallel regions; 1 without OpenMP. */
static int jb_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

JB_PRINTF(3, 4) static void jb_path(char *out, size_t cap, const char *fmt, ...) {
    va_list a;
    va_start(a, fmt);
    int n = vsnprintf(out, cap, fmt, a);
    va_end(a);
    if (n < 0 || (size_t)n >= cap)
        die("model path too long");
}

static uint64_t now_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter, frequency;
    if (QueryPerformanceCounter(&counter) && QueryPerformanceFrequency(&frequency) &&
        frequency.QuadPart > 0)
        return (uint64_t)((long double)counter.QuadPart * 1000000000.0L /
                          (long double)frequency.QuadPart);
#else
    struct timespec t;
    if (!clock_gettime(CLOCK_MONOTONIC, &t))
        return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
#endif
    die_status(JB_ERROR_INTERNAL, "monotonic clock unavailable");
    return 0;
}

static void map_file(FileMap *m, const char *path) {
#if defined(_WIN32)
    LARGE_INTEGER z;
    m->hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (m->hf == INVALID_HANDLE_VALUE) {
        m->hf = NULL;
        die2_status(JB_ERROR_IO, "open", path);
    }
    if (!GetFileSizeEx(m->hf, &z) || z.QuadPart <= 0)
        die_status(JB_ERROR_IO, "GetFileSizeEx failed");
    m->size = (uint64_t)z.QuadPart;
    if (m->size > SIZE_MAX)
        die("model is too large for this address space");
    m->hm = CreateFileMappingA(m->hf, NULL, PAGE_READONLY, 0, 0, NULL);
    if (m->hm) {
        m->map = MapViewOfFile(m->hm, FILE_MAP_READ, 0, 0, 0);
        m->mapped = m->map != NULL;
    }
#else
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0)
        die2_status(JB_ERROR_IO, "open", path);
    if (fstat(fd, &st)) {
        close(fd);
        die2_status(JB_ERROR_IO, "open", path);
    }
    if (st.st_size <= 0 || (uintmax_t)st.st_size > SIZE_MAX) {
        close(fd);
        die_status(JB_ERROR_IO, "invalid model size");
    }
    m->size = (uint64_t)st.st_size;
    m->map = mmap(NULL, (size_t)m->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m->map != MAP_FAILED)
        m->mapped = 1;
    else
        m->map = NULL;
#endif
    if (!m->map) {
        m->map = xmalloc((size_t)m->size);
        FILE *f = fopen(path, "rb");
        if (!f)
            die2_status(JB_ERROR_IO, "open", path);
        size_t got = fread(m->map, 1, (size_t)m->size, f);
        fclose(f);
        if (got != (size_t)m->size)
            die_status(JB_ERROR_IO, "short read");
    }
}

static void unmap_file(FileMap *m) {
#if defined(_WIN32)
    if (m->mapped)
        UnmapViewOfFile(m->map);
    else
        jb_release(m->map);
    if (m->hm)
        CloseHandle(m->hm);
    if (m->hf)
        CloseHandle(m->hf);
#else
    if (m->mapped)
        munmap(m->map, (size_t)m->size);
    else
        jb_release(m->map);
#endif
}

static int cmp_vocab(const void *a, const void *b) {
    const Vocab *x = a, *y = b;
    size_t n = x->n < y->n ? x->n : y->n;
    int c = memcmp(x->s, y->s, n);
    return c ? c : (x->n > y->n) - (x->n < y->n);
}

/* bsearch's key for a byte span, which a Vocab could only hold by casting
 * away const. */
typedef struct {
    const char *s;
    uint32_t n;
} VocabKey;

static int cmp_vocab_key(const void *a, const void *b) {
    const VocabKey *x = a;
    const Vocab *y = b;
    size_t n = x->n < y->n ? x->n : y->n;
    int c = memcmp(x->s, y->s, n);
    return c ? c : (x->n > y->n) - (x->n < y->n);
}

static int cmp_merge(const void *a, const void *b) {
    const Merge *x = a, *y = b;
    return x->a != y->a ? (x->a > y->a) - (x->a < y->a) : (x->b > y->b) - (x->b < y->b);
}

static void push(Tokens *t, int x) {
    if (t->n == t->cap) {
        if (t->cap > UINT32_MAX / 2)
            die("token array size overflow");
        t->cap = t->cap ? t->cap * 2 : 64;
        t->v = xrealloc(t->v, (size_t)t->cap * sizeof *t->v);
    }
    t->v[t->n++] = x;
}

static int put_utf8(uint32_t c, char *z) {
    if (c <= 0x7f) {
        z[0] = (char)c;
        return 1;
    }
    if (c <= 0x7ff) {
        z[0] = (char)(0xc0 | c >> 6);
        z[1] = (char)(0x80 | (c & 63));
        return 2;
    }
    if (c <= 0xffff) {
        z[0] = (char)(0xe0 | c >> 12);
        z[1] = (char)(0x80 | ((c >> 6) & 63));
        z[2] = (char)(0x80 | (c & 63));
        return 3;
    }
    z[0] = (char)(0xf0 | c >> 18);
    z[1] = (char)(0x80 | ((c >> 12) & 63));
    z[2] = (char)(0x80 | ((c >> 6) & 63));
    z[3] = (char)(0x80 | (c & 63));
    return 4;
}

static uint32_t next_cp(const unsigned char *s, size_t n, size_t *i) {
    size_t p = (*i)++;
    uint32_t c = s[p];
    if (c < 0x80)
        return c;
    int more = c >= 0xf0 ? 3 : c >= 0xe0 ? 2 : c >= 0xc2 ? 1 : -1;
    if (more < 0 || p + (size_t)more >= n)
        die("invalid UTF-8");
    uint32_t v = c & (0x7f >> more);
    for (int k = 0; k < more; k++) {
        unsigned q = s[(*i)++];
        if ((q & 0xc0) != 0x80)
            die("invalid UTF-8");
        v = (v << 6) | (q & 63);
    }
    if ((more == 1 && v < 0x80) || (more == 2 && v < 0x800) || (more == 3 && v < 0x10000) ||
        v > 0x10ffff || (v >= 0xd800 && v <= 0xdfff))
        die("invalid UTF-8");
    return v;
}

enum { JT_UNDEF, JT_OBJECT, JT_ARRAY, JT_STRING, JT_PRIMITIVE };

typedef struct {
    int type, start, end, parent, size;
} JTok;

typedef struct {
    const char *s;
    size_t n, pos;
    JTok *t;
    int nt, cap, depth;
} JParser;

static char *jt_string(const char *j, const JTok *t);

static int jt_new(JParser *p, int type, int start, int parent) {
    if (p->nt == p->cap) {
        if (p->cap > INT32_MAX / 2)
            die("JSON token array size overflow");
        p->cap = p->cap ? p->cap * 2 : 256;
        p->t = xrealloc(p->t, (size_t)p->cap * sizeof *p->t);
    }
    int i = p->nt++;
    p->t[i] = (JTok){type, start, -1, parent, 0};
    if (parent >= 0)
        p->t[parent].size++;
    return i;
}

static int json_parse_value(JParser *p, int parent);

static int json_primitive_valid(const char *s, size_t n) {
    if ((n == 4 && (!memcmp(s, "true", 4) || !memcmp(s, "null", 4))) ||
        (n == 5 && !memcmp(s, "false", 5)))
        return 1;
    size_t i = 0;
    if (i < n && s[i] == '-')
        i++;
    if (i >= n)
        return 0;
    if (s[i] == '0')
        i++;
    else {
        if (s[i] < '1' || s[i] > '9')
            return 0;
        while (i < n && isdigit((unsigned char)s[i]))
            i++;
    }
    if (i < n && s[i] == '.') {
        i++;
        size_t begin = i;
        while (i < n && isdigit((unsigned char)s[i]))
            i++;
        if (i == begin)
            return 0;
    }
    if (i < n && (s[i] == 'e' || s[i] == 'E')) {
        i++;
        if (i < n && (s[i] == '+' || s[i] == '-'))
            i++;
        size_t begin = i;
        while (i < n && isdigit((unsigned char)s[i]))
            i++;
        if (i == begin)
            return 0;
    }
    return i == n;
}

static int json_parse_string_tok(JParser *p, int parent) {
    int i = jt_new(p, JT_STRING, (int)++p->pos, parent);
    while (p->pos < p->n) {
        unsigned char c = p->s[p->pos++];
        if (c == '\"') {
            p->t[i].end = (int)p->pos - 1;
            return i;
        }
        if (c == '\\') {
            if (p->pos >= p->n)
                die("bad JSON escape");
            c = p->s[p->pos++];
            if (c == 'u') {
                for (int k = 0; k < 4; k++)
                    if (p->pos >= p->n || !isxdigit((unsigned char)p->s[p->pos++]))
                        die("bad JSON unicode escape");
            } else if (!c || !strchr("\"\\/bfnrt", c))
                die("bad JSON escape");
        } else if (c < 0x20)
            die("control byte in JSON string");
    }
    die("unterminated JSON string");
    return -1;
}

static int json_parse_value(JParser *p, int parent) {
    while (p->pos < p->n && isspace((unsigned char)p->s[p->pos]))
        p->pos++;
    if (p->pos >= p->n)
        die("missing JSON value");
    char c = p->s[p->pos];
    if (c == '\"')
        return json_parse_string_tok(p, parent);
    if (c == '{' || c == '[') {
        int ty = c == '{' ? JT_OBJECT : JT_ARRAY, i = jt_new(p, ty, (int)p->pos++, parent);
        if (++p->depth > JB_MAX_DEPTH)
            die("JSON nesting too deep");
        while (1) {
            while (p->pos < p->n && isspace((unsigned char)p->s[p->pos]))
                p->pos++;
            if (p->pos >= p->n)
                die("unterminated JSON container");
            if (p->s[p->pos] == (ty == JT_OBJECT ? '}' : ']')) {
                p->t[i].end = (int)++p->pos;
                p->depth--;
                return i;
            }
            if (p->t[i].size) {
                if (p->s[p->pos++] != ',')
                    die("expected JSON comma");
                while (p->pos < p->n && isspace((unsigned char)p->s[p->pos]))
                    p->pos++;
            }
            if (ty == JT_OBJECT) {
                if (p->pos >= p->n || p->s[p->pos] != '\"')
                    die("JSON object key must be a string");
                json_parse_string_tok(p, i);
                while (p->pos < p->n && isspace((unsigned char)p->s[p->pos]))
                    p->pos++;
                if (p->pos >= p->n || p->s[p->pos++] != ':')
                    die("expected JSON colon");
                json_parse_value(p, i);
            } else
                json_parse_value(p, i);
        }
    }
    int i = jt_new(p, JT_PRIMITIVE, (int)p->pos, parent);
    while (p->pos < p->n && !isspace((unsigned char)p->s[p->pos]) && !strchr(",]}:", p->s[p->pos]))
        p->pos++;
    p->t[i].end = (int)p->pos;
    if (p->t[i].end == p->t[i].start)
        die("bad JSON primitive");
    if (!json_primitive_valid(p->s + p->t[i].start, (size_t)(p->t[i].end - p->t[i].start)))
        die("bad JSON primitive");
    return i;
}

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

/* Duplicate keys are rejected: lookups would take the first value where
 * Python's json keeps the last, and canonical seeds would diverge. */
static void json_check_keys(const char *s, const JTok *t, int nt) {
    char **keys = NULL;
    int cap = 0;
    for (int obj = 0; obj < nt; obj++) {
        if (t[obj].type != JT_OBJECT || t[obj].size < 4)
            continue;
        int nk = 0;
        for (int i = obj + 1; i < nt && t[i].parent >= obj; i++)
            if (t[i].parent == obj) {
                if (nk == cap) {
                    if (cap > INT32_MAX / 2)
                        die("JSON key array size overflow");
                    cap = cap ? cap * 2 : 64;
                    keys = xrealloc(keys, (size_t)cap * sizeof *keys);
                }
                keys[nk++] = jt_string(s, &t[i]);
                i++;
            }
        if (nk > 1)
            qsort(keys, (size_t)nk, sizeof *keys, cmp_str);
        for (int i = 0; i < nk; i++) {
            if (i && !strcmp(keys[i - 1], keys[i])) {
                char message[256];
                snprintf(message, sizeof message, "duplicate JSON object key: %s", keys[i]);
                for (int k = 0; k < nk; k++)
                    jb_release(keys[k]);
                jb_release(keys);
                die(message);
            }
        }
        for (int i = 0; i < nk; i++)
            jb_release(keys[i]);
    }
    jb_release(keys);
}

static JTok *json_tokens(const char *s, size_t n, int *nt) {
    /* Token offsets are int; every caller also enforces this limit earlier. */
    if (n > JB_MAX_JSON)
        die("JSON input is too large");
    JParser parser = {s, n, 0, 0, 0, 0, 0};
    JParser *p = &parser;
    json_parse_value(p, -1);
    while (p->pos < n && isspace((unsigned char)s[p->pos]))
        p->pos++;
    if (p->pos != n)
        die("trailing JSON data");
    json_check_keys(s, p->t, p->nt);
    *nt = p->nt;
    return p->t;
}

static int jt_eq(const char *j, const JTok *t, const char *z) {
    if (t->type != JT_STRING)
        return 0;
    size_t n = (size_t)(t->end - t->start);
    if (!memchr(j + t->start, '\\', n))
        return n == strlen(z) && !memcmp(j + t->start, z, n);
    char *s = jt_string(j, t);
    int eq = !strcmp(s, z);
    jb_release(s);
    return eq;
}

static int jt_literal(const char *j, const JTok *t, const char *z) {
    return t->type == JT_PRIMITIVE && t->end - t->start == (int)strlen(z) &&
           !memcmp(j + t->start, z, strlen(z));
}

static int jt_obj_get(const char *j, JTok *t, int nt, int obj, const char *key) {
    if (obj < 0 || obj >= nt || t[obj].type != JT_OBJECT)
        return -1;
    for (int i = obj + 1; i + 1 < nt; i++) {
        if (t[i].parent < obj)
            break;
        if (t[i].parent == obj && t[i].type == JT_STRING) {
            if (jt_eq(j, &t[i], key))
                return i + 1;
            i++;
        }
    }
    return -1;
}

static int jt_nonnegative_int(const char *j, const JTok *t, const char *name) {
    if (t->type != JT_PRIMITIVE || t->end <= t->start || t->end - t->start > 9)
        die2("expected non-negative integer", name);
    int v = 0;
    for (int i = t->start; i < t->end; i++) {
        if (j[i] < '0' || j[i] > '9')
            die2("expected non-negative integer", name);
        v = v * 10 + j[i] - '0';
    }
    return v;
}

typedef struct {
    FileMap file;
    char path[768];
} DGShard;

typedef struct {
    char *name;
    const uint8_t *data;
    /* CPU-native strict AVX-512 image, owned by DGModel::exact_weights. */
    const uint8_t *exact;
    size_t exact_offset;
    uint64_t shape[4], bytes;
    int nd, dtype;
    /* The tensor's copy on the accelerator, or 0. */
    uint64_t device;
} DGTensor;

/* One NVFP4 weight matrix: packed E2M1 weights, E4M3 block scales, and the
 * global scale. */
typedef struct {
    const uint8_t *w, *s;
    const uint8_t *exact_w;
    const uint8_t *exact_s;
    float global;
} DGNvMatrix;

typedef struct {
    const char *name;
    void (*nvfp4_qdq)(float *, const float *, int, int, float);
    void (*nvfp4_layout)(float *, int, int);
    void (*nvfp4_mm)(const uint8_t *, const uint8_t *, float, const float *, float *, int, int,
                     int);
    /* h = gelu(gate x) * (up x), both products computed as nvfp4_mm does;
     * g is scratch for tokens * rows floats. */
    void (*nvfp4_gated)(const DGNvMatrix *gate, const DGNvMatrix *up, const float *x, float *g,
                        float *h, int tokens, int rows, int cols);
    void (*bf16_mm)(const uint8_t *, const float *, float *, int, int, int);
    void (*rms)(float *, const float *, const DGTensor *, int);
    double (*dot)(const float *, const float *, int);
    /* out[j] = (float)dot(a, b + j * stride, n) for j < keys. */
    void (*dots)(const float *a, const float *b, size_t stride, int keys, int n, float *out);
    /* Bit-identical to the reference kernels in strict builds. */
    int matches_reference;
} DGKernelOps;

static const DGKernelOps *dg_kernels(void);

/* A device address; 0 is none. */
typedef uint64_t DGDevice;

typedef struct DGAccelMatrix DGAccelMatrix;

/* An optional accelerator for the transformer layers, bit-identical to the
 * reference in strict builds and about as close as fast math in fast ones.
 * The caller places weights in device memory (alloc, to_device) and runs
 * each step of a layer on device addresses, in the order it issues them;
 * to_host waits for the steps before it. dg_accel() returns NULL when
 * none is built or available; alloc returns 0 when the device is out of
 * memory. free, host_free and release never fail: they run on cleanup
 * paths, including a failed call's and ones without an error frame, and a
 * device left unusable by an earlier fault fails every later call. The other
 * operations report failures through die(). */
typedef struct {
    const char *name;
    /* A handle for an NVFP4 matrix whose packed weights and block scales the
     * caller placed at w and s; release frees the handle only. */
    DGAccelMatrix *(*matrix)(DGDevice w, DGDevice s, float global, int rows, int cols);
    void (*release)(DGAccelMatrix *m);
    DGDevice (*alloc)(size_t bytes);
    void (*free)(DGDevice p);
    void (*to_device)(DGDevice dst, const void *src, size_t bytes);
    void (*to_host)(void *dst, DGDevice src, size_t bytes);
    /* Page-locked host memory, which device copies reach directly; NULL
     * when unavailable. */
    void *(*host_alloc)(size_t bytes);
    void (*host_free)(void *p);
    /* Scratch nvfp4_mm needs for this many tokens. */
    size_t (*nvfp4_scratch)(int tokens);
    /* Row t of y = m[group[t]] times row t of x, as dg_nvfp4_mm_ref computes
     * it, with x as nvfp4_qdq writes it; all of m share rows and cols, and
     * group is host memory. */
    void (*nvfp4_mm)(const DGAccelMatrix *const *m, const int *group, int tokens, DGDevice x,
                     DGDevice y, DGDevice scratch);
    /* y = w x for a row-major BF16 w, as dg_mm_data_ref computes it. */
    void (*bf16_mm)(DGDevice w, DGDevice x, DGDevice y, int tokens, int rows, int cols);
    /* Per token, as dg_rms_ref and its callers compute it: y = x * q * scale
     * (BF16; 1 when 0), then times factor / divisor (BF16 factor, when not
     * 0), then plus add (when not 0). y may be x. */
    void (*rms)(DGDevice y, DGDevice x, DGDevice scale, DGDevice factor, float divisor,
                DGDevice add, int tokens, int n);
    /* out = dg_gelu(g) * u over n values; out may be g or u. */
    void (*gelu_mul)(DGDevice out, DGDevice g, DGDevice u, size_t n);
    /* dg_nvfp4_qdq_ref, setting the device int *flag on a non-finite input
     * instead of failing. Rows are written as nvfp4_row(cols) 4-byte words
     * each: the reference's floats, or a packed form the products read. */
    void (*nvfp4_qdq)(DGDevice out, DGDevice in, int tokens, int cols, float base, DGDevice flag);
    size_t (*nvfp4_row)(int cols);
    /* Row i of dst = row index[i] of src, index being device ints. */
    void (*gather)(DGDevice dst, DGDevice src, DGDevice index, int rows, int cols);
    void (*copy)(DGDevice dst, DGDevice src, size_t bytes);
    /* dg_rope on rows of heads * hd values, stride apart per token; table
     * holds each token's cosines then sines, DG_FULL_HEAD_DIM apart. */
    void (*rope)(DGDevice x, DGDevice table, int tokens, int heads, int hd, int stride);
    /* The middle of dg_attention_segments for n = segments * seq tokens:
     * scores against a segment's cached keys, rows seg_base[s]... of k_old
     * and v_old (seg_old[s] of them), then its own, a softmax with dg_expf,
     * and the weighted values into a, each in the reference's order.
     * max_keys bounds any token's cached plus own keys; k_old holds old_rows
     * rows, and scratch (n + old_rows) * kvh * hd floats. */
    void (*attention)(DGDevice a, DGDevice q, DGDevice k, DGDevice v, DGDevice k_old,
                      DGDevice v_old, DGDevice seg_old, DGDevice seg_base, int n, int seq, int kvh,
                      int hd, int canvas, int full, int max_keys, int old_rows, DGDevice scratch);
    /* dg_ff's tail per token t: the routed experts' rows eo[slot[t, k]],
     * weighted by weight[t, k] and summed in k order, normalized by p2,
     * added to d, normalized by post, added to x, and x scaled by scalar. */
    void (*moe_tail)(DGDevice x, DGDevice eo, DGDevice slot, DGDevice weight, DGDevice d,
                     DGDevice p2, DGDevice post, float scalar, int tokens);
    /* A device table of count matrices, which matrix_table writes at dst;
     * nvfp4_grouped indexes it by expert. */
    size_t (*table_bytes)(int count);
    void (*matrix_table)(DGDevice dst, const DGAccelMatrix *const *m, int count);
    /* dg_route_token for each token's router logits: top experts and their
     * weights, DG_TOPK a token; a non-finite logit sets bit 2 of the device
     * int *flag. */
    void (*route)(DGDevice logits, DGDevice scale, int tokens, DGDevice top, DGDevice weight,
                  DGDevice flag);
    /* The routed rows grouped by expert in dg_ff's order: index[row] is a
     * row's token and slot[t, k] a token's row, and tiles, whose count goes
     * to the device int *ntiles, hold rows nvfp4_grouped multiplies
     * together. tiles holds nvfp4_scratch(tokens * DG_TOPK) bytes. */
    void (*group)(DGDevice top, int tokens, DGDevice index, DGDevice slot, DGDevice tiles,
                  DGDevice ntiles);
    /* nvfp4_mm from group's tiles, over routed rows of x. */
    void (*nvfp4_grouped)(DGDevice table, DGDevice tiles, DGDevice ntiles, int routed, DGDevice x,
                          DGDevice y, int rows, int cols);
} DGAccelOps;

static const DGAccelOps *dg_accel(void);

typedef struct {
    DGTensor *wg, *sg, *gg, *ag, *wu, *su, *gu, *wd, *sd, *gd, *ad;
    /* Gate, up and down on the accelerator, when the model uses one. */
    DGAccelMatrix *accel[3];
} DGNvExpert;

typedef struct {
    DGShard shard[11];
    DGTensor *tensor;
    size_t nt, cap;
    int nshard, nvfp4;
    DGNvExpert *nvexpert;
    float nv_a13[DG_L], nv_a2[DG_L];
    /* Set when every weight the layers use is on the accelerator, in the one
     * allocation weights; while the model loads, the accelerator any device
     * memory already placed belongs to. */
    const DGAccelOps *accel;
    DGDevice weights;
    /* The tied embedding on the accelerator in its own allocation, or 0:
     * optional, so a device without room for it keeps the layers. */
    DGDevice embedding;
    /* Each layer's gate, up and down matrices as device tables, inside
     * weights. */
    DGDevice expert_table[DG_L][3];
    /* One allocation containing all strict AVX-512 matrix images. */
    uint8_t *exact_weights;
} DGModel;

enum { DG_UNKNOWN = -1, DG_BF16, DG_U8, DG_F8E4M3, DG_F32 };

static uint64_t dg_item(int dtype) {
    return dtype == DG_BF16 ? 2 : dtype == DG_F32 ? 4 : 1;
}

typedef struct {
    Vocab *vocab, **by_id;
    Merge *merge;
    Vocab **special;
    uint32_t nv, nm, ns, maxlen;
    uint8_t special_first[256];
} DGTokenizer;

static uint64_t jt_u64(const char *j, const JTok *t, const char *name) {
    if (t->type != JT_PRIMITIVE || t->end <= t->start || t->end - t->start > 20)
        die2("expected unsigned integer", name);
    uint64_t v = 0;
    for (int i = t->start; i < t->end; i++) {
        unsigned d = (unsigned)(j[i] - '0');
        if (d > 9 || v > (UINT64_MAX - d) / 10)
            die2("bad unsigned integer", name);
        v = v * 10 + d;
    }
    return v;
}

static void dg_push_tensor(DGModel *m, DGTensor t) {
    if (m->nt == m->cap) {
        if (m->cap > SIZE_MAX / 2 / sizeof *m->tensor)
            die("tensor descriptor array size overflow");
        m->cap = m->cap ? m->cap * 2 : 1024;
        m->tensor = xrealloc(m->tensor, m->cap * sizeof *m->tensor);
    }
    m->tensor[m->nt++] = t;
}

static int dg_tensor_cmp(const void *a, const void *b) {
    const DGTensor *x = a, *y = b;
    return strcmp(x->name, y->name);
}

/* bsearch's comparison of a tensor name with a tensor. */
static int dg_tensor_key_cmp(const void *a, const void *b) {
    const DGTensor *y = b;
    return strcmp(a, y->name);
}

static int dg_is_text(const char *name) {
    return !strncmp(name, "model.decoder.", 14) ||
           (!strncmp(name, "model.encoder.language_model.", 29) && strstr(name, ".layer_scalar"));
}

static void dg_parse_shard(DGModel *m, int si, const char *dir) {
    DGShard *s = &m->shard[si];
    jb_path(s->path, sizeof s->path, "%s/model-%05d-of-%05d.safetensors", dir, si + 1, m->nshard);
    map_file(&s->file, s->path);
    if (s->file.size < 9)
        die2("short safetensors file", s->path);
    uint64_t hn = 0;
    for (int i = 0; i < 8; i++)
        hn |= (uint64_t)s->file.map[i] << (8 * i);
    if (!hn || hn > s->file.size - 8 || hn > JB_MAX_JSON)
        die2("invalid safetensors header", s->path);
    const char *j = (const char *)s->file.map + 8;
    int nt = 0;
    JTok *tok = json_tokens(j, (size_t)hn, &nt);
    if (!nt || tok[0].type != JT_OBJECT)
        die2("invalid safetensors JSON", s->path);
    for (int key = 1; key < nt; key++) {
        if (tok[key].parent != 0 || tok[key].type != JT_STRING)
            continue;
        char *name = jt_string(j, &tok[key]);
        int obj = key + 1;
        if (!strcmp(name, "__metadata__")) {
            jb_release(name);
            continue;
        }
        if (obj >= nt || tok[obj].type != JT_OBJECT)
            die2("bad safetensors descriptor", name);
        int dtype = jt_obj_get(j, tok, nt, obj, "dtype"),
            shape = jt_obj_get(j, tok, nt, obj, "shape"),
            offsets = jt_obj_get(j, tok, nt, obj, "data_offsets");
        if (dtype < 0 || shape < 0 || offsets < 0 || tok[shape].type != JT_ARRAY ||
            tok[offsets].type != JT_ARRAY)
            die2("incomplete safetensors descriptor", name);
        char *dt = jt_string(j, &tok[dtype]);
        DGTensor t = {0};
        t.name = name;
        t.dtype = DG_UNKNOWN;
        if (!strcmp(dt, "BF16"))
            t.dtype = DG_BF16;
        else if (!strcmp(dt, "U8"))
            t.dtype = DG_U8;
        else if (!strcmp(dt, "F8_E4M3"))
            t.dtype = DG_F8E4M3;
        else if (!strcmp(dt, "F32"))
            t.dtype = DG_F32;
        else if (dg_is_text(name))
            die2("unsupported DiffusionGemma text dtype", name);
        jb_release(dt);
        if (dg_is_text(name) && t.dtype != DG_BF16 &&
            !(m->nvfp4 && strstr(name, ".experts.") &&
              (t.dtype == DG_U8 || t.dtype == DG_F8E4M3 || t.dtype == DG_F32)))
            die2("unsupported DiffusionGemma text tensor encoding", name);
        for (int k = shape + 1; k < nt && tok[k].parent >= shape; k++)
            if (tok[k].parent == shape) {
                if (t.nd == 4)
                    die2("too many safetensors dimensions", name);
                t.shape[t.nd++] = jt_u64(j, &tok[k], name);
            }
        uint64_t off[2] = {0};
        int no = 0;
        for (int k = offsets + 1; k < nt && tok[k].parent >= offsets; k++)
            if (tok[k].parent == offsets) {
                if (no == 2)
                    die2("bad safetensors offsets", name);
                off[no++] = jt_u64(j, &tok[k], name);
            }
        uint64_t base = 8 + hn;
        if (no != 2 || off[1] < off[0] || base > s->file.size || off[1] > s->file.size - base)
            die2("out-of-bounds safetensors data", name);
        t.data = s->file.map + base + off[0];
        t.bytes = off[1] - off[0];
        /* Every consumer indexes tensors by shape, so the byte span must
         * match it exactly or reads run past the tensor's data. */
        if (t.dtype != DG_UNKNOWN) {
            uint64_t count = 1;
            for (int i = 0; i < t.nd; i++) {
                if (t.shape[i] && count > UINT64_MAX / t.shape[i])
                    die2("safetensors tensor size overflow", name);
                count *= t.shape[i];
            }
            if (count > UINT64_MAX / dg_item(t.dtype) || t.bytes != count * dg_item(t.dtype))
                die2("safetensors byte size does not match shape", name);
        }
        dg_push_tensor(m, t);
    }
    jb_release(tok);
}

static DGTensor *dg_tensor(DGModel *m, const char *name) {
    DGTensor *t = bsearch(name, m->tensor, m->nt, sizeof *m->tensor, dg_tensor_key_cmp);
    if (t)
        return t;
    die2("missing DiffusionGemma tensor", name);
    return NULL;
}

static DGTensor *dg_expert_tensor(DGModel *m, int l, int e, const char *tail) {
    char n[192];
    snprintf(n, sizeof n, "model.decoder.layers.%d.experts.%d.%s", l, e, tail);
    return dg_tensor(m, n);
}

static void dg_expect(DGModel *m, const char *name, int dtype, int nd, uint64_t a, uint64_t b,
                      uint64_t c) {
    DGTensor *t = dg_tensor(m, name);
    if (t->dtype != dtype || t->nd != nd || (nd > 0 && t->shape[0] != a) ||
        (nd > 1 && t->shape[1] != b) || (nd > 2 && t->shape[2] != c))
        die2("unexpected DiffusionGemma tensor shape", name);
    uint64_t n = 1;
    for (int i = 0; i < nd; i++) {
        if (t->shape[i] && n > UINT64_MAX / t->shape[i])
            die2("DiffusionGemma tensor size overflow", name);
        n *= t->shape[i];
    }
    if (t->bytes != n * dg_item(dtype))
        die2("unexpected DiffusionGemma tensor byte size", name);
}

/* Vectors are read as exactly `count` elements; their rank is not relied on. */
static void dg_expect_count(DGModel *m, const char *name, int dtype, uint64_t count) {
    DGTensor *t = dg_tensor(m, name);
    if (t->dtype != dtype || t->bytes != count * dg_item(dtype))
        die2("unexpected DiffusionGemma tensor size", name);
}

static void dg_expect_layer(DGModel *m, int l, const char *tail, int nd, uint64_t a, uint64_t b,
                            uint64_t c) {
    char n[192];
    jb_path(n, sizeof n, "model.decoder.layers.%d.%s", l, tail);
    dg_expect(m, n, DG_BF16, nd, a, b, c);
}

static void dg_expect_layer_count(DGModel *m, int l, const char *tail, uint64_t count) {
    char n[192];
    jb_path(n, sizeof n, "model.decoder.layers.%d.%s", l, tail);
    dg_expect_count(m, n, DG_BF16, count);
}

static void dg_load(DGModel *m, const char *dir) {
    memset(m, 0, sizeof *m);
    char probe[768];
    jb_path(probe, sizeof probe, "%s/model-00001-of-00002.safetensors", dir);
    FILE *pf = fopen(probe, "rb");
    if (pf) {
        fclose(pf);
        m->nshard = 2;
        m->nvfp4 = 1;
    } else
        m->nshard = 11;
    for (int i = 0; i < m->nshard; i++)
        dg_parse_shard(m, i, dir);
    qsort(m->tensor, m->nt, sizeof *m->tensor, dg_tensor_cmp);
    for (size_t i = 1; i < m->nt; i++)
        if (!strcmp(m->tensor[i - 1].name, m->tensor[i].name))
            die2("duplicate DiffusionGemma tensor", m->tensor[i].name);
    dg_expect(m, "model.decoder.embed_tokens.weight", DG_BF16, 2, DG_VOCAB, DG_H, 0);
    dg_expect_count(m, "model.decoder.norm.weight", DG_BF16, DG_H);
    /* Validate every layer, not a sample: kernels index these tensors by the
     * shapes below without further bounds checks. */
    for (int l = 0; l < DG_L; l++) {
        int full = l % DG_FULL_EVERY == DG_FULL_EVERY - 1;
        int hd = full ? DG_FULL_HEAD_DIM : DG_LOCAL_HEAD_DIM;
        int kn = (full ? DG_FULL_KV_HEADS : DG_LOCAL_KV_HEADS) * hd;
        dg_expect_layer(m, l, "self_attn.q_proj.weight", 2, DG_HEADS * hd, DG_H, 0);
        dg_expect_layer(m, l, "self_attn.k_proj.weight", 2, kn, DG_H, 0);
        if (!full)
            dg_expect_layer(m, l, "self_attn.v_proj.weight", 2, kn, DG_H, 0);
        dg_expect_layer(m, l, "self_attn.o_proj.weight", 2, DG_H, DG_HEADS * hd, 0);
        dg_expect_layer_count(m, l, "self_attn.q_norm.weight", hd);
        dg_expect_layer_count(m, l, "self_attn.k_norm.weight", hd);
        dg_expect_layer_count(m, l, "input_layernorm.weight", DG_H);
        dg_expect_layer_count(m, l, "post_attention_layernorm.weight", DG_H);
        dg_expect_layer_count(m, l, "pre_feedforward_layernorm.weight", DG_H);
        dg_expect_layer_count(m, l, "post_feedforward_layernorm_1.weight", DG_H);
        dg_expect_layer_count(m, l, "pre_feedforward_layernorm_2.weight", DG_H);
        dg_expect_layer_count(m, l, "post_feedforward_layernorm_2.weight", DG_H);
        dg_expect_layer_count(m, l, "post_feedforward_layernorm.weight", DG_H);
        dg_expect_layer(m, l, "mlp.gate_proj.weight", 2, DG_DENSE, DG_H, 0);
        dg_expect_layer(m, l, "mlp.up_proj.weight", 2, DG_DENSE, DG_H, 0);
        dg_expect_layer(m, l, "mlp.down_proj.weight", 2, DG_H, DG_DENSE, 0);
        dg_expect_layer(m, l, "router.proj.weight", 2, DG_EXPERTS, DG_H, 0);
        dg_expect_layer_count(m, l, "router.scale", DG_H);
        dg_expect_layer_count(m, l, "router.per_expert_scale", DG_EXPERTS);
        dg_expect_layer(m, l, "layer_scalar", 1, 1, 0, 0);
        if (!m->nvfp4) {
            dg_expect_layer(m, l, "experts.gate_up_proj", 3, DG_EXPERTS, DG_EXPERT_PAIR, DG_H);
            dg_expect_layer(m, l, "experts.down_proj", 3, DG_EXPERTS, DG_H, DG_MOE);
        }
    }
    if (m->nvfp4) {
        dg_expect(m, "model.decoder.layers.0.experts.0.gate_proj.weight", DG_U8, 2, DG_MOE,
                  DG_EXPERT_PAIR, 0);
        dg_expect(m, "model.decoder.layers.0.experts.0.gate_proj.weight_scale", DG_F8E4M3, 2,
                  DG_MOE, DG_MOE / 4, 0);
        dg_expect(m, "model.decoder.layers.0.experts.0.gate_proj.weight_scale_2", DG_F32, 0, 0, 0,
                  0);
        dg_expect(m, "model.decoder.layers.0.experts.0.down_proj.weight", DG_U8, 2, DG_H,
                  DG_MOE / 2, 0);
        m->nvexpert = xcalloc((size_t)DG_L * DG_EXPERTS, sizeof *m->nvexpert);
        for (int l = 0; l < DG_L; l++)
            for (int e = 0; e < DG_EXPERTS; e++) {
                DGNvExpert *v = &m->nvexpert[l * DG_EXPERTS + e];
                v->wg = dg_expert_tensor(m, l, e, "gate_proj.weight");
                v->sg = dg_expert_tensor(m, l, e, "gate_proj.weight_scale");
                v->gg = dg_expert_tensor(m, l, e, "gate_proj.weight_scale_2");
                v->ag = dg_expert_tensor(m, l, e, "gate_proj.input_scale");
                v->wu = dg_expert_tensor(m, l, e, "up_proj.weight");
                v->su = dg_expert_tensor(m, l, e, "up_proj.weight_scale");
                v->gu = dg_expert_tensor(m, l, e, "up_proj.weight_scale_2");
                v->wd = dg_expert_tensor(m, l, e, "down_proj.weight");
                v->sd = dg_expert_tensor(m, l, e, "down_proj.weight_scale");
                v->gd = dg_expert_tensor(m, l, e, "down_proj.weight_scale_2");
                v->ad = dg_expert_tensor(m, l, e, "down_proj.input_scale");
                if (v->ag->dtype != DG_F32 || v->ag->bytes != 4 || v->ad->dtype != DG_F32 ||
                    v->ad->bytes != 4)
                    die2("bad NVFP4 input scale", v->ag->name);
                float a13, a2;
                memcpy(&a13, v->ag->data, 4);
                memcpy(&a2, v->ad->data, 4);
                if (a13 > m->nv_a13[l])
                    m->nv_a13[l] = a13;
                if (a2 > m->nv_a2[l])
                    m->nv_a2[l] = a2;
            }
    }
}

static void dg_unmap_shards(DGModel *m) {
    for (int i = 0; i < m->nshard; i++)
        unmap_file(&m->shard[i].file);
    memset(m->shard, 0, sizeof m->shard);
}

/* Device memory is not tracked by error frames, so a failed load releases it
 * here, like the mapped shards. */
static void dg_release_accel(DGModel *m) {
    const DGAccelOps *a = m->accel;
    if (!a)
        return;
    if (m->nvexpert)
        for (int i = 0; i < DG_L * DG_EXPERTS; i++)
            for (int k = 0; k < 3; k++)
                if (m->nvexpert[i].accel[k]) {
                    a->release(m->nvexpert[i].accel[k]);
                    m->nvexpert[i].accel[k] = NULL;
                }
    for (size_t i = 0; i < m->nt; i++)
        m->tensor[i].device = 0;
    if (m->weights)
        a->free(m->weights);
    if (m->embedding)
        a->free(m->embedding);
    m->weights = m->embedding = 0;
    m->accel = NULL;
}

static DGTensor *dg_layer_tensor(DGModel *m, int l, const char *tail);

/* The BF16 tensors a layer uses on the accelerator. */
static const char *const dg_accel_tensors[] = {
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "post_feedforward_layernorm_1.weight",
    "pre_feedforward_layernorm_2.weight",
    "router.proj.weight",
    "router.scale",
    "router.per_expert_scale",
    "post_feedforward_layernorm_2.weight",
    "post_feedforward_layernorm.weight",
};

/* The next 256-byte-aligned place for bytes in the weights allocation at
 * base, copying src there; with base 0 it only advances *at, to size the
 * allocation. */
static DGDevice dg_place(const DGAccelOps *a, DGDevice base, size_t *at, const void *src,
                         size_t bytes) {
    *at = jb_align_up(*at, 256);
    DGDevice p = base ? base + *at : 0;
    if (base)
        a->to_device(p, src, bytes);
    *at = jb_size_add(*at, bytes);
    return p;
}

static DGNvMatrix dg_nvfp4_matrix(const DGTensor *w, const DGTensor *s, const DGTensor *g, int rows,
                                  int cols);
static void dg_prepare_exact_model(DGModel *m);

/* Places every expert matrix and every layer tensor in one device
 * allocation, or none: allocations are slow one at a time, and a device that
 * cannot hold them all leaves the model on the CPU kernels. A first pass
 * sizes the allocation and a second fills it. */
static void dg_upload_model(DGModel *m) {
    const DGAccelOps *a = m->nvfp4 ? dg_accel() : NULL;
    if (!a)
        return;
    m->accel = a;
    for (int pass = 0; pass < 2; pass++) {
        size_t at = 0;
        for (int i = 0; i < DG_L * DG_EXPERTS; i++) {
            DGNvExpert *v = &m->nvexpert[i];
            DGNvMatrix nv[3] = {dg_nvfp4_matrix(v->wg, v->sg, v->gg, DG_MOE, DG_H),
                                dg_nvfp4_matrix(v->wu, v->su, v->gu, DG_MOE, DG_H),
                                dg_nvfp4_matrix(v->wd, v->sd, v->gd, DG_H, DG_MOE)};
            for (int k = 0; k < 3; k++) {
                int rows = k < 2 ? DG_MOE : DG_H, cols = k < 2 ? DG_H : DG_MOE;
                DGDevice w = dg_place(a, m->weights, &at, nv[k].w, (size_t)rows * cols / 2),
                         sc = dg_place(a, m->weights, &at, nv[k].s, (size_t)rows * cols / 16);
                if (pass)
                    v->accel[k] = a->matrix(w, sc, nv[k].global, rows, cols);
            }
        }
        for (int l = 0; l < DG_L; l++) {
            int full = l % DG_FULL_EVERY == DG_FULL_EVERY - 1;
            for (size_t i = 0; i < sizeof dg_accel_tensors / sizeof *dg_accel_tensors; i++)
                if (!full || strcmp(dg_accel_tensors[i], "self_attn.v_proj.weight")) {
                    DGTensor *t = dg_layer_tensor(m, l, dg_accel_tensors[i]);
                    DGDevice p = dg_place(a, m->weights, &at, t->data, (size_t)t->bytes);
                    if (pass)
                        t->device = p;
                }
        }
        for (int l = 0; l < DG_L; l++)
            for (int k = 0; k < 3; k++) {
                at = jb_align_up(at, 256);
                if (pass) {
                    const DGAccelMatrix *list[DG_EXPERTS];
                    for (int e = 0; e < DG_EXPERTS; e++)
                        list[e] = m->nvexpert[l * DG_EXPERTS + e].accel[k];
                    m->expert_table[l][k] = m->weights + at;
                    a->matrix_table(m->expert_table[l][k], list, DG_EXPERTS);
                }
                at = jb_size_add(at, a->table_bytes(DG_EXPERTS));
            }
        if (!pass && !(m->weights = a->alloc(at))) {
            m->accel = NULL;
            return;
        }
    }
    /* The tied embedding, for the vocabulary logits of automatic reads, if
     * the device has room for it; otherwise the CPU computes them. */
    DGTensor *emb = dg_tensor(m, "model.decoder.embed_tokens.weight");
    if ((m->embedding = a->alloc((size_t)emb->bytes))) {
        a->to_device(m->embedding, emb->data, (size_t)emb->bytes);
        emb->device = m->embedding;
    }
}

static void dg_free(DGModel *m) {
    dg_release_accel(m);
    jb_release(m->exact_weights);
    for (size_t i = 0; i < m->nt; i++)
        jb_release(m->tensor[i].name);
    jb_release(m->tensor);
    jb_release(m->nvexpert);
    dg_unmap_shards(m);
}

static Vocab *dgt_vfind(DGTokenizer *d, const char *s, size_t n) {
    VocabKey k = {s, (uint32_t)n};
    return bsearch(&k, d->vocab, d->nv, sizeof *d->vocab, cmp_vocab_key);
}

static Merge *dgt_mfind(DGTokenizer *d, uint32_t a, uint32_t b) {
    Merge k = {a, b, 0};
    return bsearch(&k, d->merge, d->nm, sizeof *d->merge, cmp_merge);
}

static char *read_whole(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f)
        die2_status(JB_ERROR_IO, "cannot open", path);
    if (fseek(f, 0, SEEK_END) || (*n = (size_t)ftell(f)) > JB_MAX_JSON || fseek(f, 0, SEEK_SET)) {
        fclose(f);
        die2_status(JB_ERROR_IO, "invalid JSON file size", path);
    }
    char *p = jb_try_allocate(*n + 1, 0);
    if (!p) {
        fclose(f);
        jb_out_of_memory();
    }
    size_t got = fread(p, 1, *n, f);
    if (fclose(f) || got != *n)
        die2_status(JB_ERROR_IO, "cannot read", path);
    p[*n] = 0;
    return p;
}

static void dgt_load(DGTokenizer *d, const char *dir) {
    memset(d, 0, sizeof *d);
    char path[768];
    jb_path(path, sizeof path, "%s/tokenizer.json", dir);
    size_t nj;
    char *j = read_whole(path, &nj);
    int nt;
    JTok *t = json_tokens(j, nj, &nt);
    int model = jt_obj_get(j, t, nt, 0, "model");
    int vocab = jt_obj_get(j, t, nt, model, "vocab");
    int merges = jt_obj_get(j, t, nt, model, "merges");
    int added = jt_obj_get(j, t, nt, 0, "added_tokens");
    if (model < 0 || vocab < 0 || merges < 0 || added < 0 || t[vocab].type != JT_OBJECT ||
        t[merges].type != JT_ARRAY || t[added].type != JT_ARRAY)
        die("unsupported DiffusionGemma tokenizer JSON");
    uint32_t nv = (uint32_t)(t[vocab].size / 2);
    if (nv != 262144)
        die("unexpected DiffusionGemma vocabulary size");
    d->vocab = xcalloc(nv, sizeof *d->vocab);
    d->nv = nv;
    uint32_t vi = 0;
    for (int i = vocab + 1; i + 1 < nt; i++) {
        if (t[i].parent < vocab)
            break;
        if (t[i].parent != vocab)
            continue;
        if (vi == d->nv)
            die("too many DiffusionGemma vocabulary entries");
        d->vocab[vi].s = jt_string(j, &t[i]);
        d->vocab[vi].n = (uint32_t)strlen(d->vocab[vi].s);
        if (d->vocab[vi].n > d->maxlen)
            d->maxlen = d->vocab[vi].n;
        d->vocab[vi].id = jt_nonnegative_int(j, &t[i + 1], "vocabulary id");
        vi++;
        i++;
    }
    if (vi != d->nv)
        die("incomplete DiffusionGemma vocabulary");
    qsort(d->vocab, d->nv, sizeof *d->vocab, cmp_vocab);
    d->by_id = xcalloc(d->nv, sizeof *d->by_id);
    for (uint32_t i = 0; i < d->nv; i++) {
        int id = d->vocab[i].id;
        if (id < 0 || (uint32_t)id >= d->nv || d->by_id[id])
            die("invalid DiffusionGemma vocabulary id");
        d->by_id[id] = &d->vocab[i];
    }
    d->nm = (uint32_t)t[merges].size;
    d->merge = xcalloc(d->nm, sizeof *d->merge);
    uint32_t mi = 0;
    for (int i = merges + 1; i < nt; i++) {
        if (t[i].parent < merges)
            break;
        if (t[i].parent != merges)
            continue;
        int a = -1, b = -1;
        for (int k = i + 1; k < nt && t[k].parent >= i; k++)
            if (t[k].parent == i) {
                if (a < 0)
                    a = k;
                else if (b < 0)
                    b = k;
                else
                    die("bad tokenizer merge");
            }
        if (t[i].type != JT_ARRAY || a < 0 || b < 0)
            die("bad tokenizer merge");
        char *sa = jt_string(j, &t[a]), *sb = jt_string(j, &t[b]);
        Vocab *va = dgt_vfind(d, sa, strlen(sa)), *vb = dgt_vfind(d, sb, strlen(sb));
        jb_release(sa);
        jb_release(sb);
        if (!va || !vb || mi == d->nm)
            die("tokenizer merge references absent token");
        d->merge[mi] = (Merge){(uint32_t)va->id, (uint32_t)vb->id, mi};
        mi++;
    }
    if (mi != d->nm)
        die("incomplete tokenizer merges");
    qsort(d->merge, d->nm, sizeof *d->merge, cmp_merge);
    d->ns = (uint32_t)t[added].size;
    d->special = xcalloc(d->ns, sizeof *d->special);
    uint32_t si = 0;
    for (int i = added + 1; i < nt; i++) {
        if (t[i].parent < added)
            break;
        if (t[i].parent != added)
            continue;
        int content = jt_obj_get(j, t, nt, i, "content");
        int special = jt_obj_get(j, t, nt, i, "special");
        if (content < 0 || special < 0 || !jt_literal(j, &t[special], "true"))
            die("unsupported added token");
        char *s = jt_string(j, &t[content]);
        Vocab *v = dgt_vfind(d, s, strlen(s));
        jb_release(s);
        if (!v)
            die("added token absent from vocabulary");
        if (!v->n)
            die("empty added token");
        d->special_first[(unsigned char)v->s[0]] = 1;
        d->special[si++] = v;
    }
    d->ns = si;
    jb_release(t);
    jb_release(j);
}

static void dgt_free(DGTokenizer *d) {
    for (uint32_t i = 0; d->vocab && i < d->nv; i++)
        jb_release(d->vocab[i].s);
    jb_release(d->vocab);
    jb_release(d->by_id);
    jb_release(d->merge);
    jb_release(d->special);
}

typedef struct {
    uint32_t rank, pos, a, b;
} DGBpeCand;

static int dgt_bpe_less(const DGBpeCand *x, const DGBpeCand *y) {
    return x->rank != y->rank ? x->rank < y->rank : x->pos < y->pos;
}

static void dgt_bpe_push(DGTokenizer *d, DGBpeCand *heap, size_t *nh, const uint32_t *ids,
                         uint32_t i, uint32_t j) {
    Merge *m = dgt_mfind(d, ids[i], ids[j]);
    if (!m)
        return;
    size_t at = (*nh)++;
    heap[at] = (DGBpeCand){m->rank, i, ids[i], ids[j]};
    while (at && dgt_bpe_less(&heap[at], &heap[(at - 1) / 2])) {
        DGBpeCand z = heap[at];
        heap[at] = heap[(at - 1) / 2];
        heap[(at - 1) / 2] = z;
        at = (at - 1) / 2;
    }
}

static void dgt_piece(DGTokenizer *d, const char *s, size_t n, Tokens *out) {
    size_t slots = n ? n : 1, bytes = 0;
    jb_workspace_plan(&bytes, slots, sizeof(uint32_t));
    jb_workspace_plan(&bytes, slots, sizeof(uint32_t));
    jb_workspace_plan(&bytes, slots, sizeof(uint32_t));
    jb_workspace_plan(&bytes, jb_size_mul(slots, 3), sizeof(DGBpeCand));
    jb_workspace_plan(&bytes, jb_size_mul((size_t)d->maxlen, 2), sizeof(char));
    JBArena scratch;
    jb_arena_init(&scratch, bytes);
    uint32_t *ids = jb_arena_alloc(&scratch, slots, sizeof *ids, 0);
    uint32_t *next = jb_arena_alloc(&scratch, slots, sizeof *next, 0);
    uint32_t *prev = jb_arena_alloc(&scratch, slots, sizeof *prev, 0);
    DGBpeCand *heap = jb_arena_alloc(&scratch, jb_size_mul(slots, 3), sizeof *heap, 0);
    char *joined = jb_arena_alloc(&scratch, jb_size_mul((size_t)d->maxlen, 2), sizeof *joined, 0);
    uint32_t k = 0;
    for (size_t p = 0; p < n;) {
        size_t q = p;
        next_cp((const unsigned char *)s, n, &q);
        Vocab *v = dgt_vfind(d, s + p, q - p);
        if (v)
            ids[k++] = (uint32_t)v->id;
        else {
            for (; p < q; p++) {
                char z[7];
                snprintf(z, sizeof z, "<0x%02X>", (unsigned char)s[p]);
                v = dgt_vfind(d, z, 6);
                if (!v)
                    die("DiffusionGemma byte fallback token absent");
                ids[k++] = (uint32_t)v->id;
            }
            continue;
        }
        p = q;
    }
    /* Repeatedly merge the lowest-rank adjacent pair, leftmost first. A heap
     * ordered by (rank, original position) with lazy invalidation picks the
     * same pair as a full rescan, in O(k log k) instead of O(k^2). */
    for (uint32_t i = 0; i < k; i++) {
        next[i] = i + 1 < k ? i + 1 : UINT32_MAX;
        prev[i] = i ? i - 1 : UINT32_MAX;
    }
    size_t nh = 0;
    for (uint32_t i = 0; i + 1 < k; i++)
        dgt_bpe_push(d, heap, &nh, ids, i, i + 1);
    while (nh) {
        DGBpeCand c = heap[0];
        heap[0] = heap[--nh];
        for (size_t i = 0;;) {
            size_t l = 2 * i + 1, r = l + 1, m = i;
            if (l < nh && dgt_bpe_less(&heap[l], &heap[m]))
                m = l;
            if (r < nh && dgt_bpe_less(&heap[r], &heap[m]))
                m = r;
            if (m == i)
                break;
            DGBpeCand z = heap[i];
            heap[i] = heap[m];
            heap[m] = z;
            i = m;
        }
        uint32_t at = c.pos, bt = next[at];
        if (ids[at] != c.a || bt == UINT32_MAX || ids[bt] != c.b)
            continue;
        Vocab *a = d->by_id[c.a], *b = d->by_id[c.b];
        size_t joined_n = jb_size_add((size_t)a->n, b->n);
        if (joined_n > (size_t)d->maxlen * 2)
            die("DiffusionGemma BPE merge exceeds tokenizer scratch");
        memcpy(joined, a->s, a->n);
        memcpy(joined + a->n, b->s, b->n);
        Vocab *v = dgt_vfind(d, joined, joined_n);
        if (!v)
            die("DiffusionGemma merged token absent");
        ids[at] = (uint32_t)v->id;
        ids[bt] = UINT32_MAX;
        next[at] = next[bt];
        if (next[at] != UINT32_MAX)
            prev[next[at]] = at;
        if (prev[at] != UINT32_MAX)
            dgt_bpe_push(d, heap, &nh, ids, prev[at], at);
        if (next[at] != UINT32_MAX)
            dgt_bpe_push(d, heap, &nh, ids, at, next[at]);
    }
    for (uint32_t i = 0; i < k; i = next[i])
        push(out, (int)ids[i]);
    jb_arena_free(&scratch);
}

static Tokens dgt_tokenize(DGTokenizer *d, const char *s) {
    Tokens out = {0};
    size_t n = strlen(s), plain = 0, i = 0;
    while (i < n) {
        Vocab *hit = NULL;
        if (!d->special_first[(unsigned char)s[i]]) {
            i++;
            continue;
        }
        for (uint32_t q = 0; q < d->ns; q++) {
            Vocab *v = d->special[q];
            if (v->n <= n - i && !memcmp(s + i, v->s, v->n) && (!hit || v->n > hit->n))
                hit = v;
        }
        if (!hit) {
            i++;
            continue;
        }
        if (i > plain) {
            size_t zcap = (i - plain) * 3 + 1, zlen = 0;
            char *z = xmalloc(zcap);
            for (size_t q = plain; q < i; q++) {
                if (s[q] == ' ')
                    memcpy(z + zlen, "\xE2\x96\x81", 3), zlen += 3;
                else
                    z[zlen++] = s[q];
            }
            dgt_piece(d, z, zlen, &out);
            jb_release(z);
        }
        push(&out, hit->id);
        i += hit->n;
        plain = i;
    }
    if (plain < n) {
        size_t zcap = (n - plain) * 3 + 1, zlen = 0;
        char *z = xmalloc(zcap);
        for (size_t q = plain; q < n; q++) {
            if (s[q] == ' ')
                memcpy(z + zlen, "\xE2\x96\x81", 3), zlen += 3;
            else
                z[zlen++] = s[q];
        }
        dgt_piece(d, z, zlen, &out);
        jb_release(z);
    }
    return out;
}

typedef struct {
    float *k, *v;
    int n;
} DGKV;

typedef struct {
    char *schema;
    int *ids, n;
    DGKV kv[DG_L];
    JBArena metadata, kv_storage;
} DGPrefixCache;

typedef struct {
    uint32_t mt[624];
    int at;
} DGMT;

static void dg_mt_seed(DGMT *r, uint32_t seed) {
    r->mt[0] = 19650218u;
    for (int i = 1; i < 624; i++)
        r->mt[i] = 1812433253u * (r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) + i;
    int i = 1, j = 0;
    for (int q = 624; q; q--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1664525u)) + seed + j;
        i++;
        j++;
        if (i >= 624) {
            r->mt[0] = r->mt[623];
            i = 1;
        }
        if (j >= 1)
            j = 0;
    }
    for (int q = 623; q; q--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1566083941u)) - i;
        i++;
        if (i >= 624) {
            r->mt[0] = r->mt[623];
            i = 1;
        }
    }
    r->mt[0] = 0x80000000u;
    r->at = 624;
}

static uint32_t dg_mt_u32(DGMT *r) {
    if (r->at >= 624) {
        for (int i = 0; i < 624; i++) {
            uint32_t y = (r->mt[i] & 0x80000000u) | (r->mt[(i + 1) % 624] & 0x7fffffffu);
            r->mt[i] = r->mt[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfu : 0);
        }
        r->at = 0;
    }
    uint32_t y = r->mt[r->at++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

static uint32_t dg_mt_vocab(DGMT *r) {
    uint32_t x;
    do {
        x = dg_mt_u32(r) >> 13;
    } while (x >= DG_VOCAB);
    return x;
}

static float dg_bf(const uint8_t *p) {
    uint32_t u = (uint32_t)p[0] << 16 | (uint32_t)p[1] << 24;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static float dg_at(const DGTensor *t, uint64_t i) {
    return dg_bf(t->data + i * 2);
}

/* E4M3 to float by assembling the bits: normal values rebias the exponent
 * (7 to 127) and move the mantissa up; subnormals are m * 2^-9. Every E4M3
 * value is exact in float, so this equals the ldexpf formulation for all 256
 * inputs (the selftest checks each), at a fraction of the cost on the NVFP4
 * scale and quantization paths. */
static float dg_f8e4m3(uint8_t u) {
    uint32_t e = (u >> 3) & 15, m = u & 7, bits;
    float x;
    if (e) {
        bits = ((e + 120) << 23) | (m << 20);
        memcpy(&x, &bits, sizeof x);
    } else
        x = (float)m * 0x1p-9f;
    return u >> 7 ? -x : x;
}
