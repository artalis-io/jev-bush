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
 * selecting one. -DJB_SCALAR remains the portable oracle;
 * -DJB_AVX2_ONLY builds the AVX2 tier without AVX-512 variants. */
#if defined(JB_SCALAR)
#define JB_TARGET_AVX2
#define JB_TARGET_AVX512
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define JB_HAVE_AVX2 1
#if !defined(JB_AVX2_ONLY)
#define JB_HAVE_AVX512 1
#endif
#define JB_TARGET_AVX2 __attribute__((target("avx2,fma")))
#if !defined(JB_AVX2_ONLY)
#define JB_TARGET_AVX512 __attribute__((target("avx2,fma,avx512f,avx512dq,avx512bw")))
#else
#define JB_TARGET_AVX512
#endif
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

typedef void (*JBCleanupFn)(void *context);

typedef struct {
    JBCleanupFn function;
    void *context;
} JBCleanup;

#define JB_CLEANUP_MAX 16
#define JB_CLEANUP_STACK_MAX 64

typedef struct JBErrorFrame {
    jmp_buf jump;
    struct JBErrorFrame *previous;
    uint64_t mark;
    jb_status previous_status;
    unsigned cleanup_mark;
} JBErrorFrame;

static _Thread_local JBErrorFrame *jb_error_frame;
static _Thread_local jb_status jb_error_status;
static _Thread_local char jb_error_message[256];
static _Thread_local JBAllocation *jb_tracked;
static _Thread_local uint64_t jb_allocation_sequence;
static _Thread_local const jb_allocator *jb_active_allocator;
static _Thread_local JBCleanup jb_cleanup_stack[JB_CLEANUP_STACK_MAX];
static _Thread_local unsigned jb_cleanup_count;

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

/* Every frame field is set before setjmp and remains unchanged afterwards.
 * Mutable cleanups live in thread-local storage, whose values remain defined
 * after longjmp. */
static void jb_frame_enter(JBErrorFrame *f, jb_status status) {
    f->previous = jb_error_frame;
    f->previous_status = jb_error_status;
    f->mark = jb_allocation_sequence;
    f->cleanup_mark = jb_cleanup_count;
    jb_error_status = status;
    jb_error_frame = f;
}

static void jb_cleanup_push(JBErrorFrame *frame, JBCleanupFn function, void *context) {
    if (!frame || frame != jb_error_frame || !function || jb_cleanup_count < frame->cleanup_mark ||
        jb_cleanup_count - frame->cleanup_mark >= JB_CLEANUP_MAX ||
        jb_cleanup_count >= JB_CLEANUP_STACK_MAX)
        die_status(JB_ERROR_INTERNAL, "invalid error cleanup registration");
    jb_cleanup_stack[jb_cleanup_count++] = (JBCleanup){function, context};
}

static void jb_cleanup_disarm(JBErrorFrame *frame, JBCleanupFn function, void *context) {
    if (!frame || frame != jb_error_frame || jb_cleanup_count == frame->cleanup_mark)
        die_status(JB_ERROR_INTERNAL, "invalid error cleanup removal");
    JBCleanup *cleanup = &jb_cleanup_stack[jb_cleanup_count - 1];
    if (cleanup->function != function || cleanup->context != context)
        die_status(JB_ERROR_INTERNAL, "error cleanups removed out of order");
    jb_cleanup_count--;
}

static void jb_cleanup_run(JBErrorFrame *frame) {
    while (jb_cleanup_count > frame->cleanup_mark) {
        JBCleanup cleanup = jb_cleanup_stack[--jb_cleanup_count];
        cleanup.function(cleanup.context);
    }
}

/* Successful return: the outermost frame hands every allocation it made to
 * its owner (caller or session). */
static void jb_frame_leave(JBErrorFrame *f) {
    if (jb_cleanup_count != f->cleanup_mark)
        die_status(JB_ERROR_INTERNAL, "armed cleanup on successful return");
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
    jb_cleanup_run(f);
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
