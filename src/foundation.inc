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
 * workers allocate untracked. Without a frame (the CLI), die() exits. */
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
 * error frame, so it would exit the host process, and a longjmp out of the
 * region is undefined. Reaching it there is an engine bug; stop loudly. */
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

/* Strict AVX-512 assigns one output row to each vector lane. Columns are
 * consequently accumulated in exactly the scalar reference order. These
 * load-time images make the 16 rows contiguous for every column; NVFP4
 * keeps its nibbles packed and expands its block scales to lane vectors. */
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
enum { DG_EXACT_ROWS = 16 };

static size_t dg_exact_place(size_t *at, size_t bytes) {
    *at = jb_align_up(*at, 64);
    size_t p = *at;
    *at = jb_size_add(*at, bytes);
    return p;
}

static void dg_exact_pack_bf16(DGTensor *t) {
    int rows = (int)t->shape[0], cols = (int)t->shape[1];
    for (int r = 0; r < rows; r += DG_EXACT_ROWS)
        for (int c = 0; c < cols; c++)
            for (int lane = 0; lane < DG_EXACT_ROWS; lane++)
                memcpy((uint8_t *)t->exact +
                           (((size_t)r / DG_EXACT_ROWS * cols + c) * DG_EXACT_ROWS + lane) * 2,
                       t->data + ((size_t)(r + lane) * cols + c) * 2, 2);
}

static void dg_exact_pack_nvfp4(DGTensor *w, DGTensor *s, int rows, int cols) {
    for (int r = 0; r < rows; r += DG_EXACT_ROWS) {
        for (int c = 0; c < cols; c++)
            for (int lane = 0; lane < DG_EXACT_ROWS / 2; lane++) {
                int r0 = r + lane, r1 = r + lane + DG_EXACT_ROWS / 2;
                uint8_t a = w->data[(size_t)r0 * cols / 2 + c / 2],
                        b = w->data[(size_t)r1 * cols / 2 + c / 2];
                a = (uint8_t)(c & 1 ? a >> 4 : a & 15);
                b = (uint8_t)(c & 1 ? b >> 4 : b & 15);
                ((uint8_t *)w->exact)[((size_t)r / DG_EXACT_ROWS * cols + c) * 8 + lane] =
                    (uint8_t)(a | b << 4);
            }
        for (int b = 0; b < cols / 16; b++)
            for (int lane = 0; lane < DG_EXACT_ROWS; lane++)
                ((uint8_t *)
                     s->exact)[((size_t)r / DG_EXACT_ROWS * (cols / 16) + b) * DG_EXACT_ROWS +
                               lane] = s->data[(size_t)(r + lane) * cols / 16 + b];
    }
}

static void dg_prepare_exact_model(DGModel *m) {
    if (m->accel || strcmp(dg_kernels()->name, "avx512-exact"))
        return;
    size_t bytes = 0;
    for (size_t i = 0; i < m->nt; i++) {
        DGTensor *t = &m->tensor[i];
        if (t->dtype == DG_BF16 && t->nd == 2 && t->shape[0] % DG_EXACT_ROWS == 0) {
            t->exact_offset = dg_exact_place(&bytes, (size_t)t->bytes) + 1;
        }
    }
    for (int i = 0; m->nvfp4 && i < DG_L * DG_EXPERTS; i++) {
        DGNvExpert *v = &m->nvexpert[i];
        DGTensor *w[3] = {v->wg, v->wu, v->wd}, *s[3] = {v->sg, v->su, v->sd};
        int rows[3] = {DG_MOE, DG_MOE, DG_H}, cols[3] = {DG_H, DG_H, DG_MOE};
        for (int k = 0; k < 3; k++) {
            size_t wp = dg_exact_place(&bytes, (size_t)rows[k] * cols[k] / 2),
                   sp = dg_exact_place(&bytes, (size_t)rows[k] * cols[k] / 16);
            w[k]->exact_offset = wp + 1;
            s[k]->exact_offset = sp + 1;
        }
    }
    m->exact_weights = xmalloc(bytes);
    for (size_t i = 0; i < m->nt; i++)
        if (m->tensor[i].exact_offset)
            m->tensor[i].exact = m->exact_weights + m->tensor[i].exact_offset - 1;
    for (size_t i = 0; i < m->nt; i++) {
        DGTensor *t = &m->tensor[i];
        if (t->dtype == DG_BF16 && t->nd == 2 && t->exact)
            dg_exact_pack_bf16(t);
    }
    if (m->nvfp4) {
        int i;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (i = 0; i < DG_L * DG_EXPERTS; i++) {
            DGNvExpert *v = &m->nvexpert[i];
            dg_exact_pack_nvfp4(v->wg, v->sg, DG_MOE, DG_H);
            dg_exact_pack_nvfp4(v->wu, v->su, DG_MOE, DG_H);
            dg_exact_pack_nvfp4(v->wd, v->sd, DG_H, DG_MOE);
        }
    }
}
#else
static void dg_prepare_exact_model(DGModel *m) {
    (void)m;
}
#endif

static float dg_f8e4m3_round(float x) {
    if (!(x > 0))
        return 0;
    if (x >= 448)
        return 448;
    int lo = 0, hi = 126;
    while (lo + 1 < hi) {
        int m = (lo + hi) / 2;
        if (dg_f8e4m3((uint8_t)m) < x)
            lo = m;
        else
            hi = m;
    }
    float a = dg_f8e4m3((uint8_t)lo), b = dg_f8e4m3((uint8_t)hi), da = x - a, db = b - x;
    return da < db || (da == db && !(lo & 1)) ? a : b;
}

static float dg_e2m1_round(float x) {
    static const float q[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float a = fabsf(x);
    int best = 0;
    for (int i = 1; i < 8; i++) {
        float d = fabsf(a - q[i]), old = fabsf(a - q[best]);
        if (d < old || (d == old && !(i & 1)))
            best = i;
    }
    return signbit(x) ? -q[best] : q[best];
}

/* Kernels come in pairs: a portable *_ref version that is always compiled and
 * serves as the correctness reference, and an ISA-specific version selected
 * at compile time. The selftest checks both against a double-precision
 * oracle. */
static void dg_nvfp4_qdq_ref(float *out, const float *in, int tokens, int cols, float base) {
    if (!jb_finitef(base) || !(base > 0) || cols % 32)
        die("invalid NVFP4 activation scale");
    for (int t = 0; t < tokens; t++)
        for (int b = 0; b < cols / 16; b++) {
            const float *x = in + (size_t)t * cols + (size_t)b * 16;
            float *y = out + (size_t)t * cols + (size_t)b * 16, amax = 0;
            for (int k = 0; k < 16; k++) {
                if (!jb_finitef(x[k]))
                    die("non-finite NVFP4 activation");
                if (fabsf(x[k]) > amax)
                    amax = fabsf(x[k]);
            }
            float s = dg_f8e4m3_round((amax / 6) / base) * base;
            if (s == 0) {
                memset(y, 0, 16 * sizeof *y);
                continue;
            }
            for (int k = 0; k < 16; k++)
                y[k] = dg_e2m1_round(x[k] / s) * s;
        }
}

/* The reference quantizer's arithmetic, block for block, spread over the
 * OpenMP team: the vector backends use it, and it matches the reference byte
 * for byte. The scalar build keeps the serial reference as its oracle. */
static void dg_nvfp4_qdq_parallel(float *out, const float *in, int tokens, int cols, float base) {
    if (!jb_finitef(base) || !(base > 0) || cols % 32)
        die("invalid NVFP4 activation scale");
    int nb = cols / 16, blocks = tokens * nb, nonfinite = 0;
    /* die() must not run inside the parallel region: OpenMP workers have no
     * error frame, so it would exit the host process, and a longjmp out of
     * the region is undefined. Record the failure and report it afterwards. */
    {
        int z;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic, 16) if (blocks >= 256) reduction(| : nonfinite)
#endif
        for (z = 0; z < blocks; z++) {
            int t = z / nb, b = z % nb, bad = 0;
            const float *x = in + (size_t)t * cols + (size_t)b * 16;
            float *y = out + (size_t)t * cols + (size_t)b * 16, amax = 0;
            for (int k = 0; k < 16; k++) {
                bad |= !jb_finitef(x[k]);
                if (fabsf(x[k]) > amax)
                    amax = fabsf(x[k]);
            }
            if (bad) {
                nonfinite = 1;
                continue;
            }
            float s = dg_f8e4m3_round((amax / 6) / base) * base;
            if (s == 0) {
                memset(y, 0, 16 * sizeof *y);
                continue;
            }
            for (int k = 0; k < 16; k++)
                y[k] = dg_e2m1_round(x[k] / s) * s;
        }
    }
    if (nonfinite)
        die("non-finite NVFP4 activation");
}

/* Produces the activation layout the selected dg_nvfp4_mm kernel expects. */
static void dg_nvfp4_qdq(float *out, const float *in, int tokens, int cols, float base) {
    dg_kernels()->nvfp4_qdq(out, in, tokens, cols, base);
}

/* x in the dg_nvfp4_qdq_ref layout. */
static void dg_nvfp4_mm_ref(const uint8_t *wd, const uint8_t *sd, float global, const float *x,
                            float *y, int tokens, int rows, int cols) {
    {
        int r;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < rows; r++)
            for (int t = 0; t < tokens; t++) {
                const uint8_t *wp = wd + (uint64_t)r * cols / 2, *sp = sd + (uint64_t)r * cols / 16;
                const float *xp = x + (size_t)t * cols;
                float sum = 0;
                for (int b = 0; b < cols / 16; b++) {
                    float scale = dg_f8e4m3(sp[b]) * global;
                    const uint8_t *q = wp + (size_t)b * 8;
                    const float *a = xp + (size_t)b * 16;
                    for (int k = 0; k < 8; k++) {
                        uint8_t v = q[k];
                        int lo = v & 7, hi = (v >> 4) & 7;
                        float wl = (float[]){0, .5f, 1, 1.5f, 2, 3, 4, 6}[lo],
                              wh = (float[]){0, .5f, 1, 1.5f, 2, 3, 4, 6}[hi];
                        if (v & 8)
                            wl = -wl;
                        if (v & 128)
                            wh = -wh;
                        sum += scale * (wl * a[k * 2] + wh * a[k * 2 + 1]);
                    }
                }
                y[(size_t)t * rows + r] = sum;
            }
    }
}

/* expf, tanhf and the router's exp. Strict builds use these, from IEEE additions,
 * multiplications and divisions in a fixed order and nothing else: no libm
 * and no fused multiply-add, so every strict build, whatever its C library,
 * and the CUDA kernels compute the same bits. Both follow Cephes. Fast math
 * may reassociate the range reduction and flushes subnormals, so fast-math
 * builds, which make no claim to identical bits, keep the C library's. */
#if JB_STRICT_MATH
static float dg_float_bits(uint32_t u) {
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}

/* Rounds x log2(e) half away from zero to n, reduces x by n ln 2 in two
 * parts, the first exact for |n| <= 150, and scales a minimax polynomial by
 * 2^n, in two steps outside the normal exponent range. */
static float dg_expf(float x) {
    if (x != x)
        return x;
    if (x > 88.72283935546875f)
        return INFINITY;
    if (x < -103.97207641601562f)
        return 0;
    int e = (int)(x * 1.44269504088896341f + (x < 0 ? -0.5f : 0.5f));
    float n = (float)e;
    float r = x - n * 0.693359375f;
    r = r - n * -2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * (r * r) + r + 1.0f;
    if (e > 127)
        return p * dg_float_bits(254u << 23) * 2.0f;
    if (e < -126)
        return p * dg_float_bits((uint32_t)(e + 253) << 23) * 0x1p-126f;
    return p * dg_float_bits((uint32_t)(e + 127) << 23);
}

/* Odd polynomial below 0.625, 1 - 2 / (e^2|x| + 1) above, and +-1 from 10,
 * where the result rounds to 1. Tiny arguments return themselves, keeping
 * the sign of zero. */
static float dg_tanhf(float x) {
    float a = x < 0 ? -x : x;
    if (a < 0x1p-12f)
        return x;
    if (a < 0.625f) {
        float z = x * x, p = -5.70498872745e-3f;
        p = p * z + 2.06390887954e-2f;
        p = p * z - 5.37397155531e-2f;
        p = p * z + 1.33314422036e-1f;
        p = p * z - 3.33332819422e-1f;
        return p * z * x + x;
    }
    if (a >= 10.0f)
        return x < 0 ? -1.0f : 1.0f;
    float t = 1.0f - 2.0f / (dg_expf(a + a) + 1.0f);
    return x < 0 ? -t : t;
}

static double dg_double_bits(uint64_t u) {
    double x;
    memcpy(&x, &u, sizeof x);
    return x;
}

static uint64_t dg_bits_double(double x) {
    uint64_t u;
    memcpy(&u, &x, sizeof u);
    return u;
}

/* fdlibm's __ieee754_exp, under 1 ulp: x = k ln 2 + r in two parts, a
 * rational approximation of e^r, and k added to the exponent bits, scaled
 * in two steps below the normal range. The router's softmax uses it. */
static double dg_exp(double x) {
    static const double ln2_hi = 6.93147180369123816490e-01, ln2_lo = 1.90821492927058770002e-10,
                        inv_ln2 = 1.44269504088896338700e+00, p1 = 1.66666666666666019037e-01,
                        p2 = -2.77777777770155933842e-03, p3 = 6.61375632143793436117e-05,
                        p4 = -1.65339022054652515390e-06, p5 = 4.13813679705723846039e-08,
                        two_m1000 = 9.33263618503218878990e-302;
    uint64_t bits = dg_bits_double(x);
    uint32_t hx = (uint32_t)(bits >> 32) & 0x7fffffff;
    int negative = (int)(bits >> 63), k = 0;
    double hi = 0, lo = 0;
    if (hx >= 0x40862E42) {
        if (hx >= 0x7ff00000)
            return x != x ? x + x : negative ? 0.0 : x;
        if (x > 7.09782712893383973096e+02)
            return INFINITY;
        if (x < -7.45133219101941108420e+02)
            return 0.0;
    }
    if (hx > 0x3fd62e42) {
        if (hx < 0x3FF0A2B2) {
            hi = negative ? x + ln2_hi : x - ln2_hi;
            lo = negative ? -ln2_lo : ln2_lo;
            k = negative ? -1 : 1;
        } else {
            k = (int)(inv_ln2 * x + (negative ? -0.5 : 0.5));
            double t = k;
            hi = x - t * ln2_hi;
            lo = t * ln2_lo;
        }
        x = hi - lo;
    } else if (hx < 0x3e300000)
        return 1.0 + x;
    double t = x * x, c = x - t * (p1 + t * (p2 + t * (p3 + t * (p4 + t * p5)))), y;
    if (k == 0)
        return 1.0 - ((x * c) / (c - 2.0) - x);
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
    if (k >= -1021)
        return dg_double_bits(dg_bits_double(y) + ((uint64_t)(int64_t)k << 52));
    return dg_double_bits(dg_bits_double(y) + ((uint64_t)(int64_t)(k + 1000) << 52)) * two_m1000;
}
#else
static float dg_expf(float x) {
    return expf(x);
}

static float dg_tanhf(float x) {
    return tanhf(x);
}

static double dg_exp(double x) {
    return exp(x);
}
#endif

static float dg_gelu(float x) {
    return .5f * x * (1.0f + dg_tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
}

static DGNvMatrix dg_nvfp4_matrix(const DGTensor *w, const DGTensor *s, const DGTensor *g, int rows,
                                  int cols) {
    if (w->dtype != DG_U8 || w->nd != 2 || w->shape[0] != (uint64_t)rows ||
        w->shape[1] != (uint64_t)cols / 2 || s->dtype != DG_F8E4M3 || s->nd != 2 ||
        s->shape[0] != (uint64_t)rows || s->shape[1] != (uint64_t)cols / 16 || g->dtype != DG_F32 ||
        g->nd != 0 || g->bytes != 4 || cols % 32 || rows % 2)
        die2("bad NVFP4 expert tensor", w->name);
    DGNvMatrix m = {w->data, s->data, w->exact, s->exact, 0};
    memcpy(&m.global, g->data, 4);
    return m;
}

#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
static JB_TARGET_AVX512 void dg_mm_exact_avx512(const uint8_t *data, const float *x, float *y,
                                                int tokens, int rows, int cols);
static JB_TARGET_AVX512 void dg_nvfp4_exact_avx512(const uint8_t *wd, const uint8_t *sd,
                                                   float global, const float *x, float *y,
                                                   int tokens, int rows, int cols);
#endif

static void dg_nvfp4_product(const DGNvMatrix *m, const float *x, float *y, int tokens, int rows,
                             int cols) {
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
    if (m->exact_w && m->exact_s) {
        dg_nvfp4_exact_avx512(m->exact_w, m->exact_s, m->global, x, y, tokens, rows, cols);
        return;
    }
#endif
    dg_kernels()->nvfp4_mm(m->w, m->s, m->global, x, y, tokens, rows, cols);
}

/* x comes from dg_nvfp4_qdq, in the layout the selected kernel expects. */
static void dg_nvfp4_mm(const DGTensor *w, const DGTensor *s, const DGTensor *g, const float *x,
                        float *y, int tokens, int rows, int cols) {
    DGNvMatrix m = dg_nvfp4_matrix(w, s, g, rows, cols);
    dg_nvfp4_product(&m, x, y, tokens, rows, cols);
}

/* h = gelu(g) * h over n values, in chunks the faster cores can take more
 * of. */
static void dg_gelu_gate(const float *g, float *h, int n) {
    int i;
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(dynamic, 1024) if (n >= 4096)
#endif
    for (i = 0; i < n; i++)
        h[i] = dg_gelu(g[i]) * h[i];
}

/* The gated product as separate steps: two backend matrix products, then
 * GELU, for backends without a fused form. */
static void dg_nvfp4_gated_each(const DGNvMatrix *gate, const DGNvMatrix *up, const float *x,
                                float *g, float *h, int tokens, int rows, int cols) {
    dg_nvfp4_product(gate, x, g, tokens, rows, cols);
    dg_nvfp4_product(up, x, h, tokens, rows, cols);
    dg_gelu_gate(g, h, tokens * rows);
}

/* gelu(gate x) * (up x) for one expert; g is scratch. */
static void dg_nvfp4_gated(const DGNvExpert *v, const float *x, float *g, float *h, int tokens,
                           int rows, int cols) {
    DGNvMatrix gate = dg_nvfp4_matrix(v->wg, v->sg, v->gg, rows, cols),
               up = dg_nvfp4_matrix(v->wu, v->su, v->gu, rows, cols);
    dg_kernels()->nvfp4_gated(&gate, &up, x, g, h, tokens, rows, cols);
}

static void dg_mm_data_ref(const uint8_t *data, const float *x, float *y, int tokens, int rows,
                           int cols) {
    {
        int r;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < rows; r++) {
            float sum[JB_MAX_CTX];
            for (int t = 0; t < tokens; t++)
                sum[t] = 0;
            const uint8_t *p = data + (uint64_t)r * cols * 2;
            for (int c = 0; c < cols; c++) {
                float a = dg_bf(p + c * 2);
                for (int t = 0; t < tokens; t++)
                    sum[t] += a * x[(size_t)t * cols + c];
            }
            for (int t = 0; t < tokens; t++)
                y[(size_t)t * rows + r] = sum[t];
        }
    }
}

static void dg_mm_data(const uint8_t *data, const float *x, float *y, int tokens, int rows,
                       int cols) {
    if (tokens < 1 || tokens > JB_MAX_CTX)
        die("DiffusionGemma sequence exceeds context limit");
    /* The SIMD kernels tile row pairs; every model matrix has an even row
     * count, and requiring it here keeps all kernels to one contract. */
    if (rows < 2 || rows % 2)
        die("matrix row count must be even");
    dg_kernels()->bf16_mm(data, x, y, tokens, rows, cols);
}

static void dg_mm(const DGTensor *w, const float *x, float *y, int tokens, int rows, int cols) {
    if (w->nd != 2 || w->shape[0] != (uint64_t)rows || w->shape[1] != (uint64_t)cols)
        die2("bad matrix shape", w->name);
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
    if (w->exact) {
        dg_mm_exact_avx512(w->exact, x, y, tokens, rows, cols);
        return;
    }
#endif
    dg_mm_data(w->data, x, y, tokens, rows, cols);
}

static void dg_mv_slice(const DGTensor *w, uint64_t base, const float *x, float *y, int rows,
                        int cols) {
    {
        int r;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (r = 0; r < rows; r++) {
            const uint8_t *p = w->data + (base + (uint64_t)r * cols) * 2;
            float s = 0.0f;
            for (int c = 0; c < cols; c++)
                s += dg_bf(p + c * 2) * x[c];
            y[r] = s;
        }
    }
}

/* Project all answer slots together so the tied LM head is streamed once.
 * OpenJev's automatic reread test is entropy over the union of the
 * full-vocabulary top 20 and the explicitly requested label ids. */
/* logits holds n answer slots' logits over the vocabulary, which this
 * rewrites in place; terms has room for DG_VOCAB doubles. The softcap and
 * the softmax terms are computed in parallel, one value each; the maximum,
 * the top 20 and the sums run in order, as the reference's do. */
static double dg_slot_logits_entropy(int n, int **label_ids, const int *label_n,
                                     double **label_score, float *logits, double *terms) {
    if (n < 1 || n > DG_AUTO_READ_MAX_QUESTIONS)
        die("bad DiffusionGemma answer slot count");
    /* vLLM returns top-20 plus the sorted union of requested label ids at
     * every slot, capped at its per-request 128-id limit. OpenJev computes
     * entropy over that entire returned set. Keep the 128 smallest unique
     * ids without allocating a vocabulary-sized bitmap. */
    int requested[DG_CHOICE_LABELS], nr = 0;
    for (int t = 0; t < n; t++)
        for (int c = 0; c < label_n[t]; c++) {
            int id = label_ids[t][c], at = 0;
            while (at < nr && requested[at] < id)
                at++;
            if (at < nr && requested[at] == id)
                continue;
            if (nr < DG_CHOICE_LABELS) {
                memmove(requested + at + 1, requested + at, (size_t)(nr - at) * sizeof *requested);
                requested[at] = id;
                nr++;
            } else if (at < DG_CHOICE_LABELS) {
                memmove(requested + at + 1, requested + at,
                        (size_t)(DG_CHOICE_LABELS - 1 - at) * sizeof *requested);
                requested[at] = id;
            }
        }
    {
        int i, total = n * DG_VOCAB;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(static)
#endif
        for (i = 0; i < total; i++)
            logits[i] = 30.0f * dg_tanhf(logits[i] / 30.0f);
    }
    double max_entropy = 0;
    for (int t = 0; t < n; t++) {
        float *row = logits + (size_t)t * DG_VOCAB, topv[20];
        int topi[20];
        for (int k = 0; k < 20; k++) {
            topv[k] = -FLT_MAX;
            topi[k] = -1;
        }
        float mx = -FLT_MAX;
        for (int v = 0; v < DG_VOCAB; v++) {
            float z = row[v];
            if (z > mx)
                mx = z;
            /* topv descends, so nothing below its last entry enters. */
            if (!(z > topv[19]))
                continue;
            for (int k = 0; k < 20; k++)
                if (z > topv[k]) {
                    for (int q = 19; q > k; q--) {
                        topv[q] = topv[q - 1];
                        topi[q] = topi[q - 1];
                    }
                    topv[k] = z;
                    topi[k] = v;
                    break;
                }
        }
        {
            int v;
#ifdef _OPENMP
            JB_OMP();
#pragma omp parallel for schedule(static)
#endif
            for (v = 0; v < DG_VOCAB; v++)
                terms[v] = exp((double)row[v] - mx);
        }
        double den = 0;
        for (int v = 0; v < DG_VOCAB; v++)
            den += terms[v];
        double entropy = 0;
        for (int k = 0; k < 20; k++) {
            double p = exp((double)topv[k] - mx) / den;
            if (p > 0)
                entropy -= p * log(p);
        }
        for (int c = 0; c < label_n[t]; c++)
            label_score[t][c] = row[label_ids[t][c]];
        for (int c = 0; c < nr; c++) {
            int id = requested[c], seen = 0;
            for (int k = 0; k < 20; k++)
                if (topi[k] == id) {
                    seen = 1;
                    break;
                }
            if (!seen) {
                double p = exp((double)row[id] - mx) / den;
                if (p > 0)
                    entropy -= p * log(p);
            }
        }
        if (entropy > max_entropy)
            max_entropy = entropy;
    }
    return max_entropy;
}

/* The reference RMS scale factor: a sequential double sum of squares. */
static float dg_rms_inverse(const float *x, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++)
        ss += (double)x[i] * x[i];
    return 1.0f / sqrtf((float)(ss / n) + 1e-6f);
}

static void dg_rms_ref(float *y, const float *x, const DGTensor *scale, int n) {
    float q = dg_rms_inverse(x, n);
    for (int i = 0; i < n; i++)
        y[i] = x[i] * q * (scale ? dg_at(scale, i) : 1.0f);
}

static void dg_rms(float *y, const float *x, const DGTensor *scale, int n) {
    dg_kernels()->rms(y, x, scale, n);
}

static double dg_dot_ref(const float *a, const float *b, int n) {
    double s = 0;
    for (int i = 0; i < n; i++)
        s += (double)a[i] * b[i];
    return s;
}

static double dg_dot(const float *a, const float *b, int n) {
    return dg_kernels()->dot(a, b, n);
}

/* One backend dot per key, for backends without a batched form. */
static void dg_dots_each(const float *a, const float *b, size_t stride, int keys, int n,
                         float *out) {
    for (int j = 0; j < keys; j++)
        out[j] = (float)dg_dot(a, b + (size_t)j * stride, n);
}

static void dg_dots(const float *a, const float *b, size_t stride, int keys, int n, float *out) {
    dg_kernels()->dots(a, b, stride, keys, n, out);
}

/* AVX2 backend. */
#if defined(JB_HAVE_AVX2)
static JB_TARGET_AVX2 __m256 dg_bf16x8(const uint8_t *p) {
    __m128i h = _mm_loadu_si128((const __m128i *)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

static JB_TARGET_AVX2 float dg_hsum8(__m256 x) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    h = _mm_hadd_ps(h, h);
    h = _mm_hadd_ps(h, h);
    return _mm_cvtss_f32(h);
}

static JB_TARGET_AVX2 void dg_mm_data_avx2(const uint8_t *data, const float *x, float *y,
                                           int tokens, int rows, int cols) {
    if (rows % 2)
        die("AVX2 matrix row count must be even");
    {
        int r;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < rows; r += 2) {
            const uint8_t *p0 = data + (uint64_t)r * cols * 2, *p1 = p0 + (uint64_t)cols * 2;
            for (int tb = 0; tb < tokens; tb += 4) {
                int nb = tokens - tb < 4 ? tokens - tb : 4;
                __m256 a[4], b[4];
                for (int q = 0; q < nb; q++)
                    a[q] = b[q] = _mm256_setzero_ps();
                int c = 0;
                for (; c + 7 < cols; c += 8) {
                    __m256 w0 = dg_bf16x8(p0 + c * 2), w1 = dg_bf16x8(p1 + c * 2);
                    for (int q = 0; q < nb; q++) {
                        __m256 v = _mm256_loadu_ps(x + (size_t)(tb + q) * cols + c);
                        a[q] = _mm256_fmadd_ps(w0, v, a[q]);
                        b[q] = _mm256_fmadd_ps(w1, v, b[q]);
                    }
                }
                for (int q = 0; q < nb; q++) {
                    float z0 = dg_hsum8(a[q]), z1 = dg_hsum8(b[q]);
                    for (int k = c; k < cols; k++) {
                        float v = x[(size_t)(tb + q) * cols + k];
                        z0 += dg_bf(p0 + k * 2) * v;
                        z1 += dg_bf(p1 + k * 2) * v;
                    }
                    y[(size_t)(tb + q) * rows + r] = z0;
                    y[(size_t)(tb + q) * rows + r + 1] = z1;
                }
            }
        }
    }
}

static JB_TARGET_AVX2 void dg_nvfp4_weights16_avx2(const uint8_t *q, float scale, __m256 *w0,
                                                   __m256 *w1) {
    const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask = _mm_set1_epi8(15), raw = _mm_loadl_epi64((const __m128i *)q);
    __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw, mask));
    __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw, 4), mask));
    __m256 lf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
    __m256 hf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
    __m256 a = _mm256_unpacklo_ps(lf, hf), b = _mm256_unpackhi_ps(lf, hf),
           s = _mm256_set1_ps(scale * .5f);
    *w0 = _mm256_mul_ps(_mm256_permute2f128_ps(a, b, 0x20), s);
    *w1 = _mm256_mul_ps(_mm256_permute2f128_ps(a, b, 0x31), s);
}

/* Decode each packed 16-weight block entirely in registers, retaining the
 * reference interleaved lane order and reusing it across four tokens. */
static JB_TARGET_AVX2 void dg_nvfp4_mm_avx2(const uint8_t *wd, const uint8_t *sd, float global,
                                            const float *x, float *y, int tokens, int rows,
                                            int cols) {
    {
        int r;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < rows; r++) {
            const uint8_t *wp = wd + (uint64_t)r * cols / 2, *sp = sd + (uint64_t)r * cols / 16;
            for (int tb = 0; tb < tokens; tb += 4) {
                int nb = tokens - tb < 4 ? tokens - tb : 4;
                __m256 acc0[4], acc1[4];
                for (int q = 0; q < nb; q++)
                    acc0[q] = acc1[q] = _mm256_setzero_ps();
                for (int c = 0; c < cols; c += 16) {
                    __m256 w0, w1;
                    dg_nvfp4_weights16_avx2(wp + c / 2, dg_f8e4m3(sp[c / 16]) * global, &w0, &w1);
                    for (int q = 0; q < nb; q++) {
                        const float *xp = x + (size_t)(tb + q) * cols + c;
                        acc0[q] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp), acc0[q]);
                        acc1[q] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp + 8), acc1[q]);
                    }
                }
                for (int q = 0; q < nb; q++)
                    y[(size_t)(tb + q) * rows + r] = dg_hsum8(_mm256_add_ps(acc0[q], acc1[q]));
            }
        }
    }
}

static JB_TARGET_AVX2 double dg_hsum4d(__m256d x) {
    __m128d h = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
    h = _mm_hadd_pd(h, h);
    return _mm_cvtsd_f64(h);
}

static JB_TARGET_AVX2 void dg_rms_avx2(float *y, const float *x, const DGTensor *scale, int n) {
    __m256d a = _mm256_setzero_pd(), b = a;
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256d lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v)),
                hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
        a = _mm256_fmadd_pd(lo, lo, a);
        b = _mm256_fmadd_pd(hi, hi, b);
    }
    double ss = dg_hsum4d(_mm256_add_pd(a, b));
    for (; i < n; i++)
        ss += (double)x[i] * x[i];
    float q = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    i = 0;
    if (scale)
        for (; i + 7 < n; i += 8)
            _mm256_storeu_ps(y + i,
                             _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(q)),
                                           dg_bf16x8(scale->data + (size_t)i * 2)));
    else
        for (; i + 7 < n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(q)));
    for (; i < n; i++)
        y[i] = x[i] * q * (scale ? dg_at(scale, i) : 1.0f);
}

static JB_TARGET_AVX2 double dg_dot_avx2(const float *a, const float *b, int n) {
    __m256d s0 = _mm256_setzero_pd(), s1 = s0;
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(a + i), y = _mm256_loadu_ps(b + i);
        s0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(x)),
                             _mm256_cvtps_pd(_mm256_castps256_ps128(y)), s0);
        s1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(x, 1)),
                             _mm256_cvtps_pd(_mm256_extractf128_ps(y, 1)), s1);
    }
    double s = dg_hsum4d(_mm256_add_pd(s0, s1));
    for (; i < n; i++)
        s += (double)a[i] * b[i];
    return s;
}
#endif

/* AVX-512 backend. */
#if defined(JB_HAVE_AVX512)
static JB_TARGET_AVX512 __m512 dg_bf16x16(const uint8_t *p) {
    __m256i h = _mm256_loadu_si256((const __m256i *)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}

#if JB_STRICT_MATH
static JB_TARGET_AVX512 void dg_mm_exact_avx512(const uint8_t *data, const float *x, float *y,
                                                int tokens, int rows, int cols) {
    int rt;
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (rt = 0; rt < rows; rt += DG_EXACT_ROWS) {
        const uint8_t *tile = data + (size_t)rt * cols * 2;
        for (int tb = 0; tb < tokens; tb += 8) {
            int n = tokens - tb < 8 ? tokens - tb : 8;
            __m512 sum[8];
            for (int q = 0; q < n; q++)
                sum[q] = _mm512_setzero_ps();
            for (int c = 0; c < cols; c++) {
                __m512 w = dg_bf16x16(tile + (size_t)c * DG_EXACT_ROWS * 2);
                for (int q = 0; q < n; q++) {
                    __m512 p = _mm512_mul_ps(w, _mm512_set1_ps(x[(size_t)(tb + q) * cols + c]));
                    sum[q] = _mm512_add_ps(sum[q], p);
                }
            }
            for (int q = 0; q < n; q++)
                _mm512_storeu_ps(y + (size_t)(tb + q) * rows + rt, sum[q]);
        }
    }
}

static JB_TARGET_AVX512 __m512i dg_nvfp4_exact_codes(const uint8_t *p) {
    __m256i bytes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p));
    __m256i lo = _mm256_and_si256(bytes, _mm256_set1_epi32(15));
    __m256i hi = _mm256_srli_epi32(bytes, 4);
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

/* Decode 16 transposed E4M3 scale bytes without expanding them in the model
 * image. The integer bit construction and the subnormal multiply reproduce
 * dg_f8e4m3, lane for lane. */
static JB_TARGET_AVX512 __m512 dg_f8e4m3x16(const uint8_t *p) {
    __m512i code = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)p));
    __m512i exponent = _mm512_and_si512(_mm512_srli_epi32(code, 3), _mm512_set1_epi32(15));
    __m512i mantissa = _mm512_and_si512(code, _mm512_set1_epi32(7));
    __m512 normal = _mm512_castsi512_ps(
        _mm512_or_si512(_mm512_slli_epi32(_mm512_add_epi32(exponent, _mm512_set1_epi32(120)), 23),
                        _mm512_slli_epi32(mantissa, 20)));
    __m512 subnormal = _mm512_mul_ps(_mm512_cvtepi32_ps(mantissa), _mm512_set1_ps(0x1p-9f));
    __mmask16 is_subnormal = _mm512_cmpeq_epi32_mask(exponent, _mm512_setzero_si512());
    __m512 magnitude = _mm512_mask_blend_ps(is_subnormal, normal, subnormal);
    __m512i sign = _mm512_slli_epi32(_mm512_and_si512(code, _mm512_set1_epi32(128)), 24);
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(magnitude), sign));
}

#ifndef JB_NO_MAIN
static JB_TARGET_AVX512 void dg_f8e4m3x16_store(float *out, const uint8_t *p) {
    _mm512_storeu_ps(out, dg_f8e4m3x16(p));
}
#endif

static JB_TARGET_AVX512 void dg_nvfp4_exact_avx512(const uint8_t *wd, const uint8_t *sd,
                                                   float global, const float *x, float *y,
                                                   int tokens, int rows, int cols) {
    static const float lut[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
    __m512 table = _mm512_loadu_ps(lut);
    int rt;
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (rt = 0; rt < rows; rt += DG_EXACT_ROWS) {
        size_t tile = (size_t)rt / DG_EXACT_ROWS;
        for (int tb = 0; tb < tokens; tb += 8) {
            int n = tokens - tb < 8 ? tokens - tb : 8;
            __m512 sum[8];
            for (int q = 0; q < n; q++)
                sum[q] = _mm512_setzero_ps();
            for (int b = 0; b < cols / 16; b++) {
                __m512 scale =
                    _mm512_mul_ps(dg_f8e4m3x16(sd + (tile * (cols / 16) + b) * DG_EXACT_ROWS),
                                  _mm512_set1_ps(global));
                for (int k = 0; k < 8; k++) {
                    int c = b * 16 + k * 2;
                    __m512 we = _mm512_permutexvar_ps(
                        dg_nvfp4_exact_codes(wd + (tile * cols + c) * 8), table);
                    __m512 wo = _mm512_permutexvar_ps(
                        dg_nvfp4_exact_codes(wd + (tile * cols + c + 1) * 8), table);
                    for (int q = 0; q < n; q++) {
                        const float *a = x + (size_t)(tb + q) * cols + c;
                        __m512 pair = _mm512_add_ps(_mm512_mul_ps(we, _mm512_set1_ps(a[0])),
                                                    _mm512_mul_ps(wo, _mm512_set1_ps(a[1])));
                        sum[q] = _mm512_add_ps(sum[q], _mm512_mul_ps(scale, pair));
                    }
                }
            }
            for (int q = 0; q < n; q++)
                _mm512_storeu_ps(y + (size_t)(tb + q) * rows + rt, sum[q]);
        }
    }
}
#endif

static JB_TARGET_AVX512 void dg_mm_data_avx512(const uint8_t *data, const float *x, float *y,
                                               int tokens, int rows, int cols) {
    if (rows % 2)
        die("AVX-512 matrix row count must be even");
    {
        int r;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < rows; r += 2) {
            const uint8_t *p0 = data + (uint64_t)r * cols * 2, *p1 = p0 + (uint64_t)cols * 2;
            int t = 0;
            for (; t + 7 < tokens; t += 8) {
                __m512 a[8], b[8];
                for (int q = 0; q < 8; q++)
                    a[q] = b[q] = _mm512_setzero_ps();
                int c = 0;
                for (; c + 15 < cols; c += 16) {
                    __m512 w0 = dg_bf16x16(p0 + c * 2), w1 = dg_bf16x16(p1 + c * 2);
                    for (int q = 0; q < 8; q++) {
                        __m512 v = _mm512_loadu_ps(x + (size_t)(t + q) * cols + c);
                        a[q] = _mm512_fmadd_ps(w0, v, a[q]);
                        b[q] = _mm512_fmadd_ps(w1, v, b[q]);
                    }
                }
                for (int q = 0; q < 8; q++) {
                    float z0 = _mm512_reduce_add_ps(a[q]), z1 = _mm512_reduce_add_ps(b[q]);
                    for (int k = (cols & ~15); k < cols; k++) {
                        float v = x[(size_t)(t + q) * cols + k];
                        z0 += dg_bf(p0 + k * 2) * v;
                        z1 += dg_bf(p1 + k * 2) * v;
                    }
                    y[(size_t)(t + q) * rows + r] = z0;
                    y[(size_t)(t + q) * rows + r + 1] = z1;
                }
            }
            for (; t < tokens; t++) {
                __m512 s0 = _mm512_setzero_ps(), s1 = s0;
                int c = 0;
                for (; c + 15 < cols; c += 16) {
                    __m512 v = _mm512_loadu_ps(x + (size_t)t * cols + c);
                    s0 = _mm512_fmadd_ps(dg_bf16x16(p0 + c * 2), v, s0);
                    s1 = _mm512_fmadd_ps(dg_bf16x16(p1 + c * 2), v, s1);
                }
                float z0 = _mm512_reduce_add_ps(s0), z1 = _mm512_reduce_add_ps(s1);
                for (; c < cols; c++) {
                    float v = x[(size_t)t * cols + c];
                    z0 += dg_bf(p0 + c * 2) * v;
                    z1 += dg_bf(p1 + c * 2) * v;
                }
                y[(size_t)t * rows + r] = z0;
                y[(size_t)t * rows + r + 1] = z1;
            }
        }
    }
}

/* Match each packed weight byte: low-nibble activations, then high. This
 * one-time swizzle removes two activation permutes per expert row/tile. */
static void dg_nvfp4_swizzle(float *out, int tokens, int cols) {
    for (int t = 0; t < tokens; t++)
        for (int c = 0; c < cols; c += 32) {
            float *y = out + (size_t)t * cols + c, tmp[32];
            memcpy(tmp, y, sizeof tmp);
            for (int k = 0; k < 16; k++) {
                y[k] = tmp[k * 2];
                y[16 + k] = tmp[k * 2 + 1];
            }
        }
}

static void dg_nvfp4_qdq_avx512(float *out, const float *in, int tokens, int cols, float base) {
    dg_nvfp4_qdq_ref(out, in, tokens, cols, base);
    dg_nvfp4_swizzle(out, tokens, cols);
}

/* x in the dg_nvfp4_swizzle layout; rows must be even. */
static JB_TARGET_AVX512 void dg_nvfp4_mm_avx512(const uint8_t *wd, const uint8_t *sd, float global,
                                                const float *x, float *y, int tokens, int rows,
                                                int cols) {
    static const float lut[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
    __m512 table = _mm512_loadu_ps(lut);
    {
        int r;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < rows; r += 2) {
            const uint8_t *wp0 = wd + (uint64_t)r * cols / 2, *wp1 = wp0 + cols / 2;
            const uint8_t *sp0 = sd + (uint64_t)r * cols / 16, *sp1 = sp0 + cols / 16;
            for (int tb = 0; tb < tokens; tb += 8) {
                int nb = tokens - tb < 8 ? tokens - tb : 8;
                __m512 a[8], b[8];
                for (int q = 0; q < nb; q++)
                    a[q] = b[q] = _mm512_setzero_ps();
                for (int c = 0; c < cols; c += 32) {
                    __m512i z = _mm512_set1_epi32(15),
                            raw0 = _mm512_cvtepu8_epi32(
                                _mm_loadu_si128((const __m128i *)(wp0 + c / 2))),
                            raw1 = _mm512_cvtepu8_epi32(
                                _mm_loadu_si128((const __m128i *)(wp1 + c / 2)));
                    __m512 wl0 = _mm512_permutexvar_ps(_mm512_and_si512(raw0, z), table),
                           wh0 = _mm512_permutexvar_ps(_mm512_srli_epi32(raw0, 4), table);
                    __m512 wl1 = _mm512_permutexvar_ps(_mm512_and_si512(raw1, z), table),
                           wh1 = _mm512_permutexvar_ps(_mm512_srli_epi32(raw1, 4), table);
                    float a0 = dg_f8e4m3(sp0[c / 16]) * global,
                          a1 = dg_f8e4m3(sp0[c / 16 + 1]) * global,
                          b0 = dg_f8e4m3(sp1[c / 16]) * global,
                          b1 = dg_f8e4m3(sp1[c / 16 + 1]) * global;
                    __m512 sv0 =
                               _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(a0), _mm512_set1_ps(a1)),
                           sv1 =
                               _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(b0), _mm512_set1_ps(b1));
                    wl0 = _mm512_mul_ps(wl0, sv0);
                    wh0 = _mm512_mul_ps(wh0, sv0);
                    wl1 = _mm512_mul_ps(wl1, sv1);
                    wh1 = _mm512_mul_ps(wh1, sv1);
                    for (int q = 0; q < nb; q++) {
                        const float *xp = x + (size_t)(tb + q) * cols + c;
                        __m512 xe = _mm512_loadu_ps(xp), xo = _mm512_loadu_ps(xp + 16);
                        a[q] = _mm512_fmadd_ps(wl0, xe, a[q]);
                        a[q] = _mm512_fmadd_ps(wh0, xo, a[q]);
                        b[q] = _mm512_fmadd_ps(wl1, xe, b[q]);
                        b[q] = _mm512_fmadd_ps(wh1, xo, b[q]);
                    }
                }
                for (int q = 0; q < nb; q++) {
                    y[(size_t)(tb + q) * rows + r] = _mm512_reduce_add_ps(a[q]);
                    y[(size_t)(tb + q) * rows + r + 1] = _mm512_reduce_add_ps(b[q]);
                }
            }
        }
    }
}

static JB_TARGET_AVX512 void dg_rms_avx512(float *y, const float *x, const DGTensor *scale, int n) {
    __m512d a = _mm512_setzero_pd(), b = a, c = a, d = a;
    int i = 0;
    for (; i + 31 < n; i += 32) {
        __m512 x0 = _mm512_loadu_ps(x + i), x1 = _mm512_loadu_ps(x + i + 16);
        __m512d q0 = _mm512_cvtps_pd(_mm512_castps512_ps256(x0));
        __m512d q1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(x0, 1));
        __m512d q2 = _mm512_cvtps_pd(_mm512_castps512_ps256(x1));
        __m512d q3 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(x1, 1));
        a = _mm512_fmadd_pd(q0, q0, a);
        b = _mm512_fmadd_pd(q1, q1, b);
        c = _mm512_fmadd_pd(q2, q2, c);
        d = _mm512_fmadd_pd(q3, q3, d);
    }
    double ss = _mm512_reduce_add_pd(_mm512_add_pd(_mm512_add_pd(a, b), _mm512_add_pd(c, d)));
    for (; i < n; i++)
        ss += (double)x[i] * x[i];
    float q = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    i = 0;
    if (scale)
        for (; i + 15 < n; i += 16)
            _mm512_storeu_ps(y + i,
                             _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x + i), _mm512_set1_ps(q)),
                                           dg_bf16x16(scale->data + (size_t)i * 2)));
    else
        for (; i + 15 < n; i += 16)
            _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), _mm512_set1_ps(q)));
    for (; i < n; i++)
        y[i] = x[i] * q * (scale ? dg_at(scale, i) : 1.0f);
}

static JB_TARGET_AVX512 double dg_dot_avx512(const float *a, const float *b, int n) {
    __m512d s0 = _mm512_setzero_pd(), s1 = s0;
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 x = _mm512_loadu_ps(a + i), y = _mm512_loadu_ps(b + i);
        s0 = _mm512_fmadd_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(x)),
                             _mm512_cvtps_pd(_mm512_castps512_ps256(y)), s0);
        s1 = _mm512_fmadd_pd(_mm512_cvtps_pd(_mm512_extractf32x8_ps(x, 1)),
                             _mm512_cvtps_pd(_mm512_extractf32x8_ps(y, 1)), s1);
    }
    double s = _mm512_reduce_add_pd(_mm512_add_pd(s0, s1));
    for (; i < n; i++)
        s += (double)a[i] * b[i];
    return s;
}
#endif

/* NEON backend (AArch64). Each vector lane computes one output row (or one
 * attention key), and each lane accumulates in the reference kernels' order
 * with a separate multiply and add. Strict builds therefore match the
 * reference bit for bit, which the selftest checks. */
#if defined(JB_NEON)
enum { DG_NEON_TOKENS = 8, DG_NEON_KEYS = 8 };

static float32x4_t dg_bf16x4_neon(const uint8_t *p) {
    uint16x4_t h = vreinterpret_u16_u8(vld1_u8(p));
    return vreinterpretq_f32_u32(vshll_n_u16(h, 16));
}

/* Four rows of four columns become four columns of four rows. */
static void dg_transpose4_neon(float32x4_t *v) {
    float32x4_t t0 = vtrn1q_f32(v[0], v[1]), t1 = vtrn2q_f32(v[0], v[1]);
    float32x4_t t2 = vtrn1q_f32(v[2], v[3]), t3 = vtrn2q_f32(v[2], v[3]);
    v[0] = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
    v[1] = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
    v[2] = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
    v[3] = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
}

/* Rows past the end of the matrix repeat its last row; their lanes are
 * computed but never stored. */
static int dg_neon_row(int r0, int i, int rows) {
    return r0 + i < rows ? r0 + i : rows - 1;
}

static void dg_mm_data_neon(const uint8_t *data, const float *x, float *y, int tokens, int rows,
                            int cols) {
    int blocks = (rows + 3) / 4, c4 = cols & ~3;
    {
        int rb;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (rb = 0; rb < blocks; rb++) {
            int r0 = rb * 4, valid = rows - r0 < 4 ? rows - r0 : 4;
            const uint8_t *p[4];
            for (int i = 0; i < 4; i++)
                p[i] = data + (uint64_t)dg_neon_row(r0, i, rows) * cols * 2;
            for (int tb = 0; tb < tokens; tb += DG_NEON_TOKENS) {
                int nb = tokens - tb < DG_NEON_TOKENS ? tokens - tb : DG_NEON_TOKENS;
                float32x4_t acc[DG_NEON_TOKENS];
                for (int q = 0; q < nb; q++)
                    acc[q] = vdupq_n_f32(0);
                for (int c = 0; c < c4; c += 4) {
                    float32x4_t w[4];
                    for (int i = 0; i < 4; i++)
                        w[i] = dg_bf16x4_neon(p[i] + (size_t)c * 2);
                    dg_transpose4_neon(w);
                    for (int q = 0; q < nb; q++) {
                        const float *xp = x + (size_t)(tb + q) * cols + c;
                        for (int k = 0; k < 4; k++)
                            acc[q] = vaddq_f32(acc[q], vmulq_n_f32(w[k], xp[k]));
                    }
                }
                for (int q = 0; q < nb; q++) {
                    const float *xp = x + (size_t)(tb + q) * cols;
                    float sum[4];
                    vst1q_f32(sum, acc[q]);
                    for (int i = 0; i < valid; i++) {
                        for (int c = c4; c < cols; c++)
                            sum[i] += dg_bf(p[i] + (size_t)c * 2) * xp[c];
                        y[(size_t)(tb + q) * rows + r0 + i] = sum[i];
                    }
                }
            }
        }
    }
}

/* Signed E2M1 values by nibble, sign bit included; entry 8 is -0.0, which
 * the reference also produces and bit-identity must keep. */
static const float dg_e2m1_neon[16] = {0,     .5f,  1,  1.5f,  2,  3,  4,  6,
                                       -0.0f, -.5f, -1, -1.5f, -2, -3, -4, -6};

/* Four rows' float weights for byte position k from nibbles held as
 * table offsets (nibble * 4): spread each row's offset over its lane's four
 * bytes, add the byte index, and read the float's bytes from the table. */
static float32x4_t dg_e2m1x4_neon(uint8x16x4_t table, uint8x16_t offsets, int k) {
    static const uint8_t spread[4][16] = {
        {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3},
        {4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7},
        {8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11},
        {12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15}};
    static const uint8_t byte[16] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    uint8x16_t index = vaddq_u8(vqtbl1q_u8(offsets, vld1q_u8(spread[k])), vld1q_u8(byte));
    return vreinterpretq_f32_u8(vqtbl4q_u8(table, index));
}

/* One packed 16-weight block (eight bytes) from each of four rows, as eight
 * vectors of even-column weights and eight of odd-column weights, each
 * holding the four rows' values for one byte position. */
static void dg_nvfp4_weights4_neon(const uint8_t *const *q, float32x4_t *even, float32x4_t *odd) {
    const uint8x16x4_t table = vld1q_u8_x4((const uint8_t *)dg_e2m1_neon);
    uint8x8x2_t z01 = vzip_u8(vld1_u8(q[0]), vld1_u8(q[1]));
    uint8x8x2_t z23 = vzip_u8(vld1_u8(q[2]), vld1_u8(q[3]));
    for (int half = 0; half < 2; half++) {
        uint16x4x2_t z =
            vzip_u16(vreinterpret_u16_u8(z01.val[half]), vreinterpret_u16_u8(z23.val[half]));
        uint8x16_t bytes =
            vcombine_u8(vreinterpret_u8_u16(z.val[0]), vreinterpret_u8_u16(z.val[1]));
        uint8x16_t lo = vshlq_n_u8(vandq_u8(bytes, vdupq_n_u8(15)), 2);
        uint8x16_t hi = vshlq_n_u8(vshrq_n_u8(bytes, 4), 2);
        for (int k = 0; k < 4; k++) {
            even[half * 4 + k] = dg_e2m1x4_neon(table, lo, k);
            odd[half * 4 + k] = dg_e2m1x4_neon(table, hi, k);
        }
    }
}

/* dg_f8e4m3 for four rows' scale bytes: the same bit assembly, with the
 * subnormals (exponent 0) selected by mask. */
static float32x4_t dg_e4m3x4_neon(uint32x4_t u) {
    uint32x4_t e = vandq_u32(vshrq_n_u32(u, 3), vdupq_n_u32(15)), m = vandq_u32(u, vdupq_n_u32(7));
    uint32x4_t normal =
        vorrq_u32(vshlq_n_u32(vaddq_u32(e, vdupq_n_u32(120)), 23), vshlq_n_u32(m, 20));
    float32x4_t subnormal = vmulq_n_f32(vcvtq_f32_u32(m), 0x1p-9f);
    float32x4_t v =
        vbslq_f32(vcgtq_u32(e, vdupq_n_u32(0)), vreinterpretq_f32_u32(normal), subnormal);
    uint32x4_t sign = vshlq_n_u32(vshrq_n_u32(u, 7), 31);
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v), sign));
}

/* One block's contribution to one token's four rows, in the reference's
 * order: acc += scale * (even * a[2k] + odd * a[2k + 1]) for k = 0..7. The
 * activations are loaded once and used by lane. */
static float32x4_t dg_nvfp4_block_neon(float32x4_t acc, const float32x4_t *even,
                                       const float32x4_t *odd, float32x4_t scale, const float *a) {
    float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4), a2 = vld1q_f32(a + 8),
                a3 = vld1q_f32(a + 12);
#define DG_NVFP4_PAIR(k, v, even_lane, odd_lane)                                                   \
    acc = vaddq_f32(acc, vmulq_f32(scale, vaddq_f32(vmulq_laneq_f32(even[k], v, even_lane),        \
                                                    vmulq_laneq_f32(odd[k], v, odd_lane))))
    DG_NVFP4_PAIR(0, a0, 0, 1);
    DG_NVFP4_PAIR(1, a0, 2, 3);
    DG_NVFP4_PAIR(2, a1, 0, 1);
    DG_NVFP4_PAIR(3, a1, 2, 3);
    DG_NVFP4_PAIR(4, a2, 0, 1);
    DG_NVFP4_PAIR(5, a2, 2, 3);
    DG_NVFP4_PAIR(6, a3, 0, 1);
    DG_NVFP4_PAIR(7, a3, 2, 3);
#undef DG_NVFP4_PAIR
    return acc;
}

/* Four rows' products with the nb tokens from tb, in the reference's order;
 * rows past the end repeat the last row. x in the dg_nvfp4_qdq_ref layout. */
static void dg_nvfp4_rows4_neon(const DGNvMatrix *m, const float *x, int r0, int rows, int cols,
                                int tb, int nb, float32x4_t *acc) {
    const uint8_t *wp[4], *sp[4];
    for (int i = 0; i < 4; i++) {
        int r = dg_neon_row(r0, i, rows);
        wp[i] = m->w + (uint64_t)r * cols / 2;
        sp[i] = m->s + (uint64_t)r * cols / 16;
    }
    for (int t = 0; t < nb; t++)
        acc[t] = vdupq_n_f32(0);
    for (int b = 0; b < cols / 16; b++) {
        const uint8_t *q[4];
        for (int i = 0; i < 4; i++)
            q[i] = wp[i] + (size_t)b * 8;
        uint32x4_t raw = {sp[0][b], sp[1][b], sp[2][b], sp[3][b]};
        float32x4_t scale = vmulq_n_f32(dg_e4m3x4_neon(raw), m->global), even[8], odd[8];
        dg_nvfp4_weights4_neon(q, even, odd);
        for (int t = 0; t < nb; t++)
            acc[t] = dg_nvfp4_block_neon(acc[t], even, odd, scale,
                                         x + (size_t)(tb + t) * cols + (size_t)b * 16);
    }
}

static void dg_nvfp4_mm_neon(const uint8_t *wd, const uint8_t *sd, float global, const float *x,
                             float *y, int tokens, int rows, int cols) {
    DGNvMatrix m = {.w = wd, .s = sd, .global = global};
    int blocks = (rows + 3) / 4;
    {
        int rb;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (rb = 0; rb < blocks; rb++) {
            int r0 = rb * 4, valid = rows - r0 < 4 ? rows - r0 : 4;
            for (int tb = 0; tb < tokens; tb += DG_NEON_TOKENS) {
                int nb = tokens - tb < DG_NEON_TOKENS ? tokens - tb : DG_NEON_TOKENS;
                float32x4_t acc[DG_NEON_TOKENS];
                dg_nvfp4_rows4_neon(&m, x, r0, rows, cols, tb, nb, acc);
                for (int t = 0; t < nb; t++) {
                    float sum[4];
                    vst1q_f32(sum, acc[t]);
                    for (int i = 0; i < valid; i++)
                        y[(size_t)(tb + t) * rows + r0 + i] = sum[i];
                }
            }
        }
    }
}

/* Gate and up products for the same four rows, then GELU, in one parallel
 * region; each value is the one the separate steps compute. */
static void dg_nvfp4_gated_neon(const DGNvMatrix *gate, const DGNvMatrix *up, const float *x,
                                float *g, float *h, int tokens, int rows, int cols) {
    (void)g;
    int blocks = (rows + 3) / 4;
    {
        int rb;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (rb = 0; rb < blocks; rb++) {
            int r0 = rb * 4, valid = rows - r0 < 4 ? rows - r0 : 4;
            for (int tb = 0; tb < tokens; tb += DG_NEON_TOKENS) {
                int nb = tokens - tb < DG_NEON_TOKENS ? tokens - tb : DG_NEON_TOKENS;
                float32x4_t ga[DG_NEON_TOKENS], ua[DG_NEON_TOKENS];
                dg_nvfp4_rows4_neon(gate, x, r0, rows, cols, tb, nb, ga);
                dg_nvfp4_rows4_neon(up, x, r0, rows, cols, tb, nb, ua);
                for (int t = 0; t < nb; t++) {
                    float gs[4], us[4];
                    vst1q_f32(gs, ga[t]);
                    vst1q_f32(us, ua[t]);
                    for (int i = 0; i < valid; i++)
                        h[(size_t)(tb + t) * rows + r0 + i] = dg_gelu(gs[i]) * us[i];
                }
            }
        }
    }
}

/* Attention dots for eight keys at a time, one key per double lane. Each
 * lane sums (double)a[i] * b[i] in index order like dg_dot_ref, and a
 * product of two floats is exact in double, so the sums match it bit for
 * bit. Keys past the end repeat the last key and are not stored. */
static void dg_dots_neon(const float *a, const float *b, size_t stride, int keys, int n,
                         float *out) {
    int n4 = n & ~3;
    for (int j0 = 0; j0 < keys; j0 += DG_NEON_KEYS) {
        int valid = keys - j0 < DG_NEON_KEYS ? keys - j0 : DG_NEON_KEYS;
        const float *k[DG_NEON_KEYS];
        float64x2_t s[DG_NEON_KEYS / 2];
        for (int j = 0; j < DG_NEON_KEYS; j++)
            k[j] = b + (size_t)dg_neon_row(j0, j, keys) * stride;
        for (int p = 0; p < DG_NEON_KEYS / 2; p++)
            s[p] = vdupq_n_f64(0);
        for (int i = 0; i < n4; i += 4)
            for (int g = 0; g < DG_NEON_KEYS / 4; g++) {
                float32x4_t w[4];
                for (int j = 0; j < 4; j++)
                    w[j] = vld1q_f32(k[g * 4 + j] + i);
                dg_transpose4_neon(w);
                for (int c = 0; c < 4; c++) {
                    float64x2_t ac = vdupq_n_f64((double)a[i + c]);
                    s[g * 2] = vaddq_f64(s[g * 2], vmulq_f64(vcvt_f64_f32(vget_low_f32(w[c])), ac));
                    s[g * 2 + 1] = vaddq_f64(s[g * 2 + 1], vmulq_f64(vcvt_high_f64_f32(w[c]), ac));
                }
            }
        double sum[DG_NEON_KEYS];
        for (int p = 0; p < DG_NEON_KEYS / 2; p++)
            vst1q_f64(sum + p * 2, s[p]);
        for (int j = 0; j < valid; j++) {
            for (int i = n4; i < n; i++)
                sum[j] += (double)a[i] * k[j][i];
            out[j0 + j] = (float)sum[j];
        }
    }
}

static void dg_rms_neon(float *y, const float *x, const DGTensor *scale, int n) {
    float q = dg_rms_inverse(x, n);
    int i = 0;
    if (scale)
        for (; i + 3 < n; i += 4)
            vst1q_f32(y + i, vmulq_f32(vmulq_n_f32(vld1q_f32(x + i), q),
                                       dg_bf16x4_neon(scale->data + (size_t)i * 2)));
    else
        for (; i + 3 < n; i += 4)
            vst1q_f32(y + i, vmulq_n_f32(vld1q_f32(x + i), q));
    for (; i < n; i++)
        y[i] = x[i] * q * (scale ? dg_at(scale, i) : 1.0f);
}
#endif

#if defined(JB_HAVE_AVX2)
static int jb_cpu_has_avx2(void) {
#if defined(JB_SCALAR)
    return 0;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#elif defined(_MSC_VER) && defined(_M_X64)
    int leaf[4];
    __cpuid(leaf, 1);
    if (!(leaf[2] & (1 << 27)) || !(leaf[2] & (1 << 28)) || !(leaf[2] & (1 << 12)) ||
        (_xgetbv(0) & 6) != 6)
        return 0;
    __cpuidex(leaf, 7, 0);
    return !!(leaf[1] & (1 << 5));
#else
    return 0;
#endif
}
#endif

#if defined(JB_HAVE_AVX512)
static int jb_cpu_has_avx512(void) {
#if defined(JB_SCALAR)
    return 0;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512bw");
#else
    return 0;
#endif
}
#endif

static const DGKernelOps *dg_kernels(void) {
    (void)dg_nvfp4_qdq_parallel;
    static _Atomic(const DGKernelOps *) cache;
    static const DGKernelOps scalar = {
        "scalar",       dg_nvfp4_qdq_ref, NULL,       dg_nvfp4_mm_ref, dg_nvfp4_gated_each,
        dg_mm_data_ref, dg_rms_ref,       dg_dot_ref, dg_dots_each,    1};
    const DGKernelOps *cached = atomic_load_explicit(&cache, memory_order_acquire);
    if (cached)
        return cached;
    const DGKernelOps *selected = &scalar;
#if defined(JB_HAVE_AVX512)
    (void)dg_nvfp4_qdq_parallel;
#if JB_STRICT_MATH
    (void)dg_nvfp4_qdq_avx512;
    (void)dg_nvfp4_mm_avx512;
    (void)dg_mm_data_avx512;
    (void)dg_dot_avx512;
    static const DGKernelOps avx512 = {
        "avx512-exact", dg_nvfp4_qdq_ref, NULL,       dg_nvfp4_mm_ref, dg_nvfp4_gated_each,
        dg_mm_data_ref, dg_rms_avx512,    dg_dot_ref, dg_dots_each,    1};
#else
    static const DGKernelOps avx512 = {"avx512",
                                       dg_nvfp4_qdq_avx512,
                                       dg_nvfp4_swizzle,
                                       dg_nvfp4_mm_avx512,
                                       dg_nvfp4_gated_each,
                                       dg_mm_data_avx512,
                                       dg_rms_avx512,
                                       dg_dot_avx512,
                                       dg_dots_each,
                                       0};
#endif
    if (jb_cpu_has_avx512())
        selected = &avx512;
#endif
#if defined(JB_HAVE_AVX2)
    static const DGKernelOps avx2 = {
        "avx2",          dg_nvfp4_qdq_parallel, NULL,        dg_nvfp4_mm_avx2, dg_nvfp4_gated_each,
        dg_mm_data_avx2, dg_rms_avx2,           dg_dot_avx2, dg_dots_each,     0};
    if (selected == &scalar && jb_cpu_has_avx2())
        selected = &avx2;
#endif
#if defined(JB_NEON)
    static const DGKernelOps neon = {
        "neon",          dg_nvfp4_qdq_parallel, NULL,       dg_nvfp4_mm_neon, dg_nvfp4_gated_neon,
        dg_mm_data_neon, dg_rms_neon,           dg_dot_ref, dg_dots_neon,     1};
    selected = &neon;
#endif
    const DGKernelOps *expected = NULL;
    if (!atomic_compare_exchange_strong_explicit(&cache, &expected, selected, memory_order_release,
                                                 memory_order_acquire))
        selected = expected;
    return selected;
}
