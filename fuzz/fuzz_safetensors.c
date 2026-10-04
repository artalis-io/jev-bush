/*
 * libFuzzer target for the bounded safetensors metadata reader.
 *
 * Byte zero selects BF16/NVFP4 validation and raw/structured input. Raw mode
 * passes the remaining bytes directly to the parser. Structured mode reads a
 * little-endian 32-bit header length from bytes 1..4 and turns the remaining
 * bytes into a safetensors file with a valid 64-bit length prefix. This lets
 * mutations reach JSON descriptors while raw mode continues to exercise the
 * framing checks themselves.
 *
 * Build: clang -g -O1 -fsanitize=fuzzer,address,undefined \
 *        -fno-sanitize-recover=all fuzz/fuzz_safetensors.c -lm -o fuzz_safetensors
 */
#define _POSIX_C_SOURCE 200809L
#if defined(__MINGW32__) && !defined(__USE_MINGW_SETJMP_NON_SEH)
#define __USE_MINGW_SETJMP_NON_SEH
#endif

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

static long fuzz_live;

static void *fuzz_malloc(size_t size) {
    void *allocation = malloc(size);
    fuzz_live += allocation != NULL;
    return allocation;
}

static void *fuzz_calloc(size_t count, size_t size) {
    void *allocation = calloc(count, size);
    fuzz_live += allocation != NULL;
    return allocation;
}

static void *fuzz_realloc(void *allocation, size_t size) {
    void *replacement = realloc(allocation, size);
    fuzz_live += replacement && !allocation;
    return replacement;
}

static void fuzz_free(void *allocation) {
    fuzz_live -= allocation != NULL;
    free(allocation);
}

static void fuzz_exit(int status) {
    (void)status;
    abort();
}

static int fuzz_fprintf(FILE *stream, const char *format, ...) {
    (void)stream;
    (void)format;
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

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    long baseline = fuzz_live;
    JBErrorFrame frame;
    jb_frame_enter(&frame, JB_ERROR_MODEL);
    if (setjmp(frame.jump)) {
        jb_frame_fail(&frame);
    } else {
        unsigned flags = size ? data[0] : 0;
        const uint8_t *file = size ? data + 1 : data;
        size_t file_size = size ? size - 1 : 0;
        uint8_t *structured = NULL;
        if (!(flags & 2) && file_size >= 4) {
            uint32_t requested = (uint32_t)file[0] | (uint32_t)file[1] << 8 |
                                 (uint32_t)file[2] << 16 | (uint32_t)file[3] << 24;
            size_t body_size = file_size - 4;
            size_t header_size = body_size ? requested % (body_size + 1) : 0;
            structured = xmalloc(body_size + 8);
            for (int i = 0; i < 8; i++)
                structured[i] = (uint8_t)((uint64_t)header_size >> (8 * i));
            if (body_size)
                memcpy(structured + 8, file + 4, body_size);
            file = structured;
            file_size = body_size + 8;
        }

        DGModel model = {0};
        DGShard shard = {0};
        model.nvfp4 = !!(flags & 1);
        snprintf(shard.path, sizeof shard.path, "fuzz.safetensors");
        dg_parse_shard_data(&model, &shard, file, file_size);

        for (size_t i = 0; i < model.nt; i++) {
            const uint8_t *begin = file;
            const uint8_t *end = file + file_size;
            if (model.tensor[i].data < begin || model.tensor[i].data > end ||
                model.tensor[i].bytes > (uint64_t)(end - model.tensor[i].data))
                abort();
        }
        jb_release(model.tensor);
        jb_release(shard.names);
        jb_release(structured);
        jb_frame_leave(&frame);
    }
    if (fuzz_live != baseline)
        abort();
    return 0;
}
