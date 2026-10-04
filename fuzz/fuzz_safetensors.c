/*
 * libFuzzer target for the bounded safetensors metadata reader.
 *
 * Byte zero selects BF16/NVFP4 validation and raw/structured input. Printable
 * seed modes use `B` (BF16), `N` (NVFP4), or `D` (NVFP4 parsed as two shards),
 * followed by a JSON header, a blank line, and tensor bytes. Other inputs keep
 * the compact flag/length format so mutations exercise framing checks too.
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
        int seeded = flags == 'B' || flags == 'N' || flags == 'D';
        int duplicate_shard = flags == 'D';
        if (seeded) {
            size_t header_size = 0;
            while (header_size + 1 < file_size &&
                   !(file[header_size] == '\n' && file[header_size + 1] == '\n'))
                header_size++;
            if (header_size + 1 == file_size)
                header_size = file_size;
            size_t separator = header_size < file_size ? 2 : 0;
            size_t body_size = file_size - header_size - separator;
            structured = xmalloc(file_size - separator + 8);
            for (int i = 0; i < 8; i++)
                structured[i] = (uint8_t)((uint64_t)header_size >> (8 * i));
            memcpy(structured + 8, file, header_size);
            if (body_size)
                memcpy(structured + 8 + header_size, file + header_size + separator, body_size);
            file = structured;
            file_size = file_size - separator + 8;
        } else if (!(flags & 2) && file_size >= 4) {
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
        DGShard shard[2] = {0};
        model.nvfp4 = seeded ? flags != 'B' : !!(flags & 1);
        snprintf(shard[0].path, sizeof shard[0].path, "fuzz-1.safetensors");
        dg_parse_shard_data(&model, &shard[0], file, file_size);
        if (duplicate_shard) {
            snprintf(shard[1].path, sizeof shard[1].path, "fuzz-2.safetensors");
            dg_parse_shard_data(&model, &shard[1], file, file_size);
        }
        dg_finalize_tensor_index(&model);

        for (size_t i = 0; i < model.nt; i++) {
            const uint8_t *begin = file;
            const uint8_t *end = file + file_size;
            if (model.tensor[i].data < begin || model.tensor[i].data > end ||
                model.tensor[i].bytes > (uint64_t)(end - model.tensor[i].data))
                abort();
        }
        jb_release(model.tensor);
        jb_release(shard[0].names);
        jb_release(shard[1].names);
        jb_release(structured);
        jb_frame_leave(&frame);
    }
    if (fuzz_live != baseline)
        abort();
    return 0;
}
