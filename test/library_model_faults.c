#include "../jb.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    size_t calls;
    size_t fail_at;
    size_t outstanding;
} Faults;

static void *fault_allocate(void *context, size_t size) {
    Faults *faults = context;
    if (faults->calls++ == faults->fail_at)
        return NULL;
    void *allocation = malloc(size);
    if (allocation)
        faults->outstanding++;
    return allocation;
}

static void *fault_reallocate(void *context, void *allocation, size_t size) {
    Faults *faults = context;
    if (faults->calls++ == faults->fail_at)
        return NULL;
    return realloc(allocation, size);
}

static void fault_release(void *context, void *allocation) {
    Faults *faults = context;
    if (allocation) {
        faults->outstanding--;
        free(allocation);
    }
}

static char *read_file(const char *path, size_t *length) {
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END))
        return NULL;
    long end = ftell(file);
    if (end <= 0 || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return NULL;
    }
    char *data = malloc((size_t)end);
    size_t got = data ? fread(data, 1, (size_t)end, file) : 0;
    int close_error = fclose(file);
    if (!data || got != (size_t)end || close_error) {
        free(data);
        return NULL;
    }
    *length = (size_t)end;
    return data;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL REQUEST.json\n", argv[0]);
        return 2;
    }
    size_t request_length = 0;
    char *request = read_file(argv[2], &request_length);
    if (!request)
        return 1;
    Faults faults = {.fail_at = SIZE_MAX};
    jb_model_options options = JB_MODEL_OPTIONS_INIT;
    options.allocator = (jb_allocator){sizeof(jb_allocator), &faults, fault_allocate,
                                       fault_reallocate, fault_release};
    jb_model *model = NULL;
    if (jb_model_load_ex(argv[1], &options, &model) != JB_OK)
        return 1;

    size_t model_allocations = faults.outstanding;
    faults.fail_at = faults.calls;
    jb_session *session = (jb_session *)(uintptr_t)1;
    if (jb_session_create_json(model, NULL, 0, &session) != JB_ERROR_OUT_OF_MEMORY || session ||
        faults.outstanding != model_allocations)
        return 1;

    faults.fail_at = SIZE_MAX;
    if (jb_session_create_json(model, NULL, 0, &session) != JB_OK)
        return 1;
    size_t session_allocations = faults.outstanding;
    faults.fail_at = faults.calls;
    char *output = (char *)(uintptr_t)1;
    size_t output_length = 1;
    if (jb_session_decide_json(session, request, request_length, &output, &output_length) !=
            JB_ERROR_OUT_OF_MEMORY ||
        output || output_length || faults.outstanding != session_allocations)
        return 1;

    /* A failure before workspace execution begins is recoverable. */
    faults.fail_at = SIZE_MAX;
    if (jb_session_decide_json(session, request, request_length, &output, &output_length) !=
            JB_OK ||
        !output || !output_length)
        return 1;
    jb_free(output);
    jb_session_free(session);
    jb_model_free(model);
    free(request);
    if (faults.outstanding)
        return 1;
    puts("model-backed allocation failure recovery: ok");
    return 0;
}
