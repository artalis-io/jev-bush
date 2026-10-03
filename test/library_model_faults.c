#include "../jb.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t calls;
    size_t fail_at;
    size_t outstanding;
    size_t largest_size[3];
    size_t largest_call[3];
} Faults;

static void fault_observe(Faults *faults, size_t call, size_t size) {
    for (size_t i = 0; i < 3; i++)
        if (size > faults->largest_size[i]) {
            for (size_t j = 2; j > i; j--) {
                faults->largest_size[j] = faults->largest_size[j - 1];
                faults->largest_call[j] = faults->largest_call[j - 1];
            }
            faults->largest_size[i] = size;
            faults->largest_call[i] = call;
            break;
        }
}

static void fault_reset_observations(Faults *faults) {
    memset(faults->largest_size, 0, sizeof faults->largest_size);
    memset(faults->largest_call, 0, sizeof faults->largest_call);
}

static void *fault_allocate(void *context, size_t size) {
    Faults *faults = context;
    size_t call = faults->calls++;
    fault_observe(faults, call, size);
    if (call == faults->fail_at)
        return NULL;
    void *allocation = malloc(size);
    if (allocation)
        faults->outstanding++;
    return allocation;
}

static void *fault_reallocate(void *context, void *allocation, size_t size) {
    Faults *faults = context;
    size_t call = faults->calls++;
    fault_observe(faults, call, size);
    if (call == faults->fail_at)
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

static jb_model_options fault_options(Faults *faults) {
    jb_model_options options = JB_MODEL_OPTIONS_INIT;
    options.allocator = (jb_allocator){sizeof(jb_allocator), faults, fault_allocate,
                                       fault_reallocate, fault_release};
    return options;
}

static long model_mapping_count(const char *path) {
#if defined(__linux__)
    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps)
        return -1;
    long count = 0;
    char line[4096];
    while (fgets(line, sizeof line, maps))
        count += strstr(line, path) != NULL;
    fclose(maps);
    return count;
#else
    (void)path;
    return -1;
#endif
}

static int check_load_fault(const char *path, size_t fail_at) {
    long mappings = model_mapping_count(path);
    Faults faults = {.fail_at = fail_at};
    jb_model_options options = fault_options(&faults);
    jb_model *model = (jb_model *)(uintptr_t)1;
    jb_status status = jb_model_load_ex(path, &options, &model);
    if (model)
        jb_model_free(model);
    long mappings_after = model_mapping_count(path);
    return status == JB_ERROR_OUT_OF_MEMORY && !model && !faults.outstanding &&
           (mappings < 0 || mappings_after == mappings);
}

static int check_decision_fault(jb_model *model, Faults *faults, size_t model_allocations,
                                const char *request, size_t request_length, size_t offset) {
    faults->fail_at = SIZE_MAX;
    jb_session *session = NULL;
    if (jb_session_create_json(model, NULL, 0, &session) != JB_OK)
        return 0;
    faults->fail_at = faults->calls + offset;
    char *output = (char *)(uintptr_t)1;
    size_t output_length = 1;
    jb_status status =
        jb_session_decide_json(session, request, request_length, &output, &output_length);
    if (output)
        jb_free(output);
    jb_session_free(session);
    return status == JB_ERROR_OUT_OF_MEMORY && !output && !output_length &&
           faults->outstanding == model_allocations;
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
    jb_model_options options = fault_options(&faults);
    jb_model *model = NULL;
    if (jb_model_load_ex(argv[1], &options, &model) != JB_OK)
        return 1;

    size_t load_calls = faults.calls;
    size_t model_allocations = faults.outstanding;
    faults.fail_at = faults.calls;
    jb_session *session = (jb_session *)(uintptr_t)1;
    if (jb_session_create_json(model, NULL, 0, &session) != JB_ERROR_OUT_OF_MEMORY || session ||
        faults.outstanding != model_allocations)
        return 1;

    faults.fail_at = SIZE_MAX;
    if (jb_session_create_json(model, NULL, 0, &session) != JB_OK)
        return 1;
    size_t session_calls = faults.calls;
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
    size_t decision_start = faults.calls;
    fault_reset_observations(&faults);
    if (jb_session_decide_json(session, request, request_length, &output, &output_length) !=
            JB_OK ||
        !output || !output_length)
        return 1;
    size_t decision_calls = faults.calls;
    jb_free(output);
    jb_session_free(session);

    size_t decision_allocations = decision_calls - decision_start;
    const size_t decision_faults[] = {0, faults.largest_call[0] - decision_start,
                                      faults.largest_call[1] - decision_start,
                                      decision_allocations - 1};
    for (size_t i = 0; i < sizeof decision_faults / sizeof *decision_faults; i++)
        if (!check_decision_fault(model, &faults, model_allocations, request, request_length,
                                  decision_faults[i]))
            return 1;

    jb_model_free(model);
    free(request);
    if (faults.outstanding)
        return 1;

    const size_t load_faults[] = {0, load_calls / 2, load_calls - 1};
    for (size_t i = 0; i < sizeof load_faults / sizeof *load_faults; i++)
        if (!check_load_fault(argv[1], load_faults[i]))
            return 1;
    printf("model-backed allocation failure recovery: ok "
           "(load=%zu session=%zu decision=%zu total=%zu allocator calls)\n",
           load_calls, session_calls - load_calls, decision_calls - session_calls, faults.calls);
    return 0;
}
