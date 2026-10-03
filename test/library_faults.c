#include "../jb.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    size_t calls;
    size_t fail_at;
    size_t outstanding;
    size_t logs;
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

static void fault_log(void *context, jb_log_level level, const char *message) {
    Faults *faults = context;
    (void)level;
    if (message && *message)
        faults->logs++;
}

int main(void) {
    jb_abi_info abi = {.struct_size = sizeof abi};
    if (jb_get_abi_info(&abi) != JB_OK || abi.api_version != JB_API_VERSION ||
        abi.result_size != sizeof(jb_result))
        return 1;
    for (size_t fail_at = 0; fail_at < 3; fail_at++) {
        Faults faults = {.fail_at = fail_at};
        jb_model_options options = JB_MODEL_OPTIONS_INIT;
        options.allocator = (jb_allocator){sizeof(jb_allocator), &faults, fault_allocate,
                                           fault_reallocate, fault_release};
        options.log_context = &faults;
        options.log = fault_log;
        jb_model *model = NULL;
        jb_status status = jb_model_load_ex("__jev_bush_missing_model__", &options, &model);
        jb_status expected = fail_at == 0 ? JB_ERROR_OUT_OF_MEMORY : JB_ERROR_IO;
        if (status != expected || model || faults.outstanding || faults.logs != 1) {
            fprintf(stderr, "fault %zu: status=%d outstanding=%zu logs=%zu\n", fail_at, (int)status,
                    faults.outstanding, faults.logs);
            return 1;
        }
    }
    {
        Faults faults = {0};
        jb_model_options options = JB_MODEL_OPTIONS_INIT;
        options.allocator =
            (jb_allocator){sizeof(jb_allocator), &faults, fault_allocate, NULL, fault_release};
        jb_model *model = (jb_model *)(uintptr_t)1;
        if (jb_model_load_ex("unused", &options, &model) != JB_ERROR_INVALID_ARGUMENT || model ||
            faults.calls || faults.outstanding)
            return 1;
    }
    {
        jb_model_options options = JB_MODEL_OPTIONS_INIT;
        options.api_version++;
        jb_model *model = (jb_model *)(uintptr_t)1;
        if (jb_model_load_ex("unused", &options, &model) != JB_ERROR_INVALID_ARGUMENT || model)
            return 1;
    }
    puts("library fault injection: ok");
    return 0;
}
