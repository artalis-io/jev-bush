#include "../jb.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int expect(jb_status got, jb_status wanted, const char *what) {
    if (got == wanted)
        return 1;
    fprintf(stderr, "%s: got %s, wanted %s\n", what, jb_status_string(got),
            jb_status_string(wanted));
    return 0;
}

int main(void) {
    int ok = 1;
    jb_abi_info abi = {.struct_size = sizeof abi};
    ok &= !strcmp(jb_version(), "0.2.0");
    ok &= jb_api_version() == JB_API_VERSION;
    ok &= expect(jb_get_abi_info(&abi), JB_OK, "ABI info");
    ok &= abi.string_size == sizeof(jb_string) && abi.result_size == sizeof(jb_result);

    jb_model *model = (jb_model *)(uintptr_t)1;
    ok &= expect(jb_model_load(NULL, &model), JB_ERROR_INVALID_ARGUMENT, "NULL model path");
    ok &= model == NULL && *jb_last_error();

    jb_session *session = (jb_session *)(uintptr_t)1;
    ok &= expect(jb_session_create_json(NULL, NULL, 0, &session), JB_ERROR_INVALID_ARGUMENT,
                 "NULL model session");
    ok &= session == NULL;

    char *json = (char *)(uintptr_t)1;
    size_t length = 1;
    ok &= expect(jb_session_decide_json(NULL, "{}", 2, &json, &length), JB_ERROR_INVALID_ARGUMENT,
                 "NULL session decision");
    ok &= json == NULL && length == 0;

    jb_result *result = (jb_result *)(uintptr_t)1;
    jb_input input = {0};
    ok &= expect(jb_session_decide(NULL, &input, &result), JB_ERROR_INVALID_ARGUMENT,
                 "NULL typed session");
    ok &= result == NULL;

    jb_model_retain(NULL);
    jb_model_free(NULL);
    jb_session_free(NULL);
    jb_result_free(NULL);
    jb_results_free(NULL, 0);
    jb_free(NULL);

    if (!ok)
        return 1;
    puts("library API contract: ok");
    return 0;
}
