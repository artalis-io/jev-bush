#include "../jb.h"

#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    const jb_candidate candidates[] = {
        {{"true", 4}, {"The state concerns billing.", 27}},
        {{"false", 5}, {"The state does not concern billing.", 35}},
    };
    const jb_question question = {
        {"is_billing", 10},
        {"Decide whether this is a billing issue.", 39},
        JB_DECISION_BOOLEAN,
        candidates,
        2,
    };
    const jb_schema schema = {&question, 1};
    const jb_input input = {
        {"example", 7},
        {"\"I was charged twice.\"", 22},
        0,
    };
    jb_model *model = NULL;
    jb_session *session = NULL;
    jb_result *result = NULL;
    jb_status status = jb_model_load(argv[1], &model);
    if (status == JB_OK)
        status = jb_session_create(model, &schema, &session);
    if (status == JB_OK)
        status = jb_session_decide(session, &input, &result);
    if (status != JB_OK) {
        /* A session records its own calls' errors; before one exists, the
         * thread's last error describes the failure. */
        const char *error = session ? jb_session_last_error(session) : jb_last_error();
        fprintf(stderr, "jev-bush: %s: %s\n", jb_status_string(status), error);
        jb_session_free(session);
        jb_model_free(model);
        return 1;
    }
    const jb_answer *answer = &result->answers[0];
    for (size_t i = 0; i < answer->probability_count; i++)
        printf("%.*s %.17g\n", (int)answer->probabilities[i].candidate.length,
               answer->probabilities[i].candidate.data, answer->probabilities[i].probability);
    jb_result_free(result);
    jb_session_free(session);
    jb_model_free(model);
    return 0;
}
