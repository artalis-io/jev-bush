#ifndef JEV_BUSH_H
#define JEV_BUSH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JB_API_VERSION 1u

typedef struct jb_model jb_model;
typedef struct jb_session jb_session;

typedef enum {
    JB_OK = 0,
    JB_ERROR_INVALID_ARGUMENT,
    JB_ERROR_IO,
    JB_ERROR_MODEL,
    JB_ERROR_REQUEST,
    JB_ERROR_OUT_OF_MEMORY,
    JB_ERROR_INTERNAL
} jb_status;

typedef struct {
    /* Strings are counted byte spans and need not be NUL-terminated. */
    const char *data;
    size_t length;
} jb_string;

typedef enum { JB_DECISION_BOOLEAN, JB_DECISION_CHOICE, JB_DECISION_SCORE } jb_decision_type;

typedef struct {
    jb_string id;
    jb_string description;
} jb_candidate;

typedef struct {
    jb_string id;
    jb_string predicate;
    jb_decision_type type;
    const jb_candidate *candidates;
    size_t candidate_count;
} jb_question;

typedef struct {
    const jb_question *questions;
    size_t question_count;
} jb_schema;

typedef struct {
    jb_string id;
    /* A JSON value. Strings must include their JSON quotes. */
    jb_string state_json;
    uint32_t samples;
} jb_input;

typedef struct {
    jb_string candidate;
    double probability;
} jb_probability;

typedef struct {
    jb_string id;
    jb_decision_type type;
    const jb_probability *probabilities;
    size_t probability_count;
    size_t selected_candidate;
    double expected_score;
    double confidence;
} jb_answer;

typedef struct {
    const jb_answer *answers;
    size_t answer_count;
    uint32_t input_tokens;
    uint32_t prefill_tokens;
    double total_ms;
    double prefill_ms;
    double candidate_ms;
} jb_result;

const char *jb_version(void);
const char *jb_status_string(jb_status status);
/* Describes the calling thread's most recent failure. On failure, every
 * function below sets its non-NULL output parameters to NULL or zero. */
const char *jb_last_error(void);

jb_status jb_model_load(const char *model_directory, jb_model **out_model);
/* All sessions referring to a model must be freed before the model. */
void jb_model_free(jb_model *model);

/* A session owns reusable scratch, K/V, and exact schema-prefix state. It is
 * single-threaded; separate sessions may share the same immutable model. */
jb_status jb_session_create(jb_model *model, const jb_schema *schema, jb_session **out_session);
/* Passing NULL with length zero creates a dynamic JSON-only session for full
 * OpenJev requests, which is what the CLI uses. */
jb_status jb_session_create_json(jb_model *model, const char *questions_json,
                                 size_t questions_length, jb_session **out_session);
void jb_session_free(jb_session *session);

jb_status jb_session_decide(jb_session *session, const jb_input *input, jb_result **out_result);
jb_status jb_session_decide_batch(jb_session *session, const jb_input *inputs, size_t input_count,
                                  jb_result ***out_results);

/* JSON calls accept/return the existing complete OpenJev request/result shape.
 * Exact repeated schemas automatically reuse their prefix state. */
jb_status jb_session_decide_json(jb_session *session, const char *request_json,
                                 size_t request_length, char **out_json, size_t *out_length);
jb_status jb_session_decide_json_batch(jb_session *session, const char *const *request_json,
                                       const size_t *request_lengths, size_t request_count,
                                       char ***out_json, size_t **out_lengths);

void jb_result_free(jb_result *result);
void jb_results_free(jb_result **results, size_t count);
void jb_free(void *allocation);

#ifdef __cplusplus
}
#endif

#endif
