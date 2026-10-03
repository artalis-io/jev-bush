#ifndef JEV_BUSH_H
#define JEV_BUSH_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(JB_SHARED)
#if defined(JB_BUILD_SHARED)
#define JB_API __declspec(dllexport)
#else
#define JB_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) && defined(JB_SHARED)
#define JB_API __attribute__((visibility("default")))
#else
#define JB_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define JB_API_VERSION 2u

typedef struct jb_model jb_model;
typedef struct jb_session jb_session;

/* Enumerator values are part of the ABI and never change. */
typedef enum {
    JB_OK = 0,
    JB_ERROR_INVALID_ARGUMENT = 1,
    JB_ERROR_IO = 2,
    JB_ERROR_MODEL = 3,
    JB_ERROR_REQUEST = 4,
    JB_ERROR_OUT_OF_MEMORY = 5,
    JB_ERROR_INTERNAL = 6
} jb_status;

typedef enum { JB_LOG_ERROR = 0, JB_LOG_WARNING = 1, JB_LOG_INFO = 2 } jb_log_level;

typedef void *(*jb_allocate_fn)(void *context, size_t size);
typedef void *(*jb_reallocate_fn)(void *context, void *allocation, size_t size);
typedef void (*jb_release_fn)(void *context, void *allocation);
/* Log callbacks may call back into Jev Bush, including releasing the object
 * whose operation produced the message. The message is borrowed and valid
 * only for the callback. */
typedef void (*jb_log_fn)(void *context, jb_log_level level, const char *message);

/* Allocator callbacks follow malloc/realloc/free semantics, including
 * alignment suitable for every C type. They may be called concurrently by
 * separate sessions and must remain valid until the last model/session/result
 * allocation using them is released. */
typedef struct {
    size_t struct_size;
    void *context;
    jb_allocate_fn allocate;
    jb_reallocate_fn reallocate;
    jb_release_fn release;
} jb_allocator;

typedef struct {
    size_t struct_size;
    uint32_t api_version;
    jb_allocator allocator;
    void *log_context;
    jb_log_fn log;
} jb_model_options;

#define JB_MODEL_OPTIONS_INIT                                                                      \
    {                                                                                              \
        sizeof(jb_model_options), JB_API_VERSION, {sizeof(jb_allocator), NULL, NULL, NULL, NULL},  \
            NULL, NULL                                                                             \
    }

/* Existing public value layouts are permanent. Future API revisions add new
 * types or entry points instead of changing these sizes. */
typedef struct {
    size_t struct_size;
    uint32_t api_version;
    size_t string_size;
    size_t candidate_size;
    size_t question_size;
    size_t schema_size;
    size_t input_size;
    size_t probability_size;
    size_t answer_size;
    size_t result_size;
} jb_abi_info;

typedef struct {
    /* Strings are counted byte spans and need not be NUL-terminated. */
    const char *data;
    size_t length;
} jb_string;

typedef enum {
    JB_DECISION_BOOLEAN = 0,
    JB_DECISION_CHOICE = 1,
    JB_DECISION_SCORE = 2
} jb_decision_type;

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

JB_API const char *jb_version(void);
JB_API uint32_t jb_api_version(void);
JB_API jb_status jb_get_abi_info(jb_abi_info *info);
JB_API const char *jb_status_string(jb_status status);
/* Describes the calling thread's most recent failure, including calls that
 * have no session: jb_model_load and session creation. On failure, every
 * function below sets its non-NULL output parameters to NULL or zero. */
JB_API const char *jb_last_error(void);

/* A model is immutable once loaded and may be shared by sessions on any
 * threads. Each successful session creation retains it. Call retain when
 * keeping another direct reference; every direct reference is released by
 * exactly one model_free. */
JB_API jb_status jb_model_load(const char *model_directory, jb_model **out_model);
JB_API jb_status jb_model_load_ex(const char *model_directory, const jb_model_options *options,
                                  jb_model **out_model);
JB_API void jb_model_retain(jb_model *model);
JB_API void jb_model_free(jb_model *model);

/* A session owns reusable scratch, K/V, and exact schema-prefix state. It is
 * not thread-safe: one call at a time per session. Separate sessions may
 * share the same immutable model. Invalid requests leave a session reusable.
 * A failure after inference workspace execution begins poisons it: later
 * decisions return JB_ERROR_INTERNAL, and session_free remains valid. */
JB_API jb_status jb_session_create(jb_model *model, const jb_schema *schema,
                                   jb_session **out_session);
/* Passing NULL with length zero creates a dynamic JSON-only session for full
 * OpenJev requests, which is what the CLI uses. */
JB_API jb_status jb_session_create_json(jb_model *model, const char *questions_json,
                                        size_t questions_length, jb_session **out_session);
JB_API void jb_session_free(jb_session *session);
/* Describes the most recent decide call on this session: set when it fails
 * and cleared when it succeeds. Unlike jb_last_error it does not depend on
 * the calling thread, so it stays correct when a session moves between
 * threads. The string is valid until the session's next call or its free. */
JB_API const char *jb_session_last_error(const jb_session *session);

/* A result is released with jb_result_free; a batch of them, the array
 * included, with jb_results_free. Batch calls take 1 to 16 inputs. */
JB_API jb_status jb_session_decide(jb_session *session, const jb_input *input,
                                   jb_result **out_result);
JB_API jb_status jb_session_decide_batch(jb_session *session, const jb_input *inputs,
                                         size_t input_count, jb_result ***out_results);

/* JSON calls accept/return the existing complete OpenJev request/result shape.
 * Exact repeated schemas automatically reuse their prefix state. The output
 * is released with jb_free: one buffer for a single call; each string, then
 * the string array and the length array, for a batch of 1 to 16 requests. */
JB_API jb_status jb_session_decide_json(jb_session *session, const char *request_json,
                                        size_t request_length, char **out_json, size_t *out_length);
JB_API jb_status jb_session_decide_json_batch(jb_session *session, const char *const *request_json,
                                              const size_t *request_lengths, size_t request_count,
                                              char ***out_json, size_t **out_lengths);

JB_API void jb_result_free(jb_result *result);
JB_API void jb_results_free(jb_result **results, size_t count);
/* Releases a buffer or array a JSON call returned; NULL is ignored. */
JB_API void jb_free(void *allocation);

#ifdef __cplusplus
}
#endif

#endif
