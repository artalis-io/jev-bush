struct jb_model {
    DGModel model;
    DGTokenizer tokenizer;
    atomic_uint references;
    jb_allocator allocator;
    void *log_context;
    jb_log_fn log;
};

struct jb_session {
    jb_model *model;
    atomic_flag busy;
    int poisoned;
    DGPrefixCache prefix;
    DGWorkspace workspace;
    char *questions_json;
    size_t questions_length;
    /* The most recent call's error, readable from any thread. */
    char error[sizeof jb_error_message];
};

static void jb_log_message(const jb_model *model, jb_log_level level, const char *message) {
    if (model && model->log)
        model->log(model->log_context, level, message);
}

const char *jb_version(void) {
    return JB_VERSION;
}

static jb_status jb_invalid(const char *message);

uint32_t jb_api_version(void) {
    return JB_API_VERSION;
}

jb_status jb_get_abi_info(jb_abi_info *info) {
    if (!info || info->struct_size < sizeof *info)
        return jb_invalid("ABI info structure is missing or too small");
    *info = (jb_abi_info){sizeof *info,         JB_API_VERSION,         sizeof(jb_string),
                          sizeof(jb_candidate), sizeof(jb_question),    sizeof(jb_schema),
                          sizeof(jb_input),     sizeof(jb_probability), sizeof(jb_answer),
                          sizeof(jb_result),    sizeof(jb_model_info)};
    jb_error_message[0] = 0;
    return JB_OK;
}

const char *jb_status_string(jb_status status) {
    static const char *const text[] = {"ok",
                                       "invalid argument",
                                       "I/O error",
                                       "model error",
                                       "request error",
                                       "out of memory",
                                       "internal error"};
    return (unsigned)status < sizeof text / sizeof *text ? text[status] : "unknown error";
}

const char *jb_last_error(void) {
    return jb_error_message;
}

static jb_status jb_invalid(const char *message) {
    snprintf(jb_error_message, sizeof jb_error_message, "%s", message);
    return JB_ERROR_INVALID_ARGUMENT;
}

static int jb_allocator_valid(const jb_allocator *allocator) {
    return allocator->struct_size >= sizeof *allocator && allocator->allocate &&
           allocator->reallocate && allocator->release;
}

static jb_status jb_model_load_impl(const char *model_directory, jb_model **out_model,
                                    jb_allocator allocator, void *log_context, jb_log_fn log) {
    const jb_allocator *previous_allocator = jb_active_allocator;
    jb_active_allocator = &allocator;
    jb_model *volatile model = NULL;
    JBErrorFrame frame;
    jb_frame_enter(&frame, JB_ERROR_MODEL);
    if (setjmp(frame.jump)) {
        /* Memory is released with the frame; mapped shards and accelerator
         * memory are not host memory. */
        if (model) {
            dg_release_accel(&model->model);
            dg_unmap_shards(&model->model);
        }
        jb_status status = jb_frame_fail(&frame);
        jb_active_allocator = previous_allocator;
        if (log)
            log(log_context, JB_LOG_ERROR, jb_error_message);
        return status;
    }
    model = xcalloc(1, sizeof *model);
    model->allocator = allocator;
    model->log_context = log_context;
    model->log = log;
    atomic_init(&model->references, 1);
    dg_load(&model->model, model_directory);
    dgt_load(&model->tokenizer, model_directory);
    dg_upload_model(&model->model);
    dg_prepare_exact_model(&model->model);
    jb_frame_leave(&frame);
    jb_active_allocator = previous_allocator;
    jb_error_message[0] = 0;
    jb_log_message(model, JB_LOG_INFO, "model loaded");
    *out_model = model;
    return JB_OK;
}

jb_status jb_model_load_ex(const char *model_directory, const jb_model_options *options,
                           jb_model **out_model) {
    if (out_model)
        *out_model = NULL;
    if (!model_directory || !out_model)
        return jb_invalid("model directory and output pointer are required");
    jb_allocator allocator = jb_system_allocator;
    void *log_context = NULL;
    jb_log_fn log = NULL;
    if (options) {
        if (options->struct_size < sizeof *options || options->api_version != JB_API_VERSION)
            return jb_invalid("model options version or size is incompatible");
        if (options->allocator.allocate || options->allocator.reallocate ||
            options->allocator.release) {
            if (!jb_allocator_valid(&options->allocator))
                return jb_invalid("custom allocator is incomplete");
            allocator = options->allocator;
        }
        log_context = options->log_context;
        log = options->log;
    }
    return jb_model_load_impl(model_directory, out_model, allocator, log_context, log);
}

jb_status jb_model_load(const char *model_directory, jb_model **out_model) {
    return jb_model_load_ex(model_directory, NULL, out_model);
}

jb_status jb_model_get_info(const jb_model *model, jb_model_info *info) {
    if (!info || info->struct_size < sizeof *info)
        return jb_invalid("model and a complete model-info structure are required");
    if (!model) {
        memset(info, 0, sizeof *info);
        return jb_invalid("model and a complete model-info structure are required");
    }
    *info = (jb_model_info){.struct_size = sizeof *info,
                            .api_version = JB_API_VERSION,
                            .model_family = "DiffusionGemma",
                            .cpu_backend = dg_kernels()->name,
                            .accelerator = model->model.accel ? model->model.accel->name : "none",
                            .strict_math = JB_STRICT_MATH,
                            .max_batch_inputs = JB_MAX_BATCH_INPUTS,
                            .max_prompt_tokens = JB_MAX_CTX,
                            .max_canvas_tokens = DG_CANVAS_TOKENS};
    jb_error_message[0] = 0;
    return JB_OK;
}

void jb_model_retain(jb_model *model) {
    if (!model)
        return;
    unsigned references = atomic_load_explicit(&model->references, memory_order_relaxed);
    /* Saturation deliberately leaks a pathologically over-retained model
     * instead of allowing wraparound and premature destruction. */
    while (references != UINT_MAX &&
           !atomic_compare_exchange_weak_explicit(&model->references, &references, references + 1,
                                                  memory_order_relaxed, memory_order_relaxed)) {
    }
}

static void jb_model_release(jb_model *model) {
    if (!model)
        return;
    if (atomic_load_explicit(&model->references, memory_order_relaxed) == UINT_MAX)
        return;
    if (atomic_fetch_sub_explicit(&model->references, 1, memory_order_acq_rel) != 1)
        return;
    dgt_free(&model->tokenizer);
    dg_free(&model->model);
    jb_release(model);
}

void jb_model_free(jb_model *model) {
    jb_model_release(model);
}

/* Copy a caller's counted string, rejecting what would otherwise be read
 * through NULL or cut short at an embedded NUL by the JSON writer. */
static char *jb_counted_string(jb_string s, const char *what, int required) {
    if (s.length && !s.data)
        die2("typed string has a NULL pointer", what);
    if (required && !s.length)
        die2("typed string is required", what);
    if (s.length > JB_MAX_JSON)
        die2("typed string is too large", what);
    if (s.length && memchr(s.data, 0, s.length))
        die2("typed string contains NUL", what);
    char *z = xmalloc(s.length + 1);
    if (s.length)
        memcpy(z, s.data, s.length);
    z[s.length] = 0;
    return z;
}

static char *jb_schema_json(const jb_schema *schema, size_t *length) {
    if (!schema || !schema->questions || !schema->question_count ||
        schema->question_count > DG_MAX_QUESTIONS)
        return NULL;
    DGBuf out = {0};
    db_ch(&out, '{');
    for (size_t q = 0; q < schema->question_count; q++) {
        const jb_question *question = &schema->questions[q];
        if (question->type != JB_DECISION_BOOLEAN && question->type != JB_DECISION_CHOICE &&
            question->type != JB_DECISION_SCORE)
            die("unknown typed decision type");
        if (question->candidate_count && !question->candidates)
            die("typed question has a NULL candidate array");
        if (q)
            db_ch(&out, ',');
        char *id = jb_counted_string(question->id, "question id", 1);
        char *predicate = jb_counted_string(question->predicate, "question predicate", 1);
        db_json_string(&out, id, 0);
        db_mem(&out, ":{\"type\":", 9);
        const char *type = question->type == JB_DECISION_BOOLEAN  ? "noul"
                           : question->type == JB_DECISION_CHOICE ? "choice"
                                                                  : "score";
        db_json_string(&out, type, 0);
        db_mem(&out, ",\"instructions\":", 16);
        db_json_string(&out, predicate, 0);
        db_mem(&out, ",\"criteria\":", 12);
        jb_release(predicate);
        jb_release(id);
        if (question->type == JB_DECISION_BOOLEAN) {
            if (question->candidate_count != 2)
                die("boolean question needs two candidates");
            db_ch(&out, '{');
        } else
            db_ch(&out, question->type == JB_DECISION_CHOICE ? '{' : '[');
        if (question->candidate_count < 2 || question->candidate_count > JB_MAX_CAND)
            die("typed question candidate count out of range");
        for (size_t c = 0; c < question->candidate_count; c++) {
            const jb_candidate *candidate = &question->candidates[c];
            if (c)
                db_ch(&out, ',');
            if (question->type == JB_DECISION_BOOLEAN) {
                /* Boolean candidates are fixed; an id, if given, must match. */
                const char *fixed = c ? "false" : "true";
                char *given = jb_counted_string(candidate->id, "boolean candidate id", 0);
                int mismatch = *given && strcmp(given, fixed);
                jb_release(given);
                if (mismatch)
                    die("boolean candidates are \"true\" then \"false\"");
                db_json_string(&out, fixed, 0);
                db_ch(&out, ':');
            } else if (question->type == JB_DECISION_CHOICE) {
                char *z = jb_counted_string(candidate->id, "candidate id", 1);
                db_json_string(&out, z, 0);
                db_ch(&out, ':');
                jb_release(z);
            }
            char *description =
                jb_counted_string(candidate->description, "candidate description", 0);
            db_json_string(&out, description, 0);
            jb_release(description);
        }
        db_ch(&out, question->type == JB_DECISION_SCORE ? ']' : '}');
        db_ch(&out, '}');
    }
    db_ch(&out, '}');
    /* Validate with the request parser, so duplicate question or candidate
     * ids are rejected here rather than when the session parses them. */
    int nt = 0;
    jb_release(json_tokens(out.p, out.n, &nt));
    *length = out.n;
    return out.p;
}

jb_status jb_session_create_json(jb_model *model, const char *questions_json,
                                 size_t questions_length, jb_session **out_session) {
    if (out_session)
        *out_session = NULL;
    if (!model || !out_session || (!questions_json && questions_length))
        return jb_invalid("model, schema, and output pointer are required");
    if (questions_length > JB_MAX_JSON)
        return jb_invalid("schema JSON is too large");
    const jb_allocator *previous_allocator = jb_active_allocator;
    jb_active_allocator = &model->allocator;
    JBErrorFrame frame;
    jb_frame_enter(&frame, JB_ERROR_REQUEST);
    if (setjmp(frame.jump)) {
        jb_status status = jb_frame_fail(&frame);
        jb_active_allocator = previous_allocator;
        return status;
    }
    jb_session *session = xcalloc(1, sizeof *session);
    session->model = model;
    atomic_flag_clear(&session->busy);
    if (questions_json) {
        int nt = 0;
        JTok *tokens = json_tokens(questions_json, questions_length, &nt);
        if (!nt || tokens[0].type != JT_OBJECT)
            die("session schema must be a JSON object");
        jb_release(tokens);
        session->questions_json = xmalloc(questions_length + 1);
        memcpy(session->questions_json, questions_json, questions_length);
        session->questions_json[questions_length] = 0;
        session->questions_length = questions_length;
    }
    jb_model_retain(model);
    jb_frame_leave(&frame);
    *out_session = session;
    jb_active_allocator = previous_allocator;
    jb_error_message[0] = 0;
    return JB_OK;
}

jb_status jb_session_create(jb_model *model, const jb_schema *schema, jb_session **out_session) {
    if (out_session)
        *out_session = NULL;
    if (!model || !schema || !out_session)
        return jb_invalid("model, typed schema, and output pointer are required");
    const jb_allocator *previous_allocator = jb_active_allocator;
    jb_active_allocator = &model->allocator;
    JBErrorFrame frame;
    jb_frame_enter(&frame, JB_ERROR_REQUEST);
    if (setjmp(frame.jump)) {
        jb_status status = jb_frame_fail(&frame);
        jb_active_allocator = previous_allocator;
        return status;
    }
    size_t length = 0;
    char *json = jb_schema_json(schema, &length);
    if (!json) {
        jb_frame_abandon(&frame);
        jb_active_allocator = previous_allocator;
        return jb_invalid("typed schema is empty or invalid");
    }
    jb_status status = jb_session_create_json(model, json, length, out_session);
    jb_release(json);
    if (status != JB_OK) {
        jb_frame_abandon(&frame);
        jb_active_allocator = previous_allocator;
        return status;
    }
    jb_frame_leave(&frame);
    jb_active_allocator = previous_allocator;
    return JB_OK;
}

void jb_session_free(jb_session *session) {
    if (!session)
        return;
    dg_workspace_destroy(&session->workspace);
    dg_prefix_free(&session->prefix);
    jb_release(session->questions_json);
    jb_model *model = session->model;
    jb_release(session);
    jb_model_release(model);
}

static jb_status jb_session_decide_json_call(jb_session *session, const char *request_json,
                                             size_t request_length, char **out_json,
                                             size_t *out_length) {
    if (out_json)
        *out_json = NULL;
    if (out_length)
        *out_length = 0;
    if (!session || !request_json || !request_length || !out_json || !out_length)
        return jb_invalid("session, request, and output pointers are required");
    if (request_length > JB_MAX_JSON)
        return jb_invalid("request JSON is too large");
    const jb_allocator *previous_allocator = jb_active_allocator;
    jb_active_allocator = &session->model->allocator;
    volatile int accel_locked = 0;
    JBErrorFrame frame;
    jb_frame_enter(&frame, JB_ERROR_REQUEST);
    if (setjmp(frame.jump)) {
#ifdef JB_PROFILE
        jb_profile_active = NULL;
#endif
        if (session->workspace.active || session->workspace.kv_active ||
            session->workspace.control_active)
            session->poisoned = 1;
        session->workspace.active = 0;
        session->workspace.kv_active = 0;
        session->workspace.control_active = 0;
        jb_accel_call_unlock(accel_locked);
        jb_status status = jb_frame_fail(&frame);
        jb_active_allocator = previous_allocator;
        return status;
    }
    accel_locked = jb_accel_call_lock(&session->model->model);
    dg_systemone(&session->model->model, &session->model->tokenizer, request_json, request_length,
                 &session->prefix, &session->workspace, out_json, out_length);
    jb_accel_call_unlock(accel_locked);
    jb_frame_leave(&frame);
    jb_active_allocator = previous_allocator;
    jb_error_message[0] = 0;
    return JB_OK;
}

static jb_status jb_session_decide_json_batch_call(jb_session *session,
                                                   const char *const *request_json,
                                                   const size_t *request_lengths,
                                                   size_t request_count, char ***out_json,
                                                   size_t **out_lengths) {
    if (out_json)
        *out_json = NULL;
    if (out_lengths)
        *out_lengths = NULL;
    if (!session || !request_json || !request_lengths || !request_count ||
        request_count > JB_MAX_BATCH_INPUTS || !out_json || !out_lengths)
        return jb_invalid("batch arguments are invalid");
    const jb_allocator *previous_allocator = jb_active_allocator;
    jb_active_allocator = &session->model->allocator;
    volatile int accel_locked = 0;
    JBErrorFrame frame;
    jb_frame_enter(&frame, JB_ERROR_REQUEST);
    if (setjmp(frame.jump)) {
#ifdef JB_PROFILE
        jb_profile_active = NULL;
#endif
        if (session->workspace.active || session->workspace.kv_active ||
            session->workspace.control_active)
            session->poisoned = 1;
        session->workspace.active = 0;
        session->workspace.kv_active = 0;
        session->workspace.control_active = 0;
        jb_accel_call_unlock(accel_locked);
        jb_status status = jb_frame_fail(&frame);
        jb_active_allocator = previous_allocator;
        return status;
    }
    accel_locked = jb_accel_call_lock(&session->model->model);
    char **output = xcalloc(request_count, sizeof *output);
    size_t *length = xcalloc(request_count, sizeof *length);
    for (size_t i = 0; i < request_count; i++) {
        if (!request_json[i] || !request_lengths[i])
            die("batch request is empty");
        if (request_lengths[i] > JB_MAX_JSON)
            die("batch request JSON is too large");
    }
    int batched =
        request_count > 1 && dg_system_batch(&session->model->model, &session->model->tokenizer,
                                             request_json, request_lengths, (int)request_count,
                                             &session->prefix, &session->workspace, output, length);
    if (!batched)
        for (size_t i = 0; i < request_count; i++)
            dg_systemone(&session->model->model, &session->model->tokenizer, request_json[i],
                         request_lengths[i], &session->prefix, &session->workspace, &output[i],
                         &length[i]);
    jb_accel_call_unlock(accel_locked);
    jb_frame_leave(&frame);
    jb_active_allocator = previous_allocator;
    *out_json = output;
    *out_lengths = length;
    jb_error_message[0] = 0;
    return JB_OK;
}

void jb_free(void *allocation) {
    jb_release(allocation);
}

/* Parse the JSON number grammar directly, independent of the process locale. */
static double jb_token_double(const char *json, const JTok *token) {
    if (token->type != JT_PRIMITIVE || token->end <= token->start || token->end - token->start > 32)
        die("invalid numeric result field");
    char text[33], *end = NULL;
    size_t length = (size_t)(token->end - token->start);
    memcpy(text, json + token->start, length);
    text[length] = 0;
#if defined(_WIN32)
    _locale_t locale = jb_numeric_locale();
    double value = _strtod_l(text, &end, locale);
#else
    locale_t locale = jb_numeric_locale();
    locale_t previous = uselocale(locale);
    double value = strtod(text, &end);
    uselocale(previous);
#endif
    if (end != text + length || !jb_finite(value))
        die("invalid numeric result field");
    return value;
}

static uint32_t jb_token_u32(const char *json, const JTok *token) {
    double value = jb_token_double(json, token);
    if (value < 0 || value > UINT32_MAX || value != floor(value))
        die("invalid integer result field");
    return (uint32_t)value;
}

static jb_string jb_result_string(const char *json, const JTok *token, char **cursor) {
    char *decoded = jt_string(json, token);
    size_t length = strlen(decoded);
    memcpy(*cursor, decoded, length + 1);
    jb_string result = {*cursor, length};
    *cursor += length + 1;
    jb_release(decoded);
    return result;
}

static jb_result *jb_result_parse(const char *json, size_t length) {
    int nt = 0;
    JTok *tokens = json_tokens(json, length, &nt);
    int answers_at = jt_obj_get(json, tokens, nt, 0, "answers");
    if (!nt || tokens[0].type != JT_OBJECT || answers_at < 0 ||
        tokens[answers_at].type != JT_OBJECT)
        die("invalid Jev Bush result");
    int answer_keys[DG_MAX_QUESTIONS];
    int answer_count = direct_keys(tokens, nt, answers_at, answer_keys, DG_MAX_QUESTIONS);
    size_t probability_count = 0, string_bytes = 0;
    for (int a = 0; a < answer_count; a++) {
        int object = answer_keys[a] + 1;
        int probabilities = jt_obj_get(json, tokens, nt, object, "probabilities");
        if (probabilities < 0 || tokens[probabilities].type != JT_OBJECT)
            die("result answer lacks probabilities");
        string_bytes = jb_size_add(
            string_bytes, (size_t)(tokens[answer_keys[a]].end - tokens[answer_keys[a]].start) + 1);
        for (int i = probabilities + 1; i < nt && tokens[i].parent >= probabilities; i++)
            if (tokens[i].parent == probabilities && tokens[i].type == JT_STRING) {
                probability_count++;
                string_bytes =
                    jb_size_add(string_bytes, (size_t)(tokens[i].end - tokens[i].start) + 1);
                i++;
            }
    }
    size_t bytes = sizeof(jb_result);
    bytes = jb_align_up(bytes, _Alignof(jb_answer));
    size_t answers_offset = bytes;
    bytes = jb_size_add(bytes, jb_size_mul((size_t)answer_count, sizeof(jb_answer)));
    bytes = jb_align_up(bytes, _Alignof(jb_probability));
    size_t probabilities_offset = bytes;
    bytes = jb_size_add(bytes, jb_size_mul(probability_count, sizeof(jb_probability)));
    size_t strings_offset = bytes;
    bytes = jb_size_add(bytes, string_bytes);
    jb_result *result = xcalloc(bytes, 1);
    jb_answer *answers = (jb_answer *)((unsigned char *)result + answers_offset);
    jb_probability *probability =
        (jb_probability *)((unsigned char *)result + probabilities_offset);
    char *strings = (char *)result + strings_offset;
    result->answers = answers;
    result->answer_count = (size_t)answer_count;
    for (int a = 0; a < answer_count; a++) {
        int object = answer_keys[a] + 1;
        jb_answer *answer = &answers[a];
        answer->id = jb_result_string(json, &tokens[answer_keys[a]], &strings);
        int type = jt_obj_get(json, tokens, nt, object, "type");
        char *type_name = type >= 0 ? jt_string(json, &tokens[type]) : NULL;
        if (!type_name)
            die("result answer lacks type");
        answer->type = !strcmp(type_name, "noul")     ? JB_DECISION_BOOLEAN
                       : !strcmp(type_name, "choice") ? JB_DECISION_CHOICE
                                                      : JB_DECISION_SCORE;
        jb_release(type_name);
        int probabilities = jt_obj_get(json, tokens, nt, object, "probabilities");
        answer->probabilities = probability;
        size_t count = 0, selected = 0;
        double best = -1, expected = 0;
        for (int i = probabilities + 1; i < nt && tokens[i].parent >= probabilities; i++)
            if (tokens[i].parent == probabilities && tokens[i].type == JT_STRING) {
                probability[count].candidate = jb_result_string(json, &tokens[i], &strings);
                probability[count].probability = jb_token_double(json, &tokens[i + 1]);
                expected += count * probability[count].probability;
                if (probability[count].probability > best) {
                    best = probability[count].probability;
                    selected = count;
                }
                count++;
                i++;
            }
        answer->probability_count = count;
        answer->selected_candidate = selected;
        answer->expected_score = expected;
        int confidence_at = jt_obj_get(json, tokens, nt, object, "confidence");
        if (confidence_at >= 0)
            answer->confidence = jb_token_double(json, &tokens[confidence_at]);
        probability += count;
    }
    int usage = jt_obj_get(json, tokens, nt, 0, "usage");
    int timing = jt_obj_get(json, tokens, nt, 0, "timing_ms");
    if (usage >= 0) {
        int input = jt_obj_get(json, tokens, nt, usage, "input_tokens");
        int prefill = jt_obj_get(json, tokens, nt, usage, "prefill_tokens");
        if (input >= 0)
            result->input_tokens = jb_token_u32(json, &tokens[input]);
        if (prefill >= 0)
            result->prefill_tokens = jb_token_u32(json, &tokens[prefill]);
    }
    if (timing >= 0) {
        int total = jt_obj_get(json, tokens, nt, timing, "total");
        int prefill = jt_obj_get(json, tokens, nt, timing, "prefill");
        int candidates = jt_obj_get(json, tokens, nt, timing, "candidates");
        if (total >= 0)
            result->total_ms = jb_token_double(json, &tokens[total]);
        if (prefill >= 0)
            result->prefill_ms = jb_token_double(json, &tokens[prefill]);
        if (candidates >= 0)
            result->candidate_ms = jb_token_double(json, &tokens[candidates]);
    }
    jb_release(tokens);
    return result;
}

static char *jb_typed_request(const jb_session *session, const jb_input *input, size_t *length) {
    if (!session->questions_json)
        die("typed inference requires a typed or JSON session schema");
    if (!input || !input->state_json.data || !input->state_json.length)
        die("typed input needs state_json");
    if (input->state_json.length > JB_MAX_JSON || input->id.length > JB_MAX_JSON)
        die("typed input is too large");
    int nt = 0;
    JTok *state = json_tokens(input->state_json.data, input->state_json.length, &nt);
    if (!nt || state[0].parent != -1)
        die("state_json must contain exactly one JSON value");
    jb_release(state);
    DGBuf request = {0};
    db_ch(&request, '{');
    if (input->id.length) {
        char *id = jb_counted_string(input->id, "input id", 0);
        db_mem(&request, "\"id\":", 5);
        db_json_string(&request, id, 0);
        db_ch(&request, ',');
        jb_release(id);
    }
    db_mem(&request, "\"state\":", 8);
    db_mem(&request, input->state_json.data, input->state_json.length);
    db_mem(&request, ",\"questions\":", 13);
    db_mem(&request, session->questions_json, session->questions_length);
    if (input->samples)
        db_fmt(&request, ",\"samples\":%u", input->samples);
    db_ch(&request, '}');
    *length = request.n;
    return request.p;
}

static jb_status jb_session_decide_call(jb_session *session, const jb_input *input,
                                        jb_result **out_result) {
    if (out_result)
        *out_result = NULL;
    if (!session || !input || !out_result)
        return jb_invalid("session, input, and output pointer are required");
    const jb_allocator *previous_allocator = jb_active_allocator;
    jb_active_allocator = &session->model->allocator;
    JBErrorFrame frame;
    jb_frame_enter(&frame, JB_ERROR_REQUEST);
    if (setjmp(frame.jump)) {
        jb_status status = jb_frame_fail(&frame);
        jb_active_allocator = previous_allocator;
        return status;
    }
    size_t request_length = 0, output_length = 0;
    char *request = jb_typed_request(session, input, &request_length), *output = NULL;
    jb_status status =
        jb_session_decide_json_call(session, request, request_length, &output, &output_length);
    jb_release(request);
    if (status != JB_OK) {
        jb_frame_abandon(&frame);
        jb_active_allocator = previous_allocator;
        return status;
    }
    jb_result *result = jb_result_parse(output, output_length);
    jb_release(output);
    jb_frame_leave(&frame);
    jb_active_allocator = previous_allocator;
    *out_result = result;
    return JB_OK;
}

static jb_status jb_session_decide_batch_call(jb_session *session, const jb_input *inputs,
                                              size_t input_count, jb_result ***out_results) {
    if (out_results)
        *out_results = NULL;
    if (!session || !inputs || !input_count || input_count > JB_MAX_BATCH_INPUTS || !out_results)
        return jb_invalid("typed batch arguments are invalid");
    const jb_allocator *previous_allocator = jb_active_allocator;
    jb_active_allocator = &session->model->allocator;
    JBErrorFrame frame;
    jb_frame_enter(&frame, JB_ERROR_REQUEST);
    if (setjmp(frame.jump)) {
        jb_status status = jb_frame_fail(&frame);
        jb_active_allocator = previous_allocator;
        return status;
    }
    char **request = xcalloc(input_count, sizeof *request);
    size_t *request_length = xcalloc(input_count, sizeof *request_length);
    for (size_t i = 0; i < input_count; i++)
        request[i] = jb_typed_request(session, &inputs[i], &request_length[i]);
    char **output = NULL;
    size_t *output_length = NULL;
    jb_status status =
        jb_session_decide_json_batch_call(session, (const char *const *)request, request_length,
                                          input_count, &output, &output_length);
    for (size_t i = 0; i < input_count; i++)
        jb_release(request[i]);
    jb_release(request_length);
    jb_release(request);
    if (status != JB_OK) {
        jb_frame_abandon(&frame);
        jb_active_allocator = previous_allocator;
        return status;
    }
    jb_result **result = xcalloc(input_count, sizeof *result);
    for (size_t i = 0; i < input_count; i++) {
        result[i] = jb_result_parse(output[i], output_length[i]);
        jb_release(output[i]);
    }
    jb_release(output_length);
    jb_release(output);
    jb_frame_leave(&frame);
    jb_active_allocator = previous_allocator;
    *out_results = result;
    return JB_OK;
}

/* Each session call records its outcome on the session, so the error can be
 * read from whichever thread handles the session next. */
static int jb_session_acquire(jb_session *session) {
    return !session || !atomic_flag_test_and_set_explicit(&session->busy, memory_order_acquire);
}

static jb_status jb_session_release(jb_session *session, jb_status status) {
    if (!session)
        return status;
    if (status == JB_OK) {
        session->error[0] = 0;
        atomic_flag_clear_explicit(&session->busy, memory_order_release);
        return status;
    }
    snprintf(session->error, sizeof session->error, "%s", jb_error_message);
    char message[sizeof session->error];
    memcpy(message, session->error, sizeof message);
    jb_model *model = session->model;
    jb_model_retain(model);
    atomic_flag_clear_explicit(&session->busy, memory_order_release);
    /* No session access is permitted after the callback: a reentrant logger
     * may destroy it. The retained model keeps the callback and its context
     * alive for this invocation. */
    jb_log_message(model, JB_LOG_ERROR, message);
    jb_model_release(model);
    return status;
}

static jb_status jb_session_ready(jb_session *session) {
    if (!session || !session->poisoned)
        return JB_OK;
    snprintf(jb_error_message, sizeof jb_error_message,
             "session is unusable after an inference failure");
    return JB_ERROR_INTERNAL;
}

const char *jb_session_last_error(const jb_session *session) {
    return session ? session->error : "";
}

jb_status jb_session_decide_json(jb_session *session, const char *request_json,
                                 size_t request_length, char **out_json, size_t *out_length) {
    if (out_json)
        *out_json = NULL;
    if (out_length)
        *out_length = 0;
    if (!jb_session_acquire(session))
        return jb_invalid("session is already in use by another call");
    jb_status ready = jb_session_ready(session);
    if (ready != JB_OK)
        return jb_session_release(session, ready);
    return jb_session_release(
        session,
        jb_session_decide_json_call(session, request_json, request_length, out_json, out_length));
}

jb_status jb_session_decide_json_batch(jb_session *session, const char *const *request_json,
                                       const size_t *request_lengths, size_t request_count,
                                       char ***out_json, size_t **out_lengths) {
    if (out_json)
        *out_json = NULL;
    if (out_lengths)
        *out_lengths = NULL;
    if (!jb_session_acquire(session))
        return jb_invalid("session is already in use by another call");
    jb_status ready = jb_session_ready(session);
    if (ready != JB_OK)
        return jb_session_release(session, ready);
    return jb_session_release(
        session, jb_session_decide_json_batch_call(session, request_json, request_lengths,
                                                   request_count, out_json, out_lengths));
}

jb_status jb_session_decide(jb_session *session, const jb_input *input, jb_result **out_result) {
    if (out_result)
        *out_result = NULL;
    if (!jb_session_acquire(session))
        return jb_invalid("session is already in use by another call");
    jb_status ready = jb_session_ready(session);
    if (ready != JB_OK)
        return jb_session_release(session, ready);
    return jb_session_release(session, jb_session_decide_call(session, input, out_result));
}

jb_status jb_session_decide_batch(jb_session *session, const jb_input *inputs, size_t input_count,
                                  jb_result ***out_results) {
    if (out_results)
        *out_results = NULL;
    if (!jb_session_acquire(session))
        return jb_invalid("session is already in use by another call");
    jb_status ready = jb_session_ready(session);
    if (ready != JB_OK)
        return jb_session_release(session, ready);
    return jb_session_release(
        session, jb_session_decide_batch_call(session, inputs, input_count, out_results));
}

void jb_result_free(jb_result *result) {
    jb_release(result);
}

void jb_results_free(jb_result **results, size_t count) {
    if (!results)
        return;
    for (size_t i = 0; i < count; i++)
        jb_result_free(results[i]);
    jb_release(results);
}
