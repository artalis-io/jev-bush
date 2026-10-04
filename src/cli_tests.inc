#ifndef JB_NO_MAIN
/* Says on stderr when the accelerator was built and a device found, but its
 * kernels failed, so the CPU kernels run instead. */
static void dg_report_accel(void) {
    const char *log, *failure;
    if (dg_accel() || !(failure = dg_accel_failure(&log)))
        return;
    fprintf(stderr, "jb: CUDA accelerator unavailable, running on the CPU: %s\n", failure);
    if (*log)
        fprintf(stderr, "%s\n", log);
}

static int dg_decide_file(const char *dir, const char *path) {
    size_t n;
    char *j = read_all(path, &n);
    jb_model *model = NULL;
    jb_session *session = NULL;
    char *output = NULL;
    size_t output_length = 0;
    if (jb_model_load(dir, &model) != JB_OK)
        die(jb_last_error());
    if (model->model.nvfp4)
        dg_report_accel();
    if (jb_session_create_json(model, NULL, 0, &session) != JB_OK ||
        jb_session_decide_json(session, j, n, &output, &output_length) != JB_OK)
        die(jb_last_error());
    if (fwrite(output, 1, output_length, stdout) != output_length || putchar('\n') == EOF)
        die("cannot write output");
    jb_free(output);
    jb_session_free(session);
    jb_model_free(model);
    jb_release(j);
    return 0;
}

static void dg_eval_group(jb_session *session, char **row, size_t *len, int n) {
    char **output = NULL;
    size_t *output_length = NULL;
    if (jb_session_decide_json_batch(session, (const char *const *)row, len, (size_t)n, &output,
                                     &output_length) != JB_OK)
        die(jb_last_error());
    for (int i = 0; i < n; i++) {
        if (fwrite(output[i], 1, output_length[i], stdout) != output_length[i] ||
            putchar('\n') == EOF)
            die("cannot write output");
        jb_free(output[i]);
    }
    jb_free(output_length);
    jb_free(output);
    flush_output();
}

static int dg_eval_file(const char *dir, const char *path) {
    FILE *f = !strcmp(path, "-") ? stdin : fopen(path, "rb");
    if (!f)
        die2("cannot open", path);
#if defined(_WIN32)
    if (f == stdin && _setmode(_fileno(stdin), _O_BINARY) < 0)
        die("cannot set binary stdin");
#endif
    jb_model *model = NULL;
    jb_session *session = NULL;
    int mb = 1;
    const char *me = getenv("JB_MICROBATCH");
    if (me) {
        char *e;
        long v = strtol(me, &e, 10);
        if (*e || v < 1 || v > 16)
            die("JB_MICROBATCH must be 1..16");
        mb = (int)v;
    }
    if (jb_model_load(dir, &model) != JB_OK)
        die(jb_last_error());
    if (model->model.nvfp4)
        dg_report_accel();
    if (jb_session_create_json(model, NULL, 0, &session) != JB_OK)
        die(jb_last_error());
    char *line = NULL, **rows = xcalloc((size_t)mb, sizeof *rows);
    size_t cap = 0, n = 0, *lens = xcalloc((size_t)mb, sizeof *lens);
    int nr = 0, ch;
    while ((ch = fgetc(f)) != EOF) {
        if (ch == '\n') {
            while (n && line[n - 1] == '\r')
                n--;
            if (n) {
                rows[nr] = xmalloc(n);
                memcpy(rows[nr], line, n);
                lens[nr++] = n;
                if (nr == mb) {
                    dg_eval_group(session, rows, lens, nr);
                    for (int i = 0; i < nr; i++)
                        jb_release(rows[i]);
                    nr = 0;
                }
            }
            n = 0;
            continue;
        }
        if (n == cap) {
            cap = cap ? cap * 2 : 4096;
            if (cap > JB_MAX_JSON)
                die("JSONL row too large");
            line = xrealloc(line, cap);
        }
        line[n++] = (char)ch;
    }
    if (ferror(f))
        die2("cannot read", path);
    while (n && line[n - 1] == '\r')
        n--;
    if (n) {
        rows[nr] = xmalloc(n);
        memcpy(rows[nr], line, n);
        lens[nr++] = n;
    }
    if (nr) {
        dg_eval_group(session, rows, lens, nr);
        for (int i = 0; i < nr; i++)
            jb_release(rows[i]);
    }
    jb_release(lens);
    jb_release(rows);
    jb_release(line);
    if (f != stdin)
        fclose(f);
    jb_session_free(session);
    jb_model_free(model);
    return 0;
}

/* Validate a request without a model: JSON, request fields, questions, and
 * prompt construction. Choice labels use OpenJev's candidate order without
 * the tokenizer's single-token filter. The request is parsed from an
 * exact-size copy, as eval rows are, so sanitizers catch any over-read. */
static int dg_check_request(const char *path) {
    size_t n;
    char *s = read_all(path, &n), *j = xmalloc(n);
    memcpy(j, s, n);
    jb_release(s);
    DGRequest rq;
    dg_request_parse(&rq, j, n);
    (void)dg_request_seed(&rq);
    char *labels[DG_CHOICE_LABELS];
    for (int i = 0; i < DG_CHOICE_LABELS; i++) {
        char l[3];
        dg_choice_candidate(i, l);
        labels[i] = xstrdup(l);
    }
    int nq;
    DecisionWork *w = dg_questions(&rq, labels, &nq);
    for (int i = 0; i < DG_CHOICE_LABELS; i++)
        jb_release(labels[i]);
    char *sys = dg_system_prompt(rq.qj, rq.qt, rq.qnt, w, nq);
    int *zero = xcalloc((size_t)nq, sizeof *zero);
    char *text = dg_answer_template(w, nq, zero);
    printf("{\"check\":\"ok\",\"questions\":%d}\n", nq);
    jb_release(text);
    jb_release(zero);
    jb_release(sys);
    dg_questions_free(w, nq);
    dg_request_free(&rq);
    jb_release(j);
    return 0;
}

/* Kernel self-tests: every kernel, reference and selected ISA version, is
 * checked against a double-precision oracle on random data. Shapes cover
 * full 8-token blocks, token remainders, and column tails. The bound is
 * 1e-4 of the absolute-value sum: summation order may differ between
 * kernels, but an indexing or tail bug moves results by far more. */
static uint32_t jb_rng(uint64_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return (uint32_t)(*s >> 32);
}

static float jb_rng_unit(uint64_t *s) {
    return (float)(jb_rng(s) >> 8) / 8388608.0f * 2.0f - 1.0f;
}

static void jb_put_bf16(uint8_t *p, float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    p[0] = (uint8_t)(u >> 16);
    p[1] = (uint8_t)(u >> 24);
}

/* Kernels that claim bit-identity with the reference must deliver it in
 * strict builds; fast math lets the compiler reorder the reference itself. */
static void jb_check_reference_bits(const char *what, const void *got, const void *ref,
                                    size_t bytes) {
    if (dg_kernels()->matches_reference && JB_STRICT_MATH && memcmp(got, ref, bytes))
        die2("kernel self-test failed: differs from the reference", what);
}

/* Fill a kernel's output with NaN before each call, so an output the kernel
 * fails to write cannot inherit a correct value from an earlier pass. */
static void jb_poison(float *y, size_t count) {
    memset(y, 0xff, count * sizeof *y);
}

static void jb_check_close(const char *what, double got, double want, double bound) {
    if (!jb_finite(got) || fabs(got - want) > bound) {
        char z[160];
        snprintf(z, sizeof z, "%s (got %.9g, want %.9g, bound %.3g)", what, got, want, bound);
        die2("kernel self-test failed", z);
    }
}

static void dg_test_mm(uint64_t *rs) {
    static const int shape[][3] = {{2, 16, 1}, {2, 7, 1},     {2, 15, 8},  {4, 40, 3},
                                   {6, 33, 9}, {8, 2816, 17}, {16, 33, 9}, {32, 2816, 17}};
    for (size_t k = 0; k < sizeof shape / sizeof *shape; k++) {
        int rows = shape[k][0], cols = shape[k][1], tokens = shape[k][2];
        uint8_t *w = xmalloc((size_t)rows * cols * 2);
        float *x = xmalloc((size_t)tokens * cols * 4), *y = xmalloc((size_t)tokens * rows * 4);
        double *want = xmalloc((size_t)tokens * rows * sizeof *want),
               *mag = xmalloc((size_t)tokens * rows * sizeof *mag);
        for (int i = 0; i < rows * cols; i++)
            jb_put_bf16(w + (size_t)i * 2, jb_rng_unit(rs));
        for (int i = 0; i < tokens * cols; i++)
            x[i] = jb_rng_unit(rs);
        for (int t = 0; t < tokens; t++)
            for (int r = 0; r < rows; r++) {
                double s = 0, a = 0;
                for (int c = 0; c < cols; c++) {
                    double v =
                        (double)dg_bf(w + ((size_t)r * cols + c) * 2) * x[(size_t)t * cols + c];
                    s += v;
                    a += fabs(v);
                }
                want[(size_t)t * rows + r] = s;
                mag[(size_t)t * rows + r] = a;
            }
        float *yref = xmalloc((size_t)tokens * rows * 4);
        for (int pass = 0; pass < 2; pass++) {
            jb_poison(y, (size_t)tokens * rows);
            if (pass)
                dg_mm_data(w, x, y, tokens, rows, cols);
            else
                dg_mm_data_ref(w, x, y, tokens, rows, cols);
            for (int i = 0; i < tokens * rows; i++)
                jb_check_close(pass ? dg_kernels()->name : "BF16 matmul (ref)", y[i], want[i],
                               1e-4 * mag[i]);
            if (pass)
                jb_check_reference_bits("BF16 matmul", y, yref, (size_t)tokens * rows * 4);
            else
                memcpy(yref, y, (size_t)tokens * rows * 4);
        }
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
        if (jb_cpu_has_avx512() && rows % DG_EXACT_ROWS == 0) {
            DGTensor tw = {0};
            tw.data = w;
            tw.exact = xmalloc((size_t)rows * cols * 2);
            tw.shape[0] = (uint64_t)rows;
            tw.shape[1] = (uint64_t)cols;
            dg_exact_pack_bf16(&tw);
            jb_poison(y, (size_t)tokens * rows);
            dg_mm_exact_avx512(tw.exact, x, y, tokens, rows, cols);
            if (memcmp(y, yref, (size_t)tokens * rows * 4))
                die("kernel self-test failed: exact AVX-512 BF16 differs from reference");
            jb_release((void *)tw.exact);
        }
#endif
        jb_release(yref);
        jb_release(w);
        jb_release(x);
        jb_release(y);
        jb_release(want);
        jb_release(mag);
    }
}

/* An accelerator's quantized rows, row bytes each, as the reference's
 * floats: the floats themselves, or packed E2M1 nibbles, E4M3 codes and a
 * float global scale, decoded as the reference computes each value. */
static void dg_test_nvfp4_unpack(float *out, const uint8_t *in, int tokens, int cols, size_t row) {
    static const float mag[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    if (row == (size_t)cols * 4) {
        memcpy(out, in, (size_t)tokens * row);
        return;
    }
    for (int t = 0; t < tokens; t++) {
        const uint8_t *r = in + (size_t)t * row;
        float base;
        memcpy(&base, r + cols / 2 + cols / 16, 4);
        for (int b = 0; b < cols / 16; b++) {
            /* The block's scale first, as the reference computes it. */
            float scale = dg_f8e4m3(r[cols / 2 + b]) * base;
            for (int c = b * 16; c < b * 16 + 16; c++) {
                unsigned nib = c & 1 ? r[c / 2] >> 4 : r[c / 2] & 15u;
                out[(size_t)t * cols + c] = (nib & 8 ? -mag[nib & 7] : mag[nib & 7]) * scale;
            }
        }
    }
}

static void dg_test_nvfp4(uint64_t *rs) {
    static const int shape[][3] = {{2, 32, 1}, {4, 64, 9},    {8, 704, 17}, {2, 2816, 3},
                                   {6, 64, 2}, {16, 704, 17}, {32, 2816, 3}};
    static const float mag4[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    const float global = 0.01f, base = 0.02f;
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
    if (jb_cpu_has_avx512())
        for (int first = 0; first < 256; first += DG_EXACT_ROWS) {
            uint8_t code[DG_EXACT_ROWS];
            float got[DG_EXACT_ROWS], want[DG_EXACT_ROWS];
            for (int i = 0; i < DG_EXACT_ROWS; i++) {
                code[i] = (uint8_t)(first + i);
                want[i] = dg_f8e4m3(code[i]);
            }
            dg_f8e4m3x16_store(got, code);
            if (memcmp(got, want, sizeof got))
                die("kernel self-test failed: exact AVX-512 E4M3 decoder differs from reference");
        }
#endif
    for (size_t k = 0; k < sizeof shape / sizeof *shape; k++) {
        int rows = shape[k][0], cols = shape[k][1], tokens = shape[k][2];
        uint8_t *wd = xmalloc((size_t)rows * cols / 2), *sd = xmalloc((size_t)rows * cols / 16);
        float *x = xmalloc((size_t)tokens * cols * 4), *xq = xmalloc((size_t)tokens * cols * 4);
        float *xs = xmalloc((size_t)tokens * cols * 4), *y = xmalloc((size_t)tokens * rows * 4);
        double *want = xmalloc((size_t)tokens * rows * sizeof *want),
               *mag = xmalloc((size_t)tokens * rows * sizeof *mag);
        for (int i = 0; i < rows * cols / 2; i++)
            wd[i] = (uint8_t)jb_rng(rs);
        for (int i = 0; i < rows * cols / 16; i++)
            sd[i] = (uint8_t)(0x30 + jb_rng(rs) % 32);
        for (int i = 0; i < tokens * cols; i++)
            x[i] = 3 * jb_rng_unit(rs);
        dg_nvfp4_qdq_ref(xq, x, tokens, cols, base);
        for (int t = 0; t < tokens; t++)
            for (int r = 0; r < rows; r++) {
                double s = 0, a = 0;
                for (int c = 0; c < cols; c++) {
                    uint8_t b = wd[(size_t)r * cols / 2 + c / 2],
                            nib = (uint8_t)(c & 1 ? b >> 4 : b & 15);
                    float scale = dg_f8e4m3(sd[(size_t)r * cols / 16 + c / 16]) * global;
                    double v = (double)scale * (nib & 8 ? -mag4[nib & 7] : mag4[nib & 7]) *
                               xq[(size_t)t * cols + c];
                    s += v;
                    a += fabs(v);
                }
                want[(size_t)t * rows + r] = s;
                mag[(size_t)t * rows + r] = a;
            }
        jb_poison(y, (size_t)tokens * rows);
        dg_nvfp4_mm_ref(wd, sd, global, xq, y, tokens, rows, cols);
        for (int i = 0; i < tokens * rows; i++)
            jb_check_close("NVFP4 matmul (ref)", y[i], want[i], 1e-4 * mag[i]);
        float *yref = xmalloc((size_t)tokens * rows * 4);
        memcpy(yref, y, (size_t)tokens * rows * 4);
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
        if (jb_cpu_has_avx512() && rows % DG_EXACT_ROWS == 0) {
            DGTensor tw = {0}, ts = {0};
            tw.data = wd;
            ts.data = sd;
            tw.exact = xmalloc((size_t)rows * cols / 2);
            ts.exact = xmalloc((size_t)rows * cols / 16);
            dg_exact_pack_nvfp4(&tw, &ts, rows, cols);
            jb_poison(y, (size_t)tokens * rows);
            dg_nvfp4_exact_avx512(tw.exact, ts.exact, global, xq, y, tokens, rows, cols);
            if (memcmp(y, yref, (size_t)tokens * rows * 4))
                die("kernel self-test failed: exact AVX-512 NVFP4 differs from reference");
            jb_release((void *)tw.exact);
            jb_release((void *)ts.exact);
        }
#endif
        const DGAccelOps *accel = dg_accel();
        if (accel) {
            /* The accelerator's quantizer and product, against the
             * reference's. */
            size_t xb = (size_t)tokens * cols * 4, yb = (size_t)tokens * rows * 4,
                   sb = accel->nvfp4_scratch(tokens), wb = (size_t)rows * cols / 2,
                   scb = (size_t)rows * cols / 16;
            DGDevice dx = accel->alloc(xb), dq = accel->alloc(xb), dy = accel->alloc(yb),
                     ds = accel->alloc(sb), df = accel->alloc(4), dw = accel->alloc(wb),
                     dsc = accel->alloc(scb);
            if (!dx || !dq || !dy || !ds || !df || !dw || !dsc)
                die2("accelerator self-test failed: device memory", accel->name);
            accel->to_device(dw, wd, wb);
            accel->to_device(dsc, sd, scb);
            DGAccelMatrix *am = accel->matrix(dw, dsc, global, rows, cols);
            int flag = 0, *group = xcalloc((size_t)tokens, sizeof *group);
            float *xd = xmalloc(xb);
            accel->to_device(dx, x, xb);
            accel->to_device(df, &flag, sizeof flag);
            accel->nvfp4_qdq(dq, dx, tokens, cols, base, df);
            size_t row = accel->nvfp4_row(cols) * 4;
            uint8_t *packed = xmalloc((size_t)tokens * row);
            accel->to_host(packed, dq, (size_t)tokens * row);
            dg_test_nvfp4_unpack(xd, packed, tokens, cols, row);
            jb_release(packed);
            accel->to_host(&flag, df, sizeof flag);
            if (flag || memcmp(xd, xq, xb))
                die2("accelerator self-test failed: NVFP4 quantizer", accel->name);
            /* Two groups of the same matrix, every three tokens, so tiles
             * of different groups meet in one block's run. */
            const DGAccelMatrix *list[2] = {am, am};
            for (int t = 0; t < tokens; t++)
                group[t] = t / 3 % 2;
            jb_poison(y, (size_t)tokens * rows);
            accel->nvfp4_mm(list, group, tokens, dq, dy, ds);
            accel->to_host(y, dy, yb);
            for (int i = 0; i < tokens * rows; i++)
                jb_check_close(accel->name, y[i], want[i], 1e-4 * mag[i]);
            if (JB_STRICT_MATH && memcmp(y, yref, yb))
                die2("accelerator self-test failed: differs from the reference", accel->name);
            jb_release(group);
            jb_release(xd);
            accel->release(am);
            accel->free(dw);
            accel->free(dsc);
            accel->free(dx);
            accel->free(dq);
            accel->free(dy);
            accel->free(ds);
            accel->free(df);
        }
        /* The production path: QDQ into the selected kernel's layout, then
         * the validated tensor entry point. */
        dg_nvfp4_qdq(xs, x, tokens, cols, base);
        if (dg_kernels()->nvfp4_layout)
            dg_kernels()->nvfp4_layout(xq, tokens, cols);
        if (memcmp(xs, xq, (size_t)tokens * cols * 4))
            die2("kernel self-test failed", dg_kernels()->name);
        DGTensor tw = {0}, ts = {0}, tg = {0};
        tw.name = ts.name = tg.name = "selftest";
        tw.data = wd;
        tw.dtype = DG_U8;
        tw.nd = 2;
        tw.shape[0] = (uint64_t)rows;
        tw.shape[1] = (uint64_t)cols / 2;
        ts.data = sd;
        ts.dtype = DG_F8E4M3;
        ts.nd = 2;
        ts.shape[0] = (uint64_t)rows;
        ts.shape[1] = (uint64_t)cols / 16;
        tg.data = (const uint8_t *)&global;
        tg.dtype = DG_F32;
        tg.bytes = 4;
        jb_poison(y, (size_t)tokens * rows);
        dg_nvfp4_mm(&tw, &ts, &tg, xs, y, tokens, rows, cols);
        for (int i = 0; i < tokens * rows; i++)
            jb_check_close(dg_kernels()->name, y[i], want[i], 1e-4 * mag[i]);
        jb_check_reference_bits("NVFP4 matmul", y, yref, (size_t)tokens * rows * 4);
        jb_release(yref);
        /* The fused expert product, against the backend's separate steps:
         * the same bits in strict builds, whose GELU is the same scalar
         * call in both. */
        const float up_global = 0.013f;
        size_t out = (size_t)tokens * rows;
        uint8_t *wu = xmalloc((size_t)rows * cols / 2);
        float *g = xmalloc(out * 4), *u = xmalloc(out * 4), *h = xmalloc(out * 4),
              *scratch = xmalloc(out * 4);
        for (int i = 0; i < rows * cols / 2; i++)
            wu[i] = (uint8_t)(wd[i] ^ 0x5a);
        DGTensor uw = tw, ug = tg;
        uw.data = wu;
        ug.data = (const uint8_t *)&up_global;
        DGNvExpert v = {0};
        v.wg = &tw;
        v.sg = v.su = &ts;
        v.gg = &tg;
        v.wu = &uw;
        v.gu = &ug;
        dg_nvfp4_mm(&tw, &ts, &tg, xs, g, tokens, rows, cols);
        dg_nvfp4_mm(&uw, &ts, &ug, xs, u, tokens, rows, cols);
        jb_poison(h, out);
        dg_nvfp4_gated(&v, xs, scratch, h, tokens, rows, cols);
        for (size_t i = 0; i < out; i++) {
            float want_h = dg_gelu(g[i]) * u[i];
            jb_check_close(dg_kernels()->name, h[i], want_h,
                           1e-5 * (1 + fabs(g[i])) * (1 + fabs(u[i])));
            if (JB_STRICT_MATH && memcmp(&h[i], &want_h, 4))
                die2("kernel self-test failed: fused expert product differs", dg_kernels()->name);
        }
        jb_release(wu);
        jb_release(g);
        jb_release(u);
        jb_release(h);
        jb_release(scratch);
        jb_release(wd);
        jb_release(sd);
        jb_release(x);
        jb_release(xq);
        jb_release(xs);
        jb_release(y);
        jb_release(want);
        jb_release(mag);
    }
}

static void dg_test_rms_dot(uint64_t *rs) {
    static const int len[] = {1, 15, 16, 17, 31, 32, 33, 256, 2816};
    for (size_t k = 0; k < sizeof len / sizeof *len; k++) {
        int n = len[k];
        float *x = xmalloc((size_t)n * 4), *b = xmalloc((size_t)n * 4), *y = xmalloc((size_t)n * 4);
        uint8_t *sw = xmalloc((size_t)n * 2);
        for (int i = 0; i < n; i++) {
            x[i] = 4 * jb_rng_unit(rs);
            b[i] = jb_rng_unit(rs);
            jb_put_bf16(sw + (size_t)i * 2, jb_rng_unit(rs));
        }
        DGTensor scale = {0};
        scale.name = "selftest";
        scale.data = sw;
        scale.dtype = DG_BF16;
        scale.nd = 1;
        scale.shape[0] = (uint64_t)n;
        scale.bytes = (uint64_t)n * 2;
        double ss = 0, dot = 0, dmag = 0;
        for (int i = 0; i < n; i++) {
            ss += (double)x[i] * x[i];
            dot += (double)x[i] * b[i];
            dmag += fabs((double)x[i] * b[i]);
        }
        double q = 1.0 / sqrt(ss / n + 1e-6);
        float *yref = xmalloc((size_t)n * 2 * 4);
        for (int pass = 0; pass < 4; pass++) {
            const DGTensor *sc = pass & 1 ? &scale : NULL;
            jb_poison(y, (size_t)n);
            if (pass & 2)
                dg_rms(y, x, sc, n);
            else
                dg_rms_ref(y, x, sc, n);
            for (int i = 0; i < n; i++) {
                double want = x[i] * q * (sc ? dg_at(sc, i) : 1.0);
                jb_check_close(pass & 2 ? dg_kernels()->name : "RMS norm (ref)", y[i], want,
                               1e-5 * fabs(want) + 1e-12);
            }
            if (pass & 2)
                jb_check_reference_bits("RMS norm", y, yref + (size_t)(pass & 1) * n,
                                        (size_t)n * 4);
            else
                memcpy(yref + (size_t)(pass & 1) * n, y, (size_t)n * 4);
        }
        jb_release(yref);
        jb_check_close("dot (ref)", dg_dot_ref(x, b, n), dot, 1e-12 * dmag);
        jb_check_close(dg_kernels()->name, dg_dot(x, b, n), dot, 1e-12 * dmag);
        double dot_ref = dg_dot_ref(x, b, n), dot_kernel = dg_dot(x, b, n);
        jb_check_reference_bits("dot", &dot_kernel, &dot_ref, sizeof dot_ref);
        jb_release(x);
        jb_release(b);
        jb_release(y);
        jb_release(sw);
    }
}

/* The accelerator's BF16 product, RMS norm, GELU, gather and MoE tail
 * against the reference's arithmetic, bit for bit in strict builds. */
static void dg_test_accel_layer(uint64_t *rs) {
    const DGAccelOps *a = dg_accel();
    if (!a)
        return;

    enum { tokens = 11, rows = 130, cols = 2816, k = DG_TOPK };

    size_t xb = (size_t)tokens * cols * 4, yb = (size_t)tokens * rows * 4;
    uint8_t *w = xmalloc((size_t)rows * cols * 2), *sc = xmalloc((size_t)cols * 2),
            *fac = xmalloc((size_t)cols * 2), *p2 = xmalloc((size_t)cols * 2);
    float *x = xmalloc(xb), *add = xmalloc(xb), *want = xmalloc(xb), *got = xmalloc(xb),
          *y = xmalloc(yb), *yref = xmalloc(yb), *wt = xmalloc((size_t)tokens * k * 4);
    int *index = xmalloc((size_t)tokens * k * sizeof *index),
        *slot = xmalloc((size_t)tokens * k * sizeof *slot);
    for (int i = 0; i < rows * cols; i++)
        jb_put_bf16(w + (size_t)i * 2, jb_rng_unit(rs));
    for (int i = 0; i < cols; i++) {
        jb_put_bf16(sc + (size_t)i * 2, 1 + jb_rng_unit(rs) / 4);
        jb_put_bf16(fac + (size_t)i * 2, 1 + jb_rng_unit(rs) / 4);
        jb_put_bf16(p2 + (size_t)i * 2, 1 + jb_rng_unit(rs) / 4);
    }
    for (size_t i = 0; i < (size_t)tokens * cols; i++) {
        x[i] = 3 * jb_rng_unit(rs);
        add[i] = jb_rng_unit(rs);
    }
    for (int i = 0; i < tokens * k; i++) {
        index[i] = (int)(jb_rng(rs) % tokens);
        slot[i] = (int)(jb_rng(rs) % (tokens * k));
        wt[i] = jb_rng_unit(rs);
    }
    DGDevice dw = a->alloc((size_t)rows * cols * 2), dsc = a->alloc((size_t)cols * 2),
             dfac = a->alloc((size_t)cols * 2), dp2 = a->alloc((size_t)cols * 2), dx = a->alloc(xb),
             dadd = a->alloc(xb), dy = a->alloc(yb), dz = a->alloc(xb), dg = a->alloc(xb * k),
             deo = a->alloc(xb * k), dindex = a->alloc((size_t)tokens * k * 4),
             dslot = a->alloc((size_t)tokens * k * 4), dwt = a->alloc((size_t)tokens * k * 4);
    if (!dw || !dsc || !dfac || !dp2 || !dx || !dadd || !dy || !dz || !dg || !deo || !dindex ||
        !dslot || !dwt)
        die2("accelerator self-test failed: device memory", a->name);
    a->to_device(dw, w, (size_t)rows * cols * 2);
    a->to_device(dsc, sc, (size_t)cols * 2);
    a->to_device(dfac, fac, (size_t)cols * 2);
    a->to_device(dp2, p2, (size_t)cols * 2);
    a->to_device(dx, x, xb);
    a->to_device(dadd, add, xb);
    a->to_device(dindex, index, (size_t)tokens * k * 4);
    a->to_device(dslot, slot, (size_t)tokens * k * 4);
    a->to_device(dwt, wt, (size_t)tokens * k * 4);
#define DG_ACCEL_SAME(what, gotp, wantp, bytes)                                                    \
    do {                                                                                           \
        if (JB_STRICT_MATH && memcmp((gotp), (wantp), (bytes)))                                    \
            die2("accelerator self-test failed: differs from the reference", what);                \
    } while (0)
/* Fast builds' products sum in FP32 in another order than the reference's FP32
 * sums, which over 2816 terms differ by about 1e-5 relative. */
#define DG_ACCEL_CLOSE(gotp, wantp, count)                                                         \
    for (size_t i_ = 0; i_ < (size_t)(count); i_++)                                                \
    jb_check_close(a->name, (gotp)[i_], (wantp)[i_], 1e-4 * (1 + fabs((wantp)[i_])))
    /* BF16 product, on a shape for each kernel. */
    dg_mm_data_ref(w, x, yref, tokens, rows, cols);
    a->bf16_mm(dw, dx, dy, tokens, rows, cols);
    a->to_host(y, dy, yb);
    DG_ACCEL_SAME("BF16 matmul", y, yref, yb);
    DG_ACCEL_CLOSE(y, yref, (size_t)tokens * rows);
    {
        enum { big_rows = 768, big_tokens = 512 };

        size_t wb2 = (size_t)big_rows * cols * 2, xb2 = (size_t)big_tokens * cols * 4,
               yb2 = (size_t)big_tokens * big_rows * 4;
        uint8_t *w2 = xmalloc(wb2);
        float *x2 = xmalloc(xb2), *y2 = xmalloc(yb2), *y2ref = xmalloc(yb2);
        for (size_t i = 0; i < (size_t)big_rows * cols; i++)
            jb_put_bf16(w2 + i * 2, jb_rng_unit(rs));
        for (size_t i = 0; i < (size_t)big_tokens * cols; i++)
            x2[i] = jb_rng_unit(rs);
        DGDevice dw2 = a->alloc(wb2), dx2 = a->alloc(xb2), dy2 = a->alloc(yb2);
        if (!dw2 || !dx2 || !dy2)
            die2("accelerator self-test failed: device memory", a->name);
        a->to_device(dw2, w2, wb2);
        a->to_device(dx2, x2, xb2);
        dg_mm_data_ref(w2, x2, y2ref, big_tokens, big_rows, cols);
        a->bf16_mm(dw2, dx2, dy2, big_tokens, big_rows, cols);
        a->to_host(y2, dy2, yb2);
        DG_ACCEL_SAME("BF16 matmul, many tokens", y2, y2ref, yb2);
        DG_ACCEL_CLOSE(y2, y2ref, (size_t)big_tokens * big_rows);
        a->free(dw2);
        a->free(dx2);
        a->free(dy2);
        jb_release(w2);
        jb_release(x2);
        jb_release(y2);
        jb_release(y2ref);
    }
    {
        /* Many rows and few columns: enough blocks for the wide kernels. */
        enum { many_rows = 12288, few_cols = 64, few_tokens = 3 };

        size_t wb3 = (size_t)many_rows * few_cols * 2, xb3 = (size_t)few_tokens * few_cols * 4,
               yb3 = (size_t)few_tokens * many_rows * 4;
        uint8_t *w3 = xmalloc(wb3);
        float *x3 = xmalloc(xb3), *y3 = xmalloc(yb3), *y3ref = xmalloc(yb3);
        for (size_t i = 0; i < (size_t)many_rows * few_cols; i++)
            jb_put_bf16(w3 + i * 2, jb_rng_unit(rs));
        for (size_t i = 0; i < (size_t)few_tokens * few_cols; i++)
            x3[i] = jb_rng_unit(rs);
        DGDevice dw3 = a->alloc(wb3), dx3 = a->alloc(xb3), dy3 = a->alloc(yb3);
        if (!dw3 || !dx3 || !dy3)
            die2("accelerator self-test failed: device memory", a->name);
        a->to_device(dw3, w3, wb3);
        a->to_device(dx3, x3, xb3);
        dg_mm_data_ref(w3, x3, y3ref, few_tokens, many_rows, few_cols);
        a->bf16_mm(dw3, dx3, dy3, few_tokens, many_rows, few_cols);
        a->to_host(y3, dy3, yb3);
        DG_ACCEL_SAME("BF16 matmul, many rows", y3, y3ref, yb3);
        DG_ACCEL_CLOSE(y3, y3ref, (size_t)few_tokens * many_rows);
        a->free(dw3);
        a->free(dx3);
        a->free(dy3);
        jb_release(w3);
        jb_release(x3);
        jb_release(y3);
        jb_release(y3ref);
    }
    /* RMS with a scale and a residual, and with a factor. */
    DGTensor scale = {0}, factor = {0};
    scale.data = sc;
    factor.data = fac;
    float div = sqrtf(DG_H);
    for (int t = 0; t < tokens; t++) {
        float *r = want + (size_t)t * cols;
        dg_rms_ref(r, x + (size_t)t * cols, &scale, cols);
        for (int i = 0; i < cols; i++)
            r[i] += add[(size_t)t * cols + i];
    }
    a->rms(dz, dx, dsc, 0, 0, dadd, tokens, cols);
    a->to_host(got, dz, xb);
    DG_ACCEL_SAME("RMS norm with residual", got, want, xb);
    DG_ACCEL_CLOSE(got, want, (size_t)tokens * cols);
    for (int t = 0; t < tokens; t++) {
        float *r = want + (size_t)t * cols;
        dg_rms_ref(r, x + (size_t)t * cols, NULL, cols);
        for (int i = 0; i < cols; i++)
            r[i] *= dg_at(&factor, (uint64_t)i) / div;
    }
    a->rms(dz, dx, 0, dfac, div, 0, tokens, cols);
    a->to_host(got, dz, xb);
    DG_ACCEL_SAME("RMS norm with factor", got, want, xb);
    DG_ACCEL_CLOSE(got, want, (size_t)tokens * cols);
    /* GELU times up, in place. */
    for (size_t i = 0; i < (size_t)tokens * cols; i++)
        want[i] = dg_gelu(x[i]) * add[i];
    a->gelu_mul(dz, dx, dadd, (size_t)tokens * cols);
    a->to_host(got, dz, xb);
    DG_ACCEL_SAME("GELU", got, want, xb);
    /* Gather tokens * k rows, then the MoE tail over them. */
    float *gathered = xmalloc(xb * k), *eo = xmalloc(xb * k);
    for (int i = 0; i < tokens * k; i++)
        memcpy(gathered + (size_t)i * cols, x + (size_t)index[i] * cols, (size_t)cols * 4);
    a->gather(dg, dx, dindex, tokens * k, cols);
    a->to_host(eo, dg, xb * k);
    DG_ACCEL_SAME("gather", eo, gathered, xb * k);
    for (size_t i = 0; i < (size_t)tokens * k * cols; i++)
        eo[i] = jb_rng_unit(rs);
    a->to_device(deo, eo, xb * k);
    float layer_scalar = 0.75f, *mo = xmalloc((size_t)cols * 4), *sum = xmalloc((size_t)cols * 4);
    DGTensor post2 = {0};
    post2.data = p2;
    for (int t = 0; t < tokens; t++) {
        float *r = want + (size_t)t * cols;
        memcpy(r, x + (size_t)t * cols, (size_t)cols * 4);
        memset(mo, 0, (size_t)cols * 4);
        for (int j = 0; j < k; j++) {
            const float *src = eo + (size_t)slot[t * k + j] * cols;
            for (int i = 0; i < cols; i++)
                mo[i] += wt[t * k + j] * src[i];
        }
        dg_rms_ref(mo, mo, &post2, cols);
        for (int i = 0; i < cols; i++)
            sum[i] = add[(size_t)t * cols + i] + mo[i];
        dg_rms_ref(sum, sum, &scale, cols);
        for (int i = 0; i < cols; i++) {
            r[i] += sum[i];
            r[i] *= layer_scalar;
        }
    }
    a->to_device(dz, x, xb);
    a->moe_tail(dz, deo, dslot, dwt, dadd, dp2, dsc, layer_scalar, tokens);
    a->to_host(got, dz, xb);
    DG_ACCEL_SAME("MoE tail", got, want, xb);
    DG_ACCEL_CLOSE(got, want, (size_t)tokens * cols);
#undef DG_ACCEL_SAME
#undef DG_ACCEL_CLOSE
    DGDevice all[] = {dw, dsc, dfac, dp2, dx, dadd, dy, dz, dg, deo, dindex, dslot, dwt};
    for (size_t i = 0; i < sizeof all / sizeof *all; i++)
        a->free(all[i]);
    void *host[] = {w,    sc, fac,   p2,   x,        add, want, got, y,
                    yref, wt, index, slot, gathered, eo,  mo,   sum};
    for (size_t i = 0; i < sizeof host / sizeof *host; i++)
        jb_release(host[i]);
}

/* The accelerator's rotary embedding and attention against the reference's
 * loops, on a local-layer or full-layer shape: two segments sharing one
 * cache, and one without, in canvas and causal modes. Bit for bit in strict
 * builds. */
static void dg_test_accel_attention_shape(const DGAccelOps *a, uint64_t *rs, int full) {
    enum { segments = 3, seq = 4, n = segments * seq };

    int hd = full ? DG_FULL_HEAD_DIM : DG_LOCAL_HEAD_DIM;
    int kvh = full ? DG_FULL_KV_HEADS : DG_LOCAL_KV_HEADS, qn = DG_HEADS * hd, kn = kvh * hd,
        half = hd / 2;
    int seg_old[segments] = {6, 6, 0}, seg_base[segments] = {0, 0, 6}, rows = 6;
    size_t qb = (size_t)n * qn * 4, kb = (size_t)n * kn * 4, ob = (size_t)rows * kn * 4,
           tb = (size_t)n * DG_FULL_HEAD_DIM * 4;
    float *q = xmalloc(qb), *k = xmalloc(kb), *v = xmalloc(kb), *ko = xmalloc(ob),
          *vo = xmalloc(ob), *table = xcalloc((size_t)n * DG_FULL_HEAD_DIM, 4), *want = xmalloc(qb),
          *got = xmalloc(qb), *ts = xmalloc((size_t)(rows + seq) * 4);
    for (size_t i = 0; i < (size_t)n * qn; i++)
        q[i] = jb_rng_unit(rs);
    for (size_t i = 0; i < (size_t)n * kn; i++) {
        k[i] = jb_rng_unit(rs);
        v[i] = jb_rng_unit(rs);
    }
    for (size_t i = 0; i < (size_t)rows * kn; i++) {
        ko[i] = jb_rng_unit(rs);
        vo[i] = jb_rng_unit(rs);
    }
    for (int t = 0; t < n; t++)
        for (int i = 0; i < half; i++) {
            table[(size_t)t * DG_FULL_HEAD_DIM + i] = cosf((float)(t * 7 + i));
            table[(size_t)t * DG_FULL_HEAD_DIM + half + i] = sinf((float)(t * 7 + i));
        }
    DGDevice dq = a->alloc(qb), dk = a->alloc(kb), dv = a->alloc(kb), dko = a->alloc(ob),
             dvo = a->alloc(ob), dt = a->alloc(tb), dout = a->alloc(qb),
             dso = a->alloc(sizeof seg_old), dsb = a->alloc(sizeof seg_base),
             dkd = a->alloc((size_t)(n + rows) * kn * 4);
    if (!dq || !dk || !dv || !dko || !dvo || !dt || !dout || !dso || !dsb || !dkd)
        die2("accelerator self-test failed: device memory", a->name);
    a->to_device(dk, k, kb);
    a->to_device(dv, v, kb);
    a->to_device(dko, ko, ob);
    a->to_device(dvo, vo, ob);
    a->to_device(dt, table, tb);
    a->to_device(dso, seg_old, sizeof seg_old);
    a->to_device(dsb, seg_base, sizeof seg_base);
    /* Rotary embedding. */
    a->to_device(dq, q, qb);
    a->rope(dq, dt, n, DG_HEADS, hd, qn);
    for (int t = 0; t < n; t++)
        dg_rope(q + (size_t)t * qn, DG_HEADS, hd, table + (size_t)t * DG_FULL_HEAD_DIM,
                table + (size_t)t * DG_FULL_HEAD_DIM + half);
    a->to_host(got, dq, qb);
    if (JB_STRICT_MATH && memcmp(got, q, qb))
        die2("accelerator self-test failed: differs from the reference", "rotary embedding");
    /* Attention, canvas then causal. */
    for (int canvas = 1; canvas >= 0; canvas--) {
        for (int ti = 0; ti < n; ti++)
            for (int h = 0; h < DG_HEADS; h++) {
                int s = ti / seq, t = ti % seq, old = seg_old[s], kh = h / (DG_HEADS / kvh);
                int end = canvas ? old + seq : old + t + 1;
                const float *qq = q + (size_t)ti * qn + (size_t)h * hd;
                float mx = -FLT_MAX, den = 0, *oo = want + (size_t)ti * qn + (size_t)h * hd;
                for (int j = 0; j < end; j++) {
                    const float *kk = j < old ? ko + ((size_t)seg_base[s] + j) * kn
                                              : k + ((size_t)s * seq + j - old) * kn;
                    ts[j] = (float)dg_dot_ref(qq, kk + (size_t)kh * hd, hd);
                    if (ts[j] > mx)
                        mx = ts[j];
                }
                for (int j = 0; j < end; j++) {
                    ts[j] = dg_expf(ts[j] - mx);
                    den += ts[j];
                }
                memset(oo, 0, (size_t)hd * 4);
                for (int j = 0; j < end; j++) {
                    float p = ts[j] / den;
                    const float *vv = j < old ? vo + ((size_t)seg_base[s] + j) * kn
                                              : v + ((size_t)s * seq + j - old) * kn;
                    for (int d = 0; d < hd; d++)
                        oo[d] += p * vv[(size_t)kh * hd + d];
                }
            }
        a->attention(dout, dq, dk, dv, dko, dvo, dso, dsb, n, seq, kvh, hd, canvas, full,
                     rows + seq, rows, dkd);
        a->to_host(got, dout, qb);
        /* Fast builds' accelerator sums scores in FP32 on tensor cores,
         * where the reference sums in double. The test's values average 1,
         * so scores reach several hundred and FP32 sums of them carry
         * about 1e-4 of absolute error into the softmax. */
        double tolerance = JB_STRICT_MATH ? 1e-5 : 5e-4;
        for (size_t i = 0; i < (size_t)n * qn; i++)
            jb_check_close(a->name, got[i], want[i], tolerance * (1 + fabs(want[i])));
        if (JB_STRICT_MATH && memcmp(got, want, qb))
            die2("accelerator self-test failed: differs from the reference", "attention");
    }
    DGDevice all[] = {dq, dk, dv, dko, dvo, dt, dout, dso, dsb, dkd};
    for (size_t i = 0; i < sizeof all / sizeof *all; i++)
        a->free(all[i]);
    void *host[] = {q, k, v, ko, vo, table, want, got, ts};
    for (size_t i = 0; i < sizeof host / sizeof *host; i++)
        jb_release(host[i]);
}

static void dg_test_accel_attention(uint64_t *rs) {
    const DGAccelOps *a = dg_accel();
    if (!a)
        return;
    dg_test_accel_attention_shape(a, rs, 0);
    dg_test_accel_attention_shape(a, rs, 1);
}

/* The accelerator's routing against dg_route_token, with tied logits, and
 * its grouping against dg_ff's row order: rows by expert, then token, then
 * slot. */
static void dg_test_accel_routing(uint64_t *rs) {
    const DGAccelOps *a = dg_accel();
    if (!a)
        return;

    enum { tokens = 37, total = tokens * DG_TOPK };

    float *logits = xmalloc((size_t)tokens * DG_EXPERTS * 4), *tw = xmalloc(total * 4),
          *gw = xmalloc(total * 4);
    int *top = xmalloc(total * sizeof *top), *gtop = xmalloc(total * sizeof *gtop),
        *index = xmalloc(total * sizeof *index), *slot = xmalloc(total * sizeof *slot),
        *gindex = xmalloc(total * sizeof *gindex), *gslot = xmalloc(total * sizeof *gslot);
    uint8_t *scale = xmalloc(DG_EXPERTS * 2);
    for (int i = 0; i < tokens * DG_EXPERTS; i++)
        logits[i] = 4 * jb_rng_unit(rs);
    for (int t = 0; t < tokens; t += 3) /* ties, which go to the lower expert */
        logits[(size_t)t * DG_EXPERTS + 17] = logits[(size_t)t * DG_EXPERTS + 90] = 9.0f;
    for (int e = 0; e < DG_EXPERTS; e++)
        jb_put_bf16(scale + e * 2, 1 + jb_rng_unit(rs) / 4);
    DGTensor re = {0};
    re.data = scale;
    for (int t = 0; t < tokens; t++)
        dg_route_token(logits + (size_t)t * DG_EXPERTS, &re, top + t * DG_TOPK, tw + t * DG_TOPK);
    int rows = 0;
    for (int e = 0; e < DG_EXPERTS; e++)
        for (int t = 0; t < tokens; t++)
            for (int k = 0; k < DG_TOPK; k++)
                if (top[t * DG_TOPK + k] == e) {
                    index[rows] = t;
                    slot[t * DG_TOPK + k] = rows++;
                }
    size_t lb = (size_t)tokens * DG_EXPERTS * 4, ib = total * sizeof(int);
    DGDevice dl = a->alloc(lb), ds = a->alloc(DG_EXPERTS * 2), dt = a->alloc(ib),
             dw = a->alloc(total * 4), df = a->alloc(4), di = a->alloc(ib), dsl = a->alloc(ib),
             dtl = a->alloc(a->nvfp4_scratch(total)), dn = a->alloc(4);
    if (!dl || !ds || !dt || !dw || !df || !di || !dsl || !dtl || !dn)
        die2("accelerator self-test failed: device memory", a->name);
    int flag = 0, ntiles = 0;
    a->to_device(dl, logits, lb);
    a->to_device(ds, scale, DG_EXPERTS * 2);
    a->to_device(df, &flag, sizeof flag);
    a->route(dl, ds, tokens, dt, dw, df);
    a->group(dt, tokens, di, dsl, dtl, dn);
    a->to_host(gtop, dt, ib);
    a->to_host(gw, dw, total * 4);
    a->to_host(gindex, di, ib);
    a->to_host(gslot, dsl, ib);
    a->to_host(&flag, df, sizeof flag);
    a->to_host(&ntiles, dn, sizeof ntiles);
    if (flag || memcmp(gtop, top, ib) || ntiles < 1 || ntiles > total)
        die2("accelerator self-test failed: routing", a->name);
    if (JB_STRICT_MATH && memcmp(gw, tw, total * 4))
        die2("accelerator self-test failed: differs from the reference", "routing weights");
    if (memcmp(gindex, index, ib) || memcmp(gslot, slot, ib))
        die2("accelerator self-test failed: expert grouping", a->name);
    DGDevice all[] = {dl, ds, dt, dw, df, di, dsl, dtl, dn};
    for (size_t i = 0; i < sizeof all / sizeof *all; i++)
        a->free(all[i]);
    void *host[] = {logits, tw, gw, top, gtop, index, slot, gindex, gslot, scale};
    for (size_t i = 0; i < sizeof host / sizeof *host; i++)
        jb_release(host[i]);
}

/* Distance in float units in the last place from a double reference. */
static double jb_float_ulps(float got, double want) {
    double spacing = fabs(want) < FLT_MIN ? 0x1p-149 : ldexp(1.0, ilogb(want) - 23);
    return fabs((double)got - want) / spacing;
}

/* dg_expf and dg_tanhf against double libm over a sweep of every float
 * exponent, for results in the normal range, which fast math's
 * flush-to-zero leaves alone; and a hash of their bits, which strict builds
 * must reproduce on every platform and compiler, as GCC, Clang and MSVC on
 * x86-64 and GCC on AArch64 did when it was recorded. */
static void dg_test_portable_math(void) {
    uint64_t hash = 1469598103934665603u;
    double worst_exp = 0, worst_tanh = 0;
    for (uint32_t u = 0; u < 0xffffffffu - 4093; u += 4093) {
        if (((u >> 23) & 255) == 255)
            continue;
        float x, e, t;
        memcpy(&x, &u, sizeof x);
        e = dg_expf(x);
        t = dg_tanhf(x);
        uint32_t be, bt;
        memcpy(&be, &e, sizeof be);
        memcpy(&bt, &t, sizeof bt);
        hash = (hash ^ be) * 1099511628211u;
        hash = (hash ^ bt) * 1099511628211u;
        if (x > -87.0f && x < 88.7f) {
            double err = jb_float_ulps(e, exp((double)x));
            if (err > worst_exp)
                worst_exp = err;
        }
        double err = jb_float_ulps(t, tanh((double)x));
        if (err > worst_tanh)
            worst_tanh = err;
    }
    /* Fast-math builds test the C library's functions, which make no
     * one-ulp promise: glibc's tanhf is 2.07 ulp off there. */
    if (worst_exp > (JB_STRICT_MATH ? 1.0 : 4.0) || worst_tanh > (JB_STRICT_MATH ? 1.5 : 4.0))
        die("portable math self-test failed: accuracy");
        /* Fast math assumes no infinities and ignores the sign of zero. */
#if JB_STRICT_MATH
    if (dg_expf(0) != 1 || dg_expf(-INFINITY) != 0 || dg_expf(INFINITY) != INFINITY ||
        dg_tanhf(INFINITY) != 1 || dg_tanhf(-INFINITY) != -1 || !signbit(dg_tanhf(-0.0f)))
        die("portable math self-test failed: special values");
    if (hash != 0x5f90bd09fe8f8bdeu)
        die("portable math self-test failed: bits differ from the recorded ones");
    /* dg_exp: 2^20 arguments over its whole finite range, within 2 ulp of
     * the C library's exp where the result is normal, and bits that GCC,
     * Clang and MSVC on x86-64 and GCC on AArch64 all reproduced. */
    uint64_t exp_hash = 1469598103934665603u;
    double worst = 0;
    for (int i = 0; i < 1 << 20; i++) {
        double x = -746.0 + 1456.0 * i / (1 << 20), y = dg_exp(x), want = exp(x);
        uint64_t b = dg_bits_double(y);
        exp_hash = (exp_hash ^ b) * 1099511628211u;
        if (want > 0x1p-1022 && want < 0x1p1023) {
            double err = fabs(y - want) / ldexp(1.0, ilogb(want) - 52);
            if (err > worst)
                worst = err;
        }
    }
    if (worst > 2.0 || dg_exp(0) != 1 || dg_exp(-INFINITY) != 0 || dg_exp(INFINITY) != INFINITY)
        die("portable math self-test failed: exp");
    if (exp_hash != 0x22cedbf2137180efu)
        die("portable math self-test failed: exp bits differ from the recorded ones");
#else
    (void)hash;
#endif
}

/* Batched attention dots, over key counts around the kernels' key groups
 * and lengths around their column groups, with keys stored apart. */
static void dg_test_dots(uint64_t *rs) {
    static const int key_count[] = {1, 3, 7, 8, 9, 17}, len[] = {1, 3, 4, 5, 17, 256};
    for (size_t kc = 0; kc < sizeof key_count / sizeof *key_count; kc++)
        for (size_t k = 0; k < sizeof len / sizeof *len; k++) {
            int keys = key_count[kc], n = len[k];
            size_t stride = (size_t)n + 5;
            float *a = xmalloc((size_t)n * 4), *b = xmalloc(stride * keys * 4),
                  *got = xmalloc((size_t)keys * 4), *ref = xmalloc((size_t)keys * 4);
            for (int i = 0; i < n; i++)
                a[i] = 4 * jb_rng_unit(rs);
            for (size_t i = 0; i < stride * keys; i++)
                b[i] = jb_rng_unit(rs);
            jb_poison(got, (size_t)keys);
            dg_dots(a, b, stride, keys, n, got);
            for (int j = 0; j < keys; j++) {
                const float *bj = b + (size_t)j * stride;
                double want = dg_dot_ref(a, bj, n), mag = 0;
                for (int i = 0; i < n; i++)
                    mag += fabs((double)a[i] * bj[i]);
                ref[j] = (float)want;
                jb_check_close(dg_kernels()->name, got[j], want, 1e-6 * fabs(want) + 1e-12 * mag);
            }
            jb_check_reference_bits("attention dots", got, ref, (size_t)keys * 4);
            jb_release(a);
            jb_release(b);
            jb_release(got);
            jb_release(ref);
        }
}

/* A non-finite activation must come back through the error frame, even when
 * the kernel runs its blocks in parallel and a worker thread finds it. */
static void dg_test_nonfinite_activation(void) {
    enum { tokens = 8 };

    uint32_t nan_bits = 0x7fc00000u;
    float *in = xcalloc((size_t)tokens * DG_H, sizeof *in);
    float *out = xmalloc((size_t)tokens * DG_H * sizeof *out);
    memcpy(&in[(size_t)tokens * DG_H - 1], &nan_bits, sizeof nan_bits);
    JBErrorFrame frame;
    jb_frame_enter(&frame, JB_ERROR_REQUEST);
    if (!setjmp(frame.jump)) {
        dg_nvfp4_qdq(out, in, tokens, DG_H, 1.0f);
        jb_frame_leave(&frame);
        die("kernel self-test failed: non-finite NVFP4 activation accepted");
    }
    jb_frame_fail(&frame);
    if (strcmp(jb_error_message, "non-finite NVFP4 activation"))
        die2("kernel self-test failed: unexpected error", jb_error_message);
    jb_release(in);
    jb_release(out);
}

static void dg_kernel_selftest(void) {
    uint64_t rs = 0x9e3779b97f4a7c15ull;
    dg_test_mm(&rs);
    dg_test_nvfp4(&rs);
    dg_test_rms_dot(&rs);
    dg_test_dots(&rs);
    dg_test_portable_math();
    dg_test_accel_layer(&rs);
    dg_test_accel_attention(&rs);
    dg_test_accel_routing(&rs);
    dg_test_nonfinite_activation();
}

/* Kernel throughput at model shapes, for comparing ISA paths and builds.
 * "read" is a streaming-read baseline for this machine's memory bandwidth. */
static double jb_best_ms(uint64_t *ns, int reps) {
    uint64_t best = UINT64_MAX;
    for (int i = 0; i < reps; i++)
        if (ns[i] < best)
            best = ns[i];
    return (double)best / 1e6;
}

static int bench_kernels(void) {
    int use_exact = !strcmp(dg_kernels()->name, "avx512-exact");
    (void)use_exact;
    printf("{\"bench\":\"info\",\"kernels\":\"%s\",\"math\":\"%s\",\"threads\":%d}\n",
           dg_kernels()->name, JB_MATH_MODE, jb_threads());
    uint64_t rs = 0x243f6a8885a308d3ull, ns[3];
    size_t nw = (size_t)1 << 26;
    uint64_t *buf = xmalloc(nw * sizeof *buf), acc = 0;
    for (size_t i = 0; i < nw; i++)
        buf[i] = i;
    for (int rep = 0; rep < 3; rep++) {
        uint64_t t0 = now_ns(), sum = 0;
        {
            ptrdiff_t i;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum) schedule(static)
#endif
            for (i = 0; i < (ptrdiff_t)nw; i++)
                sum += buf[i];
        }
        ns[rep] = now_ns() - t0;
        acc += sum;
    }
    jb_release(buf);
    double ms = jb_best_ms(ns, 3);
    printf("{\"bench\":\"read\",\"bytes\":%zu,\"ms\":%.3f,\"gb_per_s\":%.2f,\"check\":%llu}\n",
           nw * sizeof(uint64_t), ms, (double)(nw * sizeof(uint64_t)) / ms / 1e6,
           (unsigned long long)(acc & 0xff));

    static const int mm_tokens[] = {1, 8, 64, 256};
    int rows = 4096, cols = DG_H;
    uint8_t *w = xmalloc((size_t)rows * cols * 2);
    float *x = xmalloc((size_t)256 * cols * 4), *y = xmalloc((size_t)256 * rows * 4);
    for (size_t i = 0; i < (size_t)rows * cols; i++)
        jb_put_bf16(w + i * 2, jb_rng_unit(&rs));
    for (size_t i = 0; i < (size_t)256 * cols; i++)
        x[i] = jb_rng_unit(&rs);
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
    DGTensor bw = {0};
    bw.data = w;
    if (use_exact) {
        bw.exact = xmalloc((size_t)rows * cols * 2);
        bw.shape[0] = (uint64_t)rows;
        bw.shape[1] = (uint64_t)cols;
        dg_exact_pack_bf16(&bw);
    }
#endif
    for (size_t k = 0; k < sizeof mm_tokens / sizeof *mm_tokens; k++) {
        int tokens = mm_tokens[k];
        for (int kernel = 0; kernel < 2; kernel++) {
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
            if (kernel && !use_exact && dg_kernels()->bf16_mm == dg_mm_data_ref)
#else
            if (kernel && dg_kernels()->bf16_mm == dg_mm_data_ref)
                break;
#endif
                dg_mm_data_ref(w, x, y, tokens, rows, cols);
            for (int rep = 0; rep < 3; rep++) {
                uint64_t t0 = now_ns();
                if (kernel) {
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
                    if (use_exact)
                        dg_mm_exact_avx512(bw.exact, x, y, tokens, rows, cols);
                    else
                        dg_mm_data(w, x, y, tokens, rows, cols);
#else
                    dg_mm_data(w, x, y, tokens, rows, cols);
#endif
                } else
                    dg_mm_data_ref(w, x, y, tokens, rows, cols);
                ns[rep] = now_ns() - t0;
            }
            ms = jb_best_ms(ns, 3);
            printf("{\"bench\":\"bf16_mm\",\"kernel\":\"%s\",\"rows\":%d,\"cols\":%d,\"tokens\":%d,"
                   "\"ms\":%.3f,\"gflops\":%.2f,\"weight_gb_per_s\":%.2f}\n",
                   kernel ? dg_kernels()->name : "ref", rows, cols, tokens, ms,
                   2.0 * rows * cols * tokens / ms / 1e6, 2.0 * rows * cols / ms / 1e6);
        }
    }
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
    if (bw.exact)
        jb_release((void *)bw.exact);
#endif
    jb_release(w);
    jb_release(x);
    jb_release(y);

    /* All 128 experts of one gate projection, so weights stream from memory
     * as in the model rather than staying cache-resident. */
    static const int nv_tokens[] = {1, 4, 16, 64};
    int experts = DG_EXPERTS, nr = DG_MOE, nc = DG_H;
    size_t wbytes = (size_t)nr * nc / 2, sbytes = (size_t)nr * nc / 16;
    uint8_t *wd = xmalloc(wbytes * experts), *sd = xmalloc(sbytes * experts);
    float *xr = xmalloc((size_t)64 * nc * 4), *xs = xmalloc((size_t)64 * nc * 4),
          *xq = xmalloc((size_t)64 * nc * 4), *yo = xmalloc((size_t)64 * nr * 4);
    for (size_t i = 0; i < wbytes * experts; i++)
        wd[i] = (uint8_t)jb_rng(&rs);
    for (size_t i = 0; i < sbytes * experts; i++)
        sd[i] = (uint8_t)(0x30 + jb_rng(&rs) % 32);
    for (size_t i = 0; i < (size_t)64 * nc; i++)
        xr[i] = 3 * jb_rng_unit(&rs);
    dg_nvfp4_qdq_ref(xq, xr, 64, nc, 0.02f);
    dg_nvfp4_qdq(xs, xr, 64, nc, 0.02f);
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
    size_t exact_sbytes = (size_t)nr * nc / 16;
    uint8_t *exact_w = use_exact ? xmalloc(wbytes * experts) : NULL;
    uint8_t *exact_s = use_exact ? xmalloc(exact_sbytes * experts) : NULL;
    const float exact_global = 0.01f;
    for (int e = 0; use_exact && e < experts; e++) {
        DGTensor tw = {0}, ts = {0};
        tw.data = wd + wbytes * e;
        tw.exact = exact_w + wbytes * e;
        ts.data = sd + sbytes * e;
        ts.exact = exact_s + exact_sbytes * e;
        dg_exact_pack_nvfp4(&tw, &ts, nr, nc);
    }
#endif
    for (size_t k = 0; k < sizeof nv_tokens / sizeof *nv_tokens; k++) {
        int tokens = nv_tokens[k];
        for (int kernel = 0; kernel < 2; kernel++) {
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
            if (kernel && !use_exact && dg_kernels()->nvfp4_mm == dg_nvfp4_mm_ref)
#else
            if (kernel && dg_kernels()->nvfp4_mm == dg_nvfp4_mm_ref)
                break;
#endif
                for (int rep = 0; rep < 3; rep++) {
                    uint64_t t0 = now_ns();
                    for (int e = 0; e < experts; e++) {
                        if (kernel) {
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
                            if (use_exact)
                                dg_nvfp4_exact_avx512(exact_w + wbytes * e,
                                                      exact_s + exact_sbytes * e, exact_global, xq,
                                                      yo, tokens, nr, nc);
                            else
                                dg_kernels()->nvfp4_mm(wd + wbytes * e, sd + sbytes * e, 0.01f, xs,
                                                       yo, tokens, nr, nc);
#else
                        dg_kernels()->nvfp4_mm(wd + wbytes * e, sd + sbytes * e, 0.01f, xs, yo,
                                               tokens, nr, nc);
#endif
                        } else
                            dg_nvfp4_mm_ref(wd + wbytes * e, sd + sbytes * e, 0.01f, xq, yo, tokens,
                                            nr, nc);
                    }
                    ns[rep] = now_ns() - t0;
                }
            ms = jb_best_ms(ns, 3);
            printf("{\"bench\":\"nvfp4_mm\",\"kernel\":\"%s\",\"experts\":%d,\"rows\":%d,\"cols\":%"
                   "d,\"tokens\":%d,\"ms\":%.3f,\"gflops\":%.2f,\"weight_gb_per_s\":%.2f}\n",
                   kernel ? dg_kernels()->name : "ref", experts, nr, nc, tokens, ms,
                   2.0 * experts * nr * nc * tokens / ms / 1e6,
                   (double)(wbytes + sbytes) * experts / ms / 1e6);
        }
    }
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
    if (exact_w)
        jb_release(exact_w);
    if (exact_s)
        jb_release(exact_s);
#endif
    jb_release(wd);
    jb_release(sd);
    jb_release(xr);
    jb_release(xs);
    jb_release(xq);
    jb_release(yo);
    return 0;
}

/* Numbers in results must use '.' even after a host sets a comma-decimal
 * locale. Skipped where no such locale is installed. */
static void dg_test_locale_numbers(void) {
    static const char *const names[] = {"de_DE.UTF-8", "de_DE.utf8",  "de_DE",
                                        "de-DE",       "fr_FR.UTF-8", "fr-FR"};
    const char *current = setlocale(LC_NUMERIC, NULL);
    char restore[128];
    snprintf(restore, sizeof restore, "%s", current ? current : "C");
    for (size_t i = 0; i < sizeof names / sizeof *names; i++) {
        if (!setlocale(LC_NUMERIC, names[i]))
            continue;
        if (localeconv()->decimal_point[0] == '.')
            continue;
        DGBuf b = {0};
        db_number(&b, 0.25);
        db_ch(&b, ' ');
        db_millis(&b, 1.5);
        int ok = !strcmp(b.p, "0.25 1.500");
        jb_release(b.p);
        setlocale(LC_NUMERIC, restore);
        if (!ok)
            die2("locale number self-test failed", names[i]);
        return;
    }
    setlocale(LC_NUMERIC, restore);
}

static int selftest(void) {
    uint64_t clock_previous = now_ns();
    for (int i = 0; i < 1000; i++) {
        uint64_t clock_current = now_ns();
        if (clock_current < clock_previous)
            die("monotonic clock self-test failed");
        clock_previous = clock_current;
    }
    size_t arena_bytes = 0;
    jb_workspace_plan(&arena_bytes, 7, sizeof(char));
    jb_workspace_plan(&arena_bytes, 3, sizeof(uint32_t));
    JBArena arena;
    jb_arena_init(&arena, arena_bytes);
    unsigned char *arena_bytes_out = jb_arena_alloc(&arena, 7, sizeof *arena_bytes_out, 0);
    uint32_t *arena_words = jb_arena_alloc(&arena, 3, sizeof *arena_words, 1);
    if ((uintptr_t)arena_bytes_out % 64 || (uintptr_t)arena_words % 64 || arena_words[0] ||
        arena_words[1] || arena_words[2] || arena.peak != arena.size)
        die("arena self-test failed");
    jb_arena_reset(&arena, 0);
    jb_arena_free(&arena);

    DGWorkspace reuse = {0};
    JBArena *first = dg_workspace_begin(&reuse, 128);
    (void)jb_arena_alloc(first, 128, 1, 0);
    jb_arena_reset(first, 0);
    dg_workspace_end(&reuse);
    void *first_allocation = reuse.arena.allocation;
    JBArena *second = dg_workspace_begin(&reuse, 64);
    (void)jb_arena_alloc(second, 64, 1, 0);
    jb_arena_reset(second, 0);
    dg_workspace_end(&reuse);
    if (reuse.arena.allocation != first_allocation || reuse.capacity != 128)
        die("workspace reuse self-test failed");
    JBArena *grown = dg_workspace_begin(&reuse, 256);
    (void)jb_arena_alloc(grown, 256, 1, 0);
    jb_arena_reset(grown, 0);
    dg_workspace_end(&reuse);
    if (reuse.capacity != 256)
        die("workspace growth self-test failed");

    JBArena *control = dg_control_begin(&reuse, 128);
    (void)jb_arena_alloc(control, 128, 1, 0);
    dg_control_end(&reuse);
    void *control_allocation = reuse.control_arena.allocation;
    control = dg_control_begin(&reuse, 64);
    (void)jb_arena_alloc(control, 64, 1, 0);
    dg_control_end(&reuse);
    if (reuse.control_arena.allocation != control_allocation || reuse.control_capacity != 128)
        die("control workspace reuse self-test failed");

    DGKV kv_test[DG_L * 2] = {0};
    int kv_lengths[2] = {2, 3};
    dg_worker_kv_begin(&reuse, kv_test, 2, kv_lengths);
    void *kv_allocation = reuse.kv_arena.allocation;
    for (int d = 0; d < 2; d++)
        for (int l = 0; l < DG_L; l++) {
            DGKV *one = &kv_test[d * DG_L + l];
            size_t last = (size_t)one->n * dg_kv_width(l) - 1;
            one->k[0] = (float)(d + l);
            one->v[last] = (float)(d - l);
        }
    dg_worker_kv_end(&reuse, kv_test, 2);
    int shorter = 1;
    dg_worker_kv_begin(&reuse, kv_test, 1, &shorter);
    if (reuse.kv_arena.allocation != kv_allocation)
        die("K/V workspace reuse self-test failed");
    dg_worker_kv_end(&reuse, kv_test, 1);
    dg_workspace_destroy(&reuse);

    uint8_t sh[32], sha_abc[32] = {0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40,
                                   0xde, 0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17,
                                   0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
    dg_sha256((const uint8_t *)"abc", 3, sh);
    if (memcmp(sh, sha_abc, 32))
        die("SHA-256 self-test failed");
    DGMT mt;
    dg_mt_seed(&mt, 0);
    uint32_t py_mt[5] = {201979, 220500, 21225, 135746, 254766};
    for (int i = 0; i < 5; i++)
        if (dg_mt_vocab(&mt) != py_mt[i])
            die("MT19937 self-test failed");
    const char *cj = "{\"state\":{\"b\":2,\"a\":\"é\"},\"questions\":{\"q\":{\"type\":\"noul\","
                     "\"instructions\":\"x\"}}}";
    int cnt;
    JTok *ct = json_tokens(cj, strlen(cj), &cnt);
    int cs = jt_obj_get(cj, ct, cnt, 0, "state"), cq = jt_obj_get(cj, ct, cnt, 0, "questions");
    DGBuf cb = {0};
    db_ch(&cb, '[');
    dg_json_value(&cb, cj, ct, cnt, cs, 1, 1);
    db_mem(&cb, ", ", 2);
    dg_json_value(&cb, cj, ct, cnt, cq, 1, 1);
    db_ch(&cb, ']');
    dg_sha256((const uint8_t *)cb.p, cb.n, sh);
    uint32_t cseed = (uint32_t)sh[0] << 24 | (uint32_t)sh[1] << 16 | (uint32_t)sh[2] << 8 | sh[3];
    if (cseed != 1123096946u)
        die("OpenJev canonical seed self-test failed");
    dg_mt_seed(&mt, cseed);
    if (dg_mt_vocab(&mt) != 238957u)
        die("OpenJev canvas RNG self-test failed");
    jb_release(cb.p);
    jb_release(ct);
    const char *tx = "{\"s\":\" \\u2003x\\u00a0 \",\"o\":{\"b\":2,\"a\":1},\"n\":null}";
    int txn;
    JTok *tt = json_tokens(tx, strlen(tx), &txn);
    int ts = jt_obj_get(tx, tt, txn, 0, "s"), to = jt_obj_get(tx, tt, txn, 0, "o"),
        tn = jt_obj_get(tx, tt, txn, 0, "n");
    char *sv = dg_text_of(tx, tt, txn, ts), *ov = dg_text_of(tx, tt, txn, to),
         *nv = dg_text_of(tx, tt, txn, tn);
    if (strcmp(sv, "x") || strcmp(ov, "{\"b\": 2, \"a\": 1}") || *nv)
        die("OpenJev text_of self-test failed");
    jb_release(sv);
    jb_release(ov);
    jb_release(nv);
    jb_release(tt);
    /* Build NaN and infinity from bits: under -ffinite-math-only the compiler
     * may assume no arithmetic or conversion produces them. */
    uint32_t nan_bits = 0x7fc00000u, inf_bits = 0x7f800000u;
    uint64_t dnan_bits = 0x7ff8000000000000ull, dinf_bits = 0x7ff0000000000000ull;
    float fnan, finf;
    double dnan, dinf;
    memcpy(&fnan, &nan_bits, 4);
    memcpy(&finf, &inf_bits, 4);
    memcpy(&dnan, &dnan_bits, 8);
    memcpy(&dinf, &dinf_bits, 8);
    if (jb_finitef(fnan) || jb_finitef(finf) || !jb_finitef(FLT_MAX) || jb_finite(dnan) ||
        jb_finite(dinf) || !jb_finite(DBL_MAX) || !jb_finite(0.0))
        die("finiteness self-test failed");
    const char *ek = "{\"st\\u0061te\":1}";
    int ekn;
    JTok *ekt = json_tokens(ek, strlen(ek), &ekn);
    if (jt_obj_get(ek, ekt, ekn, 0, "state") != 2)
        die("escaped JSON key self-test failed");
    jb_release(ekt);
    uint8_t one[2] = {0x80, 0x3f};
    if (dg_bf(one) != 1.0f)
        die("BF16 conversion self-test failed");
    for (int u = 0; u < 256; u++) {
        int e = (u >> 3) & 15, m = u & 7;
        float x = e ? ldexpf(1.0f + m / 8.0f, e - 7) : ldexpf((float)m, -9),
              got = dg_f8e4m3((uint8_t)u);
        x = u >> 7 ? -x : x;
        if (memcmp(&x, &got, sizeof x))
            die("E4M3 decode self-test failed");
    }
    if (dg_f8e4m3(0x01) != 0.001953125f || dg_f8e4m3(0x7e) != 448.0f ||
        dg_f8e4m3(0xfe) != -448.0f || dg_f8e4m3_round(5.25f) != 5.0f ||
        dg_e2m1_round(2.5f) != 2.0f || dg_e2m1_round(3.5f) != 4.0f)
        die("NVFP4 conversion self-test failed");
    double score[2] = {1000.0, 999.0}, prob[2];
    normalize_scores(score, 2, prob);
    if (fabs(prob[0] - 0.7310585786300049) > 1e-12 || fabs(prob[0] + prob[1] - 1.0) > 1e-15)
        die("candidate normalization self-test failed");
    double shifted[2] = {score[0] - 1234.5, score[1] - 1234.5}, shifted_prob[2];
    normalize_scores(shifted, 2, shifted_prob);
    if (shifted_prob[0] != prob[0] || shifted_prob[1] != prob[1])
        die("single-token log-softmax cancellation self-test failed");
    const char sample[] = "{\"state\":{\"ok\":true},\"candidates\":[\"yes\",\"no\"]}";
    int nt = 0;
    JTok *tokens = json_tokens(sample, sizeof sample - 1, &nt);
    int state = jt_obj_get(sample, tokens, nt, 0, "state");
    int candidates = jt_obj_get(sample, tokens, nt, 0, "candidates");
    if (nt < 7 || state < 0 || tokens[state].type != JT_OBJECT || candidates < 0 ||
        tokens[candidates].type != JT_ARRAY || tokens[candidates].size != 2)
        die("JSON self-test failed");
    jb_release(tokens);
    jb_model *missing_model = NULL;
    if (jb_model_load("__jev_bush_missing_model__", &missing_model) != JB_ERROR_IO || missing_model)
        die("public I/O status self-test failed");
    jb_model dummy_model = {0};
    dummy_model.allocator = jb_system_allocator;
    atomic_init(&dummy_model.references, 1);
    jb_session *public_session = NULL;
    if (jb_session_create_json(&dummy_model, "[]", 2, &public_session) != JB_ERROR_REQUEST ||
        public_session)
        die("public schema error self-test failed");
    if (jb_session_create_json(&dummy_model, "{}", 2, &public_session) != JB_ERROR_REQUEST ||
        public_session)
        die("empty schema acceptance self-test failed");
    static const char invalid_schema[] =
        "{\"route\":{\"type\":\"choice\",\"criteria\":{\"only\":\"One.\"}}}";
    if (jb_session_create_json(&dummy_model, invalid_schema, sizeof invalid_schema - 1,
                               &public_session) != JB_ERROR_REQUEST ||
        public_session)
        die("invalid fixed schema acceptance self-test failed");
    if (jb_session_create_json(&dummy_model, NULL, 0, &public_session) != JB_OK)
        die("public session self-test failed");
    const jb_input invalid_input = {{NULL, 0}, {"{", 1}, 0};
    jb_result *invalid_result = NULL;
    if (jb_session_decide(public_session, &invalid_input, &invalid_result) != JB_ERROR_REQUEST ||
        invalid_result)
        die("public request error self-test failed");
    jb_session_free(public_session);
    dg_kernel_selftest();
    dg_test_locale_numbers();
    dg_report_accel();
    if (dg_accel())
        printf("{\"selftest\":\"ok\",\"kernels\":\"%s\",\"accelerator\":\"%s\"}\n",
               dg_kernels()->name, dg_accel()->name);
    else
        printf("{\"selftest\":\"ok\",\"kernels\":\"%s\"}\n", dg_kernels()->name);
    return 0;
}

static void usage(void) {
    fprintf(stderr,
            "Jev Bush %s -- Please clap.\nusage:\n"
            "  jb MODEL_DIR decide REQUEST.json\n"
            "  jb MODEL_DIR eval OPENJEV.jsonl\n"
            "  jb --check-request REQUEST.json\n"
            "  jb --selftest\n"
            "  jb --bench-kernels\n",
            JB_VERSION);
}

static int finish_output(int rc) {
    if (!rc)
        flush_output();
    return rc;
}

static int jb_cli_main(int ac, char **av) {
    if (ac == 2 && !strcmp(av[1], "--selftest"))
        return finish_output(selftest());
    if (ac == 2 && !strcmp(av[1], "--bench-kernels"))
        return finish_output(bench_kernels());
    if (ac == 3 && !strcmp(av[1], "--check-request"))
        return finish_output(dg_check_request(av[2]));
    if (ac < 3) {
        usage();
        return 2;
    }
    if (!strcmp(av[2], "decide") && ac == 4)
        return finish_output(dg_decide_file(av[1], av[3]));
    if (!strcmp(av[2], "eval") && ac == 4)
        return finish_output(dg_eval_file(av[1], av[3]));
    usage();
    return 2;
}

/* CLI code uses the same tracked cleanup as the library. This is the one
 * root frame for commands and self-tests; leaf helpers never terminate the
 * process directly during ordinary execution. */
int main(int ac, char **av) {
    JBErrorFrame frame;
    jb_frame_enter(&frame, JB_ERROR_INTERNAL);
    if (setjmp(frame.jump)) {
        char message[sizeof jb_error_message];
        memcpy(message, jb_error_message, sizeof message);
        jb_frame_abandon(&frame);
        fprintf(stderr, "jb: %s\n", message);
        return 2;
    }
    int status = jb_cli_main(ac, av);
    jb_frame_leave(&frame);
    return status;
}
#endif
