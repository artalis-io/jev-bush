static float dg_f8e4m3_round(float x) {
    if (!(x > 0))
        return 0;
    if (x >= 448)
        return 448;
    int lo = 0, hi = 126;
    while (lo + 1 < hi) {
        int m = (lo + hi) / 2;
        if (dg_f8e4m3((uint8_t)m) < x)
            lo = m;
        else
            hi = m;
    }
    float a = dg_f8e4m3((uint8_t)lo), b = dg_f8e4m3((uint8_t)hi), da = x - a, db = b - x;
    return da < db || (da == db && !(lo & 1)) ? a : b;
}

static float dg_e2m1_round(float x) {
    static const float q[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float a = fabsf(x);
    int best = 0;
    for (int i = 1; i < 8; i++) {
        float d = fabsf(a - q[i]), old = fabsf(a - q[best]);
        if (d < old || (d == old && !(i & 1)))
            best = i;
    }
    return signbit(x) ? -q[best] : q[best];
}

/* Kernels come in pairs: a portable *_ref version that is always compiled and
 * serves as the correctness reference, and an ISA-specific version selected
 * at compile time. The selftest checks both against a double-precision
 * oracle. */
static void dg_nvfp4_qdq_ref(float *out, const float *in, int tokens, int cols, float base) {
    if (!jb_finitef(base) || !(base > 0) || cols % 32)
        die("invalid NVFP4 activation scale");
    for (int t = 0; t < tokens; t++)
        for (int b = 0; b < cols / 16; b++) {
            const float *x = in + (size_t)t * cols + (size_t)b * 16;
            float *y = out + (size_t)t * cols + (size_t)b * 16, amax = 0;
            for (int k = 0; k < 16; k++) {
                if (!jb_finitef(x[k]))
                    die("non-finite NVFP4 activation");
                if (fabsf(x[k]) > amax)
                    amax = fabsf(x[k]);
            }
            float s = dg_f8e4m3_round((amax / 6) / base) * base;
            if (s == 0) {
                memset(y, 0, 16 * sizeof *y);
                continue;
            }
            for (int k = 0; k < 16; k++)
                y[k] = dg_e2m1_round(x[k] / s) * s;
        }
}

/* The reference quantizer's arithmetic, block for block, spread over the
 * OpenMP team: the vector backends use it, and it matches the reference byte
 * for byte. The scalar build keeps the serial reference as its oracle. */
static void dg_nvfp4_qdq_parallel(float *out, const float *in, int tokens, int cols, float base) {
    if (!jb_finitef(base) || !(base > 0) || cols % 32)
        die("invalid NVFP4 activation scale");
    int nb = cols / 16, blocks = tokens * nb, nonfinite = 0;
    /* die() must not run inside the parallel region: OpenMP workers have no
     * error frame, so it would exit the host process, and a longjmp out of
     * the region is undefined. Record the failure and report it afterwards. */
    {
        int z;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic, 16) if (blocks >= 256) reduction(| : nonfinite)
#endif
        for (z = 0; z < blocks; z++) {
            int t = z / nb, b = z % nb, bad = 0;
            const float *x = in + (size_t)t * cols + (size_t)b * 16;
            float *y = out + (size_t)t * cols + (size_t)b * 16, amax = 0;
            for (int k = 0; k < 16; k++) {
                bad |= !jb_finitef(x[k]);
                if (fabsf(x[k]) > amax)
                    amax = fabsf(x[k]);
            }
            if (bad) {
                nonfinite = 1;
                continue;
            }
            float s = dg_f8e4m3_round((amax / 6) / base) * base;
            if (s == 0) {
                memset(y, 0, 16 * sizeof *y);
                continue;
            }
            for (int k = 0; k < 16; k++)
                y[k] = dg_e2m1_round(x[k] / s) * s;
        }
    }
    if (nonfinite)
        die("non-finite NVFP4 activation");
}

/* Produces the activation layout the selected dg_nvfp4_mm kernel expects. */
static void dg_nvfp4_qdq(float *out, const float *in, int tokens, int cols, float base) {
    dg_kernels()->nvfp4_qdq(out, in, tokens, cols, base);
}

/* x in the dg_nvfp4_qdq_ref layout. */
static void dg_nvfp4_mm_ref(const uint8_t *wd, const uint8_t *sd, float global, const float *x,
                            float *y, int tokens, int rows, int cols) {
    {
        int r;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < rows; r++)
            for (int t = 0; t < tokens; t++) {
                const uint8_t *wp = wd + (uint64_t)r * cols / 2, *sp = sd + (uint64_t)r * cols / 16;
                const float *xp = x + (size_t)t * cols;
                float sum = 0;
                for (int b = 0; b < cols / 16; b++) {
                    float scale = dg_f8e4m3(sp[b]) * global;
                    const uint8_t *q = wp + (size_t)b * 8;
                    const float *a = xp + (size_t)b * 16;
                    for (int k = 0; k < 8; k++) {
                        uint8_t v = q[k];
                        int lo = v & 7, hi = (v >> 4) & 7;
                        float wl = (float[]){0, .5f, 1, 1.5f, 2, 3, 4, 6}[lo],
                              wh = (float[]){0, .5f, 1, 1.5f, 2, 3, 4, 6}[hi];
                        if (v & 8)
                            wl = -wl;
                        if (v & 128)
                            wh = -wh;
                        sum += scale * (wl * a[k * 2] + wh * a[k * 2 + 1]);
                    }
                }
                y[(size_t)t * rows + r] = sum;
            }
    }
}

/* expf, tanhf and the router's exp. Strict builds use these, from IEEE additions,
 * multiplications and divisions in a fixed order and nothing else: no libm
 * and no fused multiply-add, so every strict build, whatever its C library,
 * and the CUDA kernels compute the same bits. Both follow Cephes. Fast math
 * may reassociate the range reduction and flushes subnormals, so fast-math
 * builds, which make no claim to identical bits, keep the C library's. */
#if JB_STRICT_MATH
static float dg_float_bits(uint32_t u) {
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}

/* Rounds x log2(e) half away from zero to n, reduces x by n ln 2 in two
 * parts, the first exact for |n| <= 150, and scales a minimax polynomial by
 * 2^n, in two steps outside the normal exponent range. */
static float dg_expf(float x) {
    if (x != x)
        return x;
    if (x > 88.72283935546875f)
        return INFINITY;
    if (x < -103.97207641601562f)
        return 0;
    int e = (int)(x * 1.44269504088896341f + (x < 0 ? -0.5f : 0.5f));
    float n = (float)e;
    float r = x - n * 0.693359375f;
    r = r - n * -2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * (r * r) + r + 1.0f;
    if (e > 127)
        return p * dg_float_bits(254u << 23) * 2.0f;
    if (e < -126)
        return p * dg_float_bits((uint32_t)(e + 253) << 23) * 0x1p-126f;
    return p * dg_float_bits((uint32_t)(e + 127) << 23);
}

/* Odd polynomial below 0.625, 1 - 2 / (e^2|x| + 1) above, and +-1 from 10,
 * where the result rounds to 1. Tiny arguments return themselves, keeping
 * the sign of zero. */
static float dg_tanhf(float x) {
    float a = x < 0 ? -x : x;
    if (a < 0x1p-12f)
        return x;
    if (a < 0.625f) {
        float z = x * x, p = -5.70498872745e-3f;
        p = p * z + 2.06390887954e-2f;
        p = p * z - 5.37397155531e-2f;
        p = p * z + 1.33314422036e-1f;
        p = p * z - 3.33332819422e-1f;
        return p * z * x + x;
    }
    if (a >= 10.0f)
        return x < 0 ? -1.0f : 1.0f;
    float t = 1.0f - 2.0f / (dg_expf(a + a) + 1.0f);
    return x < 0 ? -t : t;
}

static double dg_double_bits(uint64_t u) {
    double x;
    memcpy(&x, &u, sizeof x);
    return x;
}

static uint64_t dg_bits_double(double x) {
    uint64_t u;
    memcpy(&u, &x, sizeof u);
    return u;
}

/* fdlibm's __ieee754_exp, under 1 ulp: x = k ln 2 + r in two parts, a
 * rational approximation of e^r, and k added to the exponent bits, scaled
 * in two steps below the normal range. The router's softmax uses it. */
static double dg_exp(double x) {
    static const double ln2_hi = 6.93147180369123816490e-01, ln2_lo = 1.90821492927058770002e-10,
                        inv_ln2 = 1.44269504088896338700e+00, p1 = 1.66666666666666019037e-01,
                        p2 = -2.77777777770155933842e-03, p3 = 6.61375632143793436117e-05,
                        p4 = -1.65339022054652515390e-06, p5 = 4.13813679705723846039e-08,
                        two_m1000 = 9.33263618503218878990e-302;
    uint64_t bits = dg_bits_double(x);
    uint32_t hx = (uint32_t)(bits >> 32) & 0x7fffffff;
    int negative = (int)(bits >> 63), k = 0;
    double hi = 0, lo = 0;
    if (hx >= 0x40862E42) {
        if (hx >= 0x7ff00000)
            return x != x ? x + x : negative ? 0.0 : x;
        if (x > 7.09782712893383973096e+02)
            return INFINITY;
        if (x < -7.45133219101941108420e+02)
            return 0.0;
    }
    if (hx > 0x3fd62e42) {
        if (hx < 0x3FF0A2B2) {
            hi = negative ? x + ln2_hi : x - ln2_hi;
            lo = negative ? -ln2_lo : ln2_lo;
            k = negative ? -1 : 1;
        } else {
            k = (int)(inv_ln2 * x + (negative ? -0.5 : 0.5));
            double t = k;
            hi = x - t * ln2_hi;
            lo = t * ln2_lo;
        }
        x = hi - lo;
    } else if (hx < 0x3e300000)
        return 1.0 + x;
    double t = x * x, c = x - t * (p1 + t * (p2 + t * (p3 + t * (p4 + t * p5)))), y;
    if (k == 0)
        return 1.0 - ((x * c) / (c - 2.0) - x);
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
    if (k >= -1021)
        return dg_double_bits(dg_bits_double(y) + ((uint64_t)(int64_t)k << 52));
    return dg_double_bits(dg_bits_double(y) + ((uint64_t)(int64_t)(k + 1000) << 52)) * two_m1000;
}
#else
static float dg_expf(float x) {
    return expf(x);
}

static float dg_tanhf(float x) {
    return tanhf(x);
}

static double dg_exp(double x) {
    return exp(x);
}
#endif

static float dg_gelu(float x) {
    return .5f * x * (1.0f + dg_tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
}

static DGNvMatrix dg_nvfp4_matrix(const DGTensor *w, const DGTensor *s, const DGTensor *g, int rows,
                                  int cols) {
    if (w->dtype != DG_U8 || w->nd != 2 || w->shape[0] != (uint64_t)rows ||
        w->shape[1] != (uint64_t)cols / 2 || s->dtype != DG_F8E4M3 || s->nd != 2 ||
        s->shape[0] != (uint64_t)rows || s->shape[1] != (uint64_t)cols / 16 || g->dtype != DG_F32 ||
        g->nd != 0 || g->bytes != 4 || cols % 32 || rows % 2)
        die2("bad NVFP4 expert tensor", w->name);
    DGNvMatrix m = {w->data, s->data, w->exact, s->exact, 0};
    memcpy(&m.global, g->data, 4);
    return m;
}

#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
static JB_TARGET_AVX512 void dg_mm_exact_avx512(const uint8_t *data, const float *x, float *y,
                                                int tokens, int rows, int cols);
static JB_TARGET_AVX512 void dg_nvfp4_exact_avx512(const uint8_t *wd, const uint8_t *sd,
                                                   float global, const float *x, float *y,
                                                   int tokens, int rows, int cols);
#endif

static void dg_nvfp4_product(const DGNvMatrix *m, const float *x, float *y, int tokens, int rows,
                             int cols) {
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
    if (m->exact_w && m->exact_s) {
        dg_nvfp4_exact_avx512(m->exact_w, m->exact_s, m->global, x, y, tokens, rows, cols);
        return;
    }
#endif
    dg_kernels()->nvfp4_mm(m->w, m->s, m->global, x, y, tokens, rows, cols);
}

/* x comes from dg_nvfp4_qdq, in the layout the selected kernel expects. */
static void dg_nvfp4_mm(const DGTensor *w, const DGTensor *s, const DGTensor *g, const float *x,
                        float *y, int tokens, int rows, int cols) {
    DGNvMatrix m = dg_nvfp4_matrix(w, s, g, rows, cols);
    dg_nvfp4_product(&m, x, y, tokens, rows, cols);
}

/* h = gelu(g) * h over n values, in chunks the faster cores can take more
 * of. */
static void dg_gelu_gate(const float *g, float *h, int n) {
    int i;
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(dynamic, 1024) if (n >= 4096)
#endif
    for (i = 0; i < n; i++)
        h[i] = dg_gelu(g[i]) * h[i];
}

/* The gated product as separate steps: two backend matrix products, then
 * GELU, for backends without a fused form. */
static void dg_nvfp4_gated_each(const DGNvMatrix *gate, const DGNvMatrix *up, const float *x,
                                float *g, float *h, int tokens, int rows, int cols) {
    dg_nvfp4_product(gate, x, g, tokens, rows, cols);
    dg_nvfp4_product(up, x, h, tokens, rows, cols);
    dg_gelu_gate(g, h, tokens * rows);
}

/* gelu(gate x) * (up x) for one expert; g is scratch. */
static void dg_nvfp4_gated(const DGNvExpert *v, const float *x, float *g, float *h, int tokens,
                           int rows, int cols) {
    DGNvMatrix gate = dg_nvfp4_matrix(v->wg, v->sg, v->gg, rows, cols),
               up = dg_nvfp4_matrix(v->wu, v->su, v->gu, rows, cols);
    dg_kernels()->nvfp4_gated(&gate, &up, x, g, h, tokens, rows, cols);
}

static void dg_mm_data_ref(const uint8_t *data, const float *x, float *y, int tokens, int rows,
                           int cols) {
    {
        int r;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < rows; r++) {
            float sum[JB_MAX_CTX];
            for (int t = 0; t < tokens; t++)
                sum[t] = 0;
            const uint8_t *p = data + (uint64_t)r * cols * 2;
            for (int c = 0; c < cols; c++) {
                float a = dg_bf(p + c * 2);
                for (int t = 0; t < tokens; t++)
                    sum[t] += a * x[(size_t)t * cols + c];
            }
            for (int t = 0; t < tokens; t++)
                y[(size_t)t * rows + r] = sum[t];
        }
    }
}

static void dg_mm_data(const uint8_t *data, const float *x, float *y, int tokens, int rows,
                       int cols) {
    if (tokens < 1 || tokens > JB_MAX_CTX)
        die("DiffusionGemma sequence exceeds context limit");
    /* The SIMD kernels tile row pairs; every model matrix has an even row
     * count, and requiring it here keeps all kernels to one contract. */
    if (rows < 2 || rows % 2)
        die("matrix row count must be even");
    dg_kernels()->bf16_mm(data, x, y, tokens, rows, cols);
}

static void dg_mm(const DGTensor *w, const float *x, float *y, int tokens, int rows, int cols) {
    if (w->nd != 2 || w->shape[0] != (uint64_t)rows || w->shape[1] != (uint64_t)cols)
        die2("bad matrix shape", w->name);
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
    if (w->exact) {
        dg_mm_exact_avx512(w->exact, x, y, tokens, rows, cols);
        return;
    }
#endif
    dg_mm_data(w->data, x, y, tokens, rows, cols);
}

static void dg_mv_slice(const DGTensor *w, uint64_t base, const float *x, float *y, int rows,
                        int cols) {
    {
        int r;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (r = 0; r < rows; r++) {
            const uint8_t *p = w->data + (base + (uint64_t)r * cols) * 2;
            float s = 0.0f;
            for (int c = 0; c < cols; c++)
                s += dg_bf(p + c * 2) * x[c];
            y[r] = s;
        }
    }
}

/* Project all answer slots together so the tied LM head is streamed once.
 * OpenJev's automatic reread test is entropy over the union of the
 * full-vocabulary top 20 and the explicitly requested label ids. */
/* logits holds n answer slots' logits over the vocabulary, which this
 * rewrites in place; terms has room for DG_VOCAB doubles. The softcap and
 * the softmax terms are computed in parallel, one value each; the maximum,
 * the top 20 and the sums run in order, as the reference's do. */
static double dg_slot_logits_entropy(int n, int **label_ids, const int *label_n,
                                     double **label_score, float *logits, double *terms) {
    if (n < 1 || n > DG_AUTO_READ_MAX_QUESTIONS)
        die("bad DiffusionGemma answer slot count");
    /* vLLM returns top-20 plus the sorted union of requested label ids at
     * every slot, capped at its per-request 128-id limit. OpenJev computes
     * entropy over that entire returned set. Keep the 128 smallest unique
     * ids without allocating a vocabulary-sized bitmap. */
    int requested[DG_CHOICE_LABELS], nr = 0;
    for (int t = 0; t < n; t++)
        for (int c = 0; c < label_n[t]; c++) {
            int id = label_ids[t][c], at = 0;
            while (at < nr && requested[at] < id)
                at++;
            if (at < nr && requested[at] == id)
                continue;
            if (nr < DG_CHOICE_LABELS) {
                memmove(requested + at + 1, requested + at, (size_t)(nr - at) * sizeof *requested);
                requested[at] = id;
                nr++;
            } else if (at < DG_CHOICE_LABELS) {
                memmove(requested + at + 1, requested + at,
                        (size_t)(DG_CHOICE_LABELS - 1 - at) * sizeof *requested);
                requested[at] = id;
            }
        }
    {
        int i, total = n * DG_VOCAB;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(static)
#endif
        for (i = 0; i < total; i++)
            logits[i] = 30.0f * dg_tanhf(logits[i] / 30.0f);
    }
    double max_entropy = 0;
    for (int t = 0; t < n; t++) {
        float *row = logits + (size_t)t * DG_VOCAB, topv[20];
        int topi[20];
        for (int k = 0; k < 20; k++) {
            topv[k] = -FLT_MAX;
            topi[k] = -1;
        }
        float mx = -FLT_MAX;
        for (int v = 0; v < DG_VOCAB; v++) {
            float z = row[v];
            if (z > mx)
                mx = z;
            /* topv descends, so nothing below its last entry enters. */
            if (!(z > topv[19]))
                continue;
            for (int k = 0; k < 20; k++)
                if (z > topv[k]) {
                    for (int q = 19; q > k; q--) {
                        topv[q] = topv[q - 1];
                        topi[q] = topi[q - 1];
                    }
                    topv[k] = z;
                    topi[k] = v;
                    break;
                }
        }
        {
            int v;
#ifdef _OPENMP
            JB_OMP();
#pragma omp parallel for schedule(static)
#endif
            for (v = 0; v < DG_VOCAB; v++)
                terms[v] = exp((double)row[v] - mx);
        }
        double den = 0;
        for (int v = 0; v < DG_VOCAB; v++)
            den += terms[v];
        double entropy = 0;
        for (int k = 0; k < 20; k++) {
            double p = exp((double)topv[k] - mx) / den;
            if (p > 0)
                entropy -= p * log(p);
        }
        for (int c = 0; c < label_n[t]; c++)
            label_score[t][c] = row[label_ids[t][c]];
        for (int c = 0; c < nr; c++) {
            int id = requested[c], seen = 0;
            for (int k = 0; k < 20; k++)
                if (topi[k] == id) {
                    seen = 1;
                    break;
                }
            if (!seen) {
                double p = exp((double)row[id] - mx) / den;
                if (p > 0)
                    entropy -= p * log(p);
            }
        }
        if (entropy > max_entropy)
            max_entropy = entropy;
    }
    return max_entropy;
}

/* The reference RMS scale factor: a sequential double sum of squares. */
static float dg_rms_inverse(const float *x, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++)
        ss += (double)x[i] * x[i];
    return 1.0f / sqrtf((float)(ss / n) + 1e-6f);
}

static void dg_rms_ref(float *y, const float *x, const DGTensor *scale, int n) {
    float q = dg_rms_inverse(x, n);
    for (int i = 0; i < n; i++)
        y[i] = x[i] * q * (scale ? dg_at(scale, i) : 1.0f);
}

static void dg_rms(float *y, const float *x, const DGTensor *scale, int n) {
    dg_kernels()->rms(y, x, scale, n);
}

static double dg_dot_ref(const float *a, const float *b, int n) {
    double s = 0;
    for (int i = 0; i < n; i++)
        s += (double)a[i] * b[i];
    return s;
}

static double dg_dot(const float *a, const float *b, int n) {
    return dg_kernels()->dot(a, b, n);
}

/* One backend dot per key, for backends without a batched form. */
static void dg_dots_each(const float *a, const float *b, size_t stride, int keys, int n,
                         float *out) {
    for (int j = 0; j < keys; j++)
        out[j] = (float)dg_dot(a, b + (size_t)j * stride, n);
}

static void dg_dots(const float *a, const float *b, size_t stride, int keys, int n, float *out) {
    dg_kernels()->dots(a, b, stride, keys, n, out);
}
