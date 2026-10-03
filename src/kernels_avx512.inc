/* Strict AVX-512 assigns one output row to each vector lane. Columns are
 * consequently accumulated in exactly the scalar reference order. These
 * load-time images make the 16 rows contiguous for every column; NVFP4
 * keeps its nibbles packed and expands its block scales to lane vectors. */
#if defined(JB_HAVE_AVX512) && JB_STRICT_MATH
enum { DG_EXACT_ROWS = 16 };

static size_t dg_exact_place(size_t *at, size_t bytes) {
    *at = jb_align_up(*at, 64);
    size_t p = *at;
    *at = jb_size_add(*at, bytes);
    return p;
}

static void dg_exact_pack_bf16(DGTensor *t) {
    int rows = (int)t->shape[0], cols = (int)t->shape[1];
    for (int r = 0; r < rows; r += DG_EXACT_ROWS)
        for (int c = 0; c < cols; c++)
            for (int lane = 0; lane < DG_EXACT_ROWS; lane++)
                memcpy((uint8_t *)t->exact +
                           (((size_t)r / DG_EXACT_ROWS * cols + c) * DG_EXACT_ROWS + lane) * 2,
                       t->data + ((size_t)(r + lane) * cols + c) * 2, 2);
}

static void dg_exact_pack_nvfp4(DGTensor *w, DGTensor *s, int rows, int cols) {
    for (int r = 0; r < rows; r += DG_EXACT_ROWS) {
        for (int c = 0; c < cols; c++)
            for (int lane = 0; lane < DG_EXACT_ROWS / 2; lane++) {
                int r0 = r + lane, r1 = r + lane + DG_EXACT_ROWS / 2;
                uint8_t a = w->data[(size_t)r0 * cols / 2 + c / 2],
                        b = w->data[(size_t)r1 * cols / 2 + c / 2];
                a = (uint8_t)(c & 1 ? a >> 4 : a & 15);
                b = (uint8_t)(c & 1 ? b >> 4 : b & 15);
                ((uint8_t *)w->exact)[((size_t)r / DG_EXACT_ROWS * cols + c) * 8 + lane] =
                    (uint8_t)(a | b << 4);
            }
        for (int b = 0; b < cols / 16; b++)
            for (int lane = 0; lane < DG_EXACT_ROWS; lane++)
                ((uint8_t *)
                     s->exact)[((size_t)r / DG_EXACT_ROWS * (cols / 16) + b) * DG_EXACT_ROWS +
                               lane] = s->data[(size_t)(r + lane) * cols / 16 + b];
    }
}

static void dg_prepare_exact_model(DGModel *m) {
    if (m->accel || strcmp(dg_kernels()->name, "avx512-exact"))
        return;
    size_t bytes = 0;
    for (size_t i = 0; i < m->nt; i++) {
        DGTensor *t = &m->tensor[i];
        if (t->dtype == DG_BF16 && t->nd == 2 && t->shape[0] % DG_EXACT_ROWS == 0) {
            t->exact_offset = dg_exact_place(&bytes, (size_t)t->bytes) + 1;
        }
    }
    for (int i = 0; m->nvfp4 && i < DG_L * DG_EXPERTS; i++) {
        DGNvExpert *v = &m->nvexpert[i];
        DGTensor *w[3] = {v->wg, v->wu, v->wd}, *s[3] = {v->sg, v->su, v->sd};
        int rows[3] = {DG_MOE, DG_MOE, DG_H}, cols[3] = {DG_H, DG_H, DG_MOE};
        for (int k = 0; k < 3; k++) {
            size_t wp = dg_exact_place(&bytes, (size_t)rows[k] * cols[k] / 2),
                   sp = dg_exact_place(&bytes, (size_t)rows[k] * cols[k] / 16);
            w[k]->exact_offset = wp + 1;
            s[k]->exact_offset = sp + 1;
        }
    }
    m->exact_weights = xmalloc(bytes);
    for (size_t i = 0; i < m->nt; i++)
        if (m->tensor[i].exact_offset)
            m->tensor[i].exact = m->exact_weights + m->tensor[i].exact_offset - 1;
    for (size_t i = 0; i < m->nt; i++) {
        DGTensor *t = &m->tensor[i];
        if (t->dtype == DG_BF16 && t->nd == 2 && t->exact)
            dg_exact_pack_bf16(t);
    }
    if (m->nvfp4) {
        int i;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (i = 0; i < DG_L * DG_EXPERTS; i++) {
            DGNvExpert *v = &m->nvexpert[i];
            dg_exact_pack_nvfp4(v->wg, v->sg, DG_MOE, DG_H);
            dg_exact_pack_nvfp4(v->wu, v->su, DG_MOE, DG_H);
            dg_exact_pack_nvfp4(v->wd, v->sd, DG_H, DG_MOE);
        }
    }
}
#else
static void dg_prepare_exact_model(DGModel *m) {
    (void)m;
}
#endif

/* AVX-512 backend. */
#if defined(JB_HAVE_AVX512)
static JB_TARGET_AVX512 __m512 dg_bf16x16(const uint8_t *p) {
    __m256i h = _mm256_loadu_si256((const __m256i *)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}

#if JB_STRICT_MATH
static JB_TARGET_AVX512 void dg_mm_exact_avx512(const uint8_t *data, const float *x, float *y,
                                                int tokens, int rows, int cols) {
    int rt;
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (rt = 0; rt < rows; rt += DG_EXACT_ROWS) {
        const uint8_t *tile = data + (size_t)rt * cols * 2;
        for (int tb = 0; tb < tokens; tb += 8) {
            int n = tokens - tb < 8 ? tokens - tb : 8;
            __m512 sum[8];
            for (int q = 0; q < n; q++)
                sum[q] = _mm512_setzero_ps();
            for (int c = 0; c < cols; c++) {
                __m512 w = dg_bf16x16(tile + (size_t)c * DG_EXACT_ROWS * 2);
                for (int q = 0; q < n; q++) {
                    __m512 p = _mm512_mul_ps(w, _mm512_set1_ps(x[(size_t)(tb + q) * cols + c]));
                    sum[q] = _mm512_add_ps(sum[q], p);
                }
            }
            for (int q = 0; q < n; q++)
                _mm512_storeu_ps(y + (size_t)(tb + q) * rows + rt, sum[q]);
        }
    }
}

static JB_TARGET_AVX512 __m512i dg_nvfp4_exact_codes(const uint8_t *p) {
    __m256i bytes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p));
    __m256i lo = _mm256_and_si256(bytes, _mm256_set1_epi32(15));
    __m256i hi = _mm256_srli_epi32(bytes, 4);
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

/* Decode 16 transposed E4M3 scale bytes without expanding them in the model
 * image. The integer bit construction and the subnormal multiply reproduce
 * dg_f8e4m3, lane for lane. */
static JB_TARGET_AVX512 __m512 dg_f8e4m3x16(const uint8_t *p) {
    __m512i code = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)p));
    __m512i exponent = _mm512_and_si512(_mm512_srli_epi32(code, 3), _mm512_set1_epi32(15));
    __m512i mantissa = _mm512_and_si512(code, _mm512_set1_epi32(7));
    __m512 normal = _mm512_castsi512_ps(
        _mm512_or_si512(_mm512_slli_epi32(_mm512_add_epi32(exponent, _mm512_set1_epi32(120)), 23),
                        _mm512_slli_epi32(mantissa, 20)));
    __m512 subnormal = _mm512_mul_ps(_mm512_cvtepi32_ps(mantissa), _mm512_set1_ps(0x1p-9f));
    __mmask16 is_subnormal = _mm512_cmpeq_epi32_mask(exponent, _mm512_setzero_si512());
    __m512 magnitude = _mm512_mask_blend_ps(is_subnormal, normal, subnormal);
    __m512i sign = _mm512_slli_epi32(_mm512_and_si512(code, _mm512_set1_epi32(128)), 24);
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(magnitude), sign));
}

#ifndef JB_NO_MAIN
static JB_TARGET_AVX512 void dg_f8e4m3x16_store(float *out, const uint8_t *p) {
    _mm512_storeu_ps(out, dg_f8e4m3x16(p));
}
#endif

static JB_TARGET_AVX512 void dg_nvfp4_exact_avx512(const uint8_t *wd, const uint8_t *sd,
                                                   float global, const float *x, float *y,
                                                   int tokens, int rows, int cols) {
    static const float lut[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
    __m512 table = _mm512_loadu_ps(lut);
    int rt;
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (rt = 0; rt < rows; rt += DG_EXACT_ROWS) {
        size_t tile = (size_t)rt / DG_EXACT_ROWS;
        for (int tb = 0; tb < tokens; tb += 8) {
            int n = tokens - tb < 8 ? tokens - tb : 8;
            __m512 sum[8];
            for (int q = 0; q < n; q++)
                sum[q] = _mm512_setzero_ps();
            for (int b = 0; b < cols / 16; b++) {
                __m512 scale =
                    _mm512_mul_ps(dg_f8e4m3x16(sd + (tile * (cols / 16) + b) * DG_EXACT_ROWS),
                                  _mm512_set1_ps(global));
                for (int k = 0; k < 8; k++) {
                    int c = b * 16 + k * 2;
                    __m512 we = _mm512_permutexvar_ps(
                        dg_nvfp4_exact_codes(wd + (tile * cols + c) * 8), table);
                    __m512 wo = _mm512_permutexvar_ps(
                        dg_nvfp4_exact_codes(wd + (tile * cols + c + 1) * 8), table);
                    for (int q = 0; q < n; q++) {
                        const float *a = x + (size_t)(tb + q) * cols + c;
                        __m512 pair = _mm512_add_ps(_mm512_mul_ps(we, _mm512_set1_ps(a[0])),
                                                    _mm512_mul_ps(wo, _mm512_set1_ps(a[1])));
                        sum[q] = _mm512_add_ps(sum[q], _mm512_mul_ps(scale, pair));
                    }
                }
            }
            for (int q = 0; q < n; q++)
                _mm512_storeu_ps(y + (size_t)(tb + q) * rows + rt, sum[q]);
        }
    }
}
#endif

static JB_TARGET_AVX512 void dg_mm_data_avx512(const uint8_t *data, const float *x, float *y,
                                               int tokens, int rows, int cols) {
    if (rows % 2)
        die("AVX-512 matrix row count must be even");
    {
        int r;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < rows; r += 2) {
            const uint8_t *p0 = data + (uint64_t)r * cols * 2, *p1 = p0 + (uint64_t)cols * 2;
            int t = 0;
            for (; t + 7 < tokens; t += 8) {
                __m512 a[8], b[8];
                for (int q = 0; q < 8; q++)
                    a[q] = b[q] = _mm512_setzero_ps();
                int c = 0;
                for (; c + 15 < cols; c += 16) {
                    __m512 w0 = dg_bf16x16(p0 + c * 2), w1 = dg_bf16x16(p1 + c * 2);
                    for (int q = 0; q < 8; q++) {
                        __m512 v = _mm512_loadu_ps(x + (size_t)(t + q) * cols + c);
                        a[q] = _mm512_fmadd_ps(w0, v, a[q]);
                        b[q] = _mm512_fmadd_ps(w1, v, b[q]);
                    }
                }
                for (int q = 0; q < 8; q++) {
                    float z0 = _mm512_reduce_add_ps(a[q]), z1 = _mm512_reduce_add_ps(b[q]);
                    for (int k = (cols & ~15); k < cols; k++) {
                        float v = x[(size_t)(t + q) * cols + k];
                        z0 += dg_bf(p0 + k * 2) * v;
                        z1 += dg_bf(p1 + k * 2) * v;
                    }
                    y[(size_t)(t + q) * rows + r] = z0;
                    y[(size_t)(t + q) * rows + r + 1] = z1;
                }
            }
            for (; t < tokens; t++) {
                __m512 s0 = _mm512_setzero_ps(), s1 = s0;
                int c = 0;
                for (; c + 15 < cols; c += 16) {
                    __m512 v = _mm512_loadu_ps(x + (size_t)t * cols + c);
                    s0 = _mm512_fmadd_ps(dg_bf16x16(p0 + c * 2), v, s0);
                    s1 = _mm512_fmadd_ps(dg_bf16x16(p1 + c * 2), v, s1);
                }
                float z0 = _mm512_reduce_add_ps(s0), z1 = _mm512_reduce_add_ps(s1);
                for (; c < cols; c++) {
                    float v = x[(size_t)t * cols + c];
                    z0 += dg_bf(p0 + c * 2) * v;
                    z1 += dg_bf(p1 + c * 2) * v;
                }
                y[(size_t)t * rows + r] = z0;
                y[(size_t)t * rows + r + 1] = z1;
            }
        }
    }
}

/* Match each packed weight byte: low-nibble activations, then high. This
 * one-time swizzle removes two activation permutes per expert row/tile. */
static void dg_nvfp4_swizzle(float *out, int tokens, int cols) {
    for (int t = 0; t < tokens; t++)
        for (int c = 0; c < cols; c += 32) {
            float *y = out + (size_t)t * cols + c, tmp[32];
            memcpy(tmp, y, sizeof tmp);
            for (int k = 0; k < 16; k++) {
                y[k] = tmp[k * 2];
                y[16 + k] = tmp[k * 2 + 1];
            }
        }
}

static void dg_nvfp4_qdq_avx512(float *out, const float *in, int tokens, int cols, float base) {
    dg_nvfp4_qdq_ref(out, in, tokens, cols, base);
    dg_nvfp4_swizzle(out, tokens, cols);
}

/* x in the dg_nvfp4_swizzle layout; rows must be even. */
static JB_TARGET_AVX512 void dg_nvfp4_mm_avx512(const uint8_t *wd, const uint8_t *sd, float global,
                                                const float *x, float *y, int tokens, int rows,
                                                int cols) {
    static const float lut[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
    __m512 table = _mm512_loadu_ps(lut);
    {
        int r;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < rows; r += 2) {
            const uint8_t *wp0 = wd + (uint64_t)r * cols / 2, *wp1 = wp0 + cols / 2;
            const uint8_t *sp0 = sd + (uint64_t)r * cols / 16, *sp1 = sp0 + cols / 16;
            for (int tb = 0; tb < tokens; tb += 8) {
                int nb = tokens - tb < 8 ? tokens - tb : 8;
                __m512 a[8], b[8];
                for (int q = 0; q < nb; q++)
                    a[q] = b[q] = _mm512_setzero_ps();
                for (int c = 0; c < cols; c += 32) {
                    __m512i z = _mm512_set1_epi32(15),
                            raw0 = _mm512_cvtepu8_epi32(
                                _mm_loadu_si128((const __m128i *)(wp0 + c / 2))),
                            raw1 = _mm512_cvtepu8_epi32(
                                _mm_loadu_si128((const __m128i *)(wp1 + c / 2)));
                    __m512 wl0 = _mm512_permutexvar_ps(_mm512_and_si512(raw0, z), table),
                           wh0 = _mm512_permutexvar_ps(_mm512_srli_epi32(raw0, 4), table);
                    __m512 wl1 = _mm512_permutexvar_ps(_mm512_and_si512(raw1, z), table),
                           wh1 = _mm512_permutexvar_ps(_mm512_srli_epi32(raw1, 4), table);
                    float a0 = dg_f8e4m3(sp0[c / 16]) * global,
                          a1 = dg_f8e4m3(sp0[c / 16 + 1]) * global,
                          b0 = dg_f8e4m3(sp1[c / 16]) * global,
                          b1 = dg_f8e4m3(sp1[c / 16 + 1]) * global;
                    __m512 sv0 =
                               _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(a0), _mm512_set1_ps(a1)),
                           sv1 =
                               _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(b0), _mm512_set1_ps(b1));
                    wl0 = _mm512_mul_ps(wl0, sv0);
                    wh0 = _mm512_mul_ps(wh0, sv0);
                    wl1 = _mm512_mul_ps(wl1, sv1);
                    wh1 = _mm512_mul_ps(wh1, sv1);
                    for (int q = 0; q < nb; q++) {
                        const float *xp = x + (size_t)(tb + q) * cols + c;
                        __m512 xe = _mm512_loadu_ps(xp), xo = _mm512_loadu_ps(xp + 16);
                        a[q] = _mm512_fmadd_ps(wl0, xe, a[q]);
                        a[q] = _mm512_fmadd_ps(wh0, xo, a[q]);
                        b[q] = _mm512_fmadd_ps(wl1, xe, b[q]);
                        b[q] = _mm512_fmadd_ps(wh1, xo, b[q]);
                    }
                }
                for (int q = 0; q < nb; q++) {
                    y[(size_t)(tb + q) * rows + r] = _mm512_reduce_add_ps(a[q]);
                    y[(size_t)(tb + q) * rows + r + 1] = _mm512_reduce_add_ps(b[q]);
                }
            }
        }
    }
}

static JB_TARGET_AVX512 void dg_rms_avx512(float *y, const float *x, const DGTensor *scale, int n) {
    __m512d a = _mm512_setzero_pd(), b = a, c = a, d = a;
    int i = 0;
    for (; i + 31 < n; i += 32) {
        __m512 x0 = _mm512_loadu_ps(x + i), x1 = _mm512_loadu_ps(x + i + 16);
        __m512d q0 = _mm512_cvtps_pd(_mm512_castps512_ps256(x0));
        __m512d q1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(x0, 1));
        __m512d q2 = _mm512_cvtps_pd(_mm512_castps512_ps256(x1));
        __m512d q3 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(x1, 1));
        a = _mm512_fmadd_pd(q0, q0, a);
        b = _mm512_fmadd_pd(q1, q1, b);
        c = _mm512_fmadd_pd(q2, q2, c);
        d = _mm512_fmadd_pd(q3, q3, d);
    }
    double ss = _mm512_reduce_add_pd(_mm512_add_pd(_mm512_add_pd(a, b), _mm512_add_pd(c, d)));
    for (; i < n; i++)
        ss += (double)x[i] * x[i];
    float q = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    i = 0;
    if (scale)
        for (; i + 15 < n; i += 16)
            _mm512_storeu_ps(y + i,
                             _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x + i), _mm512_set1_ps(q)),
                                           dg_bf16x16(scale->data + (size_t)i * 2)));
    else
        for (; i + 15 < n; i += 16)
            _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), _mm512_set1_ps(q)));
    for (; i < n; i++)
        y[i] = x[i] * q * (scale ? dg_at(scale, i) : 1.0f);
}

static JB_TARGET_AVX512 double dg_dot_avx512(const float *a, const float *b, int n) {
    __m512d s0 = _mm512_setzero_pd(), s1 = s0;
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 x = _mm512_loadu_ps(a + i), y = _mm512_loadu_ps(b + i);
        s0 = _mm512_fmadd_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(x)),
                             _mm512_cvtps_pd(_mm512_castps512_ps256(y)), s0);
        s1 = _mm512_fmadd_pd(_mm512_cvtps_pd(_mm512_extractf32x8_ps(x, 1)),
                             _mm512_cvtps_pd(_mm512_extractf32x8_ps(y, 1)), s1);
    }
    double s = _mm512_reduce_add_pd(_mm512_add_pd(s0, s1));
    for (; i < n; i++)
        s += (double)a[i] * b[i];
    return s;
}
#endif
