/* AVX2 backend. */
#if defined(JB_HAVE_AVX2)
static JB_TARGET_AVX2 __m256 dg_bf16x8(const uint8_t *p) {
    __m128i h = _mm_loadu_si128((const __m128i *)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

static JB_TARGET_AVX2 float dg_hsum8(__m256 x) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    h = _mm_hadd_ps(h, h);
    h = _mm_hadd_ps(h, h);
    return _mm_cvtss_f32(h);
}

static JB_TARGET_AVX2 void dg_mm_data_avx2(const uint8_t *data, const float *x, float *y,
                                           int tokens, int rows, int cols) {
    if (rows % 2)
        die("AVX2 matrix row count must be even");
    {
        int r;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < rows; r += 2) {
            const uint8_t *p0 = data + (uint64_t)r * cols * 2, *p1 = p0 + (uint64_t)cols * 2;
            for (int tb = 0; tb < tokens; tb += 4) {
                int nb = tokens - tb < 4 ? tokens - tb : 4;
                __m256 a[4], b[4];
                for (int q = 0; q < nb; q++)
                    a[q] = b[q] = _mm256_setzero_ps();
                int c = 0;
                for (; c + 7 < cols; c += 8) {
                    __m256 w0 = dg_bf16x8(p0 + c * 2), w1 = dg_bf16x8(p1 + c * 2);
                    for (int q = 0; q < nb; q++) {
                        __m256 v = _mm256_loadu_ps(x + (size_t)(tb + q) * cols + c);
                        a[q] = _mm256_fmadd_ps(w0, v, a[q]);
                        b[q] = _mm256_fmadd_ps(w1, v, b[q]);
                    }
                }
                for (int q = 0; q < nb; q++) {
                    float z0 = dg_hsum8(a[q]), z1 = dg_hsum8(b[q]);
                    for (int k = c; k < cols; k++) {
                        float v = x[(size_t)(tb + q) * cols + k];
                        z0 += dg_bf(p0 + k * 2) * v;
                        z1 += dg_bf(p1 + k * 2) * v;
                    }
                    y[(size_t)(tb + q) * rows + r] = z0;
                    y[(size_t)(tb + q) * rows + r + 1] = z1;
                }
            }
        }
    }
}

static JB_TARGET_AVX2 void dg_nvfp4_weights16_avx2(const uint8_t *q, float scale, __m256 *w0,
                                                   __m256 *w1) {
    const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask = _mm_set1_epi8(15), raw = _mm_loadl_epi64((const __m128i *)q);
    __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw, mask));
    __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw, 4), mask));
    __m256 lf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
    __m256 hf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
    __m256 a = _mm256_unpacklo_ps(lf, hf), b = _mm256_unpackhi_ps(lf, hf),
           s = _mm256_set1_ps(scale * .5f);
    *w0 = _mm256_mul_ps(_mm256_permute2f128_ps(a, b, 0x20), s);
    *w1 = _mm256_mul_ps(_mm256_permute2f128_ps(a, b, 0x31), s);
}

/* Decode each packed 16-weight block entirely in registers, retaining the
 * reference interleaved lane order and reusing it across four tokens. */
static JB_TARGET_AVX2 void dg_nvfp4_mm_avx2(const uint8_t *wd, const uint8_t *sd, float global,
                                            const float *x, float *y, int tokens, int rows,
                                            int cols) {
    {
        int r;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < rows; r++) {
            const uint8_t *wp = wd + (uint64_t)r * cols / 2, *sp = sd + (uint64_t)r * cols / 16;
            for (int tb = 0; tb < tokens; tb += 4) {
                int nb = tokens - tb < 4 ? tokens - tb : 4;
                __m256 acc0[4], acc1[4];
                for (int q = 0; q < nb; q++)
                    acc0[q] = acc1[q] = _mm256_setzero_ps();
                for (int c = 0; c < cols; c += 16) {
                    __m256 w0, w1;
                    dg_nvfp4_weights16_avx2(wp + c / 2, dg_f8e4m3(sp[c / 16]) * global, &w0, &w1);
                    for (int q = 0; q < nb; q++) {
                        const float *xp = x + (size_t)(tb + q) * cols + c;
                        acc0[q] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp), acc0[q]);
                        acc1[q] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp + 8), acc1[q]);
                    }
                }
                for (int q = 0; q < nb; q++)
                    y[(size_t)(tb + q) * rows + r] = dg_hsum8(_mm256_add_ps(acc0[q], acc1[q]));
            }
        }
    }
}

static JB_TARGET_AVX2 double dg_hsum4d(__m256d x) {
    __m128d h = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
    h = _mm_hadd_pd(h, h);
    return _mm_cvtsd_f64(h);
}

static JB_TARGET_AVX2 void dg_rms_avx2(float *y, const float *x, const DGTensor *scale, int n) {
    __m256d a = _mm256_setzero_pd(), b = a;
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256d lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v)),
                hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
        a = _mm256_fmadd_pd(lo, lo, a);
        b = _mm256_fmadd_pd(hi, hi, b);
    }
    double ss = dg_hsum4d(_mm256_add_pd(a, b));
    for (; i < n; i++)
        ss += (double)x[i] * x[i];
    float q = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    i = 0;
    if (scale)
        for (; i + 7 < n; i += 8)
            _mm256_storeu_ps(y + i,
                             _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(q)),
                                           dg_bf16x8(scale->data + (size_t)i * 2)));
    else
        for (; i + 7 < n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(q)));
    for (; i < n; i++)
        y[i] = x[i] * q * (scale ? dg_at(scale, i) : 1.0f);
}

static JB_TARGET_AVX2 double dg_dot_avx2(const float *a, const float *b, int n) {
    __m256d s0 = _mm256_setzero_pd(), s1 = s0;
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(a + i), y = _mm256_loadu_ps(b + i);
        s0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(x)),
                             _mm256_cvtps_pd(_mm256_castps256_ps128(y)), s0);
        s1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(x, 1)),
                             _mm256_cvtps_pd(_mm256_extractf128_ps(y, 1)), s1);
    }
    double s = dg_hsum4d(_mm256_add_pd(s0, s1));
    for (; i < n; i++)
        s += (double)a[i] * b[i];
    return s;
}
#endif
