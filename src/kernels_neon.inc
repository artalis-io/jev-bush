/* NEON backend (AArch64). Each vector lane computes one output row (or one
 * attention key), and each lane accumulates in the reference kernels' order
 * with a separate multiply and add. Strict builds therefore match the
 * reference bit for bit, which the selftest checks. */
#if defined(JB_NEON)
enum { DG_NEON_TOKENS = 8, DG_NEON_KEYS = 8 };

static float32x4_t dg_bf16x4_neon(const uint8_t *p) {
    uint16x4_t h = vreinterpret_u16_u8(vld1_u8(p));
    return vreinterpretq_f32_u32(vshll_n_u16(h, 16));
}

/* Four rows of four columns become four columns of four rows. */
static void dg_transpose4_neon(float32x4_t *v) {
    float32x4_t t0 = vtrn1q_f32(v[0], v[1]), t1 = vtrn2q_f32(v[0], v[1]);
    float32x4_t t2 = vtrn1q_f32(v[2], v[3]), t3 = vtrn2q_f32(v[2], v[3]);
    v[0] = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
    v[1] = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
    v[2] = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
    v[3] = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
}

/* Rows past the end of the matrix repeat its last row; their lanes are
 * computed but never stored. */
static int dg_neon_row(int r0, int i, int rows) {
    return r0 + i < rows ? r0 + i : rows - 1;
}

static void dg_mm_data_neon(const uint8_t *data, const float *x, float *y, int tokens, int rows,
                            int cols) {
    int blocks = (rows + 3) / 4, c4 = cols & ~3;
    {
        int rb;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (rb = 0; rb < blocks; rb++) {
            int r0 = rb * 4, valid = rows - r0 < 4 ? rows - r0 : 4;
            const uint8_t *p[4];
            for (int i = 0; i < 4; i++)
                p[i] = data + (uint64_t)dg_neon_row(r0, i, rows) * cols * 2;
            for (int tb = 0; tb < tokens; tb += DG_NEON_TOKENS) {
                int nb = tokens - tb < DG_NEON_TOKENS ? tokens - tb : DG_NEON_TOKENS;
                float32x4_t acc[DG_NEON_TOKENS];
                for (int q = 0; q < nb; q++)
                    acc[q] = vdupq_n_f32(0);
                for (int c = 0; c < c4; c += 4) {
                    float32x4_t w[4];
                    for (int i = 0; i < 4; i++)
                        w[i] = dg_bf16x4_neon(p[i] + (size_t)c * 2);
                    dg_transpose4_neon(w);
                    for (int q = 0; q < nb; q++) {
                        const float *xp = x + (size_t)(tb + q) * cols + c;
                        for (int k = 0; k < 4; k++)
                            acc[q] = vaddq_f32(acc[q], vmulq_n_f32(w[k], xp[k]));
                    }
                }
                for (int q = 0; q < nb; q++) {
                    const float *xp = x + (size_t)(tb + q) * cols;
                    float sum[4];
                    vst1q_f32(sum, acc[q]);
                    for (int i = 0; i < valid; i++) {
                        for (int c = c4; c < cols; c++)
                            sum[i] += dg_bf(p[i] + (size_t)c * 2) * xp[c];
                        y[(size_t)(tb + q) * rows + r0 + i] = sum[i];
                    }
                }
            }
        }
    }
}

/* Signed E2M1 values by nibble, sign bit included; entry 8 is -0.0, which
 * the reference also produces and bit-identity must keep. */
static const float dg_e2m1_neon[16] = {0,     .5f,  1,  1.5f,  2,  3,  4,  6,
                                       -0.0f, -.5f, -1, -1.5f, -2, -3, -4, -6};

/* Four rows' float weights for byte position k from nibbles held as
 * table offsets (nibble * 4): spread each row's offset over its lane's four
 * bytes, add the byte index, and read the float's bytes from the table. */
static float32x4_t dg_e2m1x4_neon(uint8x16x4_t table, uint8x16_t offsets, int k) {
    static const uint8_t spread[4][16] = {
        {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3},
        {4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7},
        {8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11},
        {12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15}};
    static const uint8_t byte[16] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    uint8x16_t index = vaddq_u8(vqtbl1q_u8(offsets, vld1q_u8(spread[k])), vld1q_u8(byte));
    return vreinterpretq_f32_u8(vqtbl4q_u8(table, index));
}

/* One packed 16-weight block (eight bytes) from each of four rows, as eight
 * vectors of even-column weights and eight of odd-column weights, each
 * holding the four rows' values for one byte position. */
static void dg_nvfp4_weights4_neon(const uint8_t *const *q, float32x4_t *even, float32x4_t *odd) {
    const uint8x16x4_t table = vld1q_u8_x4((const uint8_t *)dg_e2m1_neon);
    uint8x8x2_t z01 = vzip_u8(vld1_u8(q[0]), vld1_u8(q[1]));
    uint8x8x2_t z23 = vzip_u8(vld1_u8(q[2]), vld1_u8(q[3]));
    for (int half = 0; half < 2; half++) {
        uint16x4x2_t z =
            vzip_u16(vreinterpret_u16_u8(z01.val[half]), vreinterpret_u16_u8(z23.val[half]));
        uint8x16_t bytes =
            vcombine_u8(vreinterpret_u8_u16(z.val[0]), vreinterpret_u8_u16(z.val[1]));
        uint8x16_t lo = vshlq_n_u8(vandq_u8(bytes, vdupq_n_u8(15)), 2);
        uint8x16_t hi = vshlq_n_u8(vshrq_n_u8(bytes, 4), 2);
        for (int k = 0; k < 4; k++) {
            even[half * 4 + k] = dg_e2m1x4_neon(table, lo, k);
            odd[half * 4 + k] = dg_e2m1x4_neon(table, hi, k);
        }
    }
}

/* dg_f8e4m3 for four rows' scale bytes: the same bit assembly, with the
 * subnormals (exponent 0) selected by mask. */
static float32x4_t dg_e4m3x4_neon(uint32x4_t u) {
    uint32x4_t e = vandq_u32(vshrq_n_u32(u, 3), vdupq_n_u32(15)), m = vandq_u32(u, vdupq_n_u32(7));
    uint32x4_t normal =
        vorrq_u32(vshlq_n_u32(vaddq_u32(e, vdupq_n_u32(120)), 23), vshlq_n_u32(m, 20));
    float32x4_t subnormal = vmulq_n_f32(vcvtq_f32_u32(m), 0x1p-9f);
    float32x4_t v =
        vbslq_f32(vcgtq_u32(e, vdupq_n_u32(0)), vreinterpretq_f32_u32(normal), subnormal);
    uint32x4_t sign = vshlq_n_u32(vshrq_n_u32(u, 7), 31);
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v), sign));
}

/* One block's contribution to one token's four rows, in the reference's
 * order: acc += scale * (even * a[2k] + odd * a[2k + 1]) for k = 0..7. The
 * activations are loaded once and used by lane. */
static float32x4_t dg_nvfp4_block_neon(float32x4_t acc, const float32x4_t *even,
                                       const float32x4_t *odd, float32x4_t scale, const float *a) {
    float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4), a2 = vld1q_f32(a + 8),
                a3 = vld1q_f32(a + 12);
#define DG_NVFP4_PAIR(k, v, even_lane, odd_lane)                                                   \
    acc = vaddq_f32(acc, vmulq_f32(scale, vaddq_f32(vmulq_laneq_f32(even[k], v, even_lane),        \
                                                    vmulq_laneq_f32(odd[k], v, odd_lane))))
    DG_NVFP4_PAIR(0, a0, 0, 1);
    DG_NVFP4_PAIR(1, a0, 2, 3);
    DG_NVFP4_PAIR(2, a1, 0, 1);
    DG_NVFP4_PAIR(3, a1, 2, 3);
    DG_NVFP4_PAIR(4, a2, 0, 1);
    DG_NVFP4_PAIR(5, a2, 2, 3);
    DG_NVFP4_PAIR(6, a3, 0, 1);
    DG_NVFP4_PAIR(7, a3, 2, 3);
#undef DG_NVFP4_PAIR
    return acc;
}

/* Four rows' products with the nb tokens from tb, in the reference's order;
 * rows past the end repeat the last row. x in the dg_nvfp4_qdq_ref layout. */
static void dg_nvfp4_rows4_neon(const DGNvMatrix *m, const float *x, int r0, int rows, int cols,
                                int tb, int nb, float32x4_t *acc) {
    const uint8_t *wp[4], *sp[4];
    for (int i = 0; i < 4; i++) {
        int r = dg_neon_row(r0, i, rows);
        wp[i] = m->w + (uint64_t)r * cols / 2;
        sp[i] = m->s + (uint64_t)r * cols / 16;
    }
    for (int t = 0; t < nb; t++)
        acc[t] = vdupq_n_f32(0);
    for (int b = 0; b < cols / 16; b++) {
        const uint8_t *q[4];
        for (int i = 0; i < 4; i++)
            q[i] = wp[i] + (size_t)b * 8;
        uint32x4_t raw = {sp[0][b], sp[1][b], sp[2][b], sp[3][b]};
        float32x4_t scale = vmulq_n_f32(dg_e4m3x4_neon(raw), m->global), even[8], odd[8];
        dg_nvfp4_weights4_neon(q, even, odd);
        for (int t = 0; t < nb; t++)
            acc[t] = dg_nvfp4_block_neon(acc[t], even, odd, scale,
                                         x + (size_t)(tb + t) * cols + (size_t)b * 16);
    }
}

static void dg_nvfp4_mm_neon(const uint8_t *wd, const uint8_t *sd, float global, const float *x,
                             float *y, int tokens, int rows, int cols) {
    DGNvMatrix m = {.w = wd, .s = sd, .global = global};
    int blocks = (rows + 3) / 4;
    {
        int rb;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (rb = 0; rb < blocks; rb++) {
            int r0 = rb * 4, valid = rows - r0 < 4 ? rows - r0 : 4;
            for (int tb = 0; tb < tokens; tb += DG_NEON_TOKENS) {
                int nb = tokens - tb < DG_NEON_TOKENS ? tokens - tb : DG_NEON_TOKENS;
                float32x4_t acc[DG_NEON_TOKENS];
                dg_nvfp4_rows4_neon(&m, x, r0, rows, cols, tb, nb, acc);
                for (int t = 0; t < nb; t++) {
                    float sum[4];
                    vst1q_f32(sum, acc[t]);
                    for (int i = 0; i < valid; i++)
                        y[(size_t)(tb + t) * rows + r0 + i] = sum[i];
                }
            }
        }
    }
}

/* Gate and up products for the same four rows, then GELU, in one parallel
 * region; each value is the one the separate steps compute. */
static void dg_nvfp4_gated_neon(const DGNvMatrix *gate, const DGNvMatrix *up, const float *x,
                                float *g, float *h, int tokens, int rows, int cols) {
    (void)g;
    int blocks = (rows + 3) / 4;
    {
        int rb;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (rb = 0; rb < blocks; rb++) {
            int r0 = rb * 4, valid = rows - r0 < 4 ? rows - r0 : 4;
            for (int tb = 0; tb < tokens; tb += DG_NEON_TOKENS) {
                int nb = tokens - tb < DG_NEON_TOKENS ? tokens - tb : DG_NEON_TOKENS;
                float32x4_t ga[DG_NEON_TOKENS], ua[DG_NEON_TOKENS];
                dg_nvfp4_rows4_neon(gate, x, r0, rows, cols, tb, nb, ga);
                dg_nvfp4_rows4_neon(up, x, r0, rows, cols, tb, nb, ua);
                for (int t = 0; t < nb; t++) {
                    float gs[4], us[4];
                    vst1q_f32(gs, ga[t]);
                    vst1q_f32(us, ua[t]);
                    for (int i = 0; i < valid; i++)
                        h[(size_t)(tb + t) * rows + r0 + i] = dg_gelu(gs[i]) * us[i];
                }
            }
        }
    }
}

/* Attention dots for eight keys at a time, one key per double lane. Each
 * lane sums (double)a[i] * b[i] in index order like dg_dot_ref, and a
 * product of two floats is exact in double, so the sums match it bit for
 * bit. Keys past the end repeat the last key and are not stored. */
static void dg_dots_neon(const float *a, const float *b, size_t stride, int keys, int n,
                         float *out) {
    int n4 = n & ~3;
    for (int j0 = 0; j0 < keys; j0 += DG_NEON_KEYS) {
        int valid = keys - j0 < DG_NEON_KEYS ? keys - j0 : DG_NEON_KEYS;
        const float *k[DG_NEON_KEYS];
        float64x2_t s[DG_NEON_KEYS / 2];
        for (int j = 0; j < DG_NEON_KEYS; j++)
            k[j] = b + (size_t)dg_neon_row(j0, j, keys) * stride;
        for (int p = 0; p < DG_NEON_KEYS / 2; p++)
            s[p] = vdupq_n_f64(0);
        for (int i = 0; i < n4; i += 4)
            for (int g = 0; g < DG_NEON_KEYS / 4; g++) {
                float32x4_t w[4];
                for (int j = 0; j < 4; j++)
                    w[j] = vld1q_f32(k[g * 4 + j] + i);
                dg_transpose4_neon(w);
                for (int c = 0; c < 4; c++) {
                    float64x2_t ac = vdupq_n_f64((double)a[i + c]);
                    s[g * 2] = vaddq_f64(s[g * 2], vmulq_f64(vcvt_f64_f32(vget_low_f32(w[c])), ac));
                    s[g * 2 + 1] = vaddq_f64(s[g * 2 + 1], vmulq_f64(vcvt_high_f64_f32(w[c]), ac));
                }
            }
        double sum[DG_NEON_KEYS];
        for (int p = 0; p < DG_NEON_KEYS / 2; p++)
            vst1q_f64(sum + p * 2, s[p]);
        for (int j = 0; j < valid; j++) {
            for (int i = n4; i < n; i++)
                sum[j] += (double)a[i] * k[j][i];
            out[j0 + j] = (float)sum[j];
        }
    }
}

static void dg_rms_neon(float *y, const float *x, const DGTensor *scale, int n) {
    float q = dg_rms_inverse(x, n);
    int i = 0;
    if (scale)
        for (; i + 3 < n; i += 4)
            vst1q_f32(y + i, vmulq_f32(vmulq_n_f32(vld1q_f32(x + i), q),
                                       dg_bf16x4_neon(scale->data + (size_t)i * 2)));
    else
        for (; i + 3 < n; i += 4)
            vst1q_f32(y + i, vmulq_n_f32(vld1q_f32(x + i), q));
    for (; i < n; i++)
        y[i] = x[i] * q * (scale ? dg_at(scale, i) : 1.0f);
}
#endif
