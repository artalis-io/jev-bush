/* CUDA accelerator for the transformer layers. The driver API and NVRTC are
 * loaded at run time and the kernels are compiled from the source below on
 * first use, so builds need neither CUDA headers nor nvcc. Without a driver,
 * NVRTC or a device, dg_accel() returns NULL and the CPU kernels run; so
 * does JB_CUDA=0 in the environment, and so do kernels that fail to compile
 * or load, which dg_accel_failure() then describes. The library never
 * writes to stderr itself; the CLI reports a failure there.
 *
 * Loading with dlopen and dlsym is the cost of needing no CUDA toolkit to
 * build: it happens only in -DJB_CUDA builds, on Linux, once, under
 * pthread_once, and only from the driver's and NVRTC's own library names.
 * Those names are searched the usual way, so whoever controls the library
 * path (LD_LIBRARY_PATH, ld.so.conf) chooses the code that runs, as with
 * any shared library. The resolved entry points live in dg_cuda, a
 * function-pointer table in writable memory because it is filled at run
 * time. It is written only inside pthread_once, before any reader, and is
 * read-only afterwards; jev-bush has no sandbox or sealed memory for an
 * overwrite to bypass, so the table is not made read-only after it is
 * filled. Every other dispatch table here, DGKernelOps and DGAccelOps
 * included, is static const.
 *
 * NVRTC compiles with --fmad=false and
 * IEEE division and square root, and every kernel keeps the reference's
 * order of operations for each value, so results match the reference bit
 * for bit, which the selftest checks. Fast builds (-ffast-math), like the
 * CPU's, give that up for speed: on GPUs with BF16 tensor cores (compute
 * capability 8.0 or later) attention runs on tensor cores in FP32-level
 * precision instead, which the selftest bounds. Each calling thread makes the context
 * current; work is ordered on the legacy default stream, and a copy to the
 * host waits for it. */
#if defined(JB_CUDA)
#if defined(_WIN32)
#error "JB_CUDA loads CUDA with dlopen, which Windows lacks"
#endif
#include <dlfcn.h>
#include <pthread.h>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef unsigned long long CUdeviceptr;
typedef int nvrtcResult;
typedef struct _nvrtcProgram *nvrtcProgram;

enum {
    DG_CU_CC_MAJOR = 75,
    DG_CU_CC_MINOR = 76,
    DG_CUDA_ROWS = 128,
    DG_CUDA_TOKENS = 8,
    DG_CUDA_BF16_TOKENS = 32,
    DG_CUDA_THREADS = 256,
    DG_CUDA_ROW_MAX = 4096,
    /* Tiles a block of the FP4 expert kernel takes, reading weights once. */
    DG_CUDA_FP4_RUN = 8,
    DG_CUDA_GRID_Y = 65535
};

/* The kernels. Device helpers repeat the host functions of the same name
 * operation for operation. dg_nvfp4_mm and dg_bf16_mm give each thread one
 * output row and each tile of up to DG_TOKENS tokens a register sum per
 * token, staging weights and activations in shared memory with coalesced
 * loads; the staging loops have fixed trip counts, so every load of a chunk
 * is issued before the first store waits for one, and staged weights are
 * padded by one row so the stores spread over the memory banks. dg_rms and dg_moe_tail stage a
 * token's row in shared memory and sum its squares on one thread, in order, in double. DG_ROWS,
 * DG_TOKENS and DG_ROW_MAX come from the compile options. */
static const char *const dg_cuda_source[] = {
    "struct dg_matrix {\n"
    "    const unsigned char *w, *s;\n"
    "    float global, pad;\n"
    "};\n"
    "\n"
    "struct dg_tile {\n"
    "    int start, count, group;\n"
    "};\n"
    "\n"
    "#define DG_BLOCKS 16\n"
    "#define DG_BCOLS 64\n"
    "\n"
    "__device__ float dg_bf(unsigned short h) {\n"
    "    return __uint_as_float((unsigned)h << 16);\n"
    "}\n"
    "\n"
    "__device__ int dg_finitef(float x) {\n"
    "    return ((__float_as_uint(x) >> 23) & 255) != 255;\n"
    "}\n"
    "\n"
    "__device__ float dg_f8e4m3(unsigned u) {\n"
    "    unsigned e = (u >> 3) & 15, m = u & 7;\n"
    "    float x = e ? __uint_as_float(((e + 120) << 23) | (m << 20)) : (float)m * 1.953125e-3f;\n"
    "    return u >> 7 ? -x : x;\n"
    "}\n"
    "\n"
    "__device__ float dg_f8e4m3_round(float x) {\n"
    "    if (!(x > 0))\n"
    "        return 0;\n"
    "    if (x >= 448)\n"
    "        return 448;\n"
    "    int lo = 0, hi = 126;\n"
    "    while (lo + 1 < hi) {\n"
    "        int m = (lo + hi) / 2;\n"
    "        if (dg_f8e4m3((unsigned)m) < x)\n"
    "            lo = m;\n"
    "        else\n"
    "            hi = m;\n"
    "    }\n"
    "    float a = dg_f8e4m3((unsigned)lo), b = dg_f8e4m3((unsigned)hi), da = x - a, db = b - x;\n"
    "    return da < db || (da == db && !(lo & 1)) ? a : b;\n"
    "}\n"
    "\n"
    "__device__ float dg_e2m1_round(float x) {\n"
    "    const float q[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};\n"
    "    float a = fabsf(x);\n"
    "    int best = 0;\n"
    "    for (int i = 1; i < 8; i++) {\n"
    "        float d = fabsf(a - q[i]), old = fabsf(a - q[best]);\n"
    "        if (d < old || (d == old && !(i & 1)))\n"
    "            best = i;\n"
    "    }\n"
    "    return signbit(x) ? -q[best] : q[best];\n"
    "}\n"
    "\n"
    "__device__ float dg_expf(float x) {\n"
    "    if (x != x)\n"
    "        return x;\n"
    "    if (x > 88.72283935546875f)\n"
    "        return __int_as_float(0x7f800000);\n"
    "    if (x < -103.97207641601562f)\n"
    "        return 0;\n"
    "    int e = (int)(x * 1.44269504088896341f + (x < 0 ? -0.5f : 0.5f));\n"
    "    float n = (float)e;\n"
    "    float r = x - n * 0.693359375f;\n"
    "    r = r - n * -2.12194440e-4f;\n"
    "    float p = 1.9875691500e-4f;\n"
    "    p = p * r + 1.3981999507e-3f;\n"
    "    p = p * r + 8.3334519073e-3f;\n"
    "    p = p * r + 4.1665795894e-2f;\n"
    "    p = p * r + 1.6666665459e-1f;\n"
    "    p = p * r + 5.0000001201e-1f;\n"
    "    p = p * (r * r) + r + 1.0f;\n"
    "    if (e > 127)\n"
    "        return p * __uint_as_float(254u << 23) * 2.0f;\n"
    "    if (e < -126)\n"
    "        return p * __uint_as_float((unsigned)(e + 253) << 23) * 0x1p-126f;\n"
    "    return p * __uint_as_float((unsigned)(e + 127) << 23);\n"
    "}\n"
    "\n"
    "__device__ float dg_tanhf(float x) {\n"
    "    float a = x < 0 ? -x : x;\n"
    "    if (a < 0x1p-12f)\n"
    "        return x;\n"
    "    if (a < 0.625f) {\n"
    "        float z = x * x, p = -5.70498872745e-3f;\n"
    "        p = p * z + 2.06390887954e-2f;\n"
    "        p = p * z - 5.37397155531e-2f;\n"
    "        p = p * z + 1.33314422036e-1f;\n"
    "        p = p * z - 3.33332819422e-1f;\n"
    "        return p * z * x + x;\n"
    "    }\n"
    "    if (a >= 10.0f)\n"
    "        return x < 0 ? -1.0f : 1.0f;\n"
    "    float t = 1.0f - 2.0f / (dg_expf(a + a) + 1.0f);\n"
    "    return x < 0 ? -t : t;\n"
    "}\n"
    "\n"
    "__device__ float dg_gelu(float x) {\n"
    "    return .5f * x * (1.0f + dg_tanhf(.7978845608028654f * (x + .044715f * x * x * x)));\n"
    "}\n"
    "\n"
    "__device__ double dg_exact_product(float a, float b) {\n"
    "    unsigned ua = __float_as_uint(a), ub = __float_as_uint(b);\n"
    "    unsigned ea = (ua >> 23) & 255, eb = (ub >> 23) & 255;\n"
    "    if (ea == 255 || eb == 255)\n"
    "        return (double)a * b;\n"
    "    unsigned long long sign = (unsigned long long)((ua ^ ub) >> 31) << 63;\n"
    "    unsigned ma = ua & 0x7fffffu, mb = ub & 0x7fffffu;\n"
    "    if (ea)\n"
    "        ma |= 0x800000u;\n"
    "    else\n"
    "        ea = 1;\n"
    "    if (eb)\n"
    "        mb |= 0x800000u;\n"
    "    else\n"
    "        eb = 1;\n"
    "    unsigned long long m = (unsigned long long)ma * mb;\n"
    "    if (!m)\n"
    "        return __longlong_as_double((long long)sign);\n"
    "    int lz = __clzll((long long)m);\n",
    "    unsigned long long e = (unsigned long long)(63 - lz + (int)ea + (int)eb - 300 + 1023);\n"
    "    unsigned long long frac = (m << (lz + 1)) >> 12;\n"
    "    return __longlong_as_double((long long)(sign | (e << 52) | frac));\n"
    "}\n"
    "\n"
    "/* dg_rms_inverse from the row's squares, which are exact in double and\n"
    " * computed in parallel beforehand; only the additions are sequential. */\n"
    "__device__ float dg_rms_inverse_squares(const double *sq, int n) {\n"
    "    double ss = 0.0;\n"
    "    for (int i = 0; i < n; i++)\n"
    "        ss += sq[i];\n"
    "    return 1.0f / sqrtf((float)(ss / n) + 1e-6f);\n"
    "}\n"
    "\n"
    "/* The inverse RMS of a staged row's squares into *q, called by every thread\n"
    " * of the block. Strict builds sum on one thread in order; fast builds\n"
    " * split the sum over the block's threads, still in double. Callers\n"
    " * synchronize before reading *q. */\n"
    "__device__ void dg_rms_inverse_block(const double *sq, int n, float *q) {\n"
    "#if DG_FAST\n"
    "    __shared__ double part[32];\n"
    "    double s = 0;\n"
    "    for (int i = threadIdx.x; i < n; i += blockDim.x)\n"
    "        s += sq[i];\n"
    "    for (int off = 16; off > 0; off >>= 1)\n"
    "        s += __shfl_down_sync(0xffffffffu, s, off);\n"
    "    if ((threadIdx.x & 31) == 0)\n"
    "        part[threadIdx.x >> 5] = s;\n"
    "    __syncthreads();\n"
    "    if (threadIdx.x == 0) {\n"
    "        double t = 0;\n"
    "        for (int w = 0; w < (int)(blockDim.x + 31) / 32; w++)\n"
    "            t += part[w];\n"
    "        *q = 1.0f / sqrtf((float)(t / n) + 1e-6f);\n"
    "    }\n"
    "#else\n"
    "    if (threadIdx.x == 0)\n"
    "        *q = dg_rms_inverse_squares(sq, n);\n"
    "#endif\n"
    "}\n"
    "\n"
    "\n"
    "/* One staged chunk's contribution to S token sums of a thread's row, in the\n"
    " * reference's order. Slots past the tile's count read stale activations and\n"
    " * are never stored; a block's threads share one tile, so choosing S by its\n"
    " * count does not diverge. */\n"
    "template <int S>\n"
    "__device__ __forceinline__ void dg_nvfp4_chunk(float *sum, const uint2 *ws, const float *ss,\n"
    "        const float *xs, const float *e2m1, int nb, int tx) {\n"
    "    for (int b = 0; b < nb; b++) {\n"
    "        float scale = ss[b * (DG_ROWS + 1) + tx];\n"
    "        uint2 q = ws[b * (DG_ROWS + 1) + tx];\n"
    "#pragma unroll\n"
    "        for (int k = 0; k < 8; k++) {\n"
    "            unsigned v = ((k < 4 ? q.x : q.y) >> (8 * (k & 3))) & 255;\n"
    "            float wl = e2m1[v & 15], wh = e2m1[v >> 4];\n"
    "#pragma unroll\n"
    "            for (int t = 0; t < S; t++)\n"
    "                sum[t] += scale * (wl * xs[t * DG_BLOCKS * 16 + b * 16 + k * 2] +\n"
    "                                   wh * xs[t * DG_BLOCKS * 16 + b * 16 + k * 2 + 1]);\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "/* dg_nvfp4_mm_ref, tiled: a block of DG_ROWS threads computes DG_ROWS rows\n"
    " * for one tile of up to DG_TOKENS tokens that share a matrix, one register\n"
    " * sum per token. Each chunk of DG_BLOCKS weight blocks of those rows, their\n"
    " * scales and the tokens' activations is staged in shared memory: every load\n"
    " * of a chunk issues before the first store, and staged weights are padded by\n"
    " * one row so the stores spread over the memory banks. The E2M1 table keeps\n"
    " * -0.0, as the reference's sign flip produces it. */\n"
    "extern \"C\" __global__ void __launch_bounds__(DG_ROWS)\n"
    "dg_nvfp4_mm(const dg_matrix *m, const dg_tile *tiles, const int *count, const float *x,\n"
    "        float *y, int rows, int cols) {\n"
    "    __shared__ float e2m1[16];\n"
    "    __shared__ uint2 ws[DG_BLOCKS][DG_ROWS + 1];\n"
    "    __shared__ float ss[DG_BLOCKS][DG_ROWS + 1];\n"
    "    __shared__ float xs[DG_TOKENS][DG_BLOCKS * 16];\n"
    "    int tx = threadIdx.x, r0 = blockIdx.x * DG_ROWS, r = r0 + tx, blocks = cols / 16;\n"
    "    int ntiles = *count;\n"
    "    if (tx < 16) {\n",
    "        const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};\n"
    "        e2m1[tx] = tx & 8 ? -mag[tx & 7] : mag[tx & 7];\n"
    "    }\n"
    "    for (int ti = blockIdx.y; ti < ntiles; ti += gridDim.y) {\n"
    "        dg_tile tile = tiles[ti];\n"
    "        dg_matrix mt = m[tile.group];\n"
    "        float sum[DG_TOKENS];\n"
    "#pragma unroll\n"
    "        for (int t = 0; t < DG_TOKENS; t++)\n"
    "            sum[t] = 0;\n"
    "        for (int b0 = 0; b0 < blocks; b0 += DG_BLOCKS) {\n"
    "            int nb = blocks - b0 < DG_BLOCKS ? blocks - b0 : DG_BLOCKS;\n"
    "            uint2 wv[DG_BLOCKS];\n"
    "            unsigned sv[DG_BLOCKS];\n"
    "            float xv[DG_TOKENS * DG_BLOCKS * 16 / DG_ROWS];\n"
    "            __syncthreads();\n"
    "#pragma unroll\n"
    "            for (int j = 0; j < DG_BLOCKS; j++) {\n"
    "                int i = tx + j * DG_ROWS, row = i / DG_BLOCKS, b = i % DG_BLOCKS;\n"
    "                int gr = r0 + row < rows ? r0 + row : rows - 1;\n"
    "                if (b < nb) {\n"
    "                    wv[j] = *(const uint2 *)(mt.w + (size_t)gr * (cols / 2) + (size_t)(b0 + "
    "b) * 8);\n"
    "                    sv[j] = mt.s[(size_t)gr * blocks + b0 + b];\n"
    "                }\n"
    "            }\n"
    "#pragma unroll\n"
    "            for (int j = 0; j < DG_TOKENS * DG_BLOCKS * 16 / DG_ROWS; j++) {\n"
    "                int i = tx + j * DG_ROWS, t = i / (DG_BLOCKS * 16), c = i % (DG_BLOCKS * "
    "16);\n"
    "                if (t < tile.count && c < nb * 16)\n"
    "                    xv[j] = x[(size_t)(tile.start + t) * cols + (size_t)b0 * 16 + c];\n"
    "            }\n"
    "#pragma unroll\n"
    "            for (int j = 0; j < DG_BLOCKS; j++) {\n"
    "                int i = tx + j * DG_ROWS, row = i / DG_BLOCKS, b = i % DG_BLOCKS;\n"
    "                if (b < nb) {\n"
    "                    ws[b][row] = wv[j];\n"
    "                    ss[b][row] = dg_f8e4m3(sv[j]) * mt.global;\n"
    "                }\n"
    "            }\n"
    "#pragma unroll\n"
    "            for (int j = 0; j < DG_TOKENS * DG_BLOCKS * 16 / DG_ROWS; j++) {\n"
    "                int i = tx + j * DG_ROWS, t = i / (DG_BLOCKS * 16), c = i % (DG_BLOCKS * "
    "16);\n"
    "                if (t < tile.count && c < nb * 16)\n"
    "                    xs[t][c] = xv[j];\n"
    "            }\n"
    "            __syncthreads();\n"
    "            if (tile.count == 1)\n"
    "                dg_nvfp4_chunk<1>(sum, &ws[0][0], &ss[0][0], &xs[0][0], e2m1, nb, tx);\n"
    "            else if (tile.count == 2)\n"
    "                dg_nvfp4_chunk<2>(sum, &ws[0][0], &ss[0][0], &xs[0][0], e2m1, nb, tx);\n"
    "            else if (tile.count <= 4)\n"
    "                dg_nvfp4_chunk<4>(sum, &ws[0][0], &ss[0][0], &xs[0][0], e2m1, nb, tx);\n"
    "            else\n"
    "                dg_nvfp4_chunk<DG_TOKENS>(sum, &ws[0][0], &ss[0][0], &xs[0][0], e2m1, nb, "
    "tx);\n"
    "        }\n"
    "        if (r < rows)\n"
    "#pragma unroll\n"
    "            for (int t = 0; t < DG_TOKENS; t++)\n"
    "                if (t < tile.count)\n"
    "                    y[(size_t)(tile.start + t) * rows + r] = sum[t];\n"
    "    }\n"
    "}\n"
    "\n"
    "/* dg_mm_data_ref: each row's BF16 weights in column order, staged in\n"
    " * pairs, times a tile of tokens. */\n"
    "extern \"C\" __global__ void __launch_bounds__(DG_ROWS)\n"
    "dg_bf16_mm(const unsigned short *w, const float *x, float *y, int tokens, int rows,\n"
    "        int cols) {\n"
    "    __shared__ unsigned ws[DG_BCOLS / 2][DG_ROWS + 1];\n"
    "    __shared__ float xs[DG_BTOKENS][DG_BCOLS];\n"
    "    int tx = threadIdx.x, r0 = blockIdx.x * DG_ROWS, r = r0 + tx;\n"
    "    int ntiles = (tokens + DG_BTOKENS - 1) / DG_BTOKENS;\n"
    "    for (int ti = blockIdx.y; ti < ntiles; ti += gridDim.y) {\n"
    "        int t0 = ti * DG_BTOKENS, count = tokens - t0 < DG_BTOKENS ? tokens - t0 : "
    "DG_BTOKENS;\n"
    "        float sum[DG_BTOKENS];\n"
    "#pragma unroll\n",
    "        for (int t = 0; t < DG_BTOKENS; t++)\n"
    "            sum[t] = 0;\n"
    "        for (int c0 = 0; c0 < cols; c0 += DG_BCOLS) {\n"
    "            int nc = cols - c0 < DG_BCOLS ? cols - c0 : DG_BCOLS, np = nc / 2;\n"
    "            __syncthreads();\n"
    "            if (nc == DG_BCOLS && !(cols & 7)) {\n"
    "                uint4 wv[DG_BCOLS / 8];\n"
    "#pragma unroll\n"
    "                for (int j = 0; j < DG_BCOLS / 8; j++) {\n"
    "                    int i = tx + j * DG_ROWS, row = i / (DG_BCOLS / 8), q = i % (DG_BCOLS / "
    "8);\n"
    "                    int gr = r0 + row < rows ? r0 + row : rows - 1;\n"
    "                    wv[j] = *(const uint4 *)(w + (size_t)gr * cols + c0 + q * 8);\n"
    "                }\n"
    "#pragma unroll\n"
    "                for (int j = 0; j < DG_BCOLS / 8; j++) {\n"
    "                    int i = tx + j * DG_ROWS, row = i / (DG_BCOLS / 8), q = i % (DG_BCOLS / "
    "8);\n"
    "                    ws[q * 4][row] = wv[j].x;\n"
    "                    ws[q * 4 + 1][row] = wv[j].y;\n"
    "                    ws[q * 4 + 2][row] = wv[j].z;\n"
    "                    ws[q * 4 + 3][row] = wv[j].w;\n"
    "                }\n"
    "            } else\n"
    "                for (int i = tx; i < DG_ROWS * np; i += DG_ROWS) {\n"
    "                    int row = i / np, pr = i % np, gr = r0 + row < rows ? r0 + row : rows - "
    "1;\n"
    "                    ws[pr][row] = ((const unsigned *)(w + (size_t)gr * cols + c0))[pr];\n"
    "                }\n"
    "            float xv[DG_BTOKENS * DG_BCOLS / DG_ROWS];\n"
    "#pragma unroll\n"
    "            for (int j = 0; j < DG_BTOKENS * DG_BCOLS / DG_ROWS; j++) {\n"
    "                int i = tx + j * DG_ROWS, t = i / DG_BCOLS, c = i % DG_BCOLS;\n"
    "                if (t < count && c < nc)\n"
    "                    xv[j] = x[(size_t)(t0 + t) * cols + c0 + c];\n"
    "            }\n"
    "#pragma unroll\n"
    "            for (int j = 0; j < DG_BTOKENS * DG_BCOLS / DG_ROWS; j++) {\n"
    "                int i = tx + j * DG_ROWS, t = i / DG_BCOLS, c = i % DG_BCOLS;\n"
    "                if (t < count && c < nc)\n"
    "                    xs[t][c] = xv[j];\n"
    "            }\n"
    "            __syncthreads();\n"
    "            for (int pr = 0; pr < np; pr++) {\n"
    "                unsigned p = ws[pr][tx];\n"
    "                float a0 = __uint_as_float(p << 16), a1 = __uint_as_float(p & 0xffff0000u);\n"
    "#pragma unroll\n"
    "                for (int t = 0; t < DG_BTOKENS; t++)\n"
    "                    if (t < count) {\n"
    "                        sum[t] += a0 * xs[t][pr * 2];\n"
    "                        sum[t] += a1 * xs[t][pr * 2 + 1];\n"
    "                    }\n"
    "            }\n"
    "        }\n"
    "        if (r < rows)\n"
    "#pragma unroll\n"
    "            for (int t = 0; t < DG_BTOKENS; t++)\n"
    "                if (t < count)\n"
    "                    y[(size_t)(t0 + t) * rows + r] = sum[t];\n"
    "    }\n"
    "}\n"
    "\n"
    "/* One token per block: y = x * q * scale, then * (factor / divisor), then\n"
    " * + add, each step when its input is given. */\n"
    "extern \"C\" __global__ void dg_rms(float *y, const float *x, const unsigned short *scale,\n"
    "        const unsigned short *factor, float divisor, const float *add, int tokens, int n) {\n"
    "    __shared__ double sq[DG_ROW_MAX];\n"
    "    __shared__ float q;\n"
    "    for (int t = blockIdx.x; t < tokens; t += gridDim.x) {\n"
    "        const float *xr = x + (size_t)t * n;\n"
    "        __syncthreads();\n"
    "        for (int i = threadIdx.x; i < n; i += blockDim.x)\n"
    "            sq[i] = (double)xr[i] * xr[i];\n"
    "        __syncthreads();\n"
    "        dg_rms_inverse_block(sq, n, &q);\n"
    "        __syncthreads();\n"
    "        for (int i = threadIdx.x; i < n; i += blockDim.x) {\n"
    "            float v = xr[i] * q;\n"
    "            if (scale)\n"
    "                v = v * dg_bf(scale[i]);\n"
    "            if (factor)\n",
    "                v = v * (dg_bf(factor[i]) / divisor);\n"
    "            if (add)\n"
    "                v = v + add[(size_t)t * n + i];\n"
    "            y[(size_t)t * n + i] = v;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void dg_gelu_mul(float *out, const float *g, const float *u,\n"
    "        long long n) {\n"
    "    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;\n"
    "         i += (long long)gridDim.x * blockDim.x)\n"
    "        out[i] = dg_gelu(g[i]) * u[i];\n"
    "}\n"
    "\n"
    "/* dg_nvfp4_qdq_parallel, one 16-value block per thread. */\n"
    "extern \"C\" __global__ void dg_nvfp4_qdq(float *out, const float *in, int tokens, int cols,\n"
    "        float base, int *flag) {\n"
    "    long long nb = cols / 16, blocks = (long long)tokens * nb;\n"
    "    for (long long z = blockIdx.x * (long long)blockDim.x + threadIdx.x; z < blocks;\n"
    "         z += (long long)gridDim.x * blockDim.x) {\n"
    "        long long t = z / nb, b = z % nb;\n"
    "        const float *x = in + t * cols + b * 16;\n"
    "        float *y = out + t * cols + b * 16, amax = 0;\n"
    "        int bad = 0;\n"
    "        for (int k = 0; k < 16; k++) {\n"
    "            bad |= !dg_finitef(x[k]);\n"
    "            if (fabsf(x[k]) > amax)\n"
    "                amax = fabsf(x[k]);\n"
    "        }\n"
    "        if (bad) {\n"
    "            atomicOr(flag, 1);\n"
    "            continue;\n"
    "        }\n"
    "        float s = dg_f8e4m3_round((amax / 6) / base) * base;\n"
    "        if (s == 0) {\n"
    "            for (int k = 0; k < 16; k++)\n"
    "                y[k] = 0;\n"
    "            continue;\n"
    "        }\n"
    "        for (int k = 0; k < 16; k++)\n"
    "            y[k] = dg_e2m1_round(x[k] / s) * s;\n"
    "    }\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void dg_gather(float *dst, const float *src, const int *index,\n"
    "        int rows, int cols) {\n"
    "    for (int r = blockIdx.x; r < rows; r += gridDim.x)\n"
    "        for (int i = threadIdx.x; i < cols; i += blockDim.x)\n"
    "            dst[(size_t)r * cols + i] = src[(size_t)index[r] * cols + i];\n"
    "}\n"
    "\n"
    "/* dg_ff's tail for one token per block, then the layer scalar. */\n"
    "extern \"C\" __global__ void dg_moe_tail(float *x, const float *eo, const int *slot,\n"
    "        const float *weight, const float *d, const unsigned short *p2,\n"
    "        const unsigned short *post, float scalar, int tokens, int n, int topk) {\n"
    "    __shared__ float row[DG_WIDTH];\n"
    "    __shared__ double sq[DG_WIDTH];\n"
    "    __shared__ float q;\n"
    "    for (int t = blockIdx.x; t < tokens; t += gridDim.x) {\n"
    "        __syncthreads();\n"
    "        for (int i = threadIdx.x; i < n; i += blockDim.x) {\n"
    "            float mo = 0;\n"
    "            for (int k = 0; k < topk; k++)\n"
    "                mo += weight[t * topk + k] * eo[(size_t)slot[t * topk + k] * n + i];\n"
    "            row[i] = mo;\n"
    "            sq[i] = (double)mo * mo;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        dg_rms_inverse_block(sq, n, &q);\n"
    "        __syncthreads();\n"
    "        for (int i = threadIdx.x; i < n; i += blockDim.x) {\n"
    "            row[i] = d[(size_t)t * n + i] + row[i] * q * dg_bf(p2[i]);\n"
    "            sq[i] = (double)row[i] * row[i];\n"
    "        }\n"
    "        __syncthreads();\n"
    "        dg_rms_inverse_block(sq, n, &q);\n"
    "        __syncthreads();\n"
    "        for (int i = threadIdx.x; i < n; i += blockDim.x) {\n"
    "            float v = x[(size_t)t * n + i] + row[i] * q * dg_bf(post[i]);\n"
    "            x[(size_t)t * n + i] = v * scalar;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "/* dg_rope over tokens * heads rows of hd values, stride apart per token;\n"
    " * each token's cosines then sines are pitch apart in table. */\n"
    "extern \"C\" __global__ void dg_rope(float *x, const float *table, int tokens, int heads, int "
    "hd,\n"
    "        int stride, int pitch) {\n"
    "    int half = hd / 2;\n",
    "    long long total = (long long)tokens * heads * half;\n"
    "    for (long long z = blockIdx.x * (long long)blockDim.x + threadIdx.x; z < total;\n"
    "         z += (long long)gridDim.x * blockDim.x) {\n"
    "        int i = (int)(z % half);\n"
    "        long long th = z / half, t = th / heads;\n"
    "        int h = (int)(th % heads);\n"
    "        float *v = x + t * stride + (long long)h * hd;\n"
    "        const float *cv = table + t * pitch, *sv = cv + half;\n"
    "        float a = v[i], b = v[i + half];\n"
    "        v[i] = a * cv[i] - b * sv[i];\n"
    "        v[i + half] = b * cv[i] + a * sv[i];\n"
    "    }\n"
    "}\n"
    "\n"
    "/* One block per token and head: each key's score is a double sum in index\n"
    " * order, as dg_dot_ref; the maximum is order-free; exponentials use\n"
    " * dg_expf; the denominator is summed in key order on one thread; and each\n"
    " * output value sums its weighted values in key order. */\n"
    "extern \"C\" __global__ void dg_attention(float *out, const float *q, const float *kt,\n"
    "        const float *v, int keys, const float *v_old, const int *seg_old,\n"
    "        const int *seg_base, int n, int seq, int heads, int kvh, int hd, int canvas, int "
    "full,\n"
    "        int window) {\n"
    "    __shared__ float ts[DG_ROW_MAX];\n"
    "    __shared__ float qs[DG_HEAD_MAX];\n"
    "    __shared__ float red[DG_THREADS];\n"
    "    __shared__ float den;\n"
    "    int tx = threadIdx.x, bd = blockDim.x, qn = heads * hd, kn = kvh * hd;\n"
    "    size_t old_rows = (size_t)keys - n;\n"
    "    for (int job = blockIdx.x; job < n * heads; job += gridDim.x) {\n"
    "        int ti = job / heads, h = job % heads, s = ti / seq, t = ti % seq, old = seg_old[s];\n"
    "        int kh = h / (heads / kvh), end = canvas ? old + seq : old + t + 1, start = 0;\n"
    "        if (!full) {\n"
    "            if (canvas) {\n"
    "                int pos = old + t;\n"
    "                start = pos >= window ? pos - window + 1 : 0;\n"
    "                if (end > pos + window)\n"
    "                    end = pos + window;\n"
    "            } else\n"
    "                start = end > window ? end - window : 0;\n"
    "        }\n"
    "        size_t cached = (size_t)seg_base[s], own = (size_t)s * seq, head = (size_t)kh * hd;\n"
    "        __syncthreads();\n"
    "        for (int i = tx; i < hd; i += bd)\n"
    "            qs[i] = q[(size_t)ti * qn + (size_t)h * hd + i];\n"
    "        __syncthreads();\n"
    "        float mx = -3.402823466e+38f;\n"
    "        for (int j = start + tx; j < end; j += bd) {\n"
    "            size_t g = j < old ? cached + j : old_rows + own + (j - old);\n"
    "            const float *kk = kt + head * keys + g;\n"
    "            double sum = 0;\n"
    "            for (int i = 0; i < hd; i++)\n"
    "                sum += dg_exact_product(qs[i], kk[(size_t)i * keys]);\n"
    "            ts[j] = (float)sum;\n"
    "            if (ts[j] > mx)\n"
    "                mx = ts[j];\n"
    "        }\n"
    "        red[tx] = mx;\n"
    "        __syncthreads();\n"
    "        for (int w = bd / 2; w > 0; w >>= 1) {\n"
    "            if (tx < w && red[tx + w] > red[tx])\n"
    "                red[tx] = red[tx + w];\n"
    "            __syncthreads();\n"
    "        }\n"
    "        mx = red[0];\n"
    "        for (int j = start + tx; j < end; j += bd)\n"
    "            ts[j] = dg_expf(ts[j] - mx);\n"
    "        __syncthreads();\n"
    "        if (tx == 0) {\n"
    "            float sum = 0;\n"
    "            for (int j = start; j < end; j++)\n"
    "                sum += ts[j];\n"
    "            den = sum;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        for (int j = start + tx; j < end; j += bd)\n"
    "            ts[j] = ts[j] / den;\n"
    "        __syncthreads();\n"
    "        for (int d = tx; d < hd; d += bd) {\n"
    "            float o = 0;\n"
    "            for (int j = start; j < end; j++)\n"
    "                o += ts[j] * (j < old ? v_old[(cached + j) * kn + head + d]\n",
    "                                      : v[(own + (j - old)) * kn + head + d]);\n"
    "            out[(size_t)ti * qn + (size_t)h * hd + d] = o;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "/* dg_mm_data_ref for column counts that are a multiple of DG_BCOLS. A block\n"
    " * covers ROWS rows and TOKENS tokens with ROWS x TY threads: thread (x, y)\n"
    " * owns row x and TOKENS / TY tokens, one register sum each in column order,\n"
    " * and the block's weights are staged once for all of them. While a chunk is\n"
    " * computed from shared memory, the next chunk's weights and activations are\n"
    " * already loading into registers. */\n"
    "template <int ROWS, int TOKENS, int TY>\n"
    "__device__ void dg_bf16_body(const unsigned short *w, const float *x, float *y, int tokens,\n"
    "        int rows, int cols) {\n"
    "    __shared__ unsigned ws[DG_BCOLS / 2][ROWS + 1];\n"
    "    __shared__ float xs[TOKENS][DG_BCOLS];\n"
    "    const int NT = ROWS * TY, WV = ROWS * DG_BCOLS / 8 / NT, XV = TOKENS * DG_BCOLS / NT,\n"
    "              TS = TOKENS / TY;\n"
    "    int tx = threadIdx.x, ty = threadIdx.y, id = ty * ROWS + tx;\n"
    "    int r0 = blockIdx.x * ROWS, r = r0 + tx;\n"
    "    int ntiles = (tokens + TOKENS - 1) / TOKENS;\n"
    "    for (int ti = blockIdx.y; ti < ntiles; ti += gridDim.y) {\n"
    "        int t0 = ti * TOKENS, count = tokens - t0 < TOKENS ? tokens - t0 : TOKENS;\n"
    "        float sum[TS];\n"
    "        uint4 wv[WV];\n"
    "        float xv[XV];\n"
    "#pragma unroll\n"
    "        for (int k = 0; k < TS; k++)\n"
    "            sum[k] = 0;\n"
    "#pragma unroll\n"
    "        for (int j = 0; j < WV; j++) {\n"
    "            int i = id + j * NT, row = i / (DG_BCOLS / 8), q = i % (DG_BCOLS / 8);\n"
    "            int gr = r0 + row < rows ? r0 + row : rows - 1;\n"
    "            wv[j] = *(const uint4 *)(w + (size_t)gr * cols + q * 8);\n"
    "        }\n"
    "#pragma unroll\n"
    "        for (int j = 0; j < XV; j++) {\n"
    "            int i = id + j * NT, t = i / DG_BCOLS, c = i % DG_BCOLS;\n"
    "            if (t < count)\n"
    "                xv[j] = x[(size_t)(t0 + t) * cols + c];\n"
    "        }\n"
    "        for (int c0 = 0; c0 < cols; c0 += DG_BCOLS) {\n"
    "            __syncthreads();\n"
    "#pragma unroll\n"
    "            for (int j = 0; j < WV; j++) {\n"
    "                int i = id + j * NT, row = i / (DG_BCOLS / 8), q = i % (DG_BCOLS / 8);\n"
    "                ws[q * 4][row] = wv[j].x;\n"
    "                ws[q * 4 + 1][row] = wv[j].y;\n"
    "                ws[q * 4 + 2][row] = wv[j].z;\n"
    "                ws[q * 4 + 3][row] = wv[j].w;\n"
    "            }\n"
    "#pragma unroll\n"
    "            for (int j = 0; j < XV; j++) {\n"
    "                int i = id + j * NT, t = i / DG_BCOLS, c = i % DG_BCOLS;\n"
    "                if (t < count)\n"
    "                    xs[t][c] = xv[j];\n"
    "            }\n"
    "            __syncthreads();\n"
    "            int next = c0 + DG_BCOLS;\n"
    "            if (next < cols) {\n"
    "#pragma unroll\n"
    "                for (int j = 0; j < WV; j++) {\n"
    "                    int i = id + j * NT, row = i / (DG_BCOLS / 8), q = i % (DG_BCOLS / 8);\n"
    "                    int gr = r0 + row < rows ? r0 + row : rows - 1;\n"
    "                    wv[j] = *(const uint4 *)(w + (size_t)gr * cols + next + q * 8);\n"
    "                }\n"
    "#pragma unroll\n"
    "                for (int j = 0; j < XV; j++) {\n"
    "                    int i = id + j * NT, t = i / DG_BCOLS, c = i % DG_BCOLS;\n"
    "                    if (t < count)\n"
    "                        xv[j] = x[(size_t)(t0 + t) * cols + next + c];\n"
    "                }\n"
    "            }\n"
    "#pragma unroll 4\n"
    "            for (int pr = 0; pr < DG_BCOLS / 2; pr++) {\n"
    "                unsigned p = ws[pr][tx];\n"
    "                float a0 = __uint_as_float(p << 16), a1 = __uint_as_float(p & 0xffff0000u);\n"
    "#pragma unroll\n"
    "                for (int k = 0; k < TS; k++) {\n",
    "                    int t = ty * TS + k;\n"
    "                    if (t < count) {\n"
    "                        sum[k] += a0 * xs[t][pr * 2];\n"
    "                        sum[k] += a1 * xs[t][pr * 2 + 1];\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "        if (r < rows)\n"
    "#pragma unroll\n"
    "            for (int k = 0; k < TS; k++) {\n"
    "                int t = ty * TS + k;\n"
    "                if (t < count)\n"
    "                    y[(size_t)(t0 + t) * rows + r] = sum[k];\n"
    "            }\n"
    "    }\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void __launch_bounds__(128) dg_bf16_mm_wide(const unsigned short *w,\n"
    "        const float *x, float *y, int tokens, int rows, int cols) {\n"
    "    dg_bf16_body<128, 32, 1>(w, x, y, tokens, rows, cols);\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void __launch_bounds__(128) dg_bf16_mm_narrow(const unsigned short "
    "*w,\n"
    "        const float *x, float *y, int tokens, int rows, int cols) {\n"
    "    dg_bf16_body<32, 32, 4>(w, x, y, tokens, rows, cols);\n"
    "}\n"
    "/* Keys transposed once per layer, cached rows then this call's, to\n"
    " * kt[c * keys + g] for column c of kv width kn: the scores' threads, which\n"
    " * own consecutive keys, then read consecutive addresses. 32 x 32 tiles\n"
    " * through shared memory keep reads and writes coalesced. */\n"
    "extern \"C\" __global__ void dg_transpose_keys(float *kt, const float *k_old, const float "
    "*k,\n"
    "        int old_rows, int keys, int kn) {\n"
    "    __shared__ float tile[32][33];\n"
    "    int g0 = blockIdx.x * 32, c0 = blockIdx.y * 32, tx = threadIdx.x, ty = threadIdx.y;\n"
    "    for (int r = ty; r < 32; r += blockDim.y) {\n"
    "        int g = g0 + r, c = c0 + tx;\n"
    "        if (g < keys && c < kn)\n"
    "            tile[r][tx] = g < old_rows ? k_old[(size_t)g * kn + c]\n"
    "                                       : k[(size_t)(g - old_rows) * kn + c];\n"
    "    }\n"
    "    __syncthreads();\n"
    "    for (int r = ty; r < 32; r += blockDim.y) {\n"
    "        int c = c0 + r, g = g0 + tx;\n"
    "        if (g < keys && c < kn)\n"
    "            kt[(size_t)c * keys + g] = tile[tx][r];\n"
    "    }\n"
    "}\n"
    "/* dg_exp, fdlibm's __ieee754_exp, as the host computes it. */\n"
    "__device__ double dg_exp(double x) {\n"
    "    const double ln2_hi = 6.93147180369123816490e-01, ln2_lo = 1.90821492927058770002e-10,\n"
    "                 inv_ln2 = 1.44269504088896338700e+00, p1 = 1.66666666666666019037e-01,\n"
    "                 p2 = -2.77777777770155933842e-03, p3 = 6.61375632143793436117e-05,\n"
    "                 p4 = -1.65339022054652515390e-06, p5 = 4.13813679705723846039e-08,\n"
    "                 two_m1000 = 9.33263618503218878990e-302;\n"
    "    unsigned long long bits = (unsigned long long)__double_as_longlong(x);\n"
    "    unsigned hx = (unsigned)(bits >> 32) & 0x7fffffffu;\n"
    "    int negative = (int)(bits >> 63), k = 0;\n"
    "    double hi = 0, lo = 0;\n"
    "    if (hx >= 0x40862E42u) {\n"
    "        if (hx >= 0x7ff00000u)\n"
    "            return x != x ? x + x : negative ? 0.0 : x;\n"
    "        if (x > 7.09782712893383973096e+02)\n"
    "            return __longlong_as_double(0x7ff0000000000000LL);\n"
    "        if (x < -7.45133219101941108420e+02)\n"
    "            return 0.0;\n"
    "    }\n"
    "    if (hx > 0x3fd62e42u) {\n"
    "        if (hx < 0x3FF0A2B2u) {\n"
    "            hi = negative ? x + ln2_hi : x - ln2_hi;\n"
    "            lo = negative ? -ln2_lo : ln2_lo;\n"
    "            k = negative ? -1 : 1;\n"
    "        } else {\n"
    "            k = (int)(inv_ln2 * x + (negative ? -0.5 : 0.5));\n"
    "            double t = k;\n"
    "            hi = x - t * ln2_hi;\n"
    "            lo = t * ln2_lo;\n"
    "        }\n"
    "        x = hi - lo;\n"
    "    } else if (hx < 0x3e300000u)\n"
    "        return 1.0 + x;\n"
    "    double t = x * x, c = x - t * (p1 + t * (p2 + t * (p3 + t * (p4 + t * p5)))), y;\n"
    "    if (k == 0)\n",
    "        return 1.0 - ((x * c) / (c - 2.0) - x);\n"
    "    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);\n"
    "    unsigned long long yb = (unsigned long long)__double_as_longlong(y);\n"
    "    if (k >= -1021)\n"
    "        return __longlong_as_double((long long)(yb + ((unsigned long long)(long long)k << "
    "52)));\n"
    "    return __longlong_as_double((long long)(yb + ((unsigned long long)(long long)(k + 1000) "
    "<< 52))) *\n"
    "           two_m1000;\n"
    "}\n"
    "\n"
    "/* dg_route_token, one token per block: each thread computes one expert's\n"
    " * exponential, and one thread does the top-k and the sums in the\n"
    " * reference's order. A non-finite logit sets bit 2 of *flag. */\n"
    "extern \"C\" __global__ void dg_route(const float *logits, const unsigned short *scale, int "
    "tokens,\n"
    "        int *top, float *weight, int *flag) {\n"
    "    __shared__ double ex[DG_EXPERTS_N];\n"
    "    __shared__ float mxs;\n"
    "    for (int t = blockIdx.x; t < tokens; t += gridDim.x) {\n"
    "        const float *rt = logits + (size_t)t * DG_EXPERTS_N;\n"
    "        int e = threadIdx.x;\n"
    "        __syncthreads();\n"
    "        if (e == 0) {\n"
    "            float mx = rt[0];\n"
    "            for (int i = 1; i < DG_EXPERTS_N; i++)\n"
    "                if (rt[i] > mx)\n"
    "                    mx = rt[i];\n"
    "            mxs = mx;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        ex[e] = dg_exp((double)rt[e] - mxs);\n"
    "        if (!dg_finitef(rt[e]))\n"
    "            atomicOr(flag, 2);\n"
    "        __syncthreads();\n"
    "        if (e == 0) {\n"
    "            int ix[DG_TOPK_N];\n"
    "            float ev[DG_TOPK_N];\n"
    "            for (int k = 0; k < DG_TOPK_N; k++) {\n"
    "                ix[k] = k;\n"
    "                ev[k] = -3.402823466e+38f;\n"
    "            }\n"
    "            for (int i = 0; i < DG_EXPERTS_N; i++)\n"
    "                for (int k = 0; k < DG_TOPK_N; k++)\n"
    "                    if (rt[i] > ev[k]) {\n"
    "                        for (int q = DG_TOPK_N - 1; q > k; q--) {\n"
    "                            ev[q] = ev[q - 1];\n"
    "                            ix[q] = ix[q - 1];\n"
    "                        }\n"
    "                        ev[k] = rt[i];\n"
    "                        ix[k] = i;\n"
    "                        break;\n"
    "                    }\n"
    "            double all = 0, sel = 0;\n"
    "            for (int i = 0; i < DG_EXPERTS_N; i++)\n"
    "                all += ex[i];\n"
    "            for (int k = 0; k < DG_TOPK_N; k++) {\n"
    "                ev[k] = (float)(dg_exp((double)ev[k] - mxs) / all);\n"
    "                sel += ev[k];\n"
    "            }\n"
    "            for (int k = 0; k < DG_TOPK_N; k++) {\n"
    "                top[(size_t)t * DG_TOPK_N + k] = ix[k];\n"
    "                weight[(size_t)t * DG_TOPK_N + k] = ev[k] / (float)sel * "
    "dg_bf(scale[ix[k]]);\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "/* The routed rows grouped by expert, in the host loop's order (expert, then\n"
    " * token, then slot), and their tiles of up to DG_TOKENS rows; one block,\n"
    " * one thread per expert. */\n"
    "extern \"C\" __global__ void dg_group(const int *top, int tokens, int *index, int *slot,\n"
    "        dg_tile *tiles, int *ntiles) {\n"
    "    __shared__ int count[DG_EXPERTS_N], rowoff[DG_EXPERTS_N], tileoff[DG_EXPERTS_N];\n"
    "    int e = threadIdx.x, c = 0;\n"
    "    for (int t = 0; t < tokens; t++)\n"
    "        for (int k = 0; k < DG_TOPK_N; k++)\n"
    "            c += top[(size_t)t * DG_TOPK_N + k] == e;\n"
    "    count[e] = c;\n"
    "    __syncthreads();\n"
    "    if (e == 0) {\n"
    "        int rows = 0, n = 0;\n"
    "        for (int i = 0; i < DG_EXPERTS_N; i++) {\n"
    "            rowoff[i] = rows;\n"
    "            tileoff[i] = n;\n"
    "            rows += count[i];\n"
    "            n += (count[i] + DG_TOKENS - 1) / DG_TOKENS;\n"
    "        }\n"
    "        *ntiles = n;\n"
    "    }\n"
    "    __syncthreads();\n"
    "    int row = rowoff[e];\n",
    "    for (int t = 0; t < tokens; t++)\n"
    "        for (int k = 0; k < DG_TOPK_N; k++)\n"
    "            if (top[(size_t)t * DG_TOPK_N + k] == e) {\n"
    "                index[row] = t;\n"
    "                slot[(size_t)t * DG_TOPK_N + k] = row;\n"
    "                row++;\n"
    "            }\n"
    "    for (int i = 0; i * DG_TOKENS < c; i++) {\n"
    "        dg_tile tl;\n"
    "        tl.start = rowoff[e] + i * DG_TOKENS;\n"
    "        tl.count = c - i * DG_TOKENS < DG_TOKENS ? c - i * DG_TOKENS : DG_TOKENS;\n"
    "        tl.group = e;\n"
    "        tiles[tileoff[e] + i] = tl;\n"
    "    }\n"
    "}\n"
    "\n"
    "#if DG_FAST\n"
    "/* Fast builds only: tensor-core kernels that give up the reference's\n"
    " * bits, as -ffast-math does on the CPU. Each FP32 operand is split into a\n"
    " * TF32 high part and a TF32 low part, and a product takes three MMAs (high\n"
    " * by high, high by low, low by high), which keeps it about as accurate as\n"
    " * an FP32 product; sums accumulate in FP32. */\n"
    "#define DG_NEG_INF __int_as_float(0xff800000)\n"
    "\n"
    "__device__ __forceinline__ void dg_tf32_split(float x, unsigned &hi, unsigned &lo) {\n"
    "    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(hi) : \"f\"(x));\n"
    "    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(lo) : \"f\"(x - __uint_as_float(hi)));\n"
    "}\n"
    "\n"
    "/* c += a b for a 16 x 8 TF32 tile a, an 8 x 8 tile b and FP32 c, in the\n"
    " * m16n8k8 fragment layout. */\n"
    "__device__ __forceinline__ void dg_mma_tf32(float *c, const unsigned *a, unsigned b0,\n"
    "        unsigned b1) {\n"
    "    asm volatile(\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 \"\n"
    "                 \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
    "                 : \"+f\"(c[0]), \"+f\"(c[1]), \"+f\"(c[2]), \"+f\"(c[3])\n"
    "                 : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"r\"(b0),\n"
    "                         \"r\"(b1));\n"
    "}\n"
    "\n"
    "/* The split product of a 16 x 8 tile with elements x[0..3] in fragment\n"
    " * order and an 8 x 8 tile with elements y0 and y1. */\n"
    "__device__ __forceinline__ void dg_mma_split(float *c, const float *x, float y0, float y1) {\n"
    "    unsigned ah[4], al[4], bh0, bl0, bh1, bl1;\n"
    "    for (int i = 0; i < 4; i++)\n"
    "        dg_tf32_split(x[i], ah[i], al[i]);\n"
    "    dg_tf32_split(y0, bh0, bl0);\n"
    "    dg_tf32_split(y1, bh1, bl1);\n"
    "    dg_mma_tf32(c, ah, bh0, bh1);\n"
    "    dg_mma_tf32(c, ah, bl0, bl1);\n"
    "    dg_mma_tf32(c, al, bh0, bh1);\n"
    "}\n"
    "\n"
    "/* The keys query t of a segment may see, as dg_attention computes them. */\n"
    "__device__ void dg_key_range(int t, int seq, int old, int canvas, int full, int window,\n"
    "        int *start, int *end) {\n"
    "    int e = canvas ? old + seq : old + t + 1, s = 0;\n"
    "    if (!full) {\n"
    "        if (canvas) {\n"
    "            int pos = old + t;\n"
    "            s = pos >= window ? pos - window + 1 : 0;\n"
    "            if (e > pos + window)\n"
    "                e = pos + window;\n"
    "        } else\n"
    "            s = e > window ? e - window : 0;\n"
    "    }\n"
    "    *start = s;\n"
    "    *end = e;\n"
    "}\n"
    "\n"
    "/* Attention in one pass over the keys, with an online softmax. A block\n"
    " * of four warps takes 16 queries of one segment and one head, and walks\n"
    " * the keys they see 32 at a time: each warp scores 8 keys against the 16\n"
    " * queries, the block rescales its running maxima and sums, and each warp\n"
    " * accumulates a quarter of the output dimensions. Cached keys and values\n"
    " * come from k_old and v_old, the segment's own from k and v. */\n"
    "template <int HD>\n"
    "__device__ void dg_attention_fast_body(float *out, const float *q, const float *k,\n"
    "        const float *v, const float *k_old, const float *v_old, const int *seg_old,\n"
    "        const int *seg_base, int seq, int heads, int kvh, int canvas, int full,\n"
    "                int window) {\n"
    "    __shared__ float pt[16][33];\n",
    "    __shared__ float m_run[16], l_run[16], rescale[16];\n"
    "    const int qn = heads * HD, kn = kvh * HD, DW = HD / 4, NT = DW / 8;\n"
    "    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;\n"
    "    int tiles = (seq + 15) / 16, s = blockIdx.x / tiles, t0 = blockIdx.x % tiles * 16;\n"
    "    int h = blockIdx.y, kh = h / (heads / kvh), old = seg_old[s];\n"
    "    const float *kc = k_old + (size_t)seg_base[s] * kn + (size_t)kh * HD,\n"
    "                *kq = k + (size_t)s * seq * kn + (size_t)kh * HD,\n"
    "                *vc = v_old + (size_t)seg_base[s] * kn + (size_t)kh * HD,\n"
    "                *vq = v + (size_t)s * seq * kn + (size_t)kh * HD;\n"
    "    int kmin = old + seq, kmax = 0;\n"
    "    for (int i = 0; i < 16 && t0 + i < seq; i++) {\n"
    "        int a, b;\n"
    "        dg_key_range(t0 + i, seq, old, canvas, full, window, &a, &b);\n"
    "        kmin = a < kmin ? a : kmin;\n"
    "        kmax = b > kmax ? b : kmax;\n"
    "    }\n"
    "    if (tid < 16) {\n"
    "        m_run[tid] = DG_NEG_INF;\n"
    "        l_run[tid] = 0;\n"
    "    }\n"
    "    float o[NT][4];\n"
    "    for (int i = 0; i < NT; i++)\n"
    "        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0;\n"
    "    int ta = t0 + g, tb = t0 + g + 8;\n"
    "    const float *qa = ta < seq ? q + ((size_t)s * seq + ta) * qn + (size_t)h * HD : 0,\n"
    "                *qb = tb < seq ? q + ((size_t)s * seq + tb) * qn + (size_t)h * HD : 0;\n"
    "    for (int k0 = kmin; k0 < kmax; k0 += 32) {\n"
    "        float sc[4] = {0, 0, 0, 0};\n"
    "        int key = k0 + warp * 8 + g;\n"
    "        const float *kr = key >= kmax ? 0 : key < old ? kc + (size_t)key * kn\n"
    "                                                         : kq + (size_t)(key - old) * kn;\n"
    "        for (int d = 0; d < HD; d += 8) {\n"
    "            float x[4] = {qa ? qa[d + c] : 0.0f, qb ? qb[d + c] : 0.0f,\n"
    "                          qa ? qa[d + c + 4] : 0.0f, qb ? qb[d + c + 4] : 0.0f};\n"
    "            dg_mma_split(sc, x, kr ? kr[d + c] : 0.0f, kr ? kr[d + c + 4] : 0.0f);\n"
    "        }\n"
    "        __syncthreads();\n"
    "        for (int e = 0; e < 4; e++) {\n"
    "            int row = g + (e >> 1) * 8, col = warp * 8 + 2 * c + (e & 1), t = t0 + row,\n"
    "                    j = k0 + col;\n"
    "            int a, b;\n"
    "            dg_key_range(t, seq, old, canvas, full, window, &a, &b);\n"
    "            pt[row][col] = t < seq && j >= a && j < b ? sc[e] : DG_NEG_INF;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        {\n"
    "            int row = tid >> 3, part = tid & 7;\n"
    "            float x[4], mx = DG_NEG_INF, sum = 0;\n"
    "            for (int i = 0; i < 4; i++) {\n"
    "                x[i] = pt[row][part * 4 + i];\n"
    "                mx = fmaxf(mx, x[i]);\n"
    "            }\n"
    "            for (int off = 1; off < 8; off <<= 1)\n"
    "                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));\n"
    "            float m_old = m_run[row], m_new = fmaxf(m_old, mx);\n"
    "            for (int i = 0; i < 4; i++) {\n"
    "                x[i] = m_new == DG_NEG_INF ? 0.0f : __expf(x[i] - m_new);\n"
    "                pt[row][part * 4 + i] = x[i];\n"
    "                sum += x[i];\n"
    "            }\n"
    "            for (int off = 1; off < 8; off <<= 1)\n"
    "                sum += __shfl_xor_sync(0xffffffffu, sum, off);\n"
    "            __syncwarp();\n"
    "            if (part == 0) {\n"
    "                float f = m_old == DG_NEG_INF ? 0.0f : __expf(m_old - m_new);\n"
    "                rescale[row] = f;\n"
    "                l_run[row] = l_run[row] * f + sum;\n"
    "                m_run[row] = m_new;\n"
    "            }\n"
    "        }\n"
    "        __syncthreads();\n"
    "        float fa = rescale[g], fb = rescale[g + 8];\n"
    "        for (int i = 0; i < NT; i++) {\n"
    "            o[i][0] *= fa;\n"
    "            o[i][1] *= fa;\n"
    "            o[i][2] *= fb;\n",
    "            o[i][3] *= fb;\n"
    "        }\n"
    "        for (int ks = 0; ks < 32; ks += 8) {\n"
    "            float p[4] = {pt[g][ks + c], pt[g + 8][ks + c], pt[g][ks + c + 4],\n"
    "                          pt[g + 8][ks + c + 4]};\n"
    "            int j0 = k0 + ks + c, j1 = j0 + 4;\n"
    "            const float *v0 = j0 >= kmax ? 0 : (j0 < old ? vc + (size_t)j0 * kn\n"
    "                                                     : vq + (size_t)(j0 - old) * kn),\n"
    "                        *v1 = j1 >= kmax ? 0 : (j1 < old ? vc + (size_t)j1 * kn\n"
    "                                                     : vq + (size_t)(j1 - old) * kn);\n"
    "            for (int i = 0; i < NT; i++) {\n"
    "                int d = warp * DW + i * 8 + g;\n"
    "                dg_mma_split(o[i], p, v0 ? v0[d] : 0.0f, v1 ? v1[d] : 0.0f);\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    float la = l_run[g], lb = l_run[g + 8];\n"
    "    for (int i = 0; i < NT; i++) {\n"
    "        int d = warp * DW + i * 8 + 2 * c;\n"
    "        if (qa) {\n"
    "            float *oa = out + ((size_t)s * seq + ta) * qn + (size_t)h * HD + d;\n"
    "            oa[0] = o[i][0] / la;\n"
    "            oa[1] = o[i][1] / la;\n"
    "        }\n"
    "        if (qb) {\n"
    "            float *ob = out + ((size_t)s * seq + tb) * qn + (size_t)h * HD + d;\n"
    "            ob[0] = o[i][2] / lb;\n"
    "            ob[1] = o[i][3] / lb;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void __launch_bounds__(128) dg_attention_fast_local(float *out,\n"
    "        const float *q, const float *k, const float *v, const float *k_old,\n"
    "                const float *v_old,\n"
    "        const int *seg_old, const int *seg_base, int seq, int heads, int kvh, int canvas,\n"
    "        int full, int window) {\n"
    "    dg_attention_fast_body<DG_LOCAL_HD>(out, q, k, v, k_old, v_old, seg_old, seg_base, seq,\n"
    "                                        heads, kvh, canvas, full, window);\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ void __launch_bounds__(128) dg_attention_fast_full(float *out,\n"
    "        const float *q, const float *k, const float *v, const float *k_old,\n"
    "                const float *v_old,\n"
    "        const int *seg_old, const int *seg_base, int seq, int heads, int kvh, int canvas,\n"
    "        int full, int window) {\n"
    "    dg_attention_fast_body<DG_HEAD_MAX>(out, q, k, v, k_old, v_old, seg_old, seg_base, seq,\n"
    "                                        heads, kvh, canvas, full, window);\n"
    "}\n"
    "#endif\n"
    "\n"
    "#if DG_FAST\n"
    "__device__ __forceinline__ unsigned dg_bf16_pair(float even, float odd) {\n"
    "    unsigned d;\n"
    "    asm(\"cvt.rn.bf16x2.f32 %0, %1, %2;\" : \"=r\"(d) : \"f\"(odd), \"f\"(even));\n"
    "    return d;\n"
    "}\n"
    "\n"
    "/* c += a b for a 16 x 16 BF16 tile a, a 16 x 8 tile b and FP32 c, in the\n"
    " * m16n8k16 fragment layout. */\n"
    "__device__ __forceinline__ void dg_mma_bf16(float *c, const unsigned *a, unsigned b0,\n"
    "        unsigned b1) {\n"
    "    asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \"\n"
    "                 \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
    "                 : \"+f\"(c[0]), \"+f\"(c[1]), \"+f\"(c[2]), \"+f\"(c[3])\n"
    "                 : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"r\"(b0),\n"
    "                         \"r\"(b1));\n"
    "}\n"
    "\n"
    "/* dg_bf16_mm on tensor cores, for column counts that are a multiple of 32.\n"
    " * Each FP32 activation is the sum of three BF16 parts, which hold all of\n"
    " * its bits, so every product is exact and only the FP32 sums round. A\n"
    " * block of eight warps covers 128 rows and 64 tokens, 32 columns at a\n"
    " * time through shared memory; each warp computes 32 rows by 32 tokens. */\n"
    "extern \"C\" __global__ void __launch_bounds__(256)\n"
    "dg_bf16_mm_fast(const unsigned short *w,\n",
    "        const float *x, float *y, int tokens, int rows, int cols) {\n"
    "    __shared__ unsigned ws[128][20];\n"
    "    __shared__ unsigned xs[3][64][20];\n"
    "    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;\n"
    "    int wr = (warp & 3) * 32, wt = (warp >> 2) * 32, r0 = blockIdx.x * 128;\n"
    "    for (int t0 = blockIdx.y * 64; t0 < tokens; t0 += gridDim.y * 64) {\n"
    "        float acc[2][4][4];\n"
    "        for (int i = 0; i < 2; i++)\n"
    "            for (int j = 0; j < 4; j++)\n"
    "                acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;\n"
    "        for (int k0 = 0; k0 < cols; k0 += 32) {\n"
    "            {\n"
    "                int row = tid >> 1, half = tid & 1, gr = r0 + row;\n"
    "                uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n"
    "                if (gr < rows) {\n"
    "                    const uint4 *p = (const uint4 *)(w + (size_t)gr * cols + k0) + half * 2;\n"
    "                    v0 = p[0];\n"
    "                    v1 = p[1];\n"
    "                }\n"
    "                unsigned *d = &ws[row][half * 8];\n"
    "                d[0] = v0.x;\n"
    "                d[1] = v0.y;\n"
    "                d[2] = v0.z;\n"
    "                d[3] = v0.w;\n"
    "                d[4] = v1.x;\n"
    "                d[5] = v1.y;\n"
    "                d[6] = v1.z;\n"
    "                d[7] = v1.w;\n"
    "            }\n"
    "            {\n"
    "                int tok = tid >> 2, part = tid & 3, gt = t0 + tok;\n"
    "                float4 f0 = make_float4(0, 0, 0, 0), f1 = f0;\n"
    "                if (gt < tokens) {\n"
    "                    const float4 *p =\n"
    "                        (const float4 *)(x + (size_t)gt * cols + k0) + part * 2;\n"
    "                    f0 = p[0];\n"
    "                    f1 = p[1];\n"
    "                }\n"
    "                float f[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};\n"
    "                for (int i = 0; i < 4; i++) {\n"
    "                    float a = f[2 * i], b = f[2 * i + 1];\n"
    "                    for (int s = 0; s < 3; s++) {\n"
    "                        unsigned p = dg_bf16_pair(a, b);\n"
    "                        xs[s][tok][part * 4 + i] = p;\n"
    "                        a -= __uint_as_float(p << 16);\n"
    "                        b -= __uint_as_float(p & 0xffff0000u);\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "            __syncthreads();\n"
    "            for (int kk = 0; kk < 16; kk += 8) {\n"
    "                unsigned a[2][4];\n"
    "                for (int i = 0; i < 2; i++) {\n"
    "                    int row = wr + i * 16 + g;\n"
    "                    a[i][0] = ws[row][kk + c];\n"
    "                    a[i][1] = ws[row + 8][kk + c];\n"
    "                    a[i][2] = ws[row][kk + c + 4];\n"
    "                    a[i][3] = ws[row + 8][kk + c + 4];\n"
    "                }\n"
    "                for (int s = 0; s < 3; s++)\n"
    "                    for (int j = 0; j < 4; j++) {\n"
    "                        int tok = wt + j * 8 + g;\n"
    "                        unsigned b0 = xs[s][tok][kk + c], b1 = xs[s][tok][kk + c + 4];\n"
    "                        dg_mma_bf16(acc[0][j], a[0], b0, b1);\n"
    "                        dg_mma_bf16(acc[1][j], a[1], b0, b1);\n"
    "                    }\n"
    "            }\n"
    "            __syncthreads();\n"
    "        }\n"
    "        for (int i = 0; i < 2; i++)\n"
    "            for (int j = 0; j < 4; j++)\n"
    "                for (int e = 0; e < 4; e++) {\n"
    "                    int r = r0 + wr + i * 16 + g + (e >> 1) * 8,\n"
    "                        t = t0 + wt + j * 8 + 2 * c + (e & 1);\n"
    "                    if (r < rows && t < tokens)\n"
    "                        y[(size_t)t * rows + r] = acc[i][j][e];\n"
    "                }\n"
    "    }\n"
    "}\n"
    "#endif\n"
    "\n"
    "#if DG_FAST\n"
    "/* dg_bf16_mm_fast for few blocks' worth of rows and tokens, as in decoding,\n",
    " * for column counts that are a multiple of 64: a block covers 32 rows and\n"
    " * 64 tokens, and its four pairs of warps each take 16 of every 64 columns,\n"
    " * so four times as many warps stream the weights. The pairs' partial sums\n"
    " * are added in shared memory at the end, always in the same order. */\n"
    "extern \"C\" __global__ void __launch_bounds__(256) dg_bf16_mm_fast_small(\n"
    "        const unsigned short *w, const float *x, float *y, int tokens, int rows, int cols) {\n"
    "    /* Staged weights, [group][row][pair], then staged activations,\n"
    "     * [group][part][token][pair], each row padded to 12 words; afterwards\n"
    "     * three groups' partial sums, [group - 1][row][token]. */\n"
    "    __shared__ unsigned smem[4 * 32 * 12 + 4 * 3 * 64 * 12];\n"
    "    unsigned *ws = smem, *xs = smem + 4 * 32 * 12;\n"
    "    float *red = (float *)xs;\n"
    "    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;\n"
    "    int gk = warp >> 1, wt = (warp & 1) * 32, gi = tid & 63, gt = tid >> 6;\n"
    "    int r0 = blockIdx.x * 32;\n"
    "    for (int t0 = blockIdx.y * 64; t0 < tokens; t0 += gridDim.y * 64) {\n"
    "        float acc[2][4][4];\n"
    "        for (int i = 0; i < 2; i++)\n"
    "            for (int j = 0; j < 4; j++)\n"
    "                acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;\n"
    "        for (int k0 = 0; k0 < cols; k0 += 64) {\n"
    "            int kc = k0 + gt * 16;\n"
    "            {\n"
    "                int row = gi >> 1, half = gi & 1, gr = r0 + row;\n"
    "                uint4 v = make_uint4(0, 0, 0, 0);\n"
    "                if (gr < rows)\n"
    "                    v = *(const uint4 *)(w + (size_t)gr * cols + kc + half * 8);\n"
    "                unsigned *d = ws + (gt * 32 + row) * 12 + half * 4;\n"
    "                d[0] = v.x;\n"
    "                d[1] = v.y;\n"
    "                d[2] = v.z;\n"
    "                d[3] = v.w;\n"
    "            }\n"
    "            for (int j = 0; j < 4; j++) {\n"
    "                int u = gi + 64 * j, tok = u >> 2, q4 = u & 3, tt = t0 + tok;\n"
    "                float4 f = make_float4(0, 0, 0, 0);\n"
    "                if (tt < tokens)\n"
    "                    f = *(const float4 *)(x + (size_t)tt * cols + kc + q4 * 4);\n"
    "                float a[2] = {f.x, f.z}, b[2] = {f.y, f.w};\n"
    "                for (int p = 0; p < 2; p++)\n"
    "                    for (int s = 0; s < 3; s++) {\n"
    "                        unsigned v = dg_bf16_pair(a[p], b[p]);\n"
    "                        xs[((gt * 3 + s) * 64 + tok) * 12 + q4 * 2 + p] = v;\n"
    "                        a[p] -= __uint_as_float(v << 16);\n"
    "                        b[p] -= __uint_as_float(v & 0xffff0000u);\n"
    "                    }\n"
    "            }\n"
    "            __syncthreads();\n"
    "            unsigned af[2][4];\n"
    "            for (int i = 0; i < 2; i++) {\n"
    "                const unsigned *wr = ws + (gk * 32 + i * 16 + g) * 12;\n"
    "                af[i][0] = wr[c];\n"
    "                af[i][1] = wr[8 * 12 + c];\n"
    "                af[i][2] = wr[c + 4];\n"
    "                af[i][3] = wr[8 * 12 + c + 4];\n"
    "            }\n"
    "            for (int s = 0; s < 3; s++)\n"
    "                for (int j = 0; j < 4; j++) {\n"
    "                    const unsigned *xr = xs + ((gk * 3 + s) * 64 + wt + j * 8 + g) * 12;\n"
    "                    dg_mma_bf16(acc[0][j], af[0], xr[c], xr[c + 4]);\n"
    "                    dg_mma_bf16(acc[1][j], af[1], xr[c], xr[c + 4]);\n"
    "                }\n"
    "            __syncthreads();\n"
    "        }\n"
    "        if (gk > 0)\n"
    "            for (int i = 0; i < 2; i++)\n"
    "                for (int j = 0; j < 4; j++)\n"
    "                    for (int e = 0; e < 4; e++) {\n"
    "                        int r = i * 16 + g + (e >> 1) * 8,\n",
    "                            t = wt + j * 8 + 2 * c + (e & 1);\n"
    "                        red[((gk - 1) * 32 + r) * 64 + t] = acc[i][j][e];\n"
    "                    }\n"
    "        __syncthreads();\n"
    "        if (gk == 0)\n"
    "            for (int i = 0; i < 2; i++)\n"
    "                for (int j = 0; j < 4; j++)\n"
    "                    for (int e = 0; e < 4; e++) {\n"
    "                        int r = i * 16 + g + (e >> 1) * 8, t = wt + j * 8 + 2 * c + (e & 1);\n"
    "                        float v = acc[i][j][e];\n"
    "                        for (int q = 0; q < 3; q++)\n"
    "                            v += red[(q * 32 + r) * 64 + t];\n"
    "                        if (r0 + r < rows && t0 + t < tokens)\n"
    "                            y[(size_t)(t0 + t) * rows + r0 + r] = v;\n"
    "                    }\n"
    "        __syncthreads();\n"
    "    }\n"
    "}\n"
    "#endif\n"
    "\n"
    "#if DG_FP4\n"
    "/* Fast builds on GPUs with block-scaled FP4 tensor cores (compute\n"
    " * capability 12.x): NVFP4 activations stay packed, as each row's E2M1\n"
    " * nibbles (low nibble first), then its E4M3 block-scale codes, then its\n"
    " * global scale as a float, and the expert products run on the FP4 tensor\n"
    " * cores, whose E2M1 and E4M3 products are exact; only the FP32 sums round. */\n"
    "\n"
    "/* A packed row's bytes: the nibbles, the scale codes and the global scale,\n"
    " * padded to 16 bytes so rows can be read 16 bytes at a time; the host's\n"
    " * dg_cuda_nvfp4_row must compute the same. */\n"
    "__device__ size_t dg_fp4_stride(int cols) {\n"
    "    return ((size_t)cols / 2 + (size_t)cols / 16 + 4 + 15) & ~(size_t)15;\n"
    "}\n"
    "\n"
    "/* The E4M3 code whose value dg_f8e4m3_round(x) returns. */\n"
    "__device__ unsigned dg_f8e4m3_code(float x) {\n"
    "    if (!(x > 0))\n"
    "        return 0;\n"
    "    if (x >= 448)\n"
    "        return 126;\n"
    "    int lo = 0, hi = 126;\n"
    "    while (lo + 1 < hi) {\n"
    "        int m = (lo + hi) / 2;\n"
    "        if (dg_f8e4m3((unsigned)m) < x)\n"
    "            lo = m;\n"
    "        else\n"
    "            hi = m;\n"
    "    }\n"
    "    float a = dg_f8e4m3((unsigned)lo), b = dg_f8e4m3((unsigned)hi), da = x - a, db = b - x;\n"
    "    return da < db || (da == db && !(lo & 1)) ? (unsigned)lo : (unsigned)hi;\n"
    "}\n"
    "\n"
    "/* The E2M1 code whose value dg_e2m1_round(x) returns. */\n"
    "__device__ unsigned dg_e2m1_code(float x) {\n"
    "    const float q[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};\n"
    "    float a = fabsf(x);\n"
    "    int best = 0;\n"
    "    for (int i = 1; i < 8; i++) {\n"
    "        float d = fabsf(a - q[i]), old = fabsf(a - q[best]);\n"
    "        if (d < old || (d == old && !(i & 1)))\n"
    "            best = i;\n"
    "    }\n"
    "    return (signbit(x) ? 8u : 0u) | (unsigned)best;\n"
    "}\n"
    "\n"
    "/* dg_nvfp4_qdq into the packed form: the same block scales and values. */\n"
    "extern \"C\" __global__ void dg_nvfp4_pack(unsigned char *out, const float *in, int tokens,\n"
    "        int cols, float base, int *flag) {\n"
    "    long long nb = cols / 16, blocks = (long long)tokens * nb;\n"
    "    size_t stride = dg_fp4_stride(cols);\n"
    "    for (long long z = blockIdx.x * (long long)blockDim.x + threadIdx.x; z < blocks;\n"
    "         z += (long long)gridDim.x * blockDim.x) {\n"
    "        long long t = z / nb, b = z % nb;\n"
    "        const float *x = in + t * cols + b * 16;\n"
    "        unsigned char *row = out + (size_t)t * stride;\n"
    "        float amax = 0;\n"
    "        int bad = 0;\n"
    "        for (int k = 0; k < 16; k++) {\n"
    "            bad |= !dg_finitef(x[k]);\n"
    "            if (fabsf(x[k]) > amax)\n"
    "                amax = fabsf(x[k]);\n"
    "        }\n"
    "        if (bad) {\n"
    "            atomicOr(flag, 1);\n"
    "            continue;\n"
    "        }\n"
    "        unsigned code = dg_f8e4m3_code((amax / 6) / base);\n"
    "        float s = dg_f8e4m3(code) * base;\n"
    "        unsigned w0 = 0, w1 = 0;\n"
    "        if (s != 0)\n",
    "            for (int k = 0; k < 16; k++) {\n"
    "                unsigned v = dg_e2m1_code(x[k] / s) << (4 * (k & 7));\n"
    "                if (k < 8)\n"
    "                    w0 |= v;\n"
    "                else\n"
    "                    w1 |= v;\n"
    "            }\n"
    "        else\n"
    "            code = 0;\n"
    "        *(unsigned *)(row + b * 8) = w0;\n"
    "        *(unsigned *)(row + b * 8 + 4) = w1;\n"
    "        row[cols / 2 + b] = (unsigned char)code;\n"
    "        if (b == 0)\n"
    "            *(float *)(row + cols / 2 + cols / 16) = base;\n"
    "    }\n"
    "}\n"
    "\n"
    "/* dg_nvfp4_mm on FP4 tensor cores, from packed activations. A block's four\n"
    " * warps cover 64 rows, a warp 16 rows, 64 columns per MMA. A block takes\n"
    " * DG_FP4_RUN consecutive tiles, and each run of them that shares a matrix\n"
    " * streams that matrix's weights and the run's activations through shared\n"
    " * memory, 256 columns at a time with coalesced 16-byte loads, the next\n"
    " * stage's loads in flight while this one is multiplied; grouped tiles keep\n"
    " * an expert's tiles together. With scale selector 0, lanes 4i and 4i + 1\n"
    " * supply rows i and i + 8 of the weights' block scales, lane 4i token i's,\n"
    " * each as four bytes for the four 16-column blocks. */\n"
    "extern \"C\" __global__ void __launch_bounds__(128) dg_nvfp4_mm_fp4(const dg_matrix *m,\n"
    "        const dg_tile *tiles, const int *count, const unsigned char *x, float *y, int rows,\n"
    "        int cols) {\n"
    "    /* [stage][row][word]: 128 bytes of a row, padded to 36 words so a\n"
    "     * warp's fragment reads spread over the banks. A weight row's padding\n"
    "     * holds nothing; an activation slot's holds its four scale words. */\n"
    "    __shared__ unsigned wsh[2][64][36];\n"
    "    __shared__ unsigned ssh[2][64][4];\n"
    "    __shared__ unsigned xsh[2][8 * DG_FP4_RUN][36];\n"
    "    __shared__ int slot_row[8 * DG_FP4_RUN];\n"
    "    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;\n"
    "    size_t wb = (size_t)cols / 2, sb = (size_t)cols / 16, stride = dg_fp4_stride(cols);\n"
    "    int ntiles = *count, r0 = blockIdx.x * 64, rw = warp * 16 + g;\n"
    "    int ra = r0 + rw, rb = ra + 8;\n"
    "    const int SLOTS = 8 * DG_FP4_RUN, XV = (SLOTS * 9 + 127) / 128;\n"
    "    for (int c0 = blockIdx.y * DG_FP4_RUN; c0 < ntiles; c0 += gridDim.y * DG_FP4_RUN) {\n"
    "        int cend = c0 + DG_FP4_RUN < ntiles ? c0 + DG_FP4_RUN : ntiles;\n"
    "        for (int t0 = c0; t0 < cend;) {\n"
    "            int group = tiles[t0].group, run = 1;\n"
    "            while (t0 + run < cend && tiles[t0 + run].group == group)\n"
    "                run++;\n"
    "            dg_matrix mt = m[group];\n"
    "            /* Slot i * 8 + j is token j of the run's tile i, or -1. */\n"
    "            for (int s = tid; s < SLOTS; s += 128) {\n"
    "                int i = s >> 3, j = s & 7, r = -1;\n"
    "                if (i < run) {\n"
    "                    dg_tile tile = tiles[t0 + i];\n"
    "                    if (j < tile.count)\n"
    "                        r = tile.start + j;\n"
    "                }\n"
    "                slot_row[s] = r;\n"
    "            }\n"
    "            __syncthreads();\n"
    "            float d[DG_FP4_RUN][4];\n"
    "#pragma unroll\n"
    "            for (int i = 0; i < DG_FP4_RUN; i++)\n"
    "                d[i][0] = d[i][1] = d[i][2] = d[i][3] = 0;\n"
    "            uint4 wv[4], xv[XV];\n"
    "            unsigned sv[2];\n"
    "            /* One stage: the weights' 64 rows by 128 bytes, four 16-byte\n"
    "             * pieces a thread, and their scales, two words a thread; and\n"
    "             * each slot's 128 bytes and 16 bytes of scales, nine 16-byte\n"
    "             * pieces a slot. */\n"
    "#define DG_FP4_LOAD(kc) \\\n"
    "    do { \\\n"
    "        int bytes_ = (cols - (kc)) / 2 < 128 ? (cols - (kc)) / 2 : 128; \\\n",
    "        for (int j_ = 0; j_ < 4; j_++) { \\\n"
    "            int u_ = tid + 128 * j_, row_ = u_ >> 3, q_ = u_ & 7; \\\n"
    "            int gr_ = r0 + row_ < rows ? r0 + row_ : rows - 1; \\\n"
    "            if (q_ * 16 < bytes_) \\\n"
    "                wv[j_] = *(const uint4 *)(mt.w + (size_t)gr_ * wb + (kc) / 2 + q_ * 16); \\\n"
    "        } \\\n"
    "        for (int j_ = 0; j_ < 2; j_++) { \\\n"
    "            int v_ = tid + 128 * j_, row_ = v_ >> 2, q_ = v_ & 3; \\\n"
    "            int gr_ = r0 + row_ < rows ? r0 + row_ : rows - 1; \\\n"
    "            if (q_ * 64 < cols - (kc)) \\\n"
    "                sv[j_] = *(const unsigned *)(mt.s + (size_t)gr_ * sb + \\\n"
    "                                             (kc) / 16 + q_ * 4); \\\n"
    "        } \\\n"
    "        for (int j_ = 0; j_ < XV; j_++) { \\\n"
    "            int u_ = tid + 128 * j_, s_ = u_ / 9, q_ = u_ % 9; \\\n"
    "            xv[j_] = make_uint4(0, 0, 0, 0); \\\n"
    "            if (s_ < SLOTS && slot_row[s_] >= 0) { \\\n"
    "                const unsigned char *r_ = x + (size_t)slot_row[s_] * stride; \\\n"
    "                if (q_ < 8 && q_ * 16 < bytes_) \\\n"
    "                    xv[j_] = *(const uint4 *)(r_ + (kc) / 2 + q_ * 16); \\\n"
    "                else if (q_ == 8) \\\n"
    "                    xv[j_] = *(const uint4 *)(r_ + wb + (kc) / 16); \\\n"
    "            } \\\n"
    "        } \\\n"
    "    } while (0)\n"
    "            DG_FP4_LOAD(0);\n"
    "            for (int kc = 0, buf = 0; kc < cols; kc += 256, buf ^= 1) {\n"
    "                for (int j = 0; j < 4; j++) {\n"
    "                    int u = tid + 128 * j;\n"
    "                    *(uint4 *)&wsh[buf][u >> 3][(u & 7) * 4] = wv[j];\n"
    "                }\n"
    "                for (int j = 0; j < 2; j++) {\n"
    "                    int v = tid + 128 * j;\n"
    "                    ssh[buf][v >> 2][v & 3] = sv[j];\n"
    "                }\n"
    "                for (int j = 0; j < XV; j++) {\n"
    "                    int u = tid + 128 * j, s = u / 9, q = u % 9;\n"
    "                    if (s < SLOTS)\n"
    "                        *(uint4 *)&xsh[buf][s][q * 4] = xv[j];\n"
    "                }\n"
    "                __syncthreads();\n"
    "                if (kc + 256 < cols)\n"
    "                    DG_FP4_LOAD(kc + 256);\n"
    "                int steps = (cols - kc) / 64 < 4 ? (cols - kc) / 64 : 4;\n"
    "                for (int s = 0; s < steps; s++) {\n"
    "                    unsigned a0 = wsh[buf][rw][s * 8 + c], a1 = wsh[buf][rw + 8][s * 8 + c],\n"
    "                             a2 = wsh[buf][rw][s * 8 + 4 + c],\n"
    "                                     a3 = wsh[buf][rw + 8][s * 8 + 4 + c];\n"
    "                    unsigned sav = ssh[buf][c == 0 ? rw : rw + 8][s];\n"
    "#pragma unroll\n"
    "                    for (int i = 0; i < DG_FP4_RUN; i++)\n"
    "                        if (i < run) {\n"
    "                            const unsigned *xs = xsh[buf][i * 8 + g];\n"
    "                            asm(\"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4\"\n"
    "                                \".block_scale\"\n"
    "                                \".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3}, \"\n"
    "                                \"{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, \"\n"
    "                                \"%10, {0, 0}, %11, {0, 0};\"\n"
    "                                : \"+f\"(d[i][0]), \"+f\"(d[i][1]), \"+f\"(d[i][2]),\n"
    "                                        \"+f\"(d[i][3])\n"
    "                                : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
    "                                        \"r\"(xs[s * 8 + c]),\n"
    "                                  \"r\"(xs[s * 8 + 4 + c]), \"r\"(sav), \"r\"(xs[32 + s]));\n"
    "                        }\n"
    "                }\n"
    "            }\n"
    "#undef DG_FP4_LOAD\n"
    "#pragma unroll\n"
    "            for (int i = 0; i < DG_FP4_RUN; i++)\n",
    "                if (i < run)\n"
    "                    for (int e = 0; e < 4; e++) {\n"
    "                        int row = e < 2 ? ra : rb, r = slot_row[i * 8 + 2 * c + (e & 1)];\n"
    "                        if (row < rows && r >= 0) {\n"
    "                            float base = *(const float *)(x + (size_t)r * stride + wb + sb);\n"
    "                            y[(size_t)r * rows + row] = d[i][e] * mt.global * base;\n"
    "                        }\n"
    "                    }\n"
    "            /* The next run rewrites the slots and buffer 0. */\n"
    "            __syncthreads();\n"
    "            t0 += run;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "#endif\n"};

/* The kernel's struct dg_tile: tokens start..start + count - 1 use matrix
 * group. */
typedef struct {
    int start, count, group;
} DGCudaTile;

/* The kernel's struct dg_matrix. */
typedef struct {
    CUdeviceptr w, s;
    float global, pad;
} DGCudaMatrix;

struct DGAccelMatrix {
    CUdeviceptr w, s;
    float global;
    int rows, cols;
};

enum {
    DG_CUDA_NVFP4_MM,
    DG_CUDA_BF16_MM,
    DG_CUDA_RMS,
    DG_CUDA_GELU_MUL,
    DG_CUDA_NVFP4_QDQ,
    DG_CUDA_GATHER,
    DG_CUDA_MOE_TAIL,
    DG_CUDA_ROPE,
    DG_CUDA_ATTENTION,
    DG_CUDA_BF16_WIDE,
    DG_CUDA_BF16_NARROW,
    DG_CUDA_TRANSPOSE_KEYS,
    DG_CUDA_ROUTE,
    DG_CUDA_GROUP,
    /* Fast builds only, from here on. */
    DG_CUDA_ATTENTION_FAST_LOCAL,
    DG_CUDA_ATTENTION_FAST_FULL,
    DG_CUDA_BF16_FAST,
    DG_CUDA_BF16_FAST_SMALL,
    /* Fast builds with FP4 tensor cores only, from here on. */
    DG_CUDA_NVFP4_PACK,
    DG_CUDA_NVFP4_MM_FP4,
    DG_CUDA_KERNELS
};

static const char *const dg_cuda_kernel_names[DG_CUDA_KERNELS] = {"dg_nvfp4_mm",
                                                                  "dg_bf16_mm",
                                                                  "dg_rms",
                                                                  "dg_gelu_mul",
                                                                  "dg_nvfp4_qdq",
                                                                  "dg_gather",
                                                                  "dg_moe_tail",
                                                                  "dg_rope",
                                                                  "dg_attention",
                                                                  "dg_bf16_mm_wide",
                                                                  "dg_bf16_mm_narrow",
                                                                  "dg_transpose_keys",
                                                                  "dg_route",
                                                                  "dg_group",
                                                                  "dg_attention_fast_local",
                                                                  "dg_attention_fast_full",
                                                                  "dg_bf16_mm_fast",
                                                                  "dg_bf16_mm_fast_small",
                                                                  "dg_nvfp4_pack",
                                                                  "dg_nvfp4_mm_fp4"};

/* The driver and NVRTC entry points, and the process-wide context and
 * kernels. */
static struct {
    CUresult (*init)(unsigned);
    CUresult (*device_get)(CUdevice *, int);
    CUresult (*device_attribute)(int *, int, CUdevice);
    CUresult (*primary_retain)(CUcontext *, CUdevice);
    CUresult (*ctx_set)(CUcontext);
    CUresult (*module_load)(CUmodule *, const void *);
    CUresult (*function_get)(CUfunction *, CUmodule, const char *);
    CUresult (*alloc)(CUdeviceptr *, size_t);
    CUresult (*free)(CUdeviceptr);
    CUresult (*htod)(CUdeviceptr, const void *, size_t);
    CUresult (*dtoh)(void *, CUdeviceptr, size_t);
    CUresult (*dtod)(CUdeviceptr, CUdeviceptr, size_t);
    CUresult (*host_alloc)(void **, size_t);
    CUresult (*host_free)(void *);
    CUresult (*launch)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                       unsigned, void *, void **, void **);
    nvrtcResult (*create)(nvrtcProgram *, const char *, const char *, int, const char *const *,
                          const char *const *);
    nvrtcResult (*compile)(nvrtcProgram, int, const char *const *);
    nvrtcResult (*ptx_size)(nvrtcProgram, size_t *);
    nvrtcResult (*ptx)(nvrtcProgram, char *);
    nvrtcResult (*log_size)(nvrtcProgram, size_t *);
    nvrtcResult (*log)(nvrtcProgram, char *);
    nvrtcResult (*destroy)(nvrtcProgram *);
    CUcontext ctx;
    CUfunction kernel[DG_CUDA_KERNELS];
    /* Why the kernels are unavailable although a device is, and NVRTC's
     * log when compiling them failed. */
    const char *failure;
    char compile_log[4096];
    int ready;
    /* The fast kernels replace exact ones: in fast builds on a GPU with
     * BF16 tensor cores (compute capability 8.0 or later). The FP4 ones
     * also need block-scaled FP4 tensor cores (12.x) and an NVRTC that
     * compiles for them. */
    int fast, fp4;
} dg_cuda;

/* dlsym returns an object pointer; copy it into the function pointer. */
static int dg_cuda_symbol(void *lib, const char *name, void *slot) {
    void *p = lib ? dlsym(lib, name) : NULL;
    memcpy(slot, &p, sizeof p);
    return p != NULL;
}

/* Compiles and loads the kernels. Runs under pthread_once, which no error
 * frame may unwind through, so it records a failure for
 * dg_accel_failure() and allocates with plain malloc. */
static void dg_cuda_compile(int major, int minor) {
    char arch[48], tile_rows[32], tile_tokens[32], bf16_tokens[32], row_max[32], head_max[32],
        local_hd[32], threads[32], width[32], experts[32], topk[32], fast[32], fp4[32], fp4_run[32];
    snprintf(fast, sizeof fast, "-DDG_FAST=%d", dg_cuda.fast);
    snprintf(fp4, sizeof fp4, "-DDG_FP4=%d", dg_cuda.fp4);
    snprintf(fp4_run, sizeof fp4_run, "-DDG_FP4_RUN=%d", DG_CUDA_FP4_RUN);
    snprintf(local_hd, sizeof local_hd, "-DDG_LOCAL_HD=%d", DG_LOCAL_HEAD_DIM);
    /* The FP4 instructions need the architecture-specific target. */
    snprintf(arch, sizeof arch, "--gpu-architecture=compute_%d%d%s", major, minor,
             dg_cuda.fp4 ? "a" : "");
    snprintf(tile_rows, sizeof tile_rows, "-DDG_ROWS=%d", DG_CUDA_ROWS);
    snprintf(tile_tokens, sizeof tile_tokens, "-DDG_TOKENS=%d", DG_CUDA_TOKENS);
    snprintf(bf16_tokens, sizeof bf16_tokens, "-DDG_BTOKENS=%d", DG_CUDA_BF16_TOKENS);
    snprintf(row_max, sizeof row_max, "-DDG_ROW_MAX=%d", DG_CUDA_ROW_MAX);
    snprintf(head_max, sizeof head_max, "-DDG_HEAD_MAX=%d", DG_FULL_HEAD_DIM);
    snprintf(threads, sizeof threads, "-DDG_THREADS=%d", DG_CUDA_THREADS);
    snprintf(width, sizeof width, "-DDG_WIDTH=%d", DG_H);
    snprintf(experts, sizeof experts, "-DDG_EXPERTS_N=%d", DG_EXPERTS);
    snprintf(topk, sizeof topk, "-DDG_TOPK_N=%d", DG_TOPK);
    const char *options[] = {arch,
                             "--fmad=false",
                             "--prec-div=true",
                             "--prec-sqrt=true",
                             "--ftz=false",
                             tile_rows,
                             tile_tokens,
                             bf16_tokens,
                             row_max,
                             head_max,
                             local_hd,
                             threads,
                             width,
                             experts,
                             topk,
                             fast,
                             fp4,
                             fp4_run};
    /* The source is split into pieces short enough for ISO C string
     * literals. */
    size_t length = 1;
    for (size_t i = 0; i < sizeof dg_cuda_source / sizeof *dg_cuda_source; i++)
        length += strlen(dg_cuda_source[i]);
    char *source = jb_try_allocate(length, 0);
    if (!source) {
        dg_cuda.failure = "out of memory";
        return;
    }
    size_t at = 0;
    for (size_t i = 0; i < sizeof dg_cuda_source / sizeof *dg_cuda_source; i++) {
        size_t n = strlen(dg_cuda_source[i]);
        memcpy(source + at, dg_cuda_source[i], n);
        at += n;
    }
    source[at] = 0;
    nvrtcProgram prog;
    int created = !dg_cuda.create(&prog, source, "jb_layers.cu", 0, NULL, NULL);
    jb_release(source);
    if (!created) {
        dg_cuda.failure = "nvrtcCreateProgram";
        return;
    }
    size_t size = 0;
    char *ptx = NULL;
    CUmodule module;
    if (dg_cuda.compile(prog, (int)(sizeof options / sizeof *options), options) ||
        dg_cuda.ptx_size(prog, &size) || !(ptx = jb_try_allocate(size, 0)) ||
        dg_cuda.ptx(prog, ptx)) {
        size_t n = 0;
        char *log = NULL;
        if (!dg_cuda.log_size(prog, &n) && n > 1 && (log = jb_try_allocate(n, 0)) &&
            !dg_cuda.log(prog, log))
            snprintf(dg_cuda.compile_log, sizeof dg_cuda.compile_log, "%s", log);
        jb_release(log);
        dg_cuda.failure = "compiling the kernels";
    } else if (dg_cuda.module_load(&module, ptx))
        dg_cuda.failure = "loading the kernels";
    else {
        dg_cuda.ready = 1;
        int kernels = dg_cuda.fp4    ? DG_CUDA_KERNELS
                      : dg_cuda.fast ? DG_CUDA_NVFP4_PACK
                                     : DG_CUDA_ATTENTION_FAST_LOCAL;
        for (int i = 0; i < kernels; i++)
            if (dg_cuda.function_get(&dg_cuda.kernel[i], module, dg_cuda_kernel_names[i])) {
                dg_cuda.failure = "loading the kernels";
                dg_cuda.ready = 0;
            }
    }
    jb_release(ptx);
    dg_cuda.destroy(&prog);
}

static void dg_cuda_open(void) {
    static const char *const nvrtc[] = {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"};
    const char *env = getenv("JB_CUDA");
    if (env && !strcmp(env, "0"))
        return;
    void *cu = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL), *rtc = NULL;
    for (size_t i = 0; i < sizeof nvrtc / sizeof *nvrtc && !rtc; i++)
        rtc = dlopen(nvrtc[i], RTLD_NOW | RTLD_LOCAL);
    int loaded = dg_cuda_symbol(cu, "cuInit", &dg_cuda.init) &
                 dg_cuda_symbol(cu, "cuDeviceGet", &dg_cuda.device_get) &
                 dg_cuda_symbol(cu, "cuDeviceGetAttribute", &dg_cuda.device_attribute) &
                 dg_cuda_symbol(cu, "cuDevicePrimaryCtxRetain", &dg_cuda.primary_retain) &
                 dg_cuda_symbol(cu, "cuCtxSetCurrent", &dg_cuda.ctx_set) &
                 dg_cuda_symbol(cu, "cuModuleLoadData", &dg_cuda.module_load) &
                 dg_cuda_symbol(cu, "cuModuleGetFunction", &dg_cuda.function_get) &
                 dg_cuda_symbol(cu, "cuMemAlloc_v2", &dg_cuda.alloc) &
                 dg_cuda_symbol(cu, "cuMemFree_v2", &dg_cuda.free) &
                 dg_cuda_symbol(cu, "cuMemcpyHtoD_v2", &dg_cuda.htod) &
                 dg_cuda_symbol(cu, "cuMemcpyDtoH_v2", &dg_cuda.dtoh) &
                 dg_cuda_symbol(cu, "cuMemcpyDtoD_v2", &dg_cuda.dtod) &
                 dg_cuda_symbol(cu, "cuMemAllocHost_v2", &dg_cuda.host_alloc) &
                 dg_cuda_symbol(cu, "cuMemFreeHost", &dg_cuda.host_free) &
                 dg_cuda_symbol(cu, "cuLaunchKernel", &dg_cuda.launch) &
                 dg_cuda_symbol(rtc, "nvrtcCreateProgram", &dg_cuda.create) &
                 dg_cuda_symbol(rtc, "nvrtcCompileProgram", &dg_cuda.compile) &
                 dg_cuda_symbol(rtc, "nvrtcGetPTXSize", &dg_cuda.ptx_size) &
                 dg_cuda_symbol(rtc, "nvrtcGetPTX", &dg_cuda.ptx) &
                 dg_cuda_symbol(rtc, "nvrtcGetProgramLogSize", &dg_cuda.log_size) &
                 dg_cuda_symbol(rtc, "nvrtcGetProgramLog", &dg_cuda.log) &
                 dg_cuda_symbol(rtc, "nvrtcDestroyProgram", &dg_cuda.destroy);
    CUdevice device;
    int major, minor;
    if (!loaded || dg_cuda.init(0) || dg_cuda.device_get(&device, 0) ||
        dg_cuda.device_attribute(&major, DG_CU_CC_MAJOR, device) ||
        dg_cuda.device_attribute(&minor, DG_CU_CC_MINOR, device) ||
        dg_cuda.primary_retain(&dg_cuda.ctx, device) || dg_cuda.ctx_set(dg_cuda.ctx))
        return;
    dg_cuda.fast = !JB_STRICT_MATH && major >= 8;
    dg_cuda.fp4 = dg_cuda.fast && major == 12;
    dg_cuda_compile(major, minor);
    if (!dg_cuda.ready && dg_cuda.fp4) {
        /* An NVRTC too old for the FP4 target: the other fast kernels. */
        dg_cuda.fp4 = 0;
        dg_cuda.failure = NULL;
        dg_cuda.compile_log[0] = 0;
        dg_cuda_compile(major, minor);
    }
}

static void dg_cuda_check(CUresult r, const char *what) {
    if (r)
        die2_status(JB_ERROR_INTERNAL, "CUDA call failed", what);
}

/* Makes the context current on the calling thread. */
static void dg_cuda_current(void) {
    dg_cuda_check(dg_cuda.ctx_set(dg_cuda.ctx), "cuCtxSetCurrent");
}

static unsigned dg_cuda_blocks(long long work, int per_block, unsigned max) {
    long long blocks = (work + per_block - 1) / per_block;
    return blocks < 1 ? 1 : blocks > max ? max : (unsigned)blocks;
}

static void dg_cuda_launch_2d(int kernel, unsigned grid_x, unsigned grid_y, unsigned block_x,
                              unsigned block_y, void **args) {
    dg_cuda_current();
    dg_cuda_check(dg_cuda.launch(dg_cuda.kernel[kernel], grid_x, grid_y, 1, block_x, block_y, 1, 0,
                                 NULL, args, NULL),
                  dg_cuda_kernel_names[kernel]);
}

static void dg_cuda_launch(int kernel, unsigned grid_x, unsigned grid_y, unsigned block,
                           void **args) {
    dg_cuda_current();
    dg_cuda_check(
        dg_cuda.launch(dg_cuda.kernel[kernel], grid_x, grid_y, 1, block, 1, 1, 0, NULL, args, NULL),
        dg_cuda_kernel_names[kernel]);
}

static DGDevice dg_cuda_alloc(size_t bytes) {
    CUdeviceptr p = 0;
    dg_cuda_current();
    return dg_cuda.alloc(&p, bytes ? bytes : 1) ? 0 : (DGDevice)p;
}

/* The frees ignore errors: after a fault every call fails, and there is
 * nothing to do about memory the context will never return. */
static void dg_cuda_free(DGDevice p) {
    if (!dg_cuda.ctx_set(dg_cuda.ctx))
        (void)dg_cuda.free((CUdeviceptr)p);
}

static void dg_cuda_to_device(DGDevice dst, const void *src, size_t bytes) {
    dg_cuda_current();
    dg_cuda_check(dg_cuda.htod((CUdeviceptr)dst, src, bytes), "cuMemcpyHtoD");
}

static void dg_cuda_to_host(void *dst, DGDevice src, size_t bytes) {
    dg_cuda_current();
    dg_cuda_check(dg_cuda.dtoh(dst, (CUdeviceptr)src, bytes), "cuMemcpyDtoH");
}

static void *dg_cuda_host_alloc(size_t bytes) {
    void *p = NULL;
    dg_cuda_current();
    return dg_cuda.host_alloc(&p, bytes ? bytes : 1) ? NULL : p;
}

static void dg_cuda_host_free(void *p) {
    if (!dg_cuda.ctx_set(dg_cuda.ctx))
        (void)dg_cuda.host_free(p);
}

static void dg_cuda_release(DGAccelMatrix *a) {
    jb_release(a);
}

static DGAccelMatrix *dg_cuda_matrix(DGDevice w, DGDevice s, float global, int rows, int cols) {
    DGAccelMatrix *a = xcalloc(1, sizeof *a);
    a->w = w;
    a->s = s;
    a->global = global;
    a->rows = rows;
    a->cols = cols;
    return a;
}

static size_t dg_cuda_align16(size_t n) {
    return (n + 15) & ~(size_t)15;
}

static size_t dg_cuda_nvfp4_scratch(int tokens) {
    return dg_cuda_align16(DG_EXPERTS * sizeof(DGCudaMatrix)) +
           dg_cuda_align16((size_t)tokens * sizeof(DGCudaTile)) + 16;
}

static size_t dg_cuda_table_bytes(int count) {
    return (size_t)count * sizeof(DGCudaMatrix);
}

static void dg_cuda_matrix_table(DGDevice dst, const DGAccelMatrix *const *m, int count) {
    DGCudaMatrix *table = xcalloc((size_t)count, sizeof *table);
    for (int i = 0; i < count; i++) {
        table[i].w = m[i]->w;
        table[i].s = m[i]->s;
        table[i].global = m[i]->global;
    }
    dg_cuda_to_device(dst, table, (size_t)count * sizeof *table);
    jb_release(table);
}

static void dg_cuda_route(DGDevice logits, DGDevice scale, int tokens, DGDevice top,
                          DGDevice weight, DGDevice flag) {
    CUdeviceptr dl = logits, ds = scale, dt = top, dw = weight, df = flag;
    void *args[] = {&dl, &ds, &tokens, &dt, &dw, &df};
    dg_cuda_launch(DG_CUDA_ROUTE, dg_cuda_blocks(tokens, 1, 1u << 30), 1, DG_EXPERTS, args);
}

static void dg_cuda_group(DGDevice top, int tokens, DGDevice index, DGDevice slot, DGDevice tiles,
                          DGDevice ntiles) {
    CUdeviceptr dt = top, di = index, ds = slot, dl = tiles, dn = ntiles;
    void *args[] = {&dt, &tokens, &di, &ds, &dl, &dn};
    dg_cuda_launch(DG_CUDA_GROUP, 1, 1, DG_EXPERTS, args);
}

/* Whether NVFP4 rows of cols values are packed for the FP4 tensor cores,
 * which take 64 columns at a time; every model matrix's count is a
 * multiple of 64. */
static int dg_cuda_packed(int cols) {
    return dg_cuda.fp4 && cols % 64 == 0;
}

/* The NVFP4 product over tiles of tokens: 128 rows and a tile a block on
 * the exact kernel, 64 rows and DG_CUDA_FP4_RUN tiles on the FP4 tensor
 * cores. */
static void dg_cuda_nvfp4_launch(int rows, int cols, long long tiles, void **args) {
    int packed = dg_cuda_packed(cols), per_block = packed ? 64 : DG_CUDA_ROWS;
    dg_cuda_launch(packed ? DG_CUDA_NVFP4_MM_FP4 : DG_CUDA_NVFP4_MM,
                   dg_cuda_blocks(rows, per_block, 1u << 30),
                   dg_cuda_blocks(tiles, packed ? DG_CUDA_FP4_RUN : 1, DG_CUDA_GRID_Y), 128, args);
}

/* The grouped product from the device's own tiles; the grid covers the
 * most tiles rows routed rows can make, and blocks past *ntiles exit. */
static void dg_cuda_nvfp4_grouped(DGDevice table, DGDevice tiles, DGDevice ntiles, int routed,
                                  DGDevice x, DGDevice y, int rows, int cols) {
    CUdeviceptr dt = table, dl = tiles, dn = ntiles, dx = x, dy = y;
    long long bound = routed / DG_CUDA_TOKENS + DG_EXPERTS;
    void *args[] = {&dt, &dl, &dn, &dx, &dy, &rows, &cols};
    dg_cuda_nvfp4_launch(rows, cols, bound < routed ? bound : routed, args);
}

/* Splits tokens into tiles of up to DG_CUDA_TOKENS consecutive tokens that
 * share a matrix, stages the matrix table and the tiles in scratch, and runs
 * one block per DG_CUDA_ROWS rows and tile. */
static void dg_cuda_nvfp4_mm(const DGAccelMatrix *const *m, const int *group, int tokens,
                             DGDevice x, DGDevice y, DGDevice scratch) {
    if (tokens <= 0)
        return;
    int count = 0, ntiles = 0;
    for (int t = 0; t < tokens; t++) {
        if (group[t] < 0 || group[t] >= DG_EXPERTS || !m[group[t]])
            die("token routed to a matrix not on the accelerator");
        if (group[t] >= count)
            count = group[t] + 1;
    }
    int rows = m[group[0]]->rows, cols = m[group[0]]->cols;
    DGCudaMatrix *table = xcalloc((size_t)count, sizeof *table);
    DGCudaTile *tiles = xmalloc((size_t)tokens * sizeof *tiles);
    for (int i = 0; i < count; i++)
        if (m[i]) {
            if (m[i]->rows != rows || m[i]->cols != cols)
                die("accelerator matrices differ in shape");
            table[i].w = m[i]->w;
            table[i].s = m[i]->s;
            table[i].global = m[i]->global;
        }
    for (int t = 0; t < tokens;) {
        int n = 1;
        while (t + n < tokens && n < DG_CUDA_TOKENS && group[t + n] == group[t])
            n++;
        tiles[ntiles++] = (DGCudaTile){t, n, group[t]};
        t += n;
    }
    CUdeviceptr dt = (CUdeviceptr)scratch,
                dl = dt + dg_cuda_align16(DG_EXPERTS * sizeof(DGCudaMatrix)),
                dn = dl + dg_cuda_align16((size_t)tokens * sizeof(DGCudaTile)), dx = x, dy = y;
    dg_cuda_to_device(dt, table, (size_t)count * sizeof *table);
    dg_cuda_to_device(dl, tiles, (size_t)ntiles * sizeof *tiles);
    dg_cuda_to_device(dn, &ntiles, sizeof ntiles);
    jb_release(table);
    jb_release(tiles);
    void *args[] = {&dt, &dl, &dn, &dx, &dy, &rows, &cols};
    dg_cuda_nvfp4_launch(rows, cols, ntiles, args);
}

/* Column counts that are a multiple of 64, which every model matrix has,
 * take the double-buffered kernels: 128 rows by 32 tokens a block when that
 * gives the GPU enough blocks, 32 rows by 32 tokens with four warps
 * splitting the tokens otherwise. */
static void dg_cuda_bf16_mm(DGDevice w, DGDevice x, DGDevice y, int tokens, int rows, int cols) {
    if (cols % 2)
        die("accelerator BF16 matrices need an even column count");
    CUdeviceptr dw = w, dx = x, dy = y;
    void *args[] = {&dw, &dx, &dy, &tokens, &rows, &cols};
    if (dg_cuda.fast && cols % 32 == 0) {
        /* 128 rows by 64 tokens a block, or 32 rows with the columns split
         * four ways when that would leave the GPU short of blocks. */
        long long wide = (long long)dg_cuda_blocks(rows, 128, 1u << 30) *
                         dg_cuda_blocks(tokens, 64, DG_CUDA_GRID_Y);
        if (wide < 96 && cols % 64 == 0)
            dg_cuda_launch(DG_CUDA_BF16_FAST_SMALL, dg_cuda_blocks(rows, 32, 1u << 30),
                           dg_cuda_blocks(tokens, 64, DG_CUDA_GRID_Y), 256, args);
        else
            dg_cuda_launch(DG_CUDA_BF16_FAST, dg_cuda_blocks(rows, 128, 1u << 30),
                           dg_cuda_blocks(tokens, 64, DG_CUDA_GRID_Y), 256, args);
        return;
    }
    if (cols % 64 == 0) {
        long long wide = (long long)dg_cuda_blocks(rows, 128, 1u << 30) *
                         dg_cuda_blocks(tokens, 32, DG_CUDA_GRID_Y);
        if (wide >= 96)
            dg_cuda_launch(DG_CUDA_BF16_WIDE, dg_cuda_blocks(rows, 128, 1u << 30),
                           dg_cuda_blocks(tokens, 32, DG_CUDA_GRID_Y), 128, args);
        else
            dg_cuda_launch_2d(DG_CUDA_BF16_NARROW, dg_cuda_blocks(rows, 32, 1u << 30),
                              dg_cuda_blocks(tokens, 32, DG_CUDA_GRID_Y), 32, 4, args);
        return;
    }
    dg_cuda_launch(DG_CUDA_BF16_MM, dg_cuda_blocks(rows, DG_CUDA_ROWS, 1u << 30),
                   dg_cuda_blocks(tokens, DG_CUDA_BF16_TOKENS, DG_CUDA_GRID_Y), DG_CUDA_ROWS, args);
}

static void dg_cuda_rms(DGDevice y, DGDevice x, DGDevice scale, DGDevice factor, float divisor,
                        DGDevice add, int tokens, int n) {
    if (n > DG_CUDA_ROW_MAX)
        die("accelerator RMS rows are too long");
    CUdeviceptr dy = y, dx = x, ds = scale, df = factor, da = add;
    void *args[] = {&dy, &dx, &ds, &df, &divisor, &da, &tokens, &n};
    dg_cuda_launch(DG_CUDA_RMS, dg_cuda_blocks(tokens, 1, 1u << 30), 1, DG_CUDA_THREADS, args);
}

static void dg_cuda_gelu_mul(DGDevice out, DGDevice g, DGDevice u, size_t n) {
    CUdeviceptr dout = out, dg = g, du = u;
    long long count = (long long)n;
    void *args[] = {&dout, &dg, &du, &count};
    dg_cuda_launch(DG_CUDA_GELU_MUL, dg_cuda_blocks(count, DG_CUDA_THREADS, 1u << 20), 1,
                   DG_CUDA_THREADS, args);
}

static void dg_cuda_nvfp4_qdq(DGDevice out, DGDevice in, int tokens, int cols, float base,
                              DGDevice flag) {
    CUdeviceptr dout = out, din = in, dflag = flag;
    void *args[] = {&dout, &din, &tokens, &cols, &base, &dflag};
    dg_cuda_launch(dg_cuda_packed(cols) ? DG_CUDA_NVFP4_PACK : DG_CUDA_NVFP4_QDQ,
                   dg_cuda_blocks((long long)tokens * (cols / 16), DG_CUDA_THREADS, 1u << 20), 1,
                   DG_CUDA_THREADS, args);
}

/* A packed row: E2M1 nibbles, E4M3 block-scale codes, and the global
 * scale as a float, padded to 16 bytes. This must match dg_fp4_stride in
 * the kernel source, which computes the same. */
static size_t dg_cuda_nvfp4_row(int cols) {
    if (!dg_cuda_packed(cols))
        return (size_t)cols;
    size_t bytes = (size_t)cols / 2 + (size_t)cols / 16 + 4;
    return ((bytes + 15) & ~(size_t)15) / 4;
}

static void dg_cuda_gather(DGDevice dst, DGDevice src, DGDevice index, int rows, int cols) {
    CUdeviceptr dd = dst, ds = src, di = index;
    void *args[] = {&dd, &ds, &di, &rows, &cols};
    dg_cuda_launch(DG_CUDA_GATHER, dg_cuda_blocks(rows, 1, 1u << 30), 1, DG_CUDA_THREADS, args);
}

static void dg_cuda_copy(DGDevice dst, DGDevice src, size_t bytes) {
    dg_cuda_current();
    dg_cuda_check(dg_cuda.dtod((CUdeviceptr)dst, (CUdeviceptr)src, bytes), "cuMemcpyDtoD");
}

static void dg_cuda_rope(DGDevice x, DGDevice table, int tokens, int heads, int hd, int stride) {
    CUdeviceptr dx = x, dt = table;
    int pitch = DG_FULL_HEAD_DIM;
    void *args[] = {&dx, &dt, &tokens, &heads, &hd, &stride, &pitch};
    dg_cuda_launch(DG_CUDA_ROPE,
                   dg_cuda_blocks((long long)tokens * heads * (hd / 2), DG_CUDA_THREADS, 1u << 20),
                   1, DG_CUDA_THREADS, args);
}

static void dg_cuda_attention(DGDevice a, DGDevice q, DGDevice k, DGDevice v, DGDevice k_old,
                              DGDevice v_old, DGDevice seg_old, DGDevice seg_base, int n, int seq,
                              int kvh, int hd, int canvas, int full, int max_keys, int old_rows,
                              DGDevice scratch) {
    if (dg_cuda.fast) {
        /* One block per 16 queries of a segment and head; no transposed
         * keys and no limit on the span. */
        if (hd != DG_LOCAL_HEAD_DIM && hd != DG_FULL_HEAD_DIM)
            die("accelerator attention head size is unsupported");
        int segments = n / seq, heads = DG_HEADS, window = DG_LOCAL_WINDOW;
        CUdeviceptr da = a, dq = q, dk = k, dv = v, dko = k_old, dvo = v_old, dso = seg_old,
                    dsb = seg_base;
        void *args[] = {&da,  &dq,  &dk,    &dv,  &dko,    &dvo,  &dso,
                        &dsb, &seq, &heads, &kvh, &canvas, &full, &window};
        dg_cuda_launch(hd == DG_FULL_HEAD_DIM ? DG_CUDA_ATTENTION_FAST_FULL
                                              : DG_CUDA_ATTENTION_FAST_LOCAL,
                       (unsigned)(segments * ((seq + 15) / 16)), DG_HEADS, 128, args);
        return;
    }
    if (max_keys > DG_CUDA_ROW_MAX || hd > DG_FULL_HEAD_DIM)
        die("accelerator attention span is too long");
    int kn = kvh * hd, keys = old_rows + n;
    CUdeviceptr dkt = scratch, dko = k_old, dk = k;
    void *targs[] = {&dkt, &dko, &dk, &old_rows, &keys, &kn};
    dg_cuda_launch_2d(DG_CUDA_TRANSPOSE_KEYS, dg_cuda_blocks(keys, 32, 1u << 30),
                      dg_cuda_blocks(kn, 32, DG_CUDA_GRID_Y), 32, 8, targs);
    CUdeviceptr da = a, dq = q, dv = v, dvo = v_old, dso = seg_old, dsb = seg_base;
    int heads = DG_HEADS, window = DG_LOCAL_WINDOW;
    void *args[] = {&da, &dq,  &dkt,   &dv,  &keys, &dvo,    &dso,  &dsb,
                    &n,  &seq, &heads, &kvh, &hd,   &canvas, &full, &window};
    dg_cuda_launch(DG_CUDA_ATTENTION, dg_cuda_blocks((long long)n * heads, 1, 1u << 30), 1,
                   DG_CUDA_THREADS, args);
}

static void dg_cuda_moe_tail(DGDevice x, DGDevice eo, DGDevice slot, DGDevice weight, DGDevice d,
                             DGDevice p2, DGDevice post, float scalar, int tokens) {
    CUdeviceptr dx = x, deo = eo, dslot = slot, dweight = weight, dd = d, dp2 = p2, dpost = post;
    int n = DG_H, topk = DG_TOPK; /* the kernel's rows are DG_H wide */
    void *args[] = {&dx, &deo, &dslot, &dweight, &dd, &dp2, &dpost, &scalar, &tokens, &n, &topk};
    dg_cuda_launch(DG_CUDA_MOE_TAIL, dg_cuda_blocks(tokens, 1, 1u << 30), 1, DG_CUDA_THREADS, args);
}

static const DGAccelOps *dg_accel(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    static const DGAccelOps cuda = {"cuda",
                                    dg_cuda_matrix,
                                    dg_cuda_release,
                                    dg_cuda_alloc,
                                    dg_cuda_free,
                                    dg_cuda_to_device,
                                    dg_cuda_to_host,
                                    dg_cuda_host_alloc,
                                    dg_cuda_host_free,
                                    dg_cuda_nvfp4_scratch,
                                    dg_cuda_nvfp4_mm,
                                    dg_cuda_bf16_mm,
                                    dg_cuda_rms,
                                    dg_cuda_gelu_mul,
                                    dg_cuda_nvfp4_qdq,
                                    dg_cuda_nvfp4_row,
                                    dg_cuda_gather,
                                    dg_cuda_copy,
                                    dg_cuda_rope,
                                    dg_cuda_attention,
                                    dg_cuda_moe_tail,
                                    dg_cuda_table_bytes,
                                    dg_cuda_matrix_table,
                                    dg_cuda_route,
                                    dg_cuda_group,
                                    dg_cuda_nvfp4_grouped};
    pthread_once(&once, dg_cuda_open);
    return dg_cuda.ready ? &cuda : NULL;
}

/* CUDA currently uses the context's legacy default stream. Keep complete
 * decisions from separate sessions from interleaving until per-session
 * streams are introduced. CPU sessions remain fully concurrent. */
static pthread_mutex_t jb_cuda_call_mutex = PTHREAD_MUTEX_INITIALIZER;

static int jb_accel_call_lock(const DGModel *model) {
    if (!model->accel)
        return 0;
    if (pthread_mutex_lock(&jb_cuda_call_mutex))
        die_status(JB_ERROR_INTERNAL, "cannot lock CUDA execution");
    return 1;
}

static void jb_accel_call_unlock(int locked) {
    if (locked)
        (void)pthread_mutex_unlock(&jb_cuda_call_mutex);
}

#ifndef JB_NO_MAIN
/* After dg_accel(): why a present device runs nothing, or NULL; *log gets
 * NVRTC's log, empty unless compiling failed. Only the CLI reports it. */
static const char *dg_accel_failure(const char **log) {
    *log = dg_cuda.compile_log;
    return dg_cuda.ready ? NULL : dg_cuda.failure;
}
#endif
#else
static const DGAccelOps *dg_accel(void) {
    return NULL;
}

static int jb_accel_call_lock(const DGModel *model) {
    (void)model;
    return 0;
}

static void jb_accel_call_unlock(int locked) {
    (void)locked;
}

#ifndef JB_NO_MAIN
static const char *dg_accel_failure(const char **log) {
    *log = "";
    return NULL;
}
#endif
#endif

