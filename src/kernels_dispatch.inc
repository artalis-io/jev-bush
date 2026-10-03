#if defined(JB_HAVE_AVX2)
static int jb_cpu_has_avx2(void) {
#if defined(JB_SCALAR)
    return 0;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#elif defined(_MSC_VER) && defined(_M_X64)
    int leaf[4];
    __cpuid(leaf, 1);
    if (!(leaf[2] & (1 << 27)) || !(leaf[2] & (1 << 28)) || !(leaf[2] & (1 << 12)) ||
        (_xgetbv(0) & 6) != 6)
        return 0;
    __cpuidex(leaf, 7, 0);
    return !!(leaf[1] & (1 << 5));
#else
    return 0;
#endif
}
#endif

#if defined(JB_HAVE_AVX512)
static int jb_cpu_has_avx512(void) {
#if defined(JB_SCALAR)
    return 0;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512bw");
#else
    return 0;
#endif
}
#endif

static const DGKernelOps *dg_kernels(void) {
    (void)dg_nvfp4_qdq_parallel;
    static _Atomic(const DGKernelOps *) cache;
    static const DGKernelOps scalar = {
        "scalar",       dg_nvfp4_qdq_ref, NULL,       dg_nvfp4_mm_ref, dg_nvfp4_gated_each,
        dg_mm_data_ref, dg_rms_ref,       dg_dot_ref, dg_dots_each,    1};
    const DGKernelOps *cached = atomic_load_explicit(&cache, memory_order_acquire);
    if (cached)
        return cached;
    const DGKernelOps *selected = &scalar;
#if defined(JB_HAVE_AVX512)
    (void)dg_nvfp4_qdq_parallel;
#if JB_STRICT_MATH
    (void)dg_nvfp4_qdq_avx512;
    (void)dg_nvfp4_mm_avx512;
    (void)dg_mm_data_avx512;
    (void)dg_dot_avx512;
    static const DGKernelOps avx512 = {
        "avx512-exact", dg_nvfp4_qdq_ref, NULL,       dg_nvfp4_mm_ref, dg_nvfp4_gated_each,
        dg_mm_data_ref, dg_rms_avx512,    dg_dot_ref, dg_dots_each,    1};
#else
    static const DGKernelOps avx512 = {"avx512",
                                       dg_nvfp4_qdq_avx512,
                                       dg_nvfp4_swizzle,
                                       dg_nvfp4_mm_avx512,
                                       dg_nvfp4_gated_each,
                                       dg_mm_data_avx512,
                                       dg_rms_avx512,
                                       dg_dot_avx512,
                                       dg_dots_each,
                                       0};
#endif
    if (jb_cpu_has_avx512())
        selected = &avx512;
#endif
#if defined(JB_HAVE_AVX2)
    static const DGKernelOps avx2 = {
        "avx2",          dg_nvfp4_qdq_parallel, NULL,        dg_nvfp4_mm_avx2, dg_nvfp4_gated_each,
        dg_mm_data_avx2, dg_rms_avx2,           dg_dot_avx2, dg_dots_each,     0};
    if (selected == &scalar && jb_cpu_has_avx2())
        selected = &avx2;
#endif
#if defined(JB_NEON)
    static const DGKernelOps neon = {
        "neon",          dg_nvfp4_qdq_parallel, NULL,       dg_nvfp4_mm_neon, dg_nvfp4_gated_neon,
        dg_mm_data_neon, dg_rms_neon,           dg_dot_ref, dg_dots_neon,     1};
    selected = &neon;
#endif
    const DGKernelOps *expected = NULL;
    if (!atomic_compare_exchange_strong_explicit(&cache, &expected, selected, memory_order_release,
                                                 memory_order_acquire))
        selected = expected;
    return selected;
}
