static DGTensor *dg_layer_tensor(DGModel *m, int l, const char *tail) {
    char n[192];
    snprintf(n, sizeof n, "model.decoder.layers.%d.%s", l, tail);
    return dg_tensor(m, n);
}

static void dg_rope(float *x, int heads, int hd, const float *cv, const float *sv) {
    int half = hd / 2;
    for (int h = 0; h < heads; h++) {
        float *v = x + (size_t)h * hd;
        for (int i = 0; i < half; i++) {
            float a = v[i], b = v[i + half];
            v[i] = a * cv[i] - b * sv[i];
            v[i + half] = b * cv[i] + a * sv[i];
        }
    }
}

static void dg_norm_heads(float *x, int heads, int hd, DGTensor *scale) {
    for (int h = 0; h < heads; h++)
        dg_rms(x + (size_t)h * hd, x + (size_t)h * hd, scale, hd);
}

enum { DG_CAUSAL, DG_CANVAS, DG_SUFFIX };

static void jb_workspace_plan(size_t *used, size_t count, size_t element_size) {
    *used = jb_align_up(*used, 64);
    *used = jb_size_add(*used, jb_size_mul(count, element_size));
}

static size_t dg_attention_workspace_bytes(int n, int old, int seq, int segments,
                                           int wrapper_arrays, int accel) {
    size_t used = 0;
    if (wrapper_arrays && segments > 1) {
        jb_workspace_plan(&used, (size_t)segments, sizeof(DGKV *));
        jb_workspace_plan(&used, (size_t)segments, sizeof(int));
    }
    /* dg_attention_accel: rotary tables and segment tables. */
    if (accel) {
        jb_workspace_plan(&used, (size_t)n * DG_FULL_HEAD_DIM, sizeof(float));
        jb_workspace_plan(&used, (size_t)segments, sizeof(int));
        jb_workspace_plan(&used, (size_t)segments, sizeof(int));
        return used;
    }
    /* Full attention has the largest Q/K/V projections in this model. */
    int qn = DG_HEADS * DG_FULL_HEAD_DIM;
    int kn = DG_FULL_KV_HEADS * DG_FULL_HEAD_DIM;
    jb_workspace_plan(&used, (size_t)n * qn, sizeof(float));
    jb_workspace_plan(&used, (size_t)n * kn, sizeof(float));
    jb_workspace_plan(&used, (size_t)n * kn, sizeof(float));
    jb_workspace_plan(&used, (size_t)n * qn, sizeof(float));
    jb_workspace_plan(&used, (size_t)n * (old + seq), sizeof(float));
    return used;
}

static size_t dg_ff_workspace_bytes(const DGModel *m, int n) {
    size_t used = 0;
#define DG_PLAN_FLOAT(count) jb_workspace_plan(&used, (size_t)(count), sizeof(float))
#define DG_PLAN_INT(count) jb_workspace_plan(&used, (size_t)(count), sizeof(int))
    /* dg_ff_accel keeps everything on the device. */
    if (m->accel)
        return used;
    DG_PLAN_FLOAT((size_t)n * DG_H);       /* z1 */
    DG_PLAN_FLOAT((size_t)n * DG_DENSE);   /* gate */
    DG_PLAN_FLOAT((size_t)n * DG_DENSE);   /* up */
    DG_PLAN_FLOAT((size_t)n * DG_H);       /* dense output */
    DG_PLAN_FLOAT((size_t)n * DG_H);       /* z2 */
    DG_PLAN_FLOAT((size_t)n * DG_H);       /* router input */
    DG_PLAN_FLOAT((size_t)n * DG_EXPERTS); /* router output */
    DG_PLAN_INT((size_t)n * DG_TOPK);
    DG_PLAN_FLOAT((size_t)n * DG_TOPK);
    DG_PLAN_FLOAT((size_t)n * DG_TOPK * DG_H); /* expert contributions */
    DG_PLAN_FLOAT((size_t)n * DG_H);           /* gathered inputs */
    DG_PLAN_FLOAT((size_t)n * DG_EXPERT_PAIR); /* gate/up output */
    DG_PLAN_FLOAT((size_t)n * DG_MOE);         /* hidden */
    DG_PLAN_FLOAT((size_t)n * DG_H);           /* expert output */
    if (m->nvfp4)
        DG_PLAN_FLOAT((size_t)n * DG_H); /* quantized expert input */
    DG_PLAN_INT((size_t)n * 2);          /* token/slot owner */
    if (m->nvfp4)
        DG_PLAN_FLOAT((size_t)n * DG_MOE); /* quantized hidden */
    DG_PLAN_FLOAT(DG_H);                   /* MoE sum */
    DG_PLAN_FLOAT(DG_H);                   /* FF sum */
#undef DG_PLAN_INT
#undef DG_PLAN_FLOAT
    return used;
}

static size_t dg_workspace_bytes(const DGModel *m, int n, int old, int seq, int segments,
                                 int wrapper_arrays) {
    size_t persistent = 0;
    if (!m->accel) { /* the residual and the normalized input */
        jb_workspace_plan(&persistent, (size_t)n * DG_H, sizeof(float));
        jb_workspace_plan(&persistent, (size_t)n * DG_H, sizeof(float));
    }
    size_t attention =
        dg_attention_workspace_bytes(n, old, seq, segments, wrapper_arrays, m->accel != NULL);
    size_t ff = dg_ff_workspace_bytes(m, n);
    return jb_size_add(persistent, attention > ff ? attention : ff);
}

typedef struct {
    JBArena arena;
    JBArena kv_arena;
    JBArena control_arena;
    size_t capacity, kv_capacity, control_capacity;
    int active, kv_active, control_active;
    /* The session's accelerator activations, or 0, and page-locked host
     * memory for the staged K/V output, both from accel. */
    const DGAccelOps *accel;
    DGDevice device;
    size_t device_bytes;
    float *pinned;
    size_t pinned_bytes;
    /* Device scratch for work outside a forward pass, from accel. */
    DGDevice scratch;
    size_t scratch_bytes;
#ifdef JB_PROFILE
    JBProfile profile;
#endif
} DGWorkspace;

static JBArena *dg_workspace_begin(DGWorkspace *workspace, size_t required) {
    if (workspace->active)
        die_status(JB_ERROR_INTERNAL, "inference workspace is already active");
    jb_arena_reserve(&workspace->arena, &workspace->capacity, required);
    workspace->active = 1;
    return &workspace->arena;
}

static void dg_workspace_end(DGWorkspace *workspace) {
    if (!workspace->active || workspace->arena.used ||
        workspace->arena.peak != workspace->arena.size)
        die_status(JB_ERROR_INTERNAL, "inference workspace plan mismatch");
    workspace->active = 0;
}

static void dg_workspace_destroy(DGWorkspace *workspace) {
    if (workspace->arena.allocation)
        jb_arena_free(&workspace->arena);
    if (workspace->kv_arena.allocation)
        jb_arena_free(&workspace->kv_arena);
    if (workspace->control_arena.allocation)
        jb_arena_free(&workspace->control_arena);
    if (workspace->device)
        workspace->accel->free(workspace->device);
    if (workspace->pinned)
        workspace->accel->host_free(workspace->pinned);
    if (workspace->scratch)
        workspace->accel->free(workspace->scratch);
    memset(workspace, 0, sizeof *workspace);
}

static int dg_kv_width(int layer);

/* Where layer l's K then V start in the staged K/V output, in floats. */
static size_t dg_kv_stage_offset(int n, int l) {
    size_t at = 0;
    for (int i = 0; i < l; i++)
        at += (size_t)n * dg_kv_width(i) * 2;
    return at;
}

/* An accelerated layer's device activations for n tokens, carved from the
 * session's device workspace. Q, K and V take the widest projections of
 * either layer kind. */
typedef struct {
    DGDevice x, res, z, q, k, v, a, g, u, d, z2, rin, route, qz2, weight, index, slot, flag, gather,
        eg, eh, eqh, eo, scratch, rope, seg_old, seg_base, kv_k, kv_v, kt, top, tiles, ntiles,
        kv_out;
    /* Rows of the widest K/V width that kv_k and kv_v each hold. */
    int kv_rows;
} DGDeviceLayout;

/* A forward call's device workspace: its base, and how many cached K/V rows
 * it holds for attention. */
typedef struct {
    DGDevice base;
    int kv_rows;
    /* Page-locked room for the staged K/V output, or NULL. */
    float *pinned;
} DGDeviceWork;

static size_t dg_device_layout(DGDeviceLayout *d, DGDevice base, int n, int kv_rows,
                               const DGAccelOps *a) {
    size_t at = 0, total = (size_t)n * DG_TOPK;
    size_t qn = DG_HEADS * DG_FULL_HEAD_DIM,
           kn = DG_LOCAL_KV_HEADS * DG_LOCAL_HEAD_DIM > DG_FULL_KV_HEADS * DG_FULL_HEAD_DIM
                    ? DG_LOCAL_KV_HEADS * DG_LOCAL_HEAD_DIM
                    : DG_FULL_KV_HEADS * DG_FULL_HEAD_DIM;
#define DG_CARVE(field, bytes)                                                                     \
    (at = jb_align_up(at, 256), d->field = base ? base + at : 0, at = jb_size_add(at, (bytes)))
    DG_CARVE(x, jb_size_mul((size_t)n, DG_H * 4));
    DG_CARVE(res, jb_size_mul((size_t)n, DG_H * 4));
    DG_CARVE(z, jb_size_mul((size_t)n, DG_H * 4));
    DG_CARVE(q, jb_size_mul((size_t)n, qn * 4));
    DG_CARVE(k, jb_size_mul((size_t)n, kn * 4));
    DG_CARVE(v, jb_size_mul((size_t)n, kn * 4));
    DG_CARVE(a, jb_size_mul((size_t)n, qn * 4));
    DG_CARVE(g, jb_size_mul((size_t)n, DG_DENSE * 4));
    DG_CARVE(u, jb_size_mul((size_t)n, DG_DENSE * 4));
    DG_CARVE(d, jb_size_mul((size_t)n, DG_H * 4));
    DG_CARVE(z2, jb_size_mul((size_t)n, DG_H * 4));
    DG_CARVE(rin, jb_size_mul((size_t)n, DG_H * 4));
    DG_CARVE(route, jb_size_mul((size_t)n, DG_EXPERTS * 4));
    DG_CARVE(qz2, jb_size_mul((size_t)n, DG_H * 4));
    DG_CARVE(weight, jb_size_mul(total, 4));
    DG_CARVE(index, jb_size_mul(total, 4));
    DG_CARVE(slot, jb_size_mul(total, 4));
    DG_CARVE(flag, 4);
    DG_CARVE(gather, jb_size_mul(total, DG_H * 4));
    DG_CARVE(eg, jb_size_mul(total, DG_MOE * 4));
    DG_CARVE(eh, jb_size_mul(total, DG_MOE * 4));
    DG_CARVE(eqh, jb_size_mul(total, DG_MOE * 4));
    DG_CARVE(eo, jb_size_mul(total, DG_H * 4));
    DG_CARVE(scratch, a->nvfp4_scratch((int)total));
    DG_CARVE(rope, jb_size_mul((size_t)n, DG_FULL_HEAD_DIM * 4));
    DG_CARVE(seg_old, jb_size_mul((size_t)n, 4));
    DG_CARVE(seg_base, jb_size_mul((size_t)n, 4));
    DG_CARVE(kv_k, jb_size_mul((size_t)kv_rows, kn * 4));
    DG_CARVE(kv_v, jb_size_mul((size_t)kv_rows, kn * 4));
    DG_CARVE(kt, jb_size_mul((size_t)n + (size_t)kv_rows, kn * 4));
    DG_CARVE(top, jb_size_mul(total, 4));
    DG_CARVE(tiles, a->nvfp4_scratch((int)total));
    DG_CARVE(ntiles, 4);
    DG_CARVE(kv_out, jb_size_mul(dg_kv_stage_offset(n, DG_L), 4));
#undef DG_CARVE
    d->kv_rows = kv_rows;
    return at;
}

/* The session's device workspace for n tokens and kv_rows cached K/V rows,
 * grown as needed; its base is 0 when the model runs on the CPU kernels. */
static DGDeviceWork dg_device_begin(DGWorkspace *owner, const DGModel *m, int n, int kv_rows) {
    DGDeviceWork work = {0, kv_rows, NULL};
    if (!m->accel)
        return work;
    DGDeviceLayout d;
    size_t bytes = dg_device_layout(&d, 0, n, kv_rows, m->accel);
    owner->accel = m->accel;
    if (owner->device_bytes < bytes) {
        if (owner->device)
            m->accel->free(owner->device);
        owner->device_bytes = 0;
        owner->device = m->accel->alloc(bytes);
        if (!owner->device)
            die("accelerator out of memory for activations");
        owner->device_bytes = bytes;
    }
    size_t staged = jb_size_mul(dg_kv_stage_offset(n, DG_L), 4);
    if (owner->pinned_bytes < staged) {
        if (owner->pinned)
            m->accel->host_free(owner->pinned);
        owner->pinned_bytes = 0;
        owner->pinned = m->accel->host_alloc(staged);
        if (!owner->pinned)
            die("accelerator out of page-locked memory");
        owner->pinned_bytes = staged;
    }
    work.base = owner->device;
    work.pinned = owner->pinned;
    return work;
}

/* The session's device scratch, at least bytes, from the accelerator its
 * forward passes use; grown as needed, and 0 when the device is out of
 * memory, for callers that can do the work on the CPU instead. */
static DGDevice dg_device_scratch(DGWorkspace *owner, size_t bytes) {
    if (owner->scratch_bytes < bytes) {
        if (owner->scratch)
            owner->accel->free(owner->scratch);
        owner->scratch_bytes = 0;
        owner->scratch = owner->accel->alloc(bytes);
        if (!owner->scratch)
            return 0;
        owner->scratch_bytes = bytes;
    }
    return owner->scratch;
}

/* dg_mm's checks, then the accelerator's product. */
static void dg_mm_accel(const DGModel *m, const DGTensor *w, DGDevice x, DGDevice y, int tokens,
                        int rows, int cols) {
    if (w->nd != 2 || w->shape[0] != (uint64_t)rows || w->shape[1] != (uint64_t)cols)
        die2("bad matrix shape", w->name);
    if (tokens < 1 || tokens > JB_MAX_CTX)
        die("DiffusionGemma sequence exceeds context limit");
    if (rows < 2 || rows % 2)
        die("matrix row count must be even");
    if (!w->device)
        die2("tensor not on the accelerator", w->name);
    m->accel->bf16_mm(w->device, x, y, tokens, rows, cols);
}

/* dg_nvfp4_qdq_ref's checks, then the accelerator's quantizer. */
static void dg_nvfp4_qdq_accel(const DGModel *m, DGDevice out, DGDevice in, int tokens, int cols,
                               float base, DGDevice flag) {
    if (!jb_finitef(base) || !(base > 0) || cols % 32)
        die("invalid NVFP4 activation scale");
    m->accel->nvfp4_qdq(out, in, tokens, cols, base, flag);
}

static JBArena *dg_control_begin(DGWorkspace *workspace, size_t required) {
    if (workspace->control_active)
        die_status(JB_ERROR_INTERNAL, "worker control storage is already active");
    jb_arena_reserve(&workspace->control_arena, &workspace->control_capacity, required);
    workspace->control_active = 1;
    return &workspace->control_arena;
}

static void dg_control_end(DGWorkspace *workspace) {
    if (!workspace->control_active ||
        workspace->control_arena.used != workspace->control_arena.size ||
        workspace->control_arena.peak != workspace->control_arena.size)
        die_status(JB_ERROR_INTERNAL, "corrupt worker control storage");
    workspace->control_active = 0;
}

static size_t dg_forward_workspace_bytes(const DGModel *m, int n, int old, int seq, int segments,
                                         int wrapper_arrays, size_t tail) {
    size_t output = 0;
    jb_workspace_plan(&output, (size_t)n * DG_H, sizeof(float));
    size_t layers = dg_workspace_bytes(m, n, old, seq, segments, wrapper_arrays);
    return jb_size_add(output, layers > tail ? layers : tail);
}

static int dg_kv_width(int layer) {
    int full = layer % DG_FULL_EVERY == DG_FULL_EVERY - 1;
    return (full ? DG_FULL_KV_HEADS : DG_LOCAL_KV_HEADS) *
           (full ? DG_FULL_HEAD_DIM : DG_LOCAL_HEAD_DIM);
}

static size_t dg_kv_bytes(int documents, const int *lengths) {
    if (documents < 1)
        die("invalid K/V document count");
    size_t bytes = 0;
    for (int d = 0; d < documents; d++)
        for (int l = 0; l < DG_L; l++) {
            size_t count = jb_size_mul((size_t)lengths[d], (size_t)dg_kv_width(l));
            jb_workspace_plan(&bytes, count, sizeof(float));
            jb_workspace_plan(&bytes, count, sizeof(float));
        }
    return bytes;
}

static void dg_kv_bind(JBArena *storage, DGKV *kv, int documents, const int *lengths) {
    for (int d = 0; d < documents; d++)
        for (int l = 0; l < DG_L; l++) {
            DGKV *one = &kv[(size_t)d * DG_L + l];
            size_t count = jb_size_mul((size_t)lengths[d], (size_t)dg_kv_width(l));
            one->n = lengths[d];
            one->k = jb_arena_alloc(storage, count, sizeof *one->k, 0);
            one->v = jb_arena_alloc(storage, count, sizeof *one->v, 0);
        }
    if (storage->peak != storage->size)
        die("K/V storage plan mismatch");
}

/* Prefix K/V has its own immutable lifetime and therefore owns an exact slab. */
static void dg_kv_init(JBArena *storage, DGKV *kv, int documents, const int *lengths) {
    if (storage->allocation)
        die("K/V storage initialized twice");
    jb_arena_init(storage, dg_kv_bytes(documents, lengths));
    dg_kv_bind(storage, kv, documents, lengths);
}

/* Per-request K/V reuses a grow-only worker slab across evaluation rows. */
static void dg_worker_kv_begin(DGWorkspace *workspace, DGKV *kv, int documents,
                               const int *lengths) {
    if (workspace->kv_active)
        die_status(JB_ERROR_INTERNAL, "worker K/V storage is already active");
    size_t required = dg_kv_bytes(documents, lengths);
    jb_arena_reserve(&workspace->kv_arena, &workspace->kv_capacity, required);
    workspace->kv_active = 1;
    dg_kv_bind(&workspace->kv_arena, kv, documents, lengths);
}

static void dg_worker_kv_end(DGWorkspace *workspace, DGKV *kv, int documents) {
    JB_TICK(kv_free_start);
    if (!workspace->kv_active || workspace->kv_arena.used != workspace->kv_arena.size ||
        workspace->kv_arena.peak != workspace->kv_arena.size)
        die_status(JB_ERROR_INTERNAL, "corrupt worker K/V ownership");
    workspace->kv_active = 0;
    memset(kv, 0, jb_size_mul(jb_size_mul((size_t)documents, DG_L), sizeof *kv));
    JB_TO(kv_free, kv_free_start);
}

static void dg_kv_free(JBArena *storage) {
    JB_TICK(kv_free_start);
    if (!storage->allocation)
        return;
    jb_arena_free(storage);
    JB_TO(kv_free, kv_free_start);
}

static void dg_attention_segments(DGModel *m, int l, float *x, int segments, int seq, int pos0,
                                  const DGKV *const *cache, const int *lens, DGKV *out, int mode,
                                  JBArena *workspace, const DGDeviceLayout *dev);

static void dg_attention(DGModel *m, int l, float *x, int n, int pos0, const DGKV *cache, DGKV *out,
                         int mode, int batch, JBArena *workspace, const DGDeviceLayout *dev) {
    if (batch <= 0 || n < 0 || n % batch || (out && batch != 1))
        die("invalid attention batch");
    const DGKV *one_cache;
    int one_len;
    size_t mark = jb_arena_mark(workspace);
    const DGKV **caches =
        batch == 1 ? &one_cache : jb_arena_alloc(workspace, (size_t)batch, sizeof *caches, 0);
    int *lens = batch == 1 ? &one_len : jb_arena_alloc(workspace, (size_t)batch, sizeof *lens, 0);
    for (int b = 0; b < batch; b++) {
        caches[b] = mode != DG_CAUSAL ? cache : NULL;
        lens[b] = n / batch;
    }
    dg_attention_segments(m, l, x, batch, n / batch, pos0, caches[0] ? caches : NULL, lens, out,
                          mode, workspace, dev);
    jb_arena_reset(workspace, mark);
}

/* Each segment's K/V output: its cached rows, then its own useful ones. */
static void dg_kv_output(DGKV *out, const DGKV *const *cache, const int *lens, int segments,
                         int seq, int kn, const float *k, const float *v) {
    for (int s = 0; s < segments; s++) {
        int old = cache ? cache[s]->n : 0, keep = lens[s];
        if (out[s].n != old + keep || !out[s].k || !out[s].v)
            die("K/V output does not match planned storage");
        if (old) {
            memcpy(out[s].k, cache[s]->k, (size_t)old * kn * 4);
            memcpy(out[s].v, cache[s]->v, (size_t)old * kn * 4);
        }
        memcpy(out[s].k + (size_t)old * kn, k + (size_t)s * seq * kn, (size_t)keep * kn * 4);
        memcpy(out[s].v + (size_t)old * kn, v + (size_t)s * seq * kn, (size_t)keep * kn * 4);
    }
}

/* dg_attention_segments on the accelerator, from dev->z to dev->z. The
 * rotary tables are computed on the host, with the C library's cosf and
 * sinf as the reference computes them. Each distinct cache's rows are
 * uploaded for the call; K and V come back only for a K/V output. */
static void dg_attention_accel(DGModel *m, int l, int segments, int seq, int pos0,
                               const DGKV *const *cache, const int *lens, DGKV *out, int mode,
                               JBArena *workspace, const DGDeviceLayout *dev) {
    const DGAccelOps *a = m->accel;
    size_t mark = jb_arena_mark(workspace);
    int n = segments * seq;
    JB_TICK(qkv_start);
    int full = l % DG_FULL_EVERY == DG_FULL_EVERY - 1;
    int hd = full ? DG_FULL_HEAD_DIM : DG_LOCAL_HEAD_DIM;
    int kvh = full ? DG_FULL_KV_HEADS : DG_LOCAL_KV_HEADS;
    int qn = DG_HEADS * hd, kn = kvh * hd;
    int widest = DG_LOCAL_KV_HEADS * DG_LOCAL_HEAD_DIM > DG_FULL_KV_HEADS * DG_FULL_HEAD_DIM
                     ? DG_LOCAL_KV_HEADS * DG_LOCAL_HEAD_DIM
                     : DG_FULL_KV_HEADS * DG_FULL_HEAD_DIM;
    DGTensor *qw = dg_layer_tensor(m, l, "self_attn.q_proj.weight");
    DGTensor *kw = dg_layer_tensor(m, l, "self_attn.k_proj.weight");
    DGTensor *vw = full ? NULL : dg_layer_tensor(m, l, "self_attn.v_proj.weight");
    DGTensor *qnrm = dg_layer_tensor(m, l, "self_attn.q_norm.weight");
    DGTensor *knrm = dg_layer_tensor(m, l, "self_attn.k_norm.weight");
    DGTensor *ow = dg_layer_tensor(m, l, "self_attn.o_proj.weight");
    float *table = jb_arena_alloc(workspace, (size_t)n * DG_FULL_HEAD_DIM, sizeof *table, 0);
    int *seg_old = jb_arena_alloc(workspace, (size_t)segments, sizeof *seg_old, 0),
        *seg_base = jb_arena_alloc(workspace, (size_t)segments, sizeof *seg_base, 0);
    /* The host's part comes first: each upload waits for the work queued
     * before it, and nothing but the input norm is queued yet. */
    JB_TO(attention_qkv, qkv_start);
    JB_TICK(kv_start);
    int half = hd / 2, rotated = full ? DG_FULL_ROPE_DIM : half;
    float inv[DG_FULL_HEAD_DIM / 2];
    for (int i = 0; i < half; i++)
        inv[i] = i < rotated ? powf(full ? 1000000.0f : 10000.0f, -(float)(2 * i) / hd) : 0;
    for (int ti = 0; ti < n; ti++) {
        int s = ti / seq, t = ti % seq, old = cache ? cache[s]->n : 0,
            pos = (cache ? old : pos0) + t;
        float *cv = table + (size_t)ti * DG_FULL_HEAD_DIM, *sv = cv + half;
        for (int i = 0; i < half; i++) {
            cv[i] = cosf(pos * inv[i]);
            sv[i] = sinf(pos * inv[i]);
        }
    }
    a->to_device(dev->rope, table, (size_t)n * DG_FULL_HEAD_DIM * 4);
    int rows = 0, max_keys = 0;
    for (int s = 0; s < segments; s++) {
        int shared = -1;
        seg_old[s] = cache ? cache[s]->n : 0;
        for (int e = 0; e < s && cache; e++)
            if (cache[e] == cache[s]) {
                shared = e;
                break;
            }
        if (shared >= 0)
            seg_base[s] = seg_base[shared];
        else {
            seg_base[s] = rows;
            if ((size_t)(rows + seg_old[s]) * kn > (size_t)dev->kv_rows * widest)
                die("accelerator K/V space exceeded");
            if (seg_old[s]) {
                a->to_device(dev->kv_k + (size_t)rows * kn * 4, cache[s]->k,
                             (size_t)seg_old[s] * kn * 4);
                a->to_device(dev->kv_v + (size_t)rows * kn * 4, cache[s]->v,
                             (size_t)seg_old[s] * kn * 4);
            }
            rows += seg_old[s];
        }
        if (seg_old[s] + seq > max_keys)
            max_keys = seg_old[s] + seq;
    }
    a->to_device(dev->seg_old, seg_old, (size_t)segments * sizeof *seg_old);
    a->to_device(dev->seg_base, seg_base, (size_t)segments * sizeof *seg_base);
    JB_TO(attention_kv, kv_start);
    JB_TICK(prepare_start);
    dg_mm_accel(m, qw, dev->z, dev->q, n, qn, DG_H);
    dg_mm_accel(m, kw, dev->z, dev->k, n, kn, DG_H);
    if (vw)
        dg_mm_accel(m, vw, dev->z, dev->v, n, kn, DG_H);
    else
        a->copy(dev->v, dev->k, (size_t)n * kn * 4);
    a->rms(dev->q, dev->q, qnrm->device, 0, 0, 0, n * DG_HEADS, hd);
    a->rms(dev->k, dev->k, knrm->device, 0, 0, 0, n * kvh, hd);
    a->rms(dev->v, dev->v, 0, 0, 0, 0, n * kvh, hd);
    a->rope(dev->q, dev->rope, n, DG_HEADS, hd, qn);
    a->rope(dev->k, dev->rope, n, kvh, hd, kn);
    JB_TO(attention_prepare, prepare_start);
    JB_TICK(core_start);
    a->attention(dev->a, dev->q, dev->k, dev->v, dev->kv_k, dev->kv_v, dev->seg_old, dev->seg_base,
                 n, seq, kvh, hd, mode == DG_CANVAS, full, max_keys, rows, dev->kt);
    JB_TO(attention_core, core_start);
    JB_TICK(output_start);
    dg_mm_accel(m, ow, dev->a, dev->z, n, DG_H, qn);
    JB_TO(attention_output, output_start);
    /* The K/V output's cached rows are host memory and are copied now; its
     * own rows are staged on the device and come down after the last
     * layer, through dg_kv_fetch, so no layer waits for them. */
    if (out) {
        for (int s = 0; s < segments; s++) {
            int old = cache ? cache[s]->n : 0;
            if (out[s].n != old + lens[s] || !out[s].k || !out[s].v)
                die("K/V output does not match planned storage");
            if (old) {
                memcpy(out[s].k, cache[s]->k, (size_t)old * kn * 4);
                memcpy(out[s].v, cache[s]->v, (size_t)old * kn * 4);
            }
        }
        DGDevice stage = dev->kv_out + dg_kv_stage_offset(n, l) * 4;
        a->copy(stage, dev->k, (size_t)n * kn * 4);
        a->copy(stage + (size_t)n * kn * 4, dev->v, (size_t)n * kn * 4);
    }
    jb_arena_reset(workspace, mark);
}

/* Execute independent equal-stride sequences through one set of projections.
 * Attention never crosses a segment boundary.  Suffix segments may have
 * different useful lengths; padding is computed but never copied into K/V.
 * With dev, the accelerator runs it from dev->z to dev->z and x is unused. */
static void dg_attention_segments(DGModel *m, int l, float *x, int segments, int seq, int pos0,
                                  const DGKV *const *cache, const int *lens, DGKV *out, int mode,
                                  JBArena *workspace, const DGDeviceLayout *dev) {
    if (dev) {
        dg_attention_accel(m, l, segments, seq, pos0, cache, lens, out, mode, workspace, dev);
        return;
    }
    size_t mark = jb_arena_mark(workspace);
    int n = segments * seq;
    JB_TICK(qkv_start);
    int full = l % DG_FULL_EVERY == DG_FULL_EVERY - 1;
    int hd = full ? DG_FULL_HEAD_DIM : DG_LOCAL_HEAD_DIM;
    int kvh = full ? DG_FULL_KV_HEADS : DG_LOCAL_KV_HEADS;
    int qn = DG_HEADS * hd, kn = kvh * hd;
    DGTensor *qw = dg_layer_tensor(m, l, "self_attn.q_proj.weight");
    DGTensor *kw = dg_layer_tensor(m, l, "self_attn.k_proj.weight");
    DGTensor *vw = full ? NULL : dg_layer_tensor(m, l, "self_attn.v_proj.weight");
    DGTensor *qnrm = dg_layer_tensor(m, l, "self_attn.q_norm.weight");
    DGTensor *knrm = dg_layer_tensor(m, l, "self_attn.k_norm.weight");
    float *q = jb_arena_alloc(workspace, (size_t)n * qn, sizeof *q, 0),
          *k = jb_arena_alloc(workspace, (size_t)n * kn, sizeof *k, 0),
          *v = jb_arena_alloc(workspace, (size_t)n * kn, sizeof *v, 0);
    dg_mm(qw, x, q, n, qn, DG_H);
    dg_mm(kw, x, k, n, kn, DG_H);
    if (vw)
        dg_mm(vw, x, v, n, kn, DG_H);
    else
        memcpy(v, k, (size_t)n * kn * 4);
    JB_TO(attention_qkv, qkv_start);
    JB_TICK(prepare_start);
    int half = hd / 2, rotated = full ? DG_FULL_ROPE_DIM : half;
    float inv[DG_FULL_HEAD_DIM / 2];
    for (int i = 0; i < half; i++)
        inv[i] = i < rotated ? powf(full ? 1000000.0f : 10000.0f, -(float)(2 * i) / hd) : 0;
    {
        int ti;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (ti = 0; ti < n; ti++) {
            int s = ti / seq, t = ti % seq, old = cache ? cache[s]->n : 0,
                pos = (cache ? old : pos0) + t;
            float cv[DG_FULL_HEAD_DIM / 2], sv[DG_FULL_HEAD_DIM / 2];
            for (int i = 0; i < half; i++) {
                cv[i] = cosf(pos * inv[i]);
                sv[i] = sinf(pos * inv[i]);
            }
            dg_norm_heads(q + (size_t)ti * qn, DG_HEADS, hd, qnrm);
            dg_norm_heads(k + (size_t)ti * kn, kvh, hd, knrm);
            dg_norm_heads(v + (size_t)ti * kn, kvh, hd, NULL);
            dg_rope(q + (size_t)ti * qn, DG_HEADS, hd, cv, sv);
            dg_rope(k + (size_t)ti * kn, kvh, hd, cv, sv);
        }
    }
    JB_TO(attention_prepare, prepare_start);
    JB_TICK(kv_start);
    int maxold = 0;
    if (cache)
        for (int s = 0; s < segments; s++)
            if (cache[s]->n > maxold)
                maxold = cache[s]->n;
    float *a = jb_arena_alloc(workspace, (size_t)n * qn, sizeof *a, 1),
          *score = jb_arena_alloc(workspace, (size_t)n * (maxold + seq), sizeof *score, 0);
    JB_TO(attention_kv, kv_start);
    JB_TICK(core_start);
    {
        int ti;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (ti = 0; ti < n; ti++)
            for (int h = 0; h < DG_HEADS; h++) {
                int s = ti / seq, t = ti % seq, old = cache ? cache[s]->n : 0,
                    kh = h / (DG_HEADS / kvh), end = mode == DG_CANVAS ? old + seq : old + t + 1,
                    start = 0;
                if (!full) {
                    if (mode == DG_CANVAS) {
                        int pos = old + t;
                        start = pos >= DG_LOCAL_WINDOW ? pos - DG_LOCAL_WINDOW + 1 : 0;
                        if (end > pos + DG_LOCAL_WINDOW)
                            end = pos + DG_LOCAL_WINDOW;
                    } else
                        start = end > DG_LOCAL_WINDOW ? end - DG_LOCAL_WINDOW : 0;
                }
                float *ts = score + (size_t)ti * (maxold + seq), mx = -FLT_MAX;
                const float *qq = q + (size_t)ti * qn + (size_t)h * hd;
                /* Cached keys first, then this call's own. */
                int split = end < old ? end : old, fresh = start > old ? start : old;
                if (start < split)
                    dg_dots(qq, cache[s]->k + (size_t)start * kn + (size_t)kh * hd, (size_t)kn,
                            split - start, hd, ts + start);
                if (fresh < end)
                    dg_dots(qq, k + ((size_t)s * seq + fresh - old) * kn + (size_t)kh * hd,
                            (size_t)kn, end - fresh, hd, ts + fresh);
                for (int j = start; j < end; j++)
                    if (ts[j] > mx)
                        mx = ts[j];
                /* Scores become their exponentials, computed once. */
                float den = 0;
                for (int j = start; j < end; j++) {
                    ts[j] = dg_expf(ts[j] - mx);
                    den += ts[j];
                }
                float *oo = a + (size_t)ti * qn + (size_t)h * hd;
                for (int j = start; j < end; j++) {
                    float p = ts[j] / den;
                    const float *vv = j < old
                                          ? cache[s]->v + (size_t)j * kn + (size_t)kh * hd
                                          : v + ((size_t)s * seq + j - old) * kn + (size_t)kh * hd;
                    for (int d = 0; d < hd; d++)
                        oo[d] += p * vv[d];
                }
            }
    }
    JB_TO(attention_core, core_start);
    JB_TICK(output_start);
    DGTensor *ow = dg_layer_tensor(m, l, "self_attn.o_proj.weight");
    dg_mm(ow, a, x, n, DG_H, qn);
    JB_TO(attention_output, output_start);
    if (out)
        dg_kv_output(out, cache, lens, segments, seq, kn, k, v);
    jb_arena_reset(workspace, mark);
}

/* Routes one token: the top DG_TOPK experts of its router logits rt, and
 * their weights, a softmax over every expert renormalized over the chosen
 * ones and scaled per expert. */
static void dg_route_token(const float *rt, const DGTensor *re, int *top, float *tw) {
    int ix[DG_TOPK];
    float ev[DG_TOPK];
    for (int k = 0; k < DG_TOPK; k++) {
        ix[k] = k;
        ev[k] = -FLT_MAX;
    }
    for (int e = 0; e < DG_EXPERTS; e++)
        for (int k = 0; k < DG_TOPK; k++)
            if (rt[e] > ev[k]) {
                for (int q = DG_TOPK - 1; q > k; q--) {
                    ev[q] = ev[q - 1];
                    ix[q] = ix[q - 1];
                }
                ev[k] = rt[e];
                ix[k] = e;
                break;
            }
    float mx = rt[0];
    for (int e = 1; e < DG_EXPERTS; e++)
        if (rt[e] > mx)
            mx = rt[e];
    double all = 0, sel = 0;
    for (int e = 0; e < DG_EXPERTS; e++)
        all += dg_exp((double)rt[e] - mx);
    for (int k = 0; k < DG_TOPK; k++) {
        ev[k] = (float)(dg_exp((double)ev[k] - mx) / all);
        sel += ev[k];
    }
    for (int k = 0; k < DG_TOPK; k++) {
        top[k] = ix[k];
        tw[k] = ev[k] / (float)sel * dg_at(re, ix[k]);
    }
}

static void dg_ff(DGModel *m, int l, float *x, int n, JBArena *workspace) {
    size_t mark = jb_arena_mark(workspace);
    JB_TICK(ff_start);
    DGTensor *pre = dg_layer_tensor(m, l, "pre_feedforward_layernorm.weight");
    DGTensor *gw = dg_layer_tensor(m, l, "mlp.gate_proj.weight"),
             *uw = dg_layer_tensor(m, l, "mlp.up_proj.weight"),
             *dw = dg_layer_tensor(m, l, "mlp.down_proj.weight");
    DGTensor *p1 = dg_layer_tensor(m, l, "post_feedforward_layernorm_1.weight");
    DGTensor *pre2 = dg_layer_tensor(m, l, "pre_feedforward_layernorm_2.weight");
    DGTensor *rw = dg_layer_tensor(m, l, "router.proj.weight"),
             *rs = dg_layer_tensor(m, l, "router.scale"),
             *re = dg_layer_tensor(m, l, "router.per_expert_scale");
    DGTensor *eg = m->nvfp4 ? NULL : dg_layer_tensor(m, l, "experts.gate_up_proj"),
             *ed = m->nvfp4 ? NULL : dg_layer_tensor(m, l, "experts.down_proj");
    DGTensor *p2 = dg_layer_tensor(m, l, "post_feedforward_layernorm_2.weight"),
             *post = dg_layer_tensor(m, l, "post_feedforward_layernorm.weight");
    float *z1 = jb_arena_alloc(workspace, (size_t)n * DG_H, sizeof *z1, 1),
          *g = jb_arena_alloc(workspace, (size_t)n * DG_DENSE, sizeof *g, 0),
          *u = jb_arena_alloc(workspace, (size_t)n * DG_DENSE, sizeof *u, 0),
          *d = jb_arena_alloc(workspace, (size_t)n * DG_H, sizeof *d, 0),
          *z2 = jb_arena_alloc(workspace, (size_t)n * DG_H, sizeof *z2, 0),
          *rin = jb_arena_alloc(workspace, (size_t)n * DG_H, sizeof *rin, 1),
          *route = jb_arena_alloc(workspace, (size_t)n * DG_EXPERTS, sizeof *route, 0);
    {
        int t;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (t = 0; t < n; t++)
            dg_rms(z1 + (size_t)t * DG_H, x + (size_t)t * DG_H, pre, DG_H);
    }
    dg_mm(gw, z1, g, n, DG_DENSE, DG_H);
    dg_mm(uw, z1, u, n, DG_DENSE, DG_H);
    {
        ptrdiff_t i;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(static)
#endif
        for (i = 0; i < (ptrdiff_t)n * DG_DENSE; i++)
            g[i] = dg_gelu(g[i]) * u[i];
    }
    dg_mm(dw, g, d, n, DG_H, DG_DENSE);
    JB_TO(dense, ff_start);
    JB_TICK(router_start);
    {
        int t;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (t = 0; t < n; t++) {
            float *r = x + (size_t)t * DG_H;
            dg_rms(d + (size_t)t * DG_H, d + (size_t)t * DG_H, p1, DG_H);
            dg_rms(z2 + (size_t)t * DG_H, r, pre2, DG_H);
            dg_rms(rin + (size_t)t * DG_H, r, NULL, DG_H);
            for (int i = 0; i < DG_H; i++)
                rin[(size_t)t * DG_H + i] *= dg_at(rs, i) / sqrtf(DG_H);
        }
    }
    dg_mm(rw, rin, route, n, DG_EXPERTS, DG_H);
    for (size_t i = 0; i < (size_t)n * DG_EXPERTS; i++)
        if (!jb_finitef(route[i]))
            die("non-finite DiffusionGemma router logits");
    int *top = jb_arena_alloc(workspace, (size_t)n * DG_TOPK, sizeof *top, 0);
    float *tw = jb_arena_alloc(workspace, (size_t)n * DG_TOPK, sizeof *tw, 0);
    {
        int t;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (t = 0; t < n; t++)
            dg_route_token(route + (size_t)t * DG_EXPERTS, re, top + (size_t)t * DG_TOPK,
                           tw + (size_t)t * DG_TOPK);
    }
    JB_TO(router, router_start);
    JB_TICK(expert_start);
    float *contrib = jb_arena_alloc(workspace, (size_t)n * DG_TOPK * DG_H, sizeof *contrib, 1),
          *gather = jb_arena_alloc(workspace, (size_t)n * DG_H, sizeof *gather, 0),
          *gu = jb_arena_alloc(workspace, (size_t)n * DG_EXPERT_PAIR, sizeof *gu, 0),
          *hid = jb_arena_alloc(workspace, (size_t)n * DG_MOE, sizeof *hid, 0),
          *eo = jb_arena_alloc(workspace, (size_t)n * DG_H, sizeof *eo, 0),
          *qz2 = m->nvfp4 ? jb_arena_alloc(workspace, (size_t)n * DG_H, sizeof *qz2, 0) : NULL,
          *qh = m->nvfp4 ? jb_arena_alloc(workspace, (size_t)n * DG_MOE, sizeof *qh, 0) : NULL;
    int *owner = jb_arena_alloc(workspace, (size_t)n * 2, sizeof *owner, 0);
    if (qz2) {
        JB_TICK(input_qdq_start);
        dg_nvfp4_qdq(qz2, z2, n, DG_H, m->nv_a13[l]);
        JB_TO(moe_input_qdq, input_qdq_start);
    }
    for (int e = 0; e < DG_EXPERTS; e++) {
        int ne = 0;
        for (int t = 0; t < n; t++)
            for (int k = 0; k < DG_TOPK; k++)
                if (top[(size_t)t * DG_TOPK + k] == e) {
                    owner[ne * 2] = t;
                    owner[ne * 2 + 1] = k;
                    memcpy(gather + (size_t)ne * DG_H, (qz2 ? qz2 : z2) + (size_t)t * DG_H,
                           DG_H * 4);
                    ne++;
                }
        if (!ne)
            continue;
        if (m->nvfp4) {
            DGNvExpert *v = &m->nvexpert[l * DG_EXPERTS + e];
            JB_TICK(gated_start);
            dg_nvfp4_gated(v, gather, gu, hid, ne, DG_MOE, DG_H);
            JB_TO(moe_gated, gated_start);
            JB_TICK(hidden_qdq_start);
            dg_nvfp4_qdq(qh, hid, ne, DG_MOE, m->nv_a2[l]);
            JB_TO(moe_hidden_qdq, hidden_qdq_start);
            JB_TICK(down_start);
            dg_nvfp4_mm(v->wd, v->sd, v->gd, qh, eo, ne, DG_H, DG_MOE);
            JB_TO(moe_down, down_start);
        } else {
            const uint8_t *gp = eg->data + (uint64_t)e * DG_EXPERT_PAIR * DG_H * 2,
                          *dp = ed->data + (uint64_t)e * DG_H * DG_MOE * 2;
            dg_mm_data(gp, gather, gu, ne, DG_EXPERT_PAIR, DG_H);
            for (int q = 0; q < ne; q++)
                for (int i = 0; i < DG_MOE; i++)
                    hid[(size_t)q * DG_MOE + i] = dg_gelu(gu[(size_t)q * DG_EXPERT_PAIR + i]) *
                                                  gu[(size_t)q * DG_EXPERT_PAIR + DG_MOE + i];
            dg_mm_data(dp, hid, eo, ne, DG_H, DG_MOE);
        }
        for (int q = 0; q < ne; q++) {
            int t = owner[q * 2], k = owner[q * 2 + 1];
            float wt = tw[(size_t)t * DG_TOPK + k],
                  *dst = contrib + ((size_t)t * DG_TOPK + k) * DG_H, *src = eo + (size_t)q * DG_H;
            for (int i = 0; i < DG_H; i++)
                dst[i] = wt * src[i];
        }
    }
    JB_TO(experts, expert_start);
    JB_TICK(ff_tail_start);
    float *mo = jb_arena_alloc(workspace, DG_H, sizeof *mo, 0),
          *sum = jb_arena_alloc(workspace, DG_H, sizeof *sum, 0);
    for (int t = 0; t < n; t++) {
        float *r = x + (size_t)t * DG_H;
        memset(mo, 0, DG_H * 4);
        for (int k = 0; k < DG_TOPK; k++) {
            float *src = contrib + ((size_t)t * DG_TOPK + k) * DG_H;
            for (int i = 0; i < DG_H; i++)
                mo[i] += src[i];
        }
        dg_rms(mo, mo, p2, DG_H);
        for (int i = 0; i < DG_H; i++)
            sum[i] = d[(size_t)t * DG_H + i] + mo[i];
        dg_rms(sum, sum, post, DG_H);
        for (int i = 0; i < DG_H; i++)
            r[i] += sum[i];
    }
    jb_arena_reset(workspace, mark);
    JB_TO(ff_other, ff_tail_start);
}

/* dg_ff on the accelerator, from d->x to d->x, including the layer scalar,
 * with routing on the device too. */
static void dg_ff_accel(DGModel *m, int l, int n, const DGDeviceLayout *d, JBArena *workspace) {
    const DGAccelOps *a = m->accel;
    size_t mark = jb_arena_mark(workspace), total = (size_t)n * DG_TOPK;
    JB_TICK(ff_start);
    DGTensor *pre = dg_layer_tensor(m, l, "pre_feedforward_layernorm.weight"),
             *gw = dg_layer_tensor(m, l, "mlp.gate_proj.weight"),
             *uw = dg_layer_tensor(m, l, "mlp.up_proj.weight"),
             *dw = dg_layer_tensor(m, l, "mlp.down_proj.weight"),
             *p1 = dg_layer_tensor(m, l, "post_feedforward_layernorm_1.weight"),
             *pre2 = dg_layer_tensor(m, l, "pre_feedforward_layernorm_2.weight"),
             *rw = dg_layer_tensor(m, l, "router.proj.weight"),
             *rs = dg_layer_tensor(m, l, "router.scale"),
             *re = dg_layer_tensor(m, l, "router.per_expert_scale"),
             *p2 = dg_layer_tensor(m, l, "post_feedforward_layernorm_2.weight"),
             *post = dg_layer_tensor(m, l, "post_feedforward_layernorm.weight");
    a->rms(d->z, d->x, pre->device, 0, 0, 0, n, DG_H);
    dg_mm_accel(m, gw, d->z, d->g, n, DG_DENSE, DG_H);
    dg_mm_accel(m, uw, d->z, d->u, n, DG_DENSE, DG_H);
    a->gelu_mul(d->g, d->g, d->u, (size_t)n * DG_DENSE);
    dg_mm_accel(m, dw, d->g, d->d, n, DG_H, DG_DENSE);
    a->rms(d->d, d->d, p1->device, 0, 0, 0, n, DG_H);
    a->rms(d->z2, d->x, pre2->device, 0, 0, 0, n, DG_H);
    a->rms(d->rin, d->x, 0, rs->device, sqrtf(DG_H), 0, n, DG_H);
    dg_mm_accel(m, rw, d->rin, d->route, n, DG_EXPERTS, DG_H);
    JB_TO(dense, ff_start);
    JB_TICK(expert_start);
    /* Routing, grouping and the experts stay on the device: nothing here
     * waits for the host. */
    int routed = (int)total;
    a->route(d->route, re->device, n, d->top, d->weight, d->flag);
    dg_nvfp4_qdq_accel(m, d->qz2, d->z2, n, DG_H, m->nv_a13[l], d->flag);
    a->group(d->top, n, d->index, d->slot, d->tiles, d->ntiles);
    a->gather(d->gather, d->qz2, d->index, routed, (int)a->nvfp4_row(DG_H));
    a->nvfp4_grouped(m->expert_table[l][0], d->tiles, d->ntiles, routed, d->gather, d->eg, DG_MOE,
                     DG_H);
    a->nvfp4_grouped(m->expert_table[l][1], d->tiles, d->ntiles, routed, d->gather, d->eh, DG_MOE,
                     DG_H);
    a->gelu_mul(d->eh, d->eg, d->eh, total * DG_MOE);
    dg_nvfp4_qdq_accel(m, d->eqh, d->eh, routed, DG_MOE, m->nv_a2[l], d->flag);
    a->nvfp4_grouped(m->expert_table[l][2], d->tiles, d->ntiles, routed, d->eqh, d->eo, DG_H,
                     DG_MOE);
    a->moe_tail(d->x, d->eo, d->slot, d->weight, d->d, p2->device, post->device,
                dg_at(dg_layer_tensor(m, l, "layer_scalar"), 0), n);
    JB_TO(experts, expert_start);
    jb_arena_reset(workspace, mark);
}

/* How a layer calls attention: dg_attention's batch form when batch > 0,
 * dg_attention_segments otherwise. */
typedef struct {
    int batch, segments, seq, pos0, mode;
    const DGKV *cache;
    const DGKV *const *caches;
    const int *lens;
    DGKV *out;
} DGAttentionCall;

/* The accelerated layers keep the activations on the device between
 * layers: they go up before the first and come back after the last, when
 * the quantizers' non-finite flag is checked too. */
static void dg_accel_enter(const DGModel *m, const DGDeviceWork *work, const float *x, int n) {
    DGDeviceLayout d;
    int bad = 0;
    dg_device_layout(&d, work->base, n, work->kv_rows, m->accel);
    m->accel->to_device(d.flag, &bad, sizeof bad);
    m->accel->to_device(d.x, x, (size_t)n * DG_H * 4);
}

static void dg_accel_leave(const DGModel *m, const DGDeviceWork *work, float *x, int n) {
    DGDeviceLayout d;
    int bad = 0;
    dg_device_layout(&d, work->base, n, work->kv_rows, m->accel);
    m->accel->to_host(x, d.x, (size_t)n * DG_H * 4);
    m->accel->to_host(&bad, d.flag, sizeof bad);
    if (bad & 2)
        die("non-finite DiffusionGemma router logits");
    if (bad)
        die("non-finite NVFP4 activation");
}

/* The staged K/V output, in one copy to page-locked memory after the last
 * layer; dg_kv_fetch then copies each layer's rows out of it. */
static void dg_kv_download(const DGModel *m, const DGDeviceWork *work, int n) {
    DGDeviceLayout d;
    dg_device_layout(&d, work->base, n, work->kv_rows, m->accel);
    m->accel->to_host(work->pinned, d.kv_out, dg_kv_stage_offset(n, DG_L) * 4);
}

/* Layer l's downloaded K/V rows into its output, for segments first onward:
 * segment s's own rows follow its cached ones there. */
static void dg_kv_fetch(const DGDeviceWork *work, int n, int l, DGKV *out, int first, int segments,
                        int seq, const int *old, const int *lens) {
    int kn = dg_kv_width(l);
    const float *k = work->pinned + dg_kv_stage_offset(n, l), *v = k + (size_t)n * kn;
    for (int s = 0; s < segments; s++) {
        size_t own = (size_t)(first + s) * seq * kn, keep = (size_t)lens[s] * kn * 4;
        memcpy(out[s].k + (size_t)old[s] * kn, k + own, keep);
        memcpy(out[s].v + (size_t)old[s] * kn, v + own, keep);
    }
}

/* dg_layer on the accelerator, from and to the device's activations; device
 * time is attributed to the next step that waits for it. */
static void dg_layer_accel(DGModel *m, int l, int n, const DGAttentionCall *c, JBArena *workspace,
                           const DGDeviceWork *work) {
    const DGAccelOps *a = m->accel;
    DGDeviceLayout d;
    dg_device_layout(&d, work->base, n, work->kv_rows, a);
    JB_TICK(layer_start);
    DGTensor *in = dg_layer_tensor(m, l, "input_layernorm.weight"),
             *pa = dg_layer_tensor(m, l, "post_attention_layernorm.weight");
    a->copy(d.res, d.x, (size_t)n * DG_H * 4);
    a->rms(d.z, d.x, in->device, 0, 0, 0, n, DG_H);
    JB_TICK(attention_start);
    if (c->batch)
        dg_attention(m, l, NULL, n, c->pos0, c->cache, c->out, c->mode, c->batch, workspace, &d);
    else
        dg_attention_segments(m, l, NULL, c->segments, c->seq, 0, c->caches, c->lens, c->out,
                              c->mode, workspace, &d);
    JB_TO(attention, attention_start);
    a->rms(d.x, d.z, pa->device, 0, 0, d.res, n, DG_H);
    dg_ff_accel(m, l, n, &d, workspace);
    JB_TO(layers, layer_start);
}

static void dg_layer(DGModel *m, int l, float *x, int n, int pos0, const DGKV *cache, DGKV *out,
                     int mode, int batch, JBArena *workspace, const DGDeviceWork *device) {
    if (device) {
        DGAttentionCall c = {batch, 0, 0, pos0, mode, cache, NULL, NULL, out};
        dg_layer_accel(m, l, n, &c, workspace, device);
        return;
    }
    size_t mark = jb_arena_mark(workspace);
    JB_TICK(layer_start);
    DGTensor *in = dg_layer_tensor(m, l, "input_layernorm.weight"),
             *pa = dg_layer_tensor(m, l, "post_attention_layernorm.weight");
    float *res = jb_arena_alloc(workspace, (size_t)n * DG_H, sizeof *res, 0),
          *z = jb_arena_alloc(workspace, (size_t)n * DG_H, sizeof *z, 0);
    memcpy(res, x, (size_t)n * DG_H * 4);
    {
        int t;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (t = 0; t < n; t++)
            dg_rms(z + (size_t)t * DG_H, x + (size_t)t * DG_H, in, DG_H);
    }
    JB_TICK(attention_start);
    dg_attention(m, l, z, n, pos0, cache, out, mode, batch, workspace, NULL);
    JB_TO(attention, attention_start);
    {
        int t;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (t = 0; t < n; t++) {
            dg_rms(x + (size_t)t * DG_H, z + (size_t)t * DG_H, pa, DG_H);
            for (int i = 0; i < DG_H; i++)
                x[(size_t)t * DG_H + i] += res[(size_t)t * DG_H + i];
        }
    }
    dg_ff(m, l, x, n, workspace);
    float sc = dg_at(dg_layer_tensor(m, l, "layer_scalar"), 0);
    for (size_t i = 0; i < (size_t)n * DG_H; i++)
        x[i] *= sc;
    jb_arena_reset(workspace, mark);
    JB_TO(layers, layer_start);
}

static void dg_layer_multi(DGModel *m, int l, float *x, int segments, int seq,
                           const DGKV *const *cache, const int *lens, DGKV *out, int mode,
                           JBArena *workspace, const DGDeviceWork *device) {
    int n = segments * seq;
    if (device) {
        DGAttentionCall c = {0, segments, seq, 0, mode, NULL, cache, lens, out};
        dg_layer_accel(m, l, n, &c, workspace, device);
        return;
    }
    size_t mark = jb_arena_mark(workspace);
    JB_TICK(layer_start);
    DGTensor *in = dg_layer_tensor(m, l, "input_layernorm.weight"),
             *pa = dg_layer_tensor(m, l, "post_attention_layernorm.weight");
    float *res = jb_arena_alloc(workspace, (size_t)n * DG_H, sizeof *res, 0),
          *z = jb_arena_alloc(workspace, (size_t)n * DG_H, sizeof *z, 0);
    memcpy(res, x, (size_t)n * DG_H * 4);
    {
        int t;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (t = 0; t < n; t++)
            dg_rms(z + (size_t)t * DG_H, x + (size_t)t * DG_H, in, DG_H);
    }
    JB_TICK(attention_start);
    dg_attention_segments(m, l, z, segments, seq, 0, cache, lens, out, mode, workspace, NULL);
    JB_TO(attention, attention_start);
    {
        int t;
#ifdef _OPENMP
        JB_OMP();
#pragma omp parallel for schedule(dynamic)
#endif
        for (t = 0; t < n; t++) {
            dg_rms(x + (size_t)t * DG_H, z + (size_t)t * DG_H, pa, DG_H);
            for (int i = 0; i < DG_H; i++)
                x[(size_t)t * DG_H + i] += res[(size_t)t * DG_H + i];
        }
    }
    dg_ff(m, l, x, n, workspace);
    float sc = dg_at(dg_layer_tensor(m, l, "layer_scalar"), 0);
    for (size_t i = 0; i < (size_t)n * DG_H; i++)
        x[i] *= sc;
    jb_arena_reset(workspace, mark);
    JB_TO(layers, layer_start);
}

static void dg_embed(DGModel *m, const int *ids, int n, int decoder, float *x) {
    DGTensor *w = dg_tensor(m, "model.decoder.embed_tokens.weight");
    float scale = sqrtf(DG_H);
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < DG_H; i++)
            x[(size_t)t * DG_H + i] = dg_at(w, (uint64_t)ids[t] * DG_H + i) * scale;
        if (decoder)
            dg_rms(x + (size_t)t * DG_H, x + (size_t)t * DG_H, NULL, DG_H);
    }
}

static void dg_final_norm(DGModel *m, float *x, int n) {
    DGTensor *w = dg_tensor(m, "model.decoder.norm.weight");
    for (int t = 0; t < n; t++)
        dg_rms(x + (size_t)t * DG_H, x + (size_t)t * DG_H, w, DG_H);
}

static void dg_prefill(DGModel *m, const int *prompt, int np, DGKV kv[DG_L], DGWorkspace *owner) {
    JBArena *workspace =
        dg_workspace_begin(owner, dg_forward_workspace_bytes(m, np, 0, np, 1, 0, 0));
    DGDeviceWork work = dg_device_begin(owner, m, np, 0);
    const DGDeviceWork *device = work.base ? &work : NULL;
    float *enc = jb_arena_alloc(workspace, (size_t)np * DG_H, sizeof *enc, 0);
    JB_TICK(embed_start);
    dg_embed(m, prompt, np, 0, enc);
    JB_TO(embedding, embed_start);
    if (device)
        dg_accel_enter(m, device, enc, np);
    for (int l = 0; l < DG_L; l++)
        dg_layer(m, l, enc, np, 0, NULL, &kv[l], DG_CAUSAL, 1, workspace, device);
    if (device) {
        int zero = 0;
        dg_accel_leave(m, device, enc, np);
        dg_kv_download(m, device, np);
        for (int l = 0; l < DG_L; l++)
            dg_kv_fetch(device, np, l, &kv[l], 0, 1, np, &zero, &np);
    }
    jb_arena_reset(workspace, 0);
    dg_workspace_end(owner);
}

static void dg_prefill_suffix(DGModel *m, const int *token, int n, const DGKV prefix[DG_L],
                              DGKV kv[DG_L], DGWorkspace *owner) {
    JBArena *workspace =
        dg_workspace_begin(owner, dg_forward_workspace_bytes(m, n, prefix[0].n, n, 1, 0, 0));
    DGDeviceWork work = dg_device_begin(owner, m, n, prefix[0].n);
    const DGDeviceWork *device = work.base ? &work : NULL;
    float *x = jb_arena_alloc(workspace, (size_t)n * DG_H, sizeof *x, 0);
    JB_TICK(embed_start);
    dg_embed(m, token, n, 0, x);
    JB_TO(embedding, embed_start);
    if (device)
        dg_accel_enter(m, device, x, n);
    for (int l = 0; l < DG_L; l++)
        dg_layer(m, l, x, n, prefix[l].n, &prefix[l], &kv[l], DG_SUFFIX, 1, workspace, device);
    if (device) {
        dg_accel_leave(m, device, x, n);
        dg_kv_download(m, device, n);
        for (int l = 0; l < DG_L; l++)
            dg_kv_fetch(device, n, l, &kv[l], 0, 1, n, &prefix[l].n, &n);
    }
    jb_arena_reset(workspace, 0);
    dg_workspace_end(owner);
}

static float *dg_decode(DGModel *m, DGKV kv[DG_L], int np, const int *canvas, int nc, int batch,
                        size_t tail, DGWorkspace *owner) {
    JBArena *workspace = dg_workspace_begin(
        owner, dg_forward_workspace_bytes(m, nc * batch, kv[0].n, nc, batch, batch > 1, tail));
    DGDeviceWork work = dg_device_begin(owner, m, nc * batch, kv[0].n);
    const DGDeviceWork *device = work.base ? &work : NULL;
    float *dec = jb_arena_alloc(workspace, (size_t)nc * batch * DG_H, sizeof *dec, 0);
    JB_TICK(embed_start);
    dg_embed(m, canvas, nc * batch, 1, dec);
    JB_TO(embedding, embed_start);
    if (device)
        dg_accel_enter(m, device, dec, nc * batch);
    for (int l = 0; l < DG_L; l++)
        dg_layer(m, l, dec, nc * batch, np, &kv[l], NULL, DG_CANVAS, batch, workspace, device);
    if (device)
        dg_accel_leave(m, device, dec, nc * batch);
    JB_TICK(norm_start);
    dg_final_norm(m, dec, nc * batch);
    JB_TO(final_norm, norm_start);
    return dec;
}

static void dg_prefill_suffix_multi(DGModel *m, const int *token, const int *lens, int batch,
                                    int seq, const DGKV prefix[DG_L], DGKV *kv, const DGKV **cache,
                                    DGKV *out, DGWorkspace *owner) {
    JBArena *workspace = dg_workspace_begin(
        owner, dg_forward_workspace_bytes(m, batch * seq, prefix[0].n, seq, batch, 0, 0));
    DGDeviceWork work = dg_device_begin(owner, m, batch * seq, prefix[0].n);
    const DGDeviceWork *device = work.base ? &work : NULL;
    float *x = jb_arena_alloc(workspace, (size_t)batch * seq * DG_H, sizeof *x, 0);
    JB_TICK(embed_start);
    dg_embed(m, token, batch * seq, 0, x);
    JB_TO(embedding, embed_start);
    if (device)
        dg_accel_enter(m, device, x, batch * seq);
    for (int l = 0; l < DG_L; l++) {
        for (int b = 0; b < batch; b++) {
            cache[b] = &prefix[l];
            out[b] = kv[(size_t)b * DG_L + l];
        }
        dg_layer_multi(m, l, x, batch, seq, cache, lens, out, DG_SUFFIX, workspace, device);
    }
    if (device) {
        dg_accel_leave(m, device, x, batch * seq);
        dg_kv_download(m, device, batch * seq);
        for (int l = 0; l < DG_L; l++)
            for (int b = 0; b < batch; b++)
                dg_kv_fetch(device, batch * seq, l, &kv[(size_t)b * DG_L + l], b, 1, seq,
                            &prefix[l].n, &lens[b]);
    }
    jb_arena_reset(workspace, 0);
    dg_workspace_end(owner);
}

static float *dg_decode_multi(DGModel *m, DGKV *kv, const int *doc, int segments, int seq,
                              const int *canvas, const DGKV **cache, int *lens, size_t tail,
                              DGWorkspace *owner) {
    int n = segments * seq;
    for (int s = 0; s < segments; s++)
        lens[s] = seq;
    int maxold = 0;
    for (int s = 0; s < segments; s++)
        for (int l = 0; l < DG_L; l++)
            if (kv[(size_t)doc[s] * DG_L + l].n > maxold)
                maxold = kv[(size_t)doc[s] * DG_L + l].n;
    JBArena *workspace =
        dg_workspace_begin(owner, dg_forward_workspace_bytes(m, n, maxold, seq, segments, 0, tail));
    DGDeviceWork work = dg_device_begin(owner, m, n, segments * maxold);
    const DGDeviceWork *device = work.base ? &work : NULL;
    float *x = jb_arena_alloc(workspace, (size_t)n * DG_H, sizeof *x, 0);
    JB_TICK(embed_start);
    dg_embed(m, canvas, n, 1, x);
    JB_TO(embedding, embed_start);
    if (device)
        dg_accel_enter(m, device, x, n);
    for (int l = 0; l < DG_L; l++) {
        for (int s = 0; s < segments; s++)
            cache[s] = &kv[(size_t)doc[s] * DG_L + l];
        dg_layer_multi(m, l, x, segments, seq, cache, lens, NULL, DG_CANVAS, workspace, device);
    }
    if (device)
        dg_accel_leave(m, device, x, n);
    JB_TICK(norm_start);
    dg_final_norm(m, x, n);
    JB_TO(final_norm, norm_start);
    return x;
}

static void dg_decode_end(DGWorkspace *workspace) {
    jb_arena_reset(&workspace->arena, 0);
    dg_workspace_end(workspace);
}

static void dg_prefix_free(DGPrefixCache *c) {
    if (!c)
        return;
    dg_kv_free(&c->kv_storage);
    if (c->metadata.allocation)
        jb_arena_free(&c->metadata);
    memset(c, 0, sizeof *c);
}

static char *jt_raw(const char *j, const JTok *t) {
    size_t n = (size_t)(t->end - t->start);
    char *z = xmalloc(n + 1);
    memcpy(z, j + t->start, n);
    z[n] = 0;
    return z;
}

static unsigned hex4(const char *s) {
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        v <<= 4;
        v += (unsigned)(s[i] <= '9' ? s[i] - '0' : (tolower((unsigned char)s[i]) - 'a' + 10));
    }
    return v;
}

static char *jt_string(const char *j, const JTok *t) {
    if (t->type != JT_STRING)
        return jt_raw(j, t);
    char *z = xmalloc((size_t)(t->end - t->start) * 3 + 1), *o = z;
    for (int i = t->start; i < t->end; i++) {
        unsigned c = (unsigned char)j[i];
        if (c != '\\') {
            *o++ = (char)c;
            continue;
        }
        c = (unsigned char)j[++i];
        if (c == 'u') {
            uint32_t u = hex4(j + i + 1);
            i += 4;
            if (u >= 0xd800 && u <= 0xdbff) {
                if (i + 6 >= t->end || j[i + 1] != '\\' || j[i + 2] != 'u')
                    die("unpaired JSON surrogate");
                uint32_t v = hex4(j + i + 3);
                if (v < 0xdc00 || v > 0xdfff)
                    die("unpaired JSON surrogate");
                u = 0x10000 + ((u - 0xd800) << 10) + (v - 0xdc00);
                i += 6;
            } else if (u >= 0xdc00 && u <= 0xdfff)
                die("unpaired JSON surrogate");
            else if (!u)
                die("NUL in JSON string is unsupported");
            o += put_utf8(u, o);
        } else {
            switch (c) {
            case 'b':
                *o++ = '\b';
                break;
            case 'f':
                *o++ = '\f';
                break;
            case 'n':
                *o++ = '\n';
                break;
            case 'r':
                *o++ = '\r';
                break;
            case 't':
                *o++ = '\t';
                break;
            default:
                *o++ = (char)c;
            }
        }
    }
    *o = 0;
    size_t at = 0, zn = (size_t)(o - z);
    while (at < zn)
        (void)next_cp((const unsigned char *)z, zn, &at);
    return z;
}

typedef struct {
    char *p;
    size_t n, cap;
} DGBuf;

static void db_need(DGBuf *b, size_t add) {
    if (add > SIZE_MAX - b->n - 1)
        die("string size overflow");
    size_t need = b->n + add + 1;
    if (need > b->cap) {
        size_t c = b->cap ? b->cap : 256;
        while (c < need) {
            if (c > SIZE_MAX / 2)
                die("string size overflow");
            c *= 2;
        }
        b->p = xrealloc(b->p, c);
        b->cap = c;
    }
}

static void db_mem(DGBuf *b, const char *p, size_t n) {
    db_need(b, n);
    memcpy(b->p + b->n, p, n);
    b->n += n;
    b->p[b->n] = 0;
}

static void db_ch(DGBuf *b, char c) {
    db_need(b, 1);
    b->p[b->n++] = c;
    b->p[b->n] = 0;
}

JB_PRINTF(2, 3) static void db_fmt(DGBuf *b, const char *fmt, ...) {
    va_list a, z;
    va_start(a, fmt);
    va_copy(z, a);
    int n = vsnprintf(NULL, 0, fmt, z);
    va_end(z);
    if (n < 0)
        die("formatting failed");
    db_need(b, (size_t)n);
    vsnprintf(b->p + b->n, (size_t)n + 1, fmt, a);
    va_end(a);
    b->n += (size_t)n;
}

/* The printf family writes the process locale's decimal separator. JSON does
 * not: remove the one punctuation sequence between the mantissa's digits and
 * replace it with '.'. No locale lookup is involved, so another thread may
 * call setlocale without racing this conversion. */
static void db_decimal(DGBuf *b, const char *z, int n) {
    if (n < 0 || n >= 64)
        die("number formatting failed");
    size_t o = 0;
    int exponent = 0, point = 0;
    char out[64];
    for (const unsigned char *c = (const unsigned char *)z; *c; c++) {
        if ((*c >= '0' && *c <= '9') || *c == '+' || *c == '-')
            out[o++] = (char)*c;
        else if (*c == 'e' || *c == 'E') {
            exponent = 1;
            out[o++] = (char)*c;
        } else if (!exponent && !point) {
            point = 1;
            out[o++] = '.';
        }
    }
    db_mem(b, out, o);
}

#if defined(_WIN32)
typedef _locale_t JBLocale;
#else
typedef locale_t JBLocale;
#endif

static JBLocale jb_numeric_locale(void) {
    static JBLocale locale;
    static atomic_int state;
    int expected = 0;
    if (atomic_compare_exchange_strong_explicit(&state, &expected, 1, memory_order_acq_rel,
                                                memory_order_acquire)) {
#if defined(_WIN32)
        locale = _create_locale(LC_NUMERIC, "C");
#else
        locale = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
#endif
        atomic_store_explicit(&state, locale ? 2 : -1, memory_order_release);
    } else
        while (atomic_load_explicit(&state, memory_order_acquire) == 1) {
        }
    if (atomic_load_explicit(&state, memory_order_acquire) != 2)
        die_status(JB_ERROR_INTERNAL, "cannot create numeric locale");
    return locale;
}

static int jb_numeric_format(char *out, size_t size, double value, int milliseconds) {
#if defined(_WIN32)
    _locale_t locale = jb_numeric_locale();
    int result = milliseconds ? _snprintf_l(out, size, "%.3f", locale, value)
                              : _snprintf_l(out, size, "%.17g", locale, value);
    return result;
#else
    locale_t locale = jb_numeric_locale();
    locale_t previous = uselocale(locale);
    int result =
        milliseconds ? snprintf(out, size, "%.3f", value) : snprintf(out, size, "%.17g", value);
    uselocale(previous);
    return result;
#endif
}

static void db_number(DGBuf *b, double value) {
    char z[64];
    db_decimal(b, z, jb_numeric_format(z, sizeof z, value, 0));
}

static void db_millis(DGBuf *b, double ms) {
    char z[64];
    db_decimal(b, z, jb_numeric_format(z, sizeof z, ms, 1));
}

static void db_json_string(DGBuf *b, const char *s, int ascii) {
    db_ch(b, '"');
    size_t n = strlen(s), i = 0;
    while (i < n) {
        size_t at = i;
        uint32_t cp = next_cp((const unsigned char *)s, n, &i);
        if (cp == '"' || cp == '\\') {
            db_ch(b, '\\');
            db_ch(b, (char)cp);
        } else if (cp == '\b')
            db_mem(b, "\\b", 2);
        else if (cp == '\f')
            db_mem(b, "\\f", 2);
        else if (cp == '\n')
            db_mem(b, "\\n", 2);
        else if (cp == '\r')
            db_mem(b, "\\r", 2);
        else if (cp == '\t')
            db_mem(b, "\\t", 2);
        else if (cp < 32 || (ascii && cp > 127)) {
            char z[16];
            if (cp <= 0xffff)
                snprintf(z, sizeof z, "\\u%04x", cp);
            else {
                cp -= 0x10000;
                snprintf(z, sizeof z, "\\u%04x\\u%04x", 0xd800 + (cp >> 10), 0xdc00 + (cp & 1023));
            }
            db_mem(b, z, strlen(z));
        } else
            db_mem(b, s + at, i - at);
    }
    db_ch(b, '"');
}

typedef struct {
    int tok;
    char *key;
} DGKey;

static int dg_key_cmp(const void *a, const void *b) {
    return strcmp(((const DGKey *)a)->key, ((const DGKey *)b)->key);
}

static void dg_json_value(DGBuf *b, const char *j, JTok *t, int nt, int at, int sort_keys,
                          int ascii) {
    if (t[at].type == JT_STRING) {
        char *s = jt_string(j, &t[at]);
        db_json_string(b, s, ascii);
        jb_release(s);
        return;
    }
    if (t[at].type == JT_PRIMITIVE) {
        db_mem(b, j + t[at].start, (size_t)(t[at].end - t[at].start));
        return;
    }
    if (t[at].type == JT_ARRAY) {
        db_ch(b, '[');
        int first = 1;
        for (int i = at + 1; i < nt; i++) {
            if (t[i].parent < at)
                break;
            if (t[i].parent == at) {
                if (!first)
                    db_mem(b, ", ", 2);
                dg_json_value(b, j, t, nt, i, sort_keys, ascii);
                first = 0;
            }
        }
        db_ch(b, ']');
        return;
    }
    int nk = t[at].size / 2;
    DGKey *k = xcalloc((size_t)nk, sizeof *k);
    int z = 0;
    for (int i = at + 1; i + 1 < nt; i++) {
        if (t[i].parent < at)
            break;
        if (t[i].parent == at) {
            k[z].tok = i;
            k[z].key = jt_string(j, &t[i]);
            z++;
            i++;
        }
    }
    if (z != nk)
        die("bad JSON object");
    if (sort_keys)
        qsort(k, (size_t)nk, sizeof *k, dg_key_cmp);
    db_ch(b, '{');
    for (int i = 0; i < nk; i++) {
        if (i)
            db_mem(b, ", ", 2);
        db_json_string(b, k[i].key, ascii);
        db_mem(b, ": ", 2);
        dg_json_value(b, j, t, nt, k[i].tok + 1, sort_keys, ascii);
        jb_release(k[i].key);
    }
    db_ch(b, '}');
    jb_release(k);
}

static char *dg_json_canonical(const char *j, JTok *t, int nt, int at, int sort_keys, int ascii) {
    DGBuf b = {0};
    dg_json_value(&b, j, t, nt, at, sort_keys, ascii);
    if (!b.p)
        return xstrdup("");
    return b.p;
}

static int dg_py_space(uint32_t c) {
    return (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 ||
           c == 0x202f || c == 0x205f || c == 0x3000;
}

static char *dg_text_of(const char *j, JTok *t, int nt, int at) {
    if (t[at].type != JT_STRING) {
        if (jt_literal(j, &t[at], "null"))
            return xstrdup("");
        return dg_json_canonical(j, t, nt, at, 0, 0);
    }
    char *s = jt_string(j, &t[at]);
    size_t n = strlen(s), p = 0, first = n, last = 0;
    while (p < n) {
        size_t start = p;
        uint32_t c = next_cp((const unsigned char *)s, n, &p);
        if (!dg_py_space(c)) {
            if (first == n)
                first = start;
            last = p;
        }
    }
    if (first == n) {
        s[0] = 0;
        return s;
    }
    memmove(s, s + first, last - first);
    s[last - first] = 0;
    return s;
}

static uint32_t dg_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void dg_sha256(const uint8_t *p, size_t n, uint8_t out[32]) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint64_t bits = (uint64_t)n * 8;
    size_t total = (n + 9 + 63) & ~(size_t)63;
    uint8_t *msg = xcalloc(total, 1);
    memcpy(msg, p, n);
    msg[n] = 0x80;
    for (int i = 0; i < 8; i++)
        msg[total - 1 - i] = (uint8_t)(bits >> (8 * i));
    for (size_t off = 0; off < total; off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)msg[off + 4 * i] << 24 | (uint32_t)msg[off + 4 * i + 1] << 16 |
                   (uint32_t)msg[off + 4 * i + 2] << 8 | msg[off + 4 * i + 3];
        for (int i = 16; i < 64; i++) {
            uint32_t a = w[i - 15], c = w[i - 2];
            w[i] = (dg_rotr(c, 17) ^ dg_rotr(c, 19) ^ (c >> 10)) + w[i - 7] +
                   (dg_rotr(a, 7) ^ dg_rotr(a, 18) ^ (a >> 3)) + w[i - 16];
        }
        uint32_t a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + (dg_rotr(e, 6) ^ dg_rotr(e, 11) ^ dg_rotr(e, 25)) +
                          ((e & f) ^ ((~e) & g)) + k[i] + w[i],
                     t2 = (dg_rotr(a, 2) ^ dg_rotr(a, 13) ^ dg_rotr(a, 22)) +
                          ((a & bb) ^ (a & c) ^ (bb & c));
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = bb;
            bb = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += bb;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }
    jb_release(msg);
    for (int i = 0; i < 8; i++)
        for (int q = 0; q < 4; q++)
            out[i * 4 + q] = (uint8_t)(h[i] >> (24 - 8 * q));
}

#ifndef JB_NO_MAIN
static char *read_all(const char *path, size_t *nn) {
    FILE *f = fopen(path, "rb");
    if (!f)
        die2("open", path);
    if (fseek(f, 0, SEEK_END))
        die("request seek failed");
    long z = ftell(f);
    if (z < 0 || (unsigned long)z > JB_MAX_JSON)
        die("request too large");
    if (fseek(f, 0, SEEK_SET))
        die("request seek failed");
    char *s = xmalloc((size_t)z + 1);
    if (fread(s, 1, (size_t)z, f) != (size_t)z)
        die("short request read");
    fclose(f);
    s[z] = 0;
    *nn = (size_t)z;
    return s;
}
#endif

static void normalize_scores(const double *score, int n, double *prob) {
    if (n < 2)
        die("at least two candidate scores required");
    double mx = score[0];
    if (!jb_finite(mx))
        die("non-finite candidate score");
    for (int i = 1; i < n; i++) {
        if (!jb_finite(score[i]))
            die("non-finite candidate score");
        if (score[i] > mx)
            mx = score[i];
    }
    double den = 0;
    for (int i = 0; i < n; i++)
        den += exp(score[i] - mx);
    if (!jb_finite(den) || den <= 0)
        die("invalid candidate normalization");
    for (int i = 0; i < n; i++)
        prob[i] = exp(score[i] - mx) / den;
}

static double confidence(const double *p, int n) {
    double h = 0;
    for (int i = 0; i < n; i++)
        if (p[i] > 0)
            h -= p[i] * log(p[i]);
    double c = 1.0 - h / log((double)n);
    if (fabs(c) < 1e-15)
        c = 0;
    return c < 0 ? 0 : c > 1 ? 1 : c;
}

static int direct_keys(const JTok *t, int nt, int obj, int *keys, int max) {
    int n = 0;
    for (int i = obj + 1; i < nt; i++) {
        if (t[i].parent < obj)
            break;
        if (t[i].parent == obj && t[i].type == JT_STRING) {
            if (n == max)
                die("too many JSON object entries");
            keys[n++] = i;
            i++;
        }
    }
    return n;
}

typedef struct {
    int key, criteria, nc;
    char *kind;
    char **cand, **label;
    double *prob;
} DecisionWork;

typedef struct {
    Tokens base;
    int start, count;
} DGCanvas;

/* OpenJev's candidate label order: A-Z, a-z, then AA, AB, ... */
static void dg_choice_candidate(int ci, char label[3]) {
    label[0] = label[1] = label[2] = 0;
    if (ci < 26)
        label[0] = (char)('A' + ci);
    else if (ci < 52)
        label[0] = (char)('a' + ci - 26);
    else {
        int j = ci - 52;
        label[0] = (char)('A' + j / 26);
        label[1] = (char)('A' + j % 26);
    }
}

static int dg_choice_inventory(DGTokenizer *tok, char *out[DG_CHOICE_LABELS]) {
    Tokens base = dgt_tokenize(tok, "q1: A");
    int seen[DG_CHOICE_LABELS], n = 0;
    for (int ci = 0; ci < 728 && n < DG_CHOICE_LABELS; ci++) {
        char label[3], text[16];
        dg_choice_candidate(ci, label);
        snprintf(text, sizeof text, "q1: %s", label);
        Tokens e = dgt_tokenize(tok, text);
        int ok = e.n == base.n && e.n > 0;
        for (uint32_t i = 0; ok && i + 1 < e.n; i++)
            if (e.v[i] != base.v[i])
                ok = 0;
        for (int i = 0; ok && i < n; i++)
            if (seen[i] == e.v[e.n - 1])
                ok = 0;
        if (ok) {
            seen[n] = e.v[e.n - 1];
            out[n++] = xstrdup(label);
        }
        jb_release(e.v);
    }
    jb_release(base.v);
    return n;
}

static char *dg_decision_label(const char *kind, int i, char **choice) {
    char z[16];
    if (!strcmp(kind, "noul"))
        return xstrdup(i ? "no" : "yes");
    if (!strcmp(kind, "score")) {
        snprintf(z, sizeof z, "%d", i);
        return xstrdup(z);
    }
    return xstrdup(choice[i]);
}

static char *dg_instruction(const char *j, JTok *t, int nt, int qo) {
    int i = jt_obj_get(j, t, nt, qo, "instructions");
    return i < 0 ? xstrdup("Answer about the state.") : dg_text_of(j, t, nt, i);
}

static char *dg_system_prompt(const char *qj, JTok *qt, int qnt, DecisionWork *w, int nq) {
    static const char intro[] =
        "Answer a fixed set of questions about the state the user provides. Each question lists "
        "its allowed answers; reply with exactly one label per question.\n";
    DGBuf b = {0};
    db_mem(&b, intro, sizeof intro - 1);
    for (int x = 0; x < nq; x++) {
        int qo = w[x].key + 1;
        char *ins = dg_instruction(qj, qt, qnt, qo);
        db_fmt(&b, "\nQuestion q%d: %s\n", x + 1, *ins ? ins : "Answer about the state.");
        jb_release(ins);
        int cr = w[x].criteria;
        for (int i = 0; i < w[x].nc; i++) {
            char *desc = NULL;
            if (cr >= 0 && qt[cr].type == JT_OBJECT) {
                int z = jt_obj_get(qj, qt, qnt, cr, w[x].cand[i]);
                if (z >= 0)
                    desc = dg_text_of(qj, qt, qnt, z);
            } else if (cr >= 0 && qt[cr].type == JT_ARRAY) {
                int zc = 0;
                for (int z = cr + 1; z < qnt; z++)
                    if (qt[z].parent == cr && zc++ == i) {
                        desc = dg_text_of(qj, qt, qnt, z);
                        break;
                    }
            }
            if (!strcmp(w[x].kind, "noul")) {
                if (desc && *desc)
                    db_fmt(&b, "  %s: %s\n", w[x].label[i], desc);
                else
                    db_fmt(&b, "  %s\n", w[x].label[i]);
            } else if (!strcmp(w[x].kind, "score"))
                db_fmt(&b, "  %s: %s\n", w[x].label[i], desc ? desc : "");
            else if (desc && *desc)
                db_fmt(&b, "  %s: %s (%s)\n", w[x].label[i], w[x].cand[i], desc);
            else
                db_fmt(&b, "  %s: %s\n", w[x].label[i], w[x].cand[i]);
            jb_release(desc);
        }
    }
    db_fmt(&b, "\n%s",
           nq <= 10 ? "Reply with one line per question, in this order, formatted as \"id: label\"."
                    : "Reply on one line with each question's id immediately followed by its "
                      "label, separated by single spaces.");
    return b.p;
}

static char *dg_answer_text_range(DecisionWork *w, int total, int start, int count,
                                  const int *pick) {
    DGBuf b = {0};
    db_mem(&b, "", 0);
    for (int z = 0; z < count; z++) {
        int i = start + z;
        if (z)
            db_ch(&b, total <= 10 ? '\n' : ' ');
        db_fmt(&b, total <= 10 ? "q%d: %s" : "q%d%s", i + 1, w[i].label[pick[i]]);
    }
    return b.p;
}

#ifndef JB_NO_MAIN
static char *dg_answer_text(DecisionWork *w, int nq, const int *pick) {
    return dg_answer_text_range(w, nq, 0, nq, pick);
}
#endif

static void db_json_token(DGBuf *b, const char *j, const JTok *t) {
    if (t->type == JT_STRING) {
        char *s = jt_string(j, t);
        db_json_string(b, s, 0);
        jb_release(s);
    } else
        db_mem(b, j + t->start, (size_t)(t->end - t->start));
}

static char *dg_format_answers(const char *qj, JTok *qt, int qnt, DecisionWork *w, int nq,
                               const char *id, uint32_t tokens, double ms, double prefill_ms,
                               double cand_ms, int reads, int canvases, const char *cache_state,
                               int cache_tokens, int processed_tokens, int microbatch,
                               size_t *output_length) {
    DGBuf out = {0};
    db_fmt(
        &out,
        "{\"model\":\"jev-bush-diffusion-%s\",\"math\":\"%s\",\"kernels\":\"%s\",\"threads\":%d,",
        JB_VERSION, JB_MATH_MODE, dg_kernels()->name, jb_threads());
    if (id) {
        db_mem(&out, "\"id\":", 5);
        db_json_string(&out, id, 0);
        db_ch(&out, ',');
    }
    db_mem(&out, "\"answers\":{", 11);
    for (int x = 0; x < nq; x++) {
        DecisionWork *d = &w[x];
        if (x)
            db_ch(&out, ',');
        char *k = jt_string(qj, &qt[d->key]);
        db_json_string(&out, k, 0);
        jb_release(k);
        db_mem(&out, ":{\"type\":", 9);
        db_json_string(&out, d->kind, 0);
        if (!strcmp(d->kind, "noul")) {
            db_fmt(&out, ",\"noul\":");
            db_number(&out, d->prob[0]);
            db_fmt(&out, ",\"probabilities\":{\"true\":");
            db_number(&out, d->prob[0]);
            db_fmt(&out, ",\"false\":");
            db_number(&out, d->prob[1]);
            db_fmt(&out, "},\"confidence\":");
            db_number(&out, confidence(d->prob, d->nc));
        } else if (!strcmp(d->kind, "choice")) {
            int top = 0;
            for (int i = 1; i < d->nc; i++)
                if (d->prob[i] > d->prob[top])
                    top = i;
            db_mem(&out, ",\"choice\":", 10);
            db_json_string(&out, d->cand[top], 0);
            db_mem(&out, ",\"probabilities\":{", 18);
            for (int i = 0; i < d->nc; i++) {
                if (i)
                    db_ch(&out, ',');
                db_json_string(&out, d->cand[i], 0);
                db_ch(&out, ':');
                db_number(&out, d->prob[i]);
            }
            db_fmt(&out, "},\"confidence\":");
            db_number(&out, confidence(d->prob, d->nc));
        } else {
            double ev = 0;
            for (int i = 0; i < d->nc; i++)
                ev += i * d->prob[i];
            db_fmt(&out, ",\"score\":");
            db_number(&out, ev);
            db_fmt(&out, ",\"legend\":{");
            int zc = 0;
            for (int z = d->criteria + 1; z < qnt; z++)
                if (qt[z].parent == d->criteria) {
                    db_fmt(&out, "%s\"%d\":", zc ? "," : "", zc);
                    zc++;
                    db_json_token(&out, qj, &qt[z]);
                }
            db_mem(&out, "},\"probabilities\":{", 19);
            for (int i = 0; i < d->nc; i++) {
                db_fmt(&out, "%s\"%d\":", i ? "," : "", i);
                db_number(&out, d->prob[i]);
            }
            db_fmt(&out, "},\"confidence\":");
            db_number(&out, confidence(d->prob, d->nc));
        }
        db_ch(&out, '}');
    }
    db_fmt(&out,
           "},\"usage\":{\"input_tokens\":%u,\"output_tokens\":0,\"prefill_tokens\":%d},"
           "\"timing_ms\":{\"total\":",
           tokens, processed_tokens);
    db_millis(&out, ms);
    db_fmt(&out, ",\"prefill\":");
    db_millis(&out, prefill_ms);
    db_fmt(&out, ",\"decode\":");
    db_millis(&out, ms - prefill_ms - cand_ms);
    db_fmt(&out, ",\"candidates\":");
    db_millis(&out, cand_ms);
    db_fmt(&out,
           ",\"reads\":%d,\"canvases\":%d,\"microbatch\":%d,\"prefix_cache\":\"%s\","
           "\"cache_tokens\":%d}}",
           reads, canvases, microbatch, cache_state, cache_tokens);
    *output_length = out.n;
    return out.p;
}

typedef struct {
    const char *j, *qj;
    JTok *t, *qt;
    int nt, qnt, si, qroot, requested;
    char *state, *reqid, *qowned;
} DGRequest;

static void dg_request_parse(DGRequest *r, const char *j, size_t len) {
    memset(r, 0, sizeof *r);
    r->j = j;
    int nt;
    JTok *t = r->t = json_tokens(j, len, &nt);
    r->nt = nt;
    if (!nt || t[0].type != JT_OBJECT)
        die("System One request must be an object");
    int si = jt_obj_get(j, t, nt, 0, "state"), qi = jt_obj_get(j, t, nt, 0, "questions"),
        ii = jt_obj_get(j, t, nt, 0, "id"), sti = jt_obj_get(j, t, nt, 0, "samples"),
        dpi = jt_obj_get(j, t, nt, 0, "steps");
    int has_samples = sti >= 0 && !jt_literal(j, &t[sti], "null"),
        requested = has_samples ? jt_nonnegative_int(j, &t[sti], "samples") : 0,
        steps = dpi >= 0 && !jt_literal(j, &t[dpi], "null")
                    ? jt_nonnegative_int(j, &t[dpi], "steps")
                    : 1;
    if (has_samples && (requested < 1 || requested > 32))
        die("samples must be 1..32");
    if (steps != 1)
        die("only one-step DiffusionGemma reads are supported");
    if (si < 0 || qi < 0 || (t[qi].type != JT_OBJECT && t[qi].type != JT_STRING))
        die("request needs state and questions object");
    r->si = si;
    r->requested = requested;
    r->state =
        t[si].type == JT_STRING ? jt_string(j, &t[si]) : dg_json_canonical(j, t, nt, si, 0, 0);
    r->reqid = ii >= 0 ? jt_string(j, &t[ii]) : NULL;
    r->qj = j;
    r->qt = t;
    r->qnt = nt;
    r->qroot = qi;
    if (t[qi].type == JT_STRING) {
        r->qowned = jt_string(j, &t[qi]);
        r->qt = json_tokens(r->qowned, strlen(r->qowned), &r->qnt);
        r->qj = r->qowned;
        r->qroot = 0;
        if (!r->qnt || r->qt[0].type != JT_OBJECT)
            die("questions string is not a JSON object");
    }
}

static void dg_request_free(DGRequest *r) {
    jb_release(r->state);
    jb_release(r->reqid);
    if (r->qowned) {
        jb_release(r->qt);
        jb_release(r->qowned);
    }
    jb_release(r->t);
}

static uint32_t dg_request_seed(const DGRequest *r) {
    DGBuf seed_json = {0};
    db_ch(&seed_json, '[');
    dg_json_value(&seed_json, r->j, r->t, r->nt, r->si, 1, 1);
    db_mem(&seed_json, ", ", 2);
    dg_json_value(&seed_json, r->qj, r->qt, r->qnt, r->qroot, 1, 1);
    db_ch(&seed_json, ']');
    uint8_t digest[32];
    dg_sha256((const uint8_t *)seed_json.p, seed_json.n, digest);
    jb_release(seed_json.p);
    return (uint32_t)digest[0] << 24 | (uint32_t)digest[1] << 16 | (uint32_t)digest[2] << 8 |
           digest[3];
}

static DecisionWork *dg_questions(const DGRequest *r, char **choice_label, int *nq_out) {
    const char *qj = r->qj;
    JTok *qt = r->qt;
    int qnt = r->qnt;
    int keys[DG_MAX_QUESTIONS], nq = direct_keys(qt, qnt, r->qroot, keys, DG_MAX_QUESTIONS);
    if (nq < 1 || nq > DG_MAX_QUESTIONS)
        die("questions must contain 1..1024 entries");
    /* Reject before any model work rather than after prefill and a read. */
    if (!r->requested && nq > DG_AUTO_READ_MAX_QUESTIONS)
        die("automatic reads support at most 128 questions; set samples");
    DecisionWork *w = xcalloc((size_t)nq, sizeof *w);
    for (int x = 0; x < nq; x++) {
        DecisionWork *d = &w[x];
        d->key = keys[x];
        int qo = keys[x] + 1, ty = jt_obj_get(qj, qt, qnt, qo, "type");
        d->criteria = jt_obj_get(qj, qt, qnt, qo, "criteria");
        if (ty < 0)
            die("question missing type");
        d->kind = jt_string(qj, &qt[ty]);
        d->cand = xcalloc(JB_MAX_CAND, sizeof *d->cand);
        if (!strcmp(d->kind, "noul")) {
            d->cand[d->nc++] = xstrdup("true");
            d->cand[d->nc++] = xstrdup("false");
        } else if (!strcmp(d->kind, "choice")) {
            if (d->criteria < 0 || qt[d->criteria].type != JT_OBJECT)
                die("choice needs criteria object");
            int ck[JB_MAX_CAND];
            d->nc = direct_keys(qt, qnt, d->criteria, ck, JB_MAX_CAND);
            for (int i = 0; i < d->nc; i++)
                d->cand[i] = jt_string(qj, &qt[ck[i]]);
        } else if (!strcmp(d->kind, "score")) {
            if (d->criteria < 0 || qt[d->criteria].type != JT_ARRAY)
                die("score needs criteria array");
            for (int z = d->criteria + 1; z < qnt; z++)
                if (qt[z].parent == d->criteria) {
                    if (d->nc >= DG_MAX_SCORE)
                        die("decision candidate count out of range");
                    char b[16];
                    snprintf(b, sizeof b, "%d", d->nc);
                    d->cand[d->nc++] = xstrdup(b);
                }
        } else
            die("unknown question type");
        if (d->nc < 2 || d->nc > DG_CHOICE_LABELS ||
            (!strcmp(d->kind, "score") && d->nc > DG_MAX_SCORE))
            die("decision candidate count out of range");
        d->label = xcalloc((size_t)d->nc, sizeof *d->label);
        for (int i = 0; i < d->nc; i++)
            d->label[i] = dg_decision_label(d->kind, i, choice_label);
        d->prob = xmalloc((size_t)d->nc * sizeof *d->prob);
    }
    *nq_out = nq;
    return w;
}

static void dg_questions_free(DecisionWork *w, int nq) {
    for (int x = 0; x < nq; x++) {
        for (int i = 0; i < w[x].nc; i++) {
            jb_release(w[x].cand[i]);
            jb_release(w[x].label[i]);
        }
        jb_release(w[x].cand);
        jb_release(w[x].label);
        jb_release(w[x].kind);
        jb_release(w[x].prob);
    }
    jb_release(w);
}

#ifndef JB_NO_MAIN
static char *dg_answer_template(DecisionWork *w, int nq, const int *pick) {
    char *a = dg_answer_text(w, nq, pick);
    DGBuf b = {0};
    db_fmt(&b, "<|channel>thought\n<channel|>%s", a);
    jb_release(a);
    return b.p;
}
#endif

static char *dg_answer_template_range(DecisionWork *w, int total, int start, int count,
                                      const int *pick) {
    char *a = dg_answer_text_range(w, total, start, count, pick);
    DGBuf b = {0};
    db_fmt(&b, "<|channel>thought\n<channel|>%s", a);
    jb_release(a);
    return b.p;
}

typedef struct {
    DGRequest rq;
    DecisionWork *w;
    int nq;
    uint32_t seed;
    char *sys, *pre, *prompt;
    Tokens pt, pre_tokens;
    int group_cap, groups, width, *slot, **ids, *id_data, *canvas;
    DGCanvas *cv;
    int requested, reads, max_reads, active;
    uint64_t candidate_ns;
} DGJob;

static void dg_job_prepare(DGJob *g, DGTokenizer *tok, const char *j, size_t len) {
    memset(g, 0, sizeof *g);
    dg_request_parse(&g->rq, j, len);
    g->requested = g->rq.requested;
    g->seed = dg_request_seed(&g->rq);
    char *choice[DG_CHOICE_LABELS];
    int nc = dg_choice_inventory(tok, choice);
    if (nc != DG_CHOICE_LABELS)
        die("cannot construct OpenJev choice labels");
    g->w = dg_questions(&g->rq, choice, &g->nq);
    for (int i = 0; i < nc; i++)
        jb_release(choice[i]);
    g->sys = dg_system_prompt(g->rq.qj, g->rq.qt, g->rq.qnt, g->w, g->nq);
    DGBuf pre = {0}, pb = {0};
    db_fmt(&pre, "<bos><|turn>system\n%s<turn|>\n<|turn>user\n", g->sys);
    g->pre = pre.p;
    g->pre_tokens = dgt_tokenize(tok, g->pre);
    db_fmt(&pb, "%s%s<turn|>\n<|turn>model\n", g->pre, g->rq.state);
    g->prompt = pb.p;
    if (strlen(g->prompt) > (size_t)JB_MAX_CTX * tok->maxlen)
        die("prompt exceeds DiffusionGemma context");
    g->pt = dgt_tokenize(tok, g->prompt);
    if (g->pt.n > JB_MAX_CTX)
        die("prompt exceeds DiffusionGemma context");
    int *zero = xcalloc((size_t)g->nq, sizeof *zero);
    g->group_cap = g->nq <= DG_CANVAS_ALIGN ? g->nq : DG_CANVAS_GROUP;
    g->groups = (g->nq + g->group_cap - 1) / g->group_cap;
    g->cv = xcalloc((size_t)g->groups, sizeof *g->cv);
    g->slot = xmalloc((size_t)g->nq * sizeof *g->slot);
    g->ids = xcalloc((size_t)g->nq, sizeof *g->ids);
    size_t candidate_ids = 0;
    for (int x = 0; x < g->nq; x++)
        candidate_ids = jb_size_add(candidate_ids, (size_t)g->w[x].nc);
    g->id_data = xmalloc(jb_size_mul(candidate_ids, sizeof *g->id_data));
    size_t candidate_at = 0;
    for (int x = 0; x < g->nq; x++) {
        g->ids[x] = g->id_data + candidate_at;
        candidate_at += (size_t)g->w[x].nc;
    }
    for (int b = 0; b < g->groups; b++) {
        DGCanvas *c = &g->cv[b];
        c->start = b * g->group_cap;
        c->count = g->nq - c->start < g->group_cap ? g->nq - c->start : g->group_cap;
        char *text = dg_answer_template_range(g->w, g->nq, c->start, c->count, zero);
        c->base = dgt_tokenize(tok, text);
        jb_release(text);
        if (c->base.n + 1 > DG_CANVAS_TOKENS)
            die("OpenJev answer template exceeds 64-token canvas");
        int cw = (int)(((c->base.n + 1 + DG_CANVAS_ALIGN - 1) / DG_CANVAS_ALIGN) * DG_CANVAS_ALIGN);
        if (cw > g->width)
            g->width = cw;
        for (int x = c->start; x < c->start + c->count; x++) {
            g->slot[x] = -1;
            for (int q = 1; q < g->w[x].nc; q++) {
                int old = zero[x];
                zero[x] = q;
                char *a = dg_answer_template_range(g->w, g->nq, c->start, c->count, zero);
                Tokens v = dgt_tokenize(tok, a);
                jb_release(a);
                zero[x] = old;
                if (v.n != c->base.n)
                    die("labels do not share one template slot");
                int diff = -1;
                for (uint32_t z = 0; z < v.n; z++)
                    if (v.v[z] != c->base.v[z]) {
                        if (diff >= 0)
                            die("label changes more than one template token");
                        diff = (int)z;
                    }
                if (diff < 0)
                    die("duplicate label token");
                if (g->slot[x] >= 0 && diff != g->slot[x])
                    die("labels do not share one template slot");
                g->slot[x] = diff;
                g->ids[x][q] = v.v[diff];
                jb_release(v.v);
            }
            if (g->slot[x] < 0)
                die("cannot resolve label slot");
            g->ids[x][0] = c->base.v[g->slot[x]];
        }
    }
    if ((uint64_t)g->width * g->groups > JB_MAX_CTX)
        die("batched answer canvases exceed DiffusionGemma context");
    g->max_reads = g->requested ? g->requested : 4;
    /* Room for as many reads' canvases as one decode takes. */
    int per_read = g->width * g->groups;
    int batch_reads = JB_MAX_CTX / per_read < g->max_reads ? JB_MAX_CTX / per_read : g->max_reads;
    g->canvas = xcalloc((size_t)per_read * (size_t)batch_reads, sizeof *g->canvas);
    for (int x = 0; x < g->nq; x++)
        memset(g->w[x].prob, 0, (size_t)g->w[x].nc * sizeof *g->w[x].prob);
    g->active = 1;
    jb_release(zero);
}

static void dg_job_free(DGJob *g) {
    for (int b = 0; b < g->groups; b++)
        jb_release(g->cv[b].base.v);
    jb_release(g->cv);
    jb_release(g->ids);
    jb_release(g->id_data);
    jb_release(g->slot);
    jb_release(g->canvas);
    jb_release(g->pt.v);
    jb_release(g->pre_tokens.v);
    jb_release(g->prompt);
    jb_release(g->pre);
    jb_release(g->sys);
    dg_questions_free(g->w, g->nq);
    dg_request_free(&g->rq);
    memset(g, 0, sizeof *g);
}

/* Read r's answer canvases into dst, width * groups tokens: each group's
 * base with the mask after it, and the read's random tokens in the answer
 * slots. */
static void dg_job_canvas(const DGJob *g, int *dst, int r) {
    memset(dst, 0, (size_t)g->width * g->groups * sizeof *dst);
    for (int b = 0; b < g->groups; b++) {
        memcpy(dst + (size_t)b * g->width, g->cv[b].base.v, (size_t)g->cv[b].base.n * sizeof *dst);
        dst[(size_t)b * g->width + g->cv[b].base.n] = DG_CANVAS_MASK_TOKEN;
    }
    DGMT rng;
    dg_mt_seed(&rng, g->seed + (uint32_t)r * 7919u);
    for (int x = 0; x < g->nq; x++)
        dst[(size_t)(x / g->group_cap) * g->width + g->slot[x]] = (int)dg_mt_vocab(&rng);
}

static size_t dg_job_score_bytes(const DGJob *g, int read) {
    size_t candidates = 0;
    int max_candidates = 0;
    for (int x = 0; x < g->nq; x++) {
        candidates = jb_size_add(candidates, (size_t)g->w[x].nc);
        if (g->w[x].nc > max_candidates)
            max_candidates = g->w[x].nc;
    }
    size_t bytes = 0;
    jb_workspace_plan(&bytes, (size_t)g->nq, sizeof(double *));
    jb_workspace_plan(&bytes, candidates, sizeof(double));
    if (!g->requested && read == 0) {
        jb_workspace_plan(&bytes, (size_t)g->nq * DG_H, sizeof(float));
        jb_workspace_plan(&bytes, (size_t)g->nq, sizeof(int));
        jb_workspace_plan(&bytes, (size_t)g->nq * DG_VOCAB, sizeof(float));
        jb_workspace_plan(&bytes, DG_VOCAB, sizeof(double));
    }
    jb_workspace_plan(&bytes, (size_t)max_candidates, sizeof(double));
    return bytes;
}

static void dg_job_score(DGTensor *emb, DGJob *g, const float *h, int segment0, int read,
                         DGWorkspace *owner) {
    uint64_t start = now_ns();
    double entropy = 0;
    size_t candidates = 0;
    int max_candidates = 0;
    for (int x = 0; x < g->nq; x++) {
        candidates = jb_size_add(candidates, (size_t)g->w[x].nc);
        if (g->w[x].nc > max_candidates)
            max_candidates = g->w[x].nc;
    }
    int score_only = !owner->active;
    JBArena *workspace =
        score_only ? dg_workspace_begin(owner, dg_job_score_bytes(g, read)) : &owner->arena;
    size_t mark = jb_arena_mark(workspace);
    double **score = jb_arena_alloc(workspace, (size_t)g->nq, sizeof *score, 0);
    double *score_data = jb_arena_alloc(workspace, candidates, sizeof *score_data, 0);
    size_t score_offset = 0;
    for (int x = 0; x < g->nq; x++) {
        score[x] = score_data + score_offset;
        score_offset += (size_t)g->w[x].nc;
    }
    if (!g->requested && read == 0) {
        float *answer_h = jb_arena_alloc(workspace, (size_t)g->nq * DG_H, sizeof *answer_h, 0);
        int *label_n = jb_arena_alloc(workspace, (size_t)g->nq, sizeof *label_n, 0);
        float *logits = jb_arena_alloc(workspace, (size_t)g->nq * DG_VOCAB, sizeof *logits, 0);
        double *terms = jb_arena_alloc(workspace, DG_VOCAB, sizeof *terms, 0);
        for (int x = 0; x < g->nq; x++) {
            memcpy(answer_h + (size_t)x * DG_H,
                   h + ((size_t)(segment0 + x / g->group_cap) * g->width + g->slot[x]) * DG_H,
                   DG_H * sizeof *answer_h);
            label_n[x] = g->w[x].nc;
        }
        /* The same product on the accelerator when it holds the embedding
         * and has room: bit for bit in strict builds. */
        size_t hb = (size_t)g->nq * DG_H * sizeof *answer_h,
               lb = (size_t)g->nq * DG_VOCAB * sizeof *logits;
        DGDevice d =
            owner->accel && emb->device ? dg_device_scratch(owner, jb_size_add(hb, lb)) : 0;
        if (d) {
            owner->accel->to_device(d, answer_h, hb);
            owner->accel->bf16_mm(emb->device, d, d + hb, g->nq, DG_VOCAB, DG_H);
            owner->accel->to_host(logits, d + hb, lb);
        } else
            dg_mm(emb, answer_h, logits, g->nq, DG_VOCAB, DG_H);
        entropy = dg_slot_logits_entropy(g->nq, g->ids, label_n, score, logits, terms);
    } else
        for (int x = 0; x < g->nq; x++)
            for (int c = 0; c < g->w[x].nc; c++) {
                float z;
                dg_mv_slice(emb, (uint64_t)g->ids[x][c] * DG_H,
                            h + ((size_t)(segment0 + x / g->group_cap) * g->width + g->slot[x]) *
                                    DG_H,
                            &z, 1, DG_H);
                score[x][c] = 30 * tanh(z / 30);
            }
    double *prob = jb_arena_alloc(workspace, (size_t)max_candidates, sizeof *prob, 0);
    for (int x = 0; x < g->nq; x++) {
        normalize_scores(score[x], g->w[x].nc, prob);
        for (int c = 0; c < g->w[x].nc; c++)
            g->w[x].prob[c] += prob[c];
    }
    jb_arena_reset(workspace, mark);
    if (score_only) {
        jb_arena_reset(workspace, 0);
        dg_workspace_end(owner);
    }
    g->candidate_ns += now_ns() - start;
    g->reads++;
    if (!g->requested && read == 0 && !(entropy > .1))
        g->active = 0;
}

static void dg_job_normalize(DGJob *g) {
    for (int x = 0; x < g->nq; x++)
        for (int c = 0; c < g->w[x].nc; c++)
            g->w[x].prob[c] /= g->reads;
}

static int dg_systemone(DGModel *m, DGTokenizer *tok, const char *j, size_t len,
                        DGPrefixCache *prefix_cache, DGWorkspace *workspace, char **output,
                        size_t *output_length) {
#ifdef JB_PROFILE
    JBProfile *profile = &workspace->profile, *previous_profile = jb_profile_active;
    memset(profile, 0, sizeof *profile);
    jb_profile_active = profile;
#endif
    DGJob job;
    dg_job_prepare(&job, tok, j, len);
    DGRequest *rq = &job.rq;
    DGTensor *emb = dg_tensor(m, "model.decoder.embed_tokens.weight");
#ifdef JB_PROFILE
    Tokens system_part = dgt_tokenize(tok, job.sys), state_part = dgt_tokenize(tok, rq->state);
    profile->prompt_tokens = job.pt.n;
    profile->system_tokens = system_part.n;
    profile->state_tokens = state_part.n;
    jb_release(system_part.v);
    jb_release(state_part.v);
#endif
#ifdef JB_PROFILE
    profile->hot = 1;
#endif
    uint64_t start = now_ns();
    DGKV kv[DG_L] = {0};
    int kv_length = (int)job.pt.n;
    dg_worker_kv_begin(workspace, kv, 1, &kv_length);
    const char *cache_state = "off";
    int cache_tokens = 0, processed_tokens = (int)job.pt.n;
    if (prefix_cache) {
        uint32_t common = 0, lim = job.pre_tokens.n < job.pt.n ? job.pre_tokens.n : job.pt.n;
        while (common < lim && job.pre_tokens.v[common] == job.pt.v[common])
            common++;
        int hit = prefix_cache->schema && !strcmp(prefix_cache->schema, job.sys) &&
                  prefix_cache->n <= (int)job.pt.n &&
                  !memcmp(prefix_cache->ids, job.pt.v, (size_t)prefix_cache->n * sizeof *job.pt.v);
        if (!hit) {
            if (!common || common >= job.pt.n)
                die("cannot construct reusable schema prefix");
            /* Build the replacement privately and install it only once it is
             * complete. If anything below fails, the entry stays empty, so
             * the next request misses and rebuilds instead of "hitting" a
             * schema whose K/V was never filled. */
            dg_prefix_free(prefix_cache);
            DGPrefixCache fresh;
            memset(&fresh, 0, sizeof fresh);
            size_t schema_bytes = strlen(job.sys) + 1, metadata_bytes = 0;
            jb_workspace_plan(&metadata_bytes, schema_bytes, sizeof(char));
            jb_workspace_plan(&metadata_bytes, common, sizeof *fresh.ids);
            jb_arena_init(&fresh.metadata, metadata_bytes);
            fresh.schema = jb_arena_alloc(&fresh.metadata, schema_bytes, sizeof *fresh.schema, 0);
            fresh.ids = jb_arena_alloc(&fresh.metadata, common, sizeof *fresh.ids, 0);
            memcpy(fresh.schema, job.sys, schema_bytes);
            fresh.n = (int)common;
            memcpy(fresh.ids, job.pt.v, (size_t)common * sizeof *fresh.ids);
            dg_prefill(m, job.pt.v, (int)job.pt.n, kv, workspace);
            int prefix_length = (int)common;
            dg_kv_init(&fresh.kv_storage, fresh.kv, 1, &prefix_length);
            for (int l = 0; l < DG_L; l++) {
                int kn = dg_kv_width(l);
                memcpy(fresh.kv[l].k, kv[l].k, (size_t)common * kn * 4);
                memcpy(fresh.kv[l].v, kv[l].v, (size_t)common * kn * 4);
            }
            jb_untrack(fresh.metadata.allocation);
            jb_untrack(fresh.kv_storage.allocation);
            *prefix_cache = fresh;
            cache_state = "miss";
        } else {
            cache_state = "hit";
            processed_tokens = (int)job.pt.n - prefix_cache->n;
            int suffix = (int)job.pt.n - prefix_cache->n;
            if (suffix <= 0)
                die("schema prefix consumes complete prompt");
            dg_prefill_suffix(m, job.pt.v + prefix_cache->n, suffix, prefix_cache->kv, kv,
                              workspace);
        }
        cache_tokens = prefix_cache->n;
    } else
        dg_prefill(m, job.pt.v, (int)job.pt.n, kv, workspace);
    uint64_t prefill_ns = now_ns() - start;
#ifdef JB_CANVAS_SEQUENTIAL
    size_t sequential_count = jb_size_mul((size_t)job.groups * job.width, DG_H);
    JBArena *sequential_storage =
        dg_control_begin(workspace, jb_size_mul(sequential_count, sizeof(float)));
    float *sequential_hidden =
        jb_arena_alloc(sequential_storage, sequential_count, sizeof *sequential_hidden, 0);
#endif
#ifdef JB_CANVAS_SEQUENTIAL
    for (int r = 0; r < job.max_reads; r++) {
        dg_job_canvas(&job, job.canvas, r);
        float *h = sequential_hidden;
        for (int b = 0; b < job.groups; b++) {
            float *one = dg_decode(m, kv, (int)job.pt.n, job.canvas + (size_t)b * job.width,
                                   job.width, 1, 0, workspace);
            memcpy(h + (size_t)b * job.width * DG_H, one, (size_t)job.width * DG_H * sizeof *h);
            dg_decode_end(workspace);
        }
        dg_job_score(emb, &job, h, 0, r, workspace);
        if (!job.active)
            break;
    }
#else
    /* Reads differ only in their random answer-slot tokens, so they share one
     * decode as extra canvases, each attending to its own tokens and the
     * prefix: the weights stream once for all of them. Automatic reads run
     * the first alone, because its entropy decides whether more follow. Each
     * read is scored in order, as it would be alone. */
    int per_read = job.width * job.groups, most = JB_MAX_CTX / per_read;
    for (int r = 0; r < job.max_reads && job.active;) {
        int count = !job.requested && r == 0 ? 1 : job.max_reads - r;
        if (count > most)
            count = most;
        for (int k = 0; k < count; k++)
            dg_job_canvas(&job, job.canvas + (size_t)k * per_read, r + k);
        size_t score_bytes = dg_job_score_bytes(&job, r);
        float *h = dg_decode(m, kv, (int)job.pt.n, job.canvas, job.width, job.groups * count,
                             score_bytes, workspace);
        for (int k = 0; k < count; k++)
            dg_job_score(emb, &job, h, k * job.groups, r + k, workspace);
        dg_decode_end(workspace);
        r += count;
    }
#endif
#ifdef JB_CANVAS_SEQUENTIAL
    dg_control_end(workspace);
#endif
#ifdef JB_PROFILE
    double kv_mb = workspace->kv_arena.size / 1048576.0;
#endif
    dg_worker_kv_end(workspace, kv, 1);
#ifdef JB_PROFILE
    profile->hot = 0;
#endif
    dg_job_normalize(&job);
    double cand_ms = job.candidate_ns / 1e6, ms = (now_ns() - start) / 1e6;
    uint32_t billed = job.pt.n * (job.requested ? job.reads : 1);
    *output = dg_format_answers(rq->qj, rq->qt, rq->qnt, job.w, job.nq, rq->reqid, billed, ms,
                                prefill_ns / 1e6, cand_ms, job.reads, job.groups, cache_state,
                                cache_tokens, processed_tokens, 1, output_length);
#ifdef JB_PROFILE
    uint64_t detailed =
        profile->moe_input_qdq + profile->moe_gated + profile->moe_hidden_qdq + profile->moe_down;
    double moe_misc =
        (double)(profile->experts >= detailed ? profile->experts - detailed : 0) / 1e6;
    uint64_t ad = profile->attention_qkv + profile->attention_prepare + profile->attention_kv +
                  profile->attention_core + profile->attention_output;
    double attention_misc = (double)(profile->attention >= ad ? profile->attention - ad : 0) / 1e6;
    uint64_t ld = profile->attention + profile->dense + profile->router + profile->experts +
                  profile->ff_other;
    double layer_other = (double)(profile->layers >= ld ? profile->layers - ld : 0) / 1e6;
    double accounted = profile->layers / 1e6 + profile->embedding / 1e6 +
                       profile->final_norm / 1e6 + profile->kv_free / 1e6 + cand_ms;
    double unaccounted = ms > accounted ? ms - accounted : 0;
    fprintf(stderr,
            "JB_PROFILE prompt_tokens=%llu system_tokens=%llu state_tokens=%llu omp_regions=%llu "
            "alloc_calls=%llu hot_alloc_calls=%llu alloc_ms=%.3f workspace_mb=%.3f kv_mb=%.3f "
            "attention=%.3f attention_qkv=%.3f attention_prepare=%.3f "
            "attention_kv=%.3f attention_core=%.3f attention_output=%.3f attention_misc=%.3f "
            "dense=%.3f router=%.3f experts=%.3f moe_input_qdq=%.3f moe_gated=%.3f "
            "moe_hidden_qdq=%.3f moe_down=%.3f moe_misc=%.3f ff_other=%.3f "
            "layer_other=%.3f embedding=%.3f final_norm=%.3f kv_free=%.3f candidates=%.3f "
            "unaccounted=%.3f total=%.3f\n",
            (unsigned long long)profile->prompt_tokens, (unsigned long long)profile->system_tokens,
            (unsigned long long)profile->state_tokens, (unsigned long long)profile->omp_regions,
            (unsigned long long)profile->alloc_calls, (unsigned long long)profile->hot_alloc_calls,
            profile->alloc_ns / 1e6, workspace->capacity / 1048576.0, kv_mb,
            profile->attention / 1e6, profile->attention_qkv / 1e6,
            profile->attention_prepare / 1e6, profile->attention_kv / 1e6,
            profile->attention_core / 1e6, profile->attention_output / 1e6, attention_misc,
            profile->dense / 1e6, profile->router / 1e6, profile->experts / 1e6,
            profile->moe_input_qdq / 1e6, profile->moe_gated / 1e6, profile->moe_hidden_qdq / 1e6,
            profile->moe_down / 1e6, moe_misc, profile->ff_other / 1e6, layer_other,
            profile->embedding / 1e6, profile->final_norm / 1e6, profile->kv_free / 1e6, cand_ms,
            unaccounted, ms);
    jb_profile_active = previous_profile;
#endif
    dg_job_free(&job);
    return 0;
}

/* Returns zero when the rows do not all hit the current exact schema entry;
 * the caller then executes them sequentially, allowing normal cache replace. */
static int dg_system_batch(DGModel *m, DGTokenizer *tok, const char *const *row, const size_t *len,
                           int batch, DGPrefixCache *prefix, DGWorkspace *workspace, char **output,
                           size_t *output_length) {
    if (batch < 2 || !prefix || !prefix->schema)
        return 0;
    DGJob *g = xcalloc((size_t)batch, sizeof *g);
    int ok = 1;
    for (int b = 0; b < batch; b++) {
        dg_job_prepare(&g[b], tok, row[b], len[b]);
        if (strcmp(g[b].sys, prefix->schema) || prefix->n > (int)g[b].pt.n ||
            memcmp(prefix->ids, g[b].pt.v, (size_t)prefix->n * sizeof *prefix->ids))
            ok = 0;
        if (b && (g[b].width != g[0].width || g[b].groups != g[0].groups || g[b].nq != g[0].nq))
            ok = 0;
    }
    if ((uint64_t)batch * g[0].groups * g[0].width > JB_MAX_CTX)
        ok = 0;
    /* Suffixes are padded to the longest one. Rows that each fit on their
     * own but not together as one microbatch run sequentially instead. */
    int seq = 0;
    for (int b = 0; ok && b < batch; b++) {
        int suffix = (int)g[b].pt.n - prefix->n;
        if (suffix > seq)
            seq = suffix;
    }
    if ((uint64_t)batch * seq > JB_MAX_CTX)
        ok = 0;
    uint64_t kv_token_bytes = 0, kv_batch_bytes = 0;
    for (int l = 0; l < DG_L; l++)
        kv_token_bytes += 2 * (uint64_t)dg_kv_width(l) * sizeof(float);
    for (int b = 0; ok && b < batch; b++)
        kv_batch_bytes += (uint64_t)g[b].pt.n * kv_token_bytes;
    if (kv_batch_bytes > JB_MICROBATCH_KV_LIMIT)
        ok = 0;
    if (!ok) {
        for (int b = 0; b < batch; b++)
            dg_job_free(&g[b]);
        jb_release(g);
        return 0;
    }
    for (int b = 0; b < batch; b++)
        if ((int)g[b].pt.n - prefix->n < 1)
            die("schema prefix consumes complete prompt");
    int width = g[0].width, maxr = 0, max_segments = 0;
    for (int b = 0; b < batch; b++) {
        max_segments += g[b].groups;
        if (g[b].max_reads > maxr)
            maxr = g[b].max_reads;
    }
    size_t control_bytes = 0;
    jb_workspace_plan(&control_bytes, (size_t)batch, sizeof(int));
    jb_workspace_plan(&control_bytes, (size_t)batch * seq, sizeof(int));
    jb_workspace_plan(&control_bytes, (size_t)batch * DG_L, sizeof(DGKV));
    jb_workspace_plan(&control_bytes, (size_t)batch, sizeof(int));
    jb_workspace_plan(&control_bytes, (size_t)max_segments * width, sizeof(int));
    jb_workspace_plan(&control_bytes, (size_t)max_segments, sizeof(int));
    jb_workspace_plan(&control_bytes, (size_t)batch, sizeof(int));
    jb_workspace_plan(&control_bytes, (size_t)max_segments, sizeof(int));
    jb_workspace_plan(&control_bytes, (size_t)max_segments, sizeof(DGKV *));
    jb_workspace_plan(&control_bytes, (size_t)batch, sizeof(DGKV));
    JBArena *control = dg_control_begin(workspace, control_bytes);
    int *sl = jb_arena_alloc(control, (size_t)batch, sizeof *sl, 0);
    int *st = jb_arena_alloc(control, (size_t)batch * seq, sizeof *st, 1);
    DGKV *kv = jb_arena_alloc(control, (size_t)batch * DG_L, sizeof *kv, 1);
    int *kv_lengths = jb_arena_alloc(control, (size_t)batch, sizeof *kv_lengths, 0);
    int *canvas = jb_arena_alloc(control, (size_t)max_segments * width, sizeof *canvas, 0),
        *doc = jb_arena_alloc(control, (size_t)max_segments, sizeof *doc, 0),
        *base = jb_arena_alloc(control, (size_t)batch, sizeof *base, 0),
        *attention_lens = jb_arena_alloc(control, (size_t)max_segments, sizeof *attention_lens, 0);
    const DGKV **attention_cache =
        jb_arena_alloc(control, (size_t)max_segments, sizeof *attention_cache, 0);
    DGKV *layer_out = jb_arena_alloc(control, (size_t)batch, sizeof *layer_out, 0);
    for (int b = 0; b < batch; b++) {
        sl[b] = (int)g[b].pt.n - prefix->n;
        kv_lengths[b] = (int)g[b].pt.n;
    }
    for (int b = 0; b < batch; b++)
        memcpy(st + (size_t)b * seq, g[b].pt.v + prefix->n, (size_t)sl[b] * sizeof *st);
    dg_worker_kv_begin(workspace, kv, batch, kv_lengths);
    uint64_t start = now_ns();
    dg_prefill_suffix_multi(m, st, sl, batch, seq, prefix->kv, kv, attention_cache, layer_out,
                            workspace);
    uint64_t prefill_ns = now_ns() - start;
    DGTensor *emb = dg_tensor(m, "model.decoder.embed_tokens.weight");
    for (int r = 0; r < maxr; r++) {
        int segments = 0;
        for (int b = 0; b < batch; b++)
            if (g[b].active && r < g[b].max_reads)
                segments += g[b].groups;
        if (!segments)
            break;
        int at = 0;
        for (int b = 0; b < batch; b++) {
            base[b] = -1;
            if (!g[b].active || r >= g[b].max_reads)
                continue;
            base[b] = at;
            dg_job_canvas(&g[b], g[b].canvas, r);
            for (int q = 0; q < g[b].groups; q++) {
                memcpy(canvas + (size_t)at * width, g[b].canvas + (size_t)q * width,
                       (size_t)width * sizeof *canvas);
                doc[at++] = b;
            }
        }
        size_t score_bytes = 0;
        for (int b = 0; b < batch; b++)
            if (base[b] >= 0) {
                size_t one = dg_job_score_bytes(&g[b], r);
                if (one > score_bytes)
                    score_bytes = one;
            }
        float *h = dg_decode_multi(m, kv, doc, segments, width, canvas, attention_cache,
                                   attention_lens, score_bytes, workspace);
        for (int b = 0; b < batch; b++)
            if (base[b] >= 0) {
                dg_job_score(emb, &g[b], h, base[b], r, workspace);
            }
        dg_decode_end(workspace);
    }
    uint64_t total_ns = now_ns() - start;
    dg_worker_kv_end(workspace, kv, batch);
    dg_control_end(workspace);
    for (int b = 0; b < batch; b++) {
        dg_job_normalize(&g[b]);
        uint32_t billed = g[b].pt.n * (g[b].requested ? g[b].reads : 1);
        output[b] = dg_format_answers(
            g[b].rq.qj, g[b].rq.qt, g[b].rq.qnt, g[b].w, g[b].nq, g[b].rq.reqid, billed,
            total_ns / 1e6, prefill_ns / 1e6, g[b].candidate_ns / 1e6, g[b].reads, g[b].groups,
            "hit", prefix->n, (int)g[b].pt.n - prefix->n, batch, &output_length[b]);
        dg_job_free(&g[b]);
    }
    jb_release(g);
    return 1;
}
