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
