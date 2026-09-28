/*
 * Jev Bush -- CPU-first bounded decisions with DiffusionGemma.
 *
 * Build: cc -O3 -march=native -ffast-math -std=c11 -Wall -Wextra
 *        -pedantic -fopenmp jb.c -lm -o jb
 * The engine reads the exact public DiffusionGemma safetensors layout directly.
 */
#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define JB_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__) && defined(__FMA__)
#define JB_AVX2 1
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define JB_VERSION "0.1.0"
#define JB_MAX_CTX 4096
#define JB_MAX_CAND 255
#define JB_MAX_JSON (64u * 1024u * 1024u)
#define JB_MAX_DEPTH 256

#if defined(__FAST_MATH__)
#define JB_MATH_MODE "fast"
#else
#define JB_MATH_MODE "strict"
#endif

#ifdef JB_PROFILE
typedef struct {
    uint64_t layers, attention, attention_qkv, attention_prepare;
    uint64_t attention_kv, attention_core, attention_output;
    uint64_t dense, router, experts, ff_other, embedding, final_norm, kv_free;
    uint64_t moe_input_qdq, moe_gate, moe_up, moe_activation;
    uint64_t moe_hidden_qdq, moe_down;
    uint64_t prompt_tokens, system_tokens, state_tokens;
    uint64_t omp_regions, alloc_calls, alloc_ns;
} JBProfile;
static JBProfile jb_profile;
#define JB_TICK(name) uint64_t name = now_ns()
#define JB_TO(field, name) (jb_profile.field += now_ns() - (name))
#define JB_OMP() (jb_profile.omp_regions++)
#else
#define JB_TICK(name)
#define JB_TO(field, name)
#define JB_OMP()
#endif

typedef struct {
    int id;
    char *s;
    uint32_t n;
} Vocab;
typedef struct {
    uint32_t a, b, rank;
} Merge;
typedef struct {
    uint8_t *map;
    uint64_t size;
    int mapped;
#if defined(_WIN32)
    HANDLE hf, hm;
#endif
} FileMap;
typedef struct {
    int *v;
    uint32_t n, cap;
} Tokens;
static uint64_t now_ns(void);
static void die(const char *s) {
    fprintf(stderr, "jb: %s\n", s);
    exit(2);
}
static void die2(const char *a, const char *b) {
    fprintf(stderr, "jb: %s: %s\n", a, b);
    exit(2);
}
static void *xmalloc(size_t n) {
#ifdef JB_PROFILE
    uint64_t start = now_ns();
#endif
    void *p = malloc(n ? n : 1);
#ifdef JB_PROFILE
    jb_profile.alloc_ns += now_ns() - start;
    jb_profile.alloc_calls++;
#endif
    if (!p)
        die("out of memory");
    return p;
}
static void *xcalloc(size_t n, size_t z) {
#ifdef JB_PROFILE
    uint64_t start = now_ns();
#endif
    void *p = calloc(n ? n : 1, z);
#ifdef JB_PROFILE
    jb_profile.alloc_ns += now_ns() - start;
    jb_profile.alloc_calls++;
#endif
    if (!p)
        die("out of memory");
    return p;
}
static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *z = xmalloc(n);
    memcpy(z, s, n);
    return z;
}
static void *xrealloc(void *p, size_t n) {
#ifdef JB_PROFILE
    uint64_t start = now_ns();
#endif
    p = realloc(p, n ? n : 1);
#ifdef JB_PROFILE
    jb_profile.alloc_ns += now_ns() - start;
    jb_profile.alloc_calls++;
#endif
    if (!p)
        die("out of memory");
    return p;
}
/* Bit-level finiteness tests: -ffast-math lets the compiler assume NaN and
 * infinity never occur, which can fold isfinite() and NaN comparisons away. */
static int jb_finitef(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return (u & 0x7f800000u) != 0x7f800000u;
}
static int jb_finite(double x) {
    uint64_t u;
    memcpy(&u, &x, sizeof u);
    return (u & 0x7ff0000000000000ull) != 0x7ff0000000000000ull;
}
/* Threads OpenMP will use for parallel regions; 1 without OpenMP. */
static int jb_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
static void jb_path(char *out, size_t cap, const char *fmt, ...) {
    va_list a;
    va_start(a, fmt);
    int n = vsnprintf(out, cap, fmt, a);
    va_end(a);
    if (n < 0 || (size_t)n >= cap)
        die("model path too long");
}
static uint64_t now_ns(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}
static void map_file(FileMap *m, const char *path) {
#if defined(_WIN32)
    LARGE_INTEGER z;
    m->hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (m->hf == INVALID_HANDLE_VALUE)
        die2("open", path);
    if (!GetFileSizeEx(m->hf, &z) || z.QuadPart <= 0)
        die("GetFileSizeEx failed");
    m->size = (uint64_t)z.QuadPart;
    if (m->size > SIZE_MAX)
        die("model is too large for this address space");
    m->hm = CreateFileMappingA(m->hf, NULL, PAGE_READONLY, 0, 0, NULL);
    if (m->hm) {
        m->map = MapViewOfFile(m->hm, FILE_MAP_READ, 0, 0, 0);
        m->mapped = m->map != NULL;
    }
#else
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st))
        die2("open", path);
    if (st.st_size <= 0 || (uintmax_t)st.st_size > SIZE_MAX)
        die("invalid model size");
    m->size = (uint64_t)st.st_size;
    m->map = mmap(NULL, (size_t)m->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m->map != MAP_FAILED)
        m->mapped = 1;
    else
        m->map = NULL;
#endif
    if (!m->map) {
        FILE *f = fopen(path, "rb");
        if (!f)
            die2("open", path);
        m->map = xmalloc((size_t)m->size);
        if (fread(m->map, 1, (size_t)m->size, f) != (size_t)m->size)
            die("short read");
        fclose(f);
    }
}
static void unmap_file(FileMap *m) {
#if defined(_WIN32)
    if (m->mapped)
        UnmapViewOfFile(m->map);
    else
        free(m->map);
    if (m->hm)
        CloseHandle(m->hm);
    if (m->hf)
        CloseHandle(m->hf);
#else
    if (m->mapped)
        munmap(m->map, (size_t)m->size);
    else
        free(m->map);
#endif
}

static int cmp_vocab(const void *a, const void *b) {
    const Vocab *x = a, *y = b;
    size_t n = x->n < y->n ? x->n : y->n;
    int c = memcmp(x->s, y->s, n);
    return c ? c : (x->n > y->n) - (x->n < y->n);
}
static int cmp_merge(const void *a, const void *b) {
    const Merge *x = a, *y = b;
    return x->a != y->a ? (x->a > y->a) - (x->a < y->a) : (x->b > y->b) - (x->b < y->b);
}

static void push(Tokens *t, int x) {
    if (t->n == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 64;
        t->v = xrealloc(t->v, (size_t)t->cap * sizeof *t->v);
    }
    t->v[t->n++] = x;
}
static int put_utf8(uint32_t c, char *z) {
    if (c <= 0x7f) {
        z[0] = (char)c;
        return 1;
    }
    if (c <= 0x7ff) {
        z[0] = (char)(0xc0 | c >> 6);
        z[1] = (char)(0x80 | (c & 63));
        return 2;
    }
    if (c <= 0xffff) {
        z[0] = (char)(0xe0 | c >> 12);
        z[1] = (char)(0x80 | ((c >> 6) & 63));
        z[2] = (char)(0x80 | (c & 63));
        return 3;
    }
    z[0] = (char)(0xf0 | c >> 18);
    z[1] = (char)(0x80 | ((c >> 12) & 63));
    z[2] = (char)(0x80 | ((c >> 6) & 63));
    z[3] = (char)(0x80 | (c & 63));
    return 4;
}
static uint32_t next_cp(const unsigned char *s, size_t n, size_t *i) {
    size_t p = (*i)++;
    uint32_t c = s[p];
    if (c < 0x80)
        return c;
    int more = c >= 0xf0 ? 3 : c >= 0xe0 ? 2 : c >= 0xc2 ? 1 : -1;
    if (more < 0 || p + (size_t)more >= n)
        die("invalid UTF-8");
    uint32_t v = c & (0x7f >> more);
    for (int k = 0; k < more; k++) {
        unsigned q = s[(*i)++];
        if ((q & 0xc0) != 0x80)
            die("invalid UTF-8");
        v = (v << 6) | (q & 63);
    }
    if ((more == 1 && v < 0x80) || (more == 2 && v < 0x800) || (more == 3 && v < 0x10000) ||
        v > 0x10ffff || (v >= 0xd800 && v <= 0xdfff))
        die("invalid UTF-8");
    return v;
}
enum { JT_UNDEF, JT_OBJECT, JT_ARRAY, JT_STRING, JT_PRIMITIVE };
typedef struct {
    int type, start, end, parent, size;
} JTok;
typedef struct {
    const char *s;
    size_t n, pos;
    JTok *t;
    int nt, cap, depth;
} JParser;
static char *jt_string(const char *j, const JTok *t);
static int jt_new(JParser *p, int type, int start, int parent) {
    if (p->nt == p->cap) {
        p->cap = p->cap ? p->cap * 2 : 256;
        p->t = xrealloc(p->t, (size_t)p->cap * sizeof *p->t);
    }
    int i = p->nt++;
    p->t[i] = (JTok){type, start, -1, parent, 0};
    if (parent >= 0)
        p->t[parent].size++;
    return i;
}
static int json_parse_value(JParser *p, int parent);
static int json_primitive_valid(const char *s, size_t n) {
    if ((n == 4 && (!memcmp(s, "true", 4) || !memcmp(s, "null", 4))) ||
        (n == 5 && !memcmp(s, "false", 5)))
        return 1;
    size_t i = 0;
    if (i < n && s[i] == '-')
        i++;
    if (i >= n)
        return 0;
    if (s[i] == '0')
        i++;
    else {
        if (s[i] < '1' || s[i] > '9')
            return 0;
        while (i < n && isdigit((unsigned char)s[i]))
            i++;
    }
    if (i < n && s[i] == '.') {
        i++;
        size_t begin = i;
        while (i < n && isdigit((unsigned char)s[i]))
            i++;
        if (i == begin)
            return 0;
    }
    if (i < n && (s[i] == 'e' || s[i] == 'E')) {
        i++;
        if (i < n && (s[i] == '+' || s[i] == '-'))
            i++;
        size_t begin = i;
        while (i < n && isdigit((unsigned char)s[i]))
            i++;
        if (i == begin)
            return 0;
    }
    return i == n;
}
static int json_parse_string_tok(JParser *p, int parent) {
    int i = jt_new(p, JT_STRING, (int)++p->pos, parent);
    while (p->pos < p->n) {
        unsigned char c = p->s[p->pos++];
        if (c == '\"') {
            p->t[i].end = (int)p->pos - 1;
            return i;
        }
        if (c == '\\') {
            if (p->pos >= p->n)
                die("bad JSON escape");
            c = p->s[p->pos++];
            if (c == 'u') {
                for (int k = 0; k < 4; k++)
                    if (p->pos >= p->n || !isxdigit((unsigned char)p->s[p->pos++]))
                        die("bad JSON unicode escape");
            } else if (!c || !strchr("\"\\/bfnrt", c))
                die("bad JSON escape");
        } else if (c < 0x20)
            die("control byte in JSON string");
    }
    die("unterminated JSON string");
    return -1;
}
static int json_parse_value(JParser *p, int parent) {
    while (p->pos < p->n && isspace((unsigned char)p->s[p->pos]))
        p->pos++;
    if (p->pos >= p->n)
        die("missing JSON value");
    char c = p->s[p->pos];
    if (c == '\"')
        return json_parse_string_tok(p, parent);
    if (c == '{' || c == '[') {
        int ty = c == '{' ? JT_OBJECT : JT_ARRAY, i = jt_new(p, ty, (int)p->pos++, parent);
        if (++p->depth > JB_MAX_DEPTH)
            die("JSON nesting too deep");
        while (1) {
            while (p->pos < p->n && isspace((unsigned char)p->s[p->pos]))
                p->pos++;
            if (p->pos >= p->n)
                die("unterminated JSON container");
            if (p->s[p->pos] == (ty == JT_OBJECT ? '}' : ']')) {
                p->t[i].end = (int)++p->pos;
                p->depth--;
                return i;
            }
            if (p->t[i].size) {
                if (p->s[p->pos++] != ',')
                    die("expected JSON comma");
                while (p->pos < p->n && isspace((unsigned char)p->s[p->pos]))
                    p->pos++;
            }
            if (ty == JT_OBJECT) {
                if (p->pos >= p->n || p->s[p->pos] != '\"')
                    die("JSON object key must be a string");
                json_parse_string_tok(p, i);
                while (p->pos < p->n && isspace((unsigned char)p->s[p->pos]))
                    p->pos++;
                if (p->pos >= p->n || p->s[p->pos++] != ':')
                    die("expected JSON colon");
                json_parse_value(p, i);
            } else
                json_parse_value(p, i);
        }
    }
    int i = jt_new(p, JT_PRIMITIVE, (int)p->pos, parent);
    while (p->pos < p->n && !isspace((unsigned char)p->s[p->pos]) && !strchr(",]}:", p->s[p->pos]))
        p->pos++;
    p->t[i].end = (int)p->pos;
    if (p->t[i].end == p->t[i].start)
        die("bad JSON primitive");
    if (!json_primitive_valid(p->s + p->t[i].start, (size_t)(p->t[i].end - p->t[i].start)))
        die("bad JSON primitive");
    return i;
}
static int cmp_str(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}
/* Duplicate keys are rejected: lookups would take the first value where
 * Python's json keeps the last, and canonical seeds would diverge. */
static void json_check_keys(const char *s, const JTok *t, int nt) {
    char **keys = NULL;
    int cap = 0;
    for (int obj = 0; obj < nt; obj++) {
        if (t[obj].type != JT_OBJECT || t[obj].size < 4)
            continue;
        int nk = 0;
        for (int i = obj + 1; i < nt && t[i].parent >= obj; i++)
            if (t[i].parent == obj) {
                if (nk == cap) {
                    cap = cap ? cap * 2 : 64;
                    keys = xrealloc(keys, (size_t)cap * sizeof *keys);
                }
                keys[nk++] = jt_string(s, &t[i]);
                i++;
            }
        qsort(keys, (size_t)nk, sizeof *keys, cmp_str);
        for (int i = 0; i < nk; i++) {
            if (i && !strcmp(keys[i - 1], keys[i]))
                die2("duplicate JSON object key", keys[i]);
        }
        for (int i = 0; i < nk; i++)
            free(keys[i]);
    }
    free(keys);
}
static JTok *json_tokens(const char *s, size_t n, int *nt) {
    JParser p = {s, n, 0, 0, 0, 0, 0};
    json_parse_value(&p, -1);
    while (p.pos < n && isspace((unsigned char)s[p.pos]))
        p.pos++;
    if (p.pos != n)
        die("trailing JSON data");
    json_check_keys(s, p.t, p.nt);
    *nt = p.nt;
    return p.t;
}
static int jt_eq(const char *j, const JTok *t, const char *z) {
    if (t->type != JT_STRING)
        return 0;
    size_t n = (size_t)(t->end - t->start);
    if (!memchr(j + t->start, '\\', n))
        return n == strlen(z) && !memcmp(j + t->start, z, n);
    char *s = jt_string(j, t);
    int eq = !strcmp(s, z);
    free(s);
    return eq;
}
static int jt_literal(const char *j, const JTok *t, const char *z) {
    return t->type == JT_PRIMITIVE && t->end - t->start == (int)strlen(z) &&
           !memcmp(j + t->start, z, strlen(z));
}
static int jt_obj_get(const char *j, JTok *t, int nt, int obj, const char *key) {
    if (obj < 0 || obj >= nt || t[obj].type != JT_OBJECT)
        return -1;
    for (int i = obj + 1; i + 1 < nt; i++) {
        if (t[i].parent < obj)
            break;
        if (t[i].parent == obj && t[i].type == JT_STRING) {
            if (jt_eq(j, &t[i], key))
                return i + 1;
            i++;
        }
    }
    return -1;
}
static int jt_nonnegative_int(const char *j, const JTok *t, const char *name) {
    if (t->type != JT_PRIMITIVE || t->end <= t->start || t->end - t->start > 9)
        die2("expected non-negative integer", name);
    int v = 0;
    for (int i = t->start; i < t->end; i++) {
        if (j[i] < '0' || j[i] > '9')
            die2("expected non-negative integer", name);
        v = v * 10 + j[i] - '0';
    }
    return v;
}

typedef struct {
    FileMap file;
    char path[768];
} DGShard;
typedef struct {
    char *name;
    const uint8_t *data;
    uint64_t shape[4], bytes;
    int nd, dtype;
} DGTensor;
typedef struct {
    const char *name;
    void (*nvfp4_qdq)(float *, const float *, int, int, float);
    void (*nvfp4_layout)(float *, int, int);
    void (*nvfp4_mm)(const uint8_t *, const uint8_t *, float, const float *, float *, int, int,
                     int);
    void (*bf16_mm)(const uint8_t *, const float *, float *, int, int, int);
    void (*rms)(float *, const float *, const DGTensor *, int);
    double (*dot)(const float *, const float *, int);
} DGKernelOps;
static const DGKernelOps *dg_kernels(void);
typedef struct {
    DGTensor *wg, *sg, *gg, *ag, *wu, *su, *gu, *wd, *sd, *gd, *ad;
} DGNvExpert;
typedef struct {
    DGShard shard[11];
    DGTensor *tensor;
    size_t nt, cap;
    int nshard, nvfp4;
    DGNvExpert *nvexpert;
    float nv_a13[30], nv_a2[30];
} DGModel;

enum { DG_UNKNOWN = -1, DG_BF16, DG_U8, DG_F8E4M3, DG_F32 };
static uint64_t dg_item(int dtype) {
    return dtype == DG_BF16 ? 2 : dtype == DG_F32 ? 4 : 1;
}

typedef struct {
    Vocab *vocab, **by_id;
    Merge *merge;
    Vocab **special;
    uint32_t nv, nm, ns, maxlen;
    uint8_t special_first[256];
} DGTokenizer;

static uint64_t jt_u64(const char *j, const JTok *t, const char *name) {
    if (t->type != JT_PRIMITIVE || t->end <= t->start || t->end - t->start > 20)
        die2("expected unsigned integer", name);
    uint64_t v = 0;
    for (int i = t->start; i < t->end; i++) {
        unsigned d = (unsigned)(j[i] - '0');
        if (d > 9 || v > (UINT64_MAX - d) / 10)
            die2("bad unsigned integer", name);
        v = v * 10 + d;
    }
    return v;
}
static void dg_push_tensor(DGModel *m, DGTensor t) {
    if (m->nt == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 1024;
        m->tensor = xrealloc(m->tensor, m->cap * sizeof *m->tensor);
    }
    m->tensor[m->nt++] = t;
}
static int dg_tensor_cmp(const void *a, const void *b) {
    const DGTensor *x = a, *y = b;
    return strcmp(x->name, y->name);
}
static int dg_is_text(const char *name) {
    return !strncmp(name, "model.decoder.", 14) ||
           (!strncmp(name, "model.encoder.language_model.", 29) && strstr(name, ".layer_scalar"));
}
static void dg_parse_shard(DGModel *m, int si, const char *dir) {
    DGShard *s = &m->shard[si];
    jb_path(s->path, sizeof s->path, "%s/model-%05d-of-%05d.safetensors", dir, si + 1, m->nshard);
    map_file(&s->file, s->path);
    if (s->file.size < 9)
        die2("short safetensors file", s->path);
    uint64_t hn = 0;
    for (int i = 0; i < 8; i++)
        hn |= (uint64_t)s->file.map[i] << (8 * i);
    if (!hn || hn > s->file.size - 8 || hn > JB_MAX_JSON)
        die2("invalid safetensors header", s->path);
    const char *j = (const char *)s->file.map + 8;
    int nt = 0;
    JTok *tok = json_tokens(j, (size_t)hn, &nt);
    if (!nt || tok[0].type != JT_OBJECT)
        die2("invalid safetensors JSON", s->path);
    for (int key = 1; key < nt; key++) {
        if (tok[key].parent != 0 || tok[key].type != JT_STRING)
            continue;
        char *name = jt_string(j, &tok[key]);
        int obj = key + 1;
        if (!strcmp(name, "__metadata__")) {
            free(name);
            continue;
        }
        if (obj >= nt || tok[obj].type != JT_OBJECT)
            die2("bad safetensors descriptor", name);
        int dtype = jt_obj_get(j, tok, nt, obj, "dtype"),
            shape = jt_obj_get(j, tok, nt, obj, "shape"),
            offsets = jt_obj_get(j, tok, nt, obj, "data_offsets");
        if (dtype < 0 || shape < 0 || offsets < 0 || tok[shape].type != JT_ARRAY ||
            tok[offsets].type != JT_ARRAY)
            die2("incomplete safetensors descriptor", name);
        char *dt = jt_string(j, &tok[dtype]);
        DGTensor t = {0};
        t.name = name;
        t.dtype = DG_UNKNOWN;
        if (!strcmp(dt, "BF16"))
            t.dtype = DG_BF16;
        else if (!strcmp(dt, "U8"))
            t.dtype = DG_U8;
        else if (!strcmp(dt, "F8_E4M3"))
            t.dtype = DG_F8E4M3;
        else if (!strcmp(dt, "F32"))
            t.dtype = DG_F32;
        else if (dg_is_text(name))
            die2("unsupported DiffusionGemma text dtype", name);
        free(dt);
        if (dg_is_text(name) && t.dtype != DG_BF16 &&
            !(m->nvfp4 && strstr(name, ".experts.") &&
              (t.dtype == DG_U8 || t.dtype == DG_F8E4M3 || t.dtype == DG_F32)))
            die2("unsupported DiffusionGemma text tensor encoding", name);
        for (int k = shape + 1; k < nt && tok[k].parent >= shape; k++)
            if (tok[k].parent == shape) {
                if (t.nd == 4)
                    die2("too many safetensors dimensions", name);
                t.shape[t.nd++] = jt_u64(j, &tok[k], name);
            }
        uint64_t off[2] = {0};
        int no = 0;
        for (int k = offsets + 1; k < nt && tok[k].parent >= offsets; k++)
            if (tok[k].parent == offsets) {
                if (no == 2)
                    die2("bad safetensors offsets", name);
                off[no++] = jt_u64(j, &tok[k], name);
            }
        uint64_t base = 8 + hn;
        if (no != 2 || off[1] < off[0] || base > s->file.size || off[1] > s->file.size - base)
            die2("out-of-bounds safetensors data", name);
        t.data = s->file.map + base + off[0];
        t.bytes = off[1] - off[0];
        /* Every consumer indexes tensors by shape, so the byte span must
         * match it exactly or reads run past the tensor's data. */
        if (t.dtype != DG_UNKNOWN) {
            uint64_t count = 1;
            for (int i = 0; i < t.nd; i++) {
                if (t.shape[i] && count > UINT64_MAX / t.shape[i])
                    die2("safetensors tensor size overflow", name);
                count *= t.shape[i];
            }
            if (count > UINT64_MAX / dg_item(t.dtype) || t.bytes != count * dg_item(t.dtype))
                die2("safetensors byte size does not match shape", name);
        }
        dg_push_tensor(m, t);
    }
    free(tok);
}
static DGTensor *dg_tensor(DGModel *m, const char *name) {
    DGTensor key = {0};
    key.name = (char *)name;
    DGTensor *t = bsearch(&key, m->tensor, m->nt, sizeof *m->tensor, dg_tensor_cmp);
    if (t)
        return t;
    die2("missing DiffusionGemma tensor", name);
    return NULL;
}
static DGTensor *dg_expert_tensor(DGModel *m, int l, int e, const char *tail) {
    char n[192];
    snprintf(n, sizeof n, "model.decoder.layers.%d.experts.%d.%s", l, e, tail);
    return dg_tensor(m, n);
}
static void dg_expect(DGModel *m, const char *name, int dtype, int nd, uint64_t a, uint64_t b,
                      uint64_t c) {
    DGTensor *t = dg_tensor(m, name);
    if (t->dtype != dtype || t->nd != nd || (nd > 0 && t->shape[0] != a) ||
        (nd > 1 && t->shape[1] != b) || (nd > 2 && t->shape[2] != c))
        die2("unexpected DiffusionGemma tensor shape", name);
    uint64_t n = 1;
    for (int i = 0; i < nd; i++) {
        if (t->shape[i] && n > UINT64_MAX / t->shape[i])
            die2("DiffusionGemma tensor size overflow", name);
        n *= t->shape[i];
    }
    if (t->bytes != n * dg_item(dtype))
        die2("unexpected DiffusionGemma tensor byte size", name);
}
/* Vectors are read as exactly `count` elements; their rank is not relied on. */
static void dg_expect_count(DGModel *m, const char *name, int dtype, uint64_t count) {
    DGTensor *t = dg_tensor(m, name);
    if (t->dtype != dtype || t->bytes != count * dg_item(dtype))
        die2("unexpected DiffusionGemma tensor size", name);
}
static void dg_expect_layer(DGModel *m, int l, const char *tail, int nd, uint64_t a, uint64_t b,
                            uint64_t c) {
    char n[192];
    jb_path(n, sizeof n, "model.decoder.layers.%d.%s", l, tail);
    dg_expect(m, n, DG_BF16, nd, a, b, c);
}
static void dg_expect_layer_count(DGModel *m, int l, const char *tail, uint64_t count) {
    char n[192];
    jb_path(n, sizeof n, "model.decoder.layers.%d.%s", l, tail);
    dg_expect_count(m, n, DG_BF16, count);
}
static void dg_load(DGModel *m, const char *dir) {
    memset(m, 0, sizeof *m);
    char probe[768];
    jb_path(probe, sizeof probe, "%s/model-00001-of-00002.safetensors", dir);
    FILE *pf = fopen(probe, "rb");
    if (pf) {
        fclose(pf);
        m->nshard = 2;
        m->nvfp4 = 1;
    } else
        m->nshard = 11;
    for (int i = 0; i < m->nshard; i++)
        dg_parse_shard(m, i, dir);
    qsort(m->tensor, m->nt, sizeof *m->tensor, dg_tensor_cmp);
    for (size_t i = 1; i < m->nt; i++)
        if (!strcmp(m->tensor[i - 1].name, m->tensor[i].name))
            die2("duplicate DiffusionGemma tensor", m->tensor[i].name);
    dg_expect(m, "model.decoder.embed_tokens.weight", DG_BF16, 2, 262144, 2816, 0);
    dg_expect_count(m, "model.decoder.norm.weight", DG_BF16, 2816);
    /* Validate every layer, not a sample: kernels index these tensors by the
     * shapes below without further bounds checks. */
    for (int l = 0; l < 30; l++) {
        int full = l % 6 == 5, hd = full ? 512 : 256, kn = (full ? 2 : 8) * hd;
        dg_expect_layer(m, l, "self_attn.q_proj.weight", 2, 16u * hd, 2816, 0);
        dg_expect_layer(m, l, "self_attn.k_proj.weight", 2, kn, 2816, 0);
        if (!full)
            dg_expect_layer(m, l, "self_attn.v_proj.weight", 2, kn, 2816, 0);
        dg_expect_layer(m, l, "self_attn.o_proj.weight", 2, 2816, 16u * hd, 0);
        dg_expect_layer_count(m, l, "self_attn.q_norm.weight", hd);
        dg_expect_layer_count(m, l, "self_attn.k_norm.weight", hd);
        dg_expect_layer_count(m, l, "input_layernorm.weight", 2816);
        dg_expect_layer_count(m, l, "post_attention_layernorm.weight", 2816);
        dg_expect_layer_count(m, l, "pre_feedforward_layernorm.weight", 2816);
        dg_expect_layer_count(m, l, "post_feedforward_layernorm_1.weight", 2816);
        dg_expect_layer_count(m, l, "pre_feedforward_layernorm_2.weight", 2816);
        dg_expect_layer_count(m, l, "post_feedforward_layernorm_2.weight", 2816);
        dg_expect_layer_count(m, l, "post_feedforward_layernorm.weight", 2816);
        dg_expect_layer(m, l, "mlp.gate_proj.weight", 2, 2112, 2816, 0);
        dg_expect_layer(m, l, "mlp.up_proj.weight", 2, 2112, 2816, 0);
        dg_expect_layer(m, l, "mlp.down_proj.weight", 2, 2816, 2112, 0);
        dg_expect_layer(m, l, "router.proj.weight", 2, 128, 2816, 0);
        dg_expect_layer_count(m, l, "router.scale", 2816);
        dg_expect_layer_count(m, l, "router.per_expert_scale", 128);
        dg_expect_layer(m, l, "layer_scalar", 1, 1, 0, 0);
        if (!m->nvfp4) {
            dg_expect_layer(m, l, "experts.gate_up_proj", 3, 128, 1408, 2816);
            dg_expect_layer(m, l, "experts.down_proj", 3, 128, 2816, 704);
        }
    }
    if (m->nvfp4) {
        dg_expect(m, "model.decoder.layers.0.experts.0.gate_proj.weight", DG_U8, 2, 704, 1408, 0);
        dg_expect(m, "model.decoder.layers.0.experts.0.gate_proj.weight_scale", DG_F8E4M3, 2, 704,
                  176, 0);
        dg_expect(m, "model.decoder.layers.0.experts.0.gate_proj.weight_scale_2", DG_F32, 0, 0, 0,
                  0);
        dg_expect(m, "model.decoder.layers.0.experts.0.down_proj.weight", DG_U8, 2, 2816, 352, 0);
        m->nvexpert = xcalloc(30u * 128, sizeof *m->nvexpert);
        for (int l = 0; l < 30; l++)
            for (int e = 0; e < 128; e++) {
                DGNvExpert *v = &m->nvexpert[l * 128 + e];
                v->wg = dg_expert_tensor(m, l, e, "gate_proj.weight");
                v->sg = dg_expert_tensor(m, l, e, "gate_proj.weight_scale");
                v->gg = dg_expert_tensor(m, l, e, "gate_proj.weight_scale_2");
                v->ag = dg_expert_tensor(m, l, e, "gate_proj.input_scale");
                v->wu = dg_expert_tensor(m, l, e, "up_proj.weight");
                v->su = dg_expert_tensor(m, l, e, "up_proj.weight_scale");
                v->gu = dg_expert_tensor(m, l, e, "up_proj.weight_scale_2");
                v->wd = dg_expert_tensor(m, l, e, "down_proj.weight");
                v->sd = dg_expert_tensor(m, l, e, "down_proj.weight_scale");
                v->gd = dg_expert_tensor(m, l, e, "down_proj.weight_scale_2");
                v->ad = dg_expert_tensor(m, l, e, "down_proj.input_scale");
                if (v->ag->dtype != DG_F32 || v->ag->bytes != 4 || v->ad->dtype != DG_F32 ||
                    v->ad->bytes != 4)
                    die2("bad NVFP4 input scale", v->ag->name);
                float a13, a2;
                memcpy(&a13, v->ag->data, 4);
                memcpy(&a2, v->ad->data, 4);
                if (a13 > m->nv_a13[l])
                    m->nv_a13[l] = a13;
                if (a2 > m->nv_a2[l])
                    m->nv_a2[l] = a2;
            }
    }
}
static void dg_free(DGModel *m) {
    for (size_t i = 0; i < m->nt; i++)
        free(m->tensor[i].name);
    free(m->tensor);
    free(m->nvexpert);
    for (int i = 0; i < m->nshard; i++)
        unmap_file(&m->shard[i].file);
}

static Vocab *dgt_vfind(DGTokenizer *d, const char *s, size_t n) {
    Vocab k = {0, (char *)s, (uint32_t)n};
    return bsearch(&k, d->vocab, d->nv, sizeof *d->vocab, cmp_vocab);
}
static Merge *dgt_mfind(DGTokenizer *d, uint32_t a, uint32_t b) {
    Merge k = {a, b, 0};
    return bsearch(&k, d->merge, d->nm, sizeof *d->merge, cmp_merge);
}
static char *read_whole(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f)
        die2("cannot open", path);
    if (fseek(f, 0, SEEK_END) || (*n = (size_t)ftell(f)) > JB_MAX_JSON || fseek(f, 0, SEEK_SET))
        die2("invalid JSON file size", path);
    char *p = xmalloc(*n + 1);
    if (fread(p, 1, *n, f) != *n || fclose(f))
        die2("cannot read", path);
    p[*n] = 0;
    return p;
}
static void dgt_load(DGTokenizer *d, const char *dir) {
    memset(d, 0, sizeof *d);
    char path[768];
    jb_path(path, sizeof path, "%s/tokenizer.json", dir);
    size_t nj;
    char *j = read_whole(path, &nj);
    int nt;
    JTok *t = json_tokens(j, nj, &nt);
    int model = jt_obj_get(j, t, nt, 0, "model");
    int vocab = jt_obj_get(j, t, nt, model, "vocab");
    int merges = jt_obj_get(j, t, nt, model, "merges");
    int added = jt_obj_get(j, t, nt, 0, "added_tokens");
    if (model < 0 || vocab < 0 || merges < 0 || added < 0 || t[vocab].type != JT_OBJECT ||
        t[merges].type != JT_ARRAY || t[added].type != JT_ARRAY)
        die("unsupported DiffusionGemma tokenizer JSON");
    d->nv = (uint32_t)(t[vocab].size / 2);
    if (d->nv != 262144)
        die("unexpected DiffusionGemma vocabulary size");
    d->vocab = xcalloc(d->nv, sizeof *d->vocab);
    uint32_t vi = 0;
    for (int i = vocab + 1; i + 1 < nt; i++) {
        if (t[i].parent < vocab)
            break;
        if (t[i].parent != vocab)
            continue;
        if (vi == d->nv)
            die("too many DiffusionGemma vocabulary entries");
        d->vocab[vi].s = jt_string(j, &t[i]);
        d->vocab[vi].n = (uint32_t)strlen(d->vocab[vi].s);
        if (d->vocab[vi].n > d->maxlen)
            d->maxlen = d->vocab[vi].n;
        d->vocab[vi].id = jt_nonnegative_int(j, &t[i + 1], "vocabulary id");
        vi++;
        i++;
    }
    if (vi != d->nv)
        die("incomplete DiffusionGemma vocabulary");
    qsort(d->vocab, d->nv, sizeof *d->vocab, cmp_vocab);
    d->by_id = xcalloc(d->nv, sizeof *d->by_id);
    for (uint32_t i = 0; i < d->nv; i++) {
        int id = d->vocab[i].id;
        if (id < 0 || (uint32_t)id >= d->nv || d->by_id[id])
            die("invalid DiffusionGemma vocabulary id");
        d->by_id[id] = &d->vocab[i];
    }
    d->nm = (uint32_t)t[merges].size;
    d->merge = xcalloc(d->nm, sizeof *d->merge);
    uint32_t mi = 0;
    for (int i = merges + 1; i < nt; i++) {
        if (t[i].parent < merges)
            break;
        if (t[i].parent != merges)
            continue;
        int a = -1, b = -1;
        for (int k = i + 1; k < nt && t[k].parent >= i; k++)
            if (t[k].parent == i) {
                if (a < 0)
                    a = k;
                else if (b < 0)
                    b = k;
                else
                    die("bad tokenizer merge");
            }
        if (t[i].type != JT_ARRAY || a < 0 || b < 0)
            die("bad tokenizer merge");
        char *sa = jt_string(j, &t[a]), *sb = jt_string(j, &t[b]);
        Vocab *va = dgt_vfind(d, sa, strlen(sa)), *vb = dgt_vfind(d, sb, strlen(sb));
        free(sa);
        free(sb);
        if (!va || !vb || mi == d->nm)
            die("tokenizer merge references absent token");
        d->merge[mi] = (Merge){(uint32_t)va->id, (uint32_t)vb->id, mi};
        mi++;
    }
    if (mi != d->nm)
        die("incomplete tokenizer merges");
    qsort(d->merge, d->nm, sizeof *d->merge, cmp_merge);
    d->ns = (uint32_t)t[added].size;
    d->special = xcalloc(d->ns, sizeof *d->special);
    uint32_t si = 0;
    for (int i = added + 1; i < nt; i++) {
        if (t[i].parent < added)
            break;
        if (t[i].parent != added)
            continue;
        int content = jt_obj_get(j, t, nt, i, "content");
        int special = jt_obj_get(j, t, nt, i, "special");
        if (content < 0 || special < 0 || !jt_literal(j, &t[special], "true"))
            die("unsupported added token");
        char *s = jt_string(j, &t[content]);
        Vocab *v = dgt_vfind(d, s, strlen(s));
        free(s);
        if (!v)
            die("added token absent from vocabulary");
        if (!v->n)
            die("empty added token");
        d->special_first[(unsigned char)v->s[0]] = 1;
        d->special[si++] = v;
    }
    d->ns = si;
    free(t);
    free(j);
}
static void dgt_free(DGTokenizer *d) {
    for (uint32_t i = 0; i < d->nv; i++)
        free(d->vocab[i].s);
    free(d->vocab);
    free(d->by_id);
    free(d->merge);
    free(d->special);
}
typedef struct {
    uint32_t rank, pos, a, b;
} DGBpeCand;
static int dgt_bpe_less(const DGBpeCand *x, const DGBpeCand *y) {
    return x->rank != y->rank ? x->rank < y->rank : x->pos < y->pos;
}
static void dgt_bpe_push(DGTokenizer *d, DGBpeCand *heap, size_t *nh, const uint32_t *ids,
                         uint32_t i, uint32_t j) {
    Merge *m = dgt_mfind(d, ids[i], ids[j]);
    if (!m)
        return;
    size_t at = (*nh)++;
    heap[at] = (DGBpeCand){m->rank, i, ids[i], ids[j]};
    while (at && dgt_bpe_less(&heap[at], &heap[(at - 1) / 2])) {
        DGBpeCand z = heap[at];
        heap[at] = heap[(at - 1) / 2];
        heap[(at - 1) / 2] = z;
        at = (at - 1) / 2;
    }
}
static void dgt_piece(DGTokenizer *d, const char *s, size_t n, Tokens *out) {
    uint32_t *ids = xmalloc((n ? n : 1) * sizeof *ids);
    uint32_t k = 0;
    for (size_t p = 0; p < n;) {
        size_t q = p;
        next_cp((const unsigned char *)s, n, &q);
        Vocab *v = dgt_vfind(d, s + p, q - p);
        if (v)
            ids[k++] = (uint32_t)v->id;
        else {
            for (; p < q; p++) {
                char z[7];
                snprintf(z, sizeof z, "<0x%02X>", (unsigned char)s[p]);
                v = dgt_vfind(d, z, 6);
                if (!v)
                    die("DiffusionGemma byte fallback token absent");
                ids[k++] = (uint32_t)v->id;
            }
            continue;
        }
        p = q;
    }
    /* Repeatedly merge the lowest-rank adjacent pair, leftmost first. A heap
     * ordered by (rank, original position) with lazy invalidation picks the
     * same pair as a full rescan, in O(k log k) instead of O(k^2). */
    uint32_t *next = xmalloc((size_t)(k ? k : 1) * sizeof *next),
             *prev = xmalloc((size_t)(k ? k : 1) * sizeof *prev);
    for (uint32_t i = 0; i < k; i++) {
        next[i] = i + 1 < k ? i + 1 : UINT32_MAX;
        prev[i] = i ? i - 1 : UINT32_MAX;
    }
    DGBpeCand *heap = xmalloc((size_t)(k ? k : 1) * 3 * sizeof *heap);
    size_t nh = 0;
    for (uint32_t i = 0; i + 1 < k; i++)
        dgt_bpe_push(d, heap, &nh, ids, i, i + 1);
    while (nh) {
        DGBpeCand c = heap[0];
        heap[0] = heap[--nh];
        for (size_t i = 0;;) {
            size_t l = 2 * i + 1, r = l + 1, m = i;
            if (l < nh && dgt_bpe_less(&heap[l], &heap[m]))
                m = l;
            if (r < nh && dgt_bpe_less(&heap[r], &heap[m]))
                m = r;
            if (m == i)
                break;
            DGBpeCand z = heap[i];
            heap[i] = heap[m];
            heap[m] = z;
            i = m;
        }
        uint32_t at = c.pos, bt = next[at];
        if (ids[at] != c.a || bt == UINT32_MAX || ids[bt] != c.b)
            continue;
        Vocab *a = d->by_id[c.a], *b = d->by_id[c.b];
        char *z = xmalloc((size_t)a->n + b->n);
        memcpy(z, a->s, a->n);
        memcpy(z + a->n, b->s, b->n);
        Vocab *v = dgt_vfind(d, z, (size_t)a->n + b->n);
        free(z);
        if (!v)
            die("DiffusionGemma merged token absent");
        ids[at] = (uint32_t)v->id;
        ids[bt] = UINT32_MAX;
        next[at] = next[bt];
        if (next[at] != UINT32_MAX)
            prev[next[at]] = at;
        if (prev[at] != UINT32_MAX)
            dgt_bpe_push(d, heap, &nh, ids, prev[at], at);
        if (next[at] != UINT32_MAX)
            dgt_bpe_push(d, heap, &nh, ids, at, next[at]);
    }
    for (uint32_t i = 0; i < k; i = next[i])
        push(out, (int)ids[i]);
    free(heap);
    free(prev);
    free(next);
    free(ids);
}
static Tokens dgt_tokenize(DGTokenizer *d, const char *s) {
    Tokens out = {0};
    size_t n = strlen(s), plain = 0, i = 0;
    while (i < n) {
        Vocab *hit = NULL;
        if (!d->special_first[(unsigned char)s[i]]) {
            i++;
            continue;
        }
        for (uint32_t q = 0; q < d->ns; q++) {
            Vocab *v = d->special[q];
            if (v->n <= n - i && !memcmp(s + i, v->s, v->n) && (!hit || v->n > hit->n))
                hit = v;
        }
        if (!hit) {
            i++;
            continue;
        }
        if (i > plain) {
            size_t zcap = (i - plain) * 3 + 1, zlen = 0;
            char *z = xmalloc(zcap);
            for (size_t q = plain; q < i; q++) {
                if (s[q] == ' ')
                    memcpy(z + zlen, "\xE2\x96\x81", 3), zlen += 3;
                else
                    z[zlen++] = s[q];
            }
            dgt_piece(d, z, zlen, &out);
            free(z);
        }
        push(&out, hit->id);
        i += hit->n;
        plain = i;
    }
    if (plain < n) {
        size_t zcap = (n - plain) * 3 + 1, zlen = 0;
        char *z = xmalloc(zcap);
        for (size_t q = plain; q < n; q++) {
            if (s[q] == ' ')
                memcpy(z + zlen, "\xE2\x96\x81", 3), zlen += 3;
            else
                z[zlen++] = s[q];
        }
        dgt_piece(d, z, zlen, &out);
        free(z);
    }
    return out;
}
#define DG_H 2816
#define DG_L 30
#define DG_HEADS 16
#define DG_VOCAB 262144
#define DG_DENSE 2112
#define DG_MOE 704
#define DG_TOPK 8
typedef struct {
    float *k, *v;
    int n;
} DGKV;
typedef struct {
    char *schema;
    int *ids, n;
    DGKV kv[DG_L];
} DGPrefixCache;
typedef struct {
    uint32_t mt[624];
    int at;
} DGMT;
static void dg_mt_seed(DGMT *r, uint32_t seed) {
    r->mt[0] = 19650218u;
    for (int i = 1; i < 624; i++)
        r->mt[i] = 1812433253u * (r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) + i;
    int i = 1, j = 0;
    for (int q = 624; q; q--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1664525u)) + seed + j;
        i++;
        j++;
        if (i >= 624) {
            r->mt[0] = r->mt[623];
            i = 1;
        }
        if (j >= 1)
            j = 0;
    }
    for (int q = 623; q; q--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1566083941u)) - i;
        i++;
        if (i >= 624) {
            r->mt[0] = r->mt[623];
            i = 1;
        }
    }
    r->mt[0] = 0x80000000u;
    r->at = 624;
}
static uint32_t dg_mt_u32(DGMT *r) {
    if (r->at >= 624) {
        for (int i = 0; i < 624; i++) {
            uint32_t y = (r->mt[i] & 0x80000000u) | (r->mt[(i + 1) % 624] & 0x7fffffffu);
            r->mt[i] = r->mt[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfu : 0);
        }
        r->at = 0;
    }
    uint32_t y = r->mt[r->at++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}
static uint32_t dg_mt_vocab(DGMT *r) {
    uint32_t x;
    do {
        x = dg_mt_u32(r) >> 13;
    } while (x >= DG_VOCAB);
    return x;
}

static float dg_bf(const uint8_t *p) {
    uint32_t u = (uint32_t)p[0] << 16 | (uint32_t)p[1] << 24;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static float dg_at(const DGTensor *t, uint64_t i) {
    return dg_bf(t->data + i * 2);
}
static float dg_f8e4m3(uint8_t u) {
    int sign = u >> 7, e = (u >> 3) & 15, m = u & 7;
    float x = e ? ldexpf(1.0f + m / 8.0f, e - 7) : ldexpf((float)m, -9);
    return sign ? -x : x;
}
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
#if defined(JB_AVX2)
static void dg_nvfp4_qdq_avx2(float *out, const float *in, int tokens, int cols, float base) {
    if (!jb_finitef(base) || !(base > 0) || cols % 32)
        die("invalid NVFP4 activation scale");
    int nb = cols / 16, blocks = tokens * nb;
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static) if (blocks >= 256)
#endif
    for (int z = 0; z < blocks; z++) {
        int t = z / nb, b = z % nb;
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
#endif
#if defined(JB_AVX512)
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
#endif
/* Produces the activation layout the selected dg_nvfp4_mm kernel expects. */
static void dg_nvfp4_qdq(float *out, const float *in, int tokens, int cols, float base) {
    dg_kernels()->nvfp4_qdq(out, in, tokens, cols, base);
}
/* x in the dg_nvfp4_qdq_ref layout. */
static void dg_nvfp4_mm_ref(const uint8_t *wd, const uint8_t *sd, float global, const float *x,
                            float *y, int tokens, int rows, int cols) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r++)
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
#if defined(JB_AVX512)
/* x in the dg_nvfp4_swizzle layout; rows must be even. */
static void dg_nvfp4_mm_avx512(const uint8_t *wd, const uint8_t *sd, float global, const float *x,
                               float *y, int tokens, int rows, int cols) {
    static const float lut[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
    __m512 table = _mm512_loadu_ps(lut);
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r += 2) {
        const uint8_t *wp0 = wd + (uint64_t)r * cols / 2, *wp1 = wp0 + cols / 2;
        const uint8_t *sp0 = sd + (uint64_t)r * cols / 16, *sp1 = sp0 + cols / 16;
        for (int tb = 0; tb < tokens; tb += 8) {
            int nb = tokens - tb < 8 ? tokens - tb : 8;
            __m512 a[8], b[8];
            for (int q = 0; q < nb; q++)
                a[q] = b[q] = _mm512_setzero_ps();
            for (int c = 0; c < cols; c += 32) {
                __m512i z = _mm512_set1_epi32(15),
                        raw0 =
                            _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(wp0 + c / 2))),
                        raw1 =
                            _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(wp1 + c / 2)));
                __m512 wl0 = _mm512_permutexvar_ps(_mm512_and_si512(raw0, z), table),
                       wh0 = _mm512_permutexvar_ps(_mm512_srli_epi32(raw0, 4), table);
                __m512 wl1 = _mm512_permutexvar_ps(_mm512_and_si512(raw1, z), table),
                       wh1 = _mm512_permutexvar_ps(_mm512_srli_epi32(raw1, 4), table);
                float a0 = dg_f8e4m3(sp0[c / 16]) * global,
                      a1 = dg_f8e4m3(sp0[c / 16 + 1]) * global,
                      b0 = dg_f8e4m3(sp1[c / 16]) * global,
                      b1 = dg_f8e4m3(sp1[c / 16 + 1]) * global;
                __m512 sv0 = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(a0), _mm512_set1_ps(a1)),
                       sv1 = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(b0), _mm512_set1_ps(b1));
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
#endif
#if defined(JB_AVX2)
static float dg_hsum8(__m256 x);
static void dg_nvfp4_weights16_avx2(const uint8_t *q, float scale, __m256 *w0, __m256 *w1) {
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
static void dg_nvfp4_mm_avx2(const uint8_t *wd, const uint8_t *sd, float global, const float *x,
                             float *y, int tokens, int rows, int cols) {
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r++) {
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
#endif
/* x comes from dg_nvfp4_qdq, in the layout the selected kernel expects. */
static void dg_nvfp4_mm(const DGTensor *w, const DGTensor *s, const DGTensor *g, const float *x,
                        float *y, int tokens, int rows, int cols) {
    if (w->dtype != DG_U8 || w->nd != 2 || w->shape[0] != (uint64_t)rows ||
        w->shape[1] != (uint64_t)cols / 2 || s->dtype != DG_F8E4M3 || s->nd != 2 ||
        s->shape[0] != (uint64_t)rows || s->shape[1] != (uint64_t)cols / 16 || g->dtype != DG_F32 ||
        g->nd != 0 || g->bytes != 4 || cols % 32 || rows % 2)
        die2("bad NVFP4 expert tensor", w->name);
    float global;
    memcpy(&global, g->data, 4);
    dg_kernels()->nvfp4_mm(w->data, s->data, global, x, y, tokens, rows, cols);
}
#if defined(JB_AVX512)
static __m512 dg_bf16x16(const uint8_t *p) {
    __m256i h = _mm256_loadu_si256((const __m256i *)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}
static void dg_mm_data_avx512(const uint8_t *data, const float *x, float *y, int tokens, int rows,
                              int cols) {
    if (rows % 2)
        die("AVX-512 matrix row count must be even");
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r += 2) {
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
#endif
#if defined(JB_AVX2)
static __m256 dg_bf16x8(const uint8_t *p) {
    __m128i h = _mm_loadu_si128((const __m128i *)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}
static float dg_hsum8(__m256 x) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    h = _mm_hadd_ps(h, h);
    h = _mm_hadd_ps(h, h);
    return _mm_cvtss_f32(h);
}
static void dg_mm_data_avx2(const uint8_t *data, const float *x, float *y, int tokens, int rows,
                            int cols) {
    if (rows % 2)
        die("AVX2 matrix row count must be even");
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r += 2) {
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
#endif
static void dg_mm_data_ref(const uint8_t *data, const float *x, float *y, int tokens, int rows,
                           int cols) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r++) {
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
static void dg_mm_data(const uint8_t *data, const float *x, float *y, int tokens, int rows,
                       int cols) {
    if (tokens < 1 || tokens > JB_MAX_CTX)
        die("DiffusionGemma sequence exceeds context limit");
    dg_kernels()->bf16_mm(data, x, y, tokens, rows, cols);
}
static void dg_mm(const DGTensor *w, const float *x, float *y, int tokens, int rows, int cols) {
    if (w->nd != 2 || w->shape[0] != (uint64_t)rows || w->shape[1] != (uint64_t)cols)
        die2("bad matrix shape", w->name);
    dg_mm_data(w->data, x, y, tokens, rows, cols);
}
static void dg_mv_slice(const DGTensor *w, uint64_t base, const float *x, float *y, int rows,
                        int cols) {
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r++) {
        const uint8_t *p = w->data + (base + (uint64_t)r * cols) * 2;
        float s = 0.0f;
        for (int c = 0; c < cols; c++)
            s += dg_bf(p + c * 2) * x[c];
        y[r] = s;
    }
}
/* Project all answer slots together so the tied LM head is streamed once.
 * OpenJev's automatic reread test is entropy over the union of the
 * full-vocabulary top 20 and the explicitly requested label ids. */
static double dg_slot_logits_entropy(const DGTensor *w, const float *hidden, int n, int **label_ids,
                                     const int *label_n, double **label_score) {
    if (n < 1 || n > 128)
        die("bad DiffusionGemma answer slot count");
    /* vLLM returns top-20 plus the sorted union of requested label ids at
     * every slot, capped at its per-request 128-id limit. OpenJev computes
     * entropy over that entire returned set. Keep the 128 smallest unique
     * ids without allocating a vocabulary-sized bitmap. */
    int requested[128], nr = 0;
    for (int t = 0; t < n; t++)
        for (int c = 0; c < label_n[t]; c++) {
            int id = label_ids[t][c], at = 0;
            while (at < nr && requested[at] < id)
                at++;
            if (at < nr && requested[at] == id)
                continue;
            if (nr < 128) {
                memmove(requested + at + 1, requested + at, (size_t)(nr - at) * sizeof *requested);
                requested[at] = id;
                nr++;
            } else if (at < 128) {
                memmove(requested + at + 1, requested + at, (size_t)(127 - at) * sizeof *requested);
                requested[at] = id;
            }
        }
    float *logits = xmalloc((size_t)n * DG_VOCAB * sizeof *logits);
    dg_mm_data(w->data, hidden, logits, n, DG_VOCAB, DG_H);
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
            float z = 30.0f * tanhf(row[v] / 30.0f);
            row[v] = z;
            if (z > mx)
                mx = z;
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
        double den = 0;
        for (int v = 0; v < DG_VOCAB; v++)
            den += exp((double)row[v] - mx);
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
    free(logits);
    return max_entropy;
}
static void dg_rms_ref(float *y, const float *x, const DGTensor *scale, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++)
        ss += (double)x[i] * x[i];
    float q = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    for (int i = 0; i < n; i++)
        y[i] = x[i] * q * (scale ? dg_at(scale, i) : 1.0f);
}
#if defined(JB_AVX512)
static void dg_rms_avx512(float *y, const float *x, const DGTensor *scale, int n) {
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
#endif
#if defined(JB_AVX2)
static double dg_hsum4d(__m256d x) {
    __m128d h = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
    h = _mm_hadd_pd(h, h);
    return _mm_cvtsd_f64(h);
}
static void dg_rms_avx2(float *y, const float *x, const DGTensor *scale, int n) {
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
#endif
static void dg_rms(float *y, const float *x, const DGTensor *scale, int n) {
    dg_kernels()->rms(y, x, scale, n);
}
static float dg_gelu(float x) {
    return .5f * x * (1.0f + tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
}
static double dg_dot_ref(const float *a, const float *b, int n) {
    double s = 0;
    for (int i = 0; i < n; i++)
        s += (double)a[i] * b[i];
    return s;
}
#if defined(JB_AVX512)
static double dg_dot_avx512(const float *a, const float *b, int n) {
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
#if defined(JB_AVX2)
static double dg_dot_avx2(const float *a, const float *b, int n) {
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
static double dg_dot(const float *a, const float *b, int n) {
    return dg_kernels()->dot(a, b, n);
}
static const DGKernelOps *dg_kernels(void) {
#if defined(JB_AVX512)
    static const DGKernelOps selected = {"avx512",           dg_nvfp4_qdq_avx512, dg_nvfp4_swizzle,
                                         dg_nvfp4_mm_avx512, dg_mm_data_avx512,   dg_rms_avx512,
                                         dg_dot_avx512};
    return &selected;
#elif defined(JB_AVX2)
    static const DGKernelOps selected = {"avx2",           dg_nvfp4_qdq_avx2, NULL,
                                         dg_nvfp4_mm_avx2, dg_mm_data_avx2,   dg_rms_avx2,
                                         dg_dot_avx2};
    return &selected;
#else
    static const DGKernelOps scalar = {"scalar",       dg_nvfp4_qdq_ref, NULL,      dg_nvfp4_mm_ref,
                                       dg_mm_data_ref, dg_rms_ref,       dg_dot_ref};
    return &scalar;
#endif
}
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
static void dg_attention(DGModel *m, int l, float *x, int n, int pos0, const DGKV *cache, DGKV *out,
                         int mode, int batch) {
    JB_TICK(qkv_start);
    int full = l % 6 == 5, hd = full ? 512 : 256, kvh = full ? 2 : 8;
    int qn = DG_HEADS * hd, kn = kvh * hd;
    DGTensor *qw = dg_layer_tensor(m, l, "self_attn.q_proj.weight");
    DGTensor *kw = dg_layer_tensor(m, l, "self_attn.k_proj.weight");
    DGTensor *vw = full ? NULL : dg_layer_tensor(m, l, "self_attn.v_proj.weight");
    DGTensor *qnrm = dg_layer_tensor(m, l, "self_attn.q_norm.weight");
    DGTensor *knrm = dg_layer_tensor(m, l, "self_attn.k_norm.weight");
    float *q = xmalloc((size_t)n * qn * 4), *k = xmalloc((size_t)n * kn * 4),
          *v = xmalloc((size_t)n * kn * 4);
    dg_mm(qw, x, q, n, qn, DG_H);
    dg_mm(kw, x, k, n, kn, DG_H);
    if (vw)
        dg_mm(vw, x, v, n, kn, DG_H);
    else
        memcpy(v, k, (size_t)n * kn * 4);
    JB_TO(attention_qkv, qkv_start);
    JB_TICK(prepare_start);
    int half = hd / 2, rotated = full ? 64 : half;
    float inv[256];
    for (int i = 0; i < half; i++)
        inv[i] = i < rotated ? powf(full ? 1000000.0f : 10000.0f, -(float)(2 * i) / hd) : 0.0f;
    int seq = n / batch;
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n; t++) {
        int pos = t % seq;
        float cv[256], sv[256];
        for (int i = 0; i < half; i++) {
            cv[i] = cosf((pos0 + pos) * inv[i]);
            sv[i] = sinf((pos0 + pos) * inv[i]);
        }
        dg_norm_heads(q + (size_t)t * qn, DG_HEADS, hd, qnrm);
        dg_norm_heads(k + (size_t)t * kn, kvh, hd, knrm);
        dg_norm_heads(v + (size_t)t * kn, kvh, hd, NULL);
        dg_rope(q + (size_t)t * qn, DG_HEADS, hd, cv, sv);
        dg_rope(k + (size_t)t * kn, kvh, hd, cv, sv);
    }
    JB_TO(attention_prepare, prepare_start);
    JB_TICK(kv_start);
    int old = mode != DG_CAUSAL && cache ? cache->n : 0;
    float *a = xcalloc((size_t)n * qn, 4), *score = xmalloc((size_t)n * (old + seq) * 4);
    JB_TO(attention_kv, kv_start);
    JB_TICK(core_start);
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int ti = 0; ti < n; ti++)
        for (int h = 0; h < DG_HEADS; h++) {
            int b = ti / seq, t = ti % seq;
            float *tscore = score + (size_t)ti * (old + seq);
            int kh = h / (DG_HEADS / kvh), end = mode == DG_CANVAS ? old + seq : old + t + 1,
                start = 0;
            if (!full) {
                if (mode == DG_CANVAS) {
                    /* OpenJev's non-causal decoder symmetrizes Gemma's 1024-token
                     * window around each absolute query position. */
                    int pos = old + t;
                    start = pos >= 1024 ? pos - 1024 + 1 : 0;
                    if (end > pos + 1024)
                        end = pos + 1024;
                } else
                    start = end > 1024 ? end - 1024 : 0;
            }
            float mx = -FLT_MAX;
            const float *qq = q + (size_t)ti * qn + (size_t)h * hd;
            for (int j = start; j < end; j++) {
                const float *kk = j < old ? cache->k + (size_t)j * kn + (size_t)kh * hd
                                          : k + ((size_t)b * seq + j - old) * kn + (size_t)kh * hd;
                tscore[j] = (float)dg_dot(qq, kk, hd);
                if (tscore[j] > mx)
                    mx = tscore[j];
            }
            float den = 0;
            for (int j = start; j < end; j++)
                den += expf(tscore[j] - mx);
            float *oo = a + (size_t)ti * qn + (size_t)h * hd;
            for (int j = start; j < end; j++) {
                float p = expf(tscore[j] - mx) / den;
                const float *vv = j < old ? cache->v + (size_t)j * kn + (size_t)kh * hd
                                          : v + ((size_t)b * seq + j - old) * kn + (size_t)kh * hd;
                for (int d = 0; d < hd; d++)
                    oo[d] += p * vv[d];
            }
        }
    JB_TO(attention_core, core_start);
    JB_TICK(output_start);
    DGTensor *ow = dg_layer_tensor(m, l, "self_attn.o_proj.weight");
    dg_mm(ow, a, x, n, DG_H, qn);
    JB_TO(attention_output, output_start);
    if (out) {
        out->n = old + n;
        if (!old) {
            out->k = k;
            out->v = v;
            k = v = NULL;
        } else {
            out->k = xmalloc((size_t)(old + n) * kn * 4);
            out->v = xmalloc((size_t)(old + n) * kn * 4);
            memcpy(out->k, cache->k, (size_t)old * kn * 4);
            memcpy(out->v, cache->v, (size_t)old * kn * 4);
            memcpy(out->k + (size_t)old * kn, k, (size_t)n * kn * 4);
            memcpy(out->v + (size_t)old * kn, v, (size_t)n * kn * 4);
        }
    }
    free(q);
    free(k);
    free(v);
    free(a);
    free(score);
}
/* Execute independent equal-stride sequences through one set of projections.
 * Attention never crosses a segment boundary.  Suffix segments may have
 * different useful lengths; padding is computed but never copied into K/V. */
static void dg_attention_multi(DGModel *m, int l, float *x, int segments, int seq,
                               const DGKV *const *cache, const int *lens, DGKV *out, int mode) {
    int n = segments * seq;
    JB_TICK(qkv_start);
    int full = l % 6 == 5, hd = full ? 512 : 256, kvh = full ? 2 : 8, qn = DG_HEADS * hd,
        kn = kvh * hd;
    DGTensor *qw = dg_layer_tensor(m, l, "self_attn.q_proj.weight");
    DGTensor *kw = dg_layer_tensor(m, l, "self_attn.k_proj.weight");
    DGTensor *vw = full ? NULL : dg_layer_tensor(m, l, "self_attn.v_proj.weight");
    DGTensor *qnrm = dg_layer_tensor(m, l, "self_attn.q_norm.weight");
    DGTensor *knrm = dg_layer_tensor(m, l, "self_attn.k_norm.weight");
    float *q = xmalloc((size_t)n * qn * 4), *k = xmalloc((size_t)n * kn * 4),
          *v = xmalloc((size_t)n * kn * 4);
    dg_mm(qw, x, q, n, qn, DG_H);
    dg_mm(kw, x, k, n, kn, DG_H);
    if (vw)
        dg_mm(vw, x, v, n, kn, DG_H);
    else
        memcpy(v, k, (size_t)n * kn * 4);
    JB_TO(attention_qkv, qkv_start);
    JB_TICK(prepare_start);
    int half = hd / 2, rotated = full ? 64 : half;
    float inv[256];
    for (int i = 0; i < half; i++)
        inv[i] = i < rotated ? powf(full ? 1000000.0f : 10000.0f, -(float)(2 * i) / hd) : 0;
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int ti = 0; ti < n; ti++) {
        int s = ti / seq, t = ti % seq, pos = cache[s]->n + t;
        float cv[256], sv[256];
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
    JB_TO(attention_prepare, prepare_start);
    JB_TICK(kv_start);
    int maxold = 0;
    for (int s = 0; s < segments; s++)
        if (cache[s]->n > maxold)
            maxold = cache[s]->n;
    float *a = xcalloc((size_t)n * qn, 4), *score = xmalloc((size_t)n * (maxold + seq) * 4);
    JB_TO(attention_kv, kv_start);
    JB_TICK(core_start);
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int ti = 0; ti < n; ti++)
        for (int h = 0; h < DG_HEADS; h++) {
            int s = ti / seq, t = ti % seq, old = cache[s]->n, kh = h / (DG_HEADS / kvh),
                end = mode == DG_CANVAS ? old + seq : old + t + 1, start = 0;
            if (!full) {
                if (mode == DG_CANVAS) {
                    int pos = old + t;
                    start = pos >= 1024 ? pos - 1024 + 1 : 0;
                    if (end > pos + 1024)
                        end = pos + 1024;
                } else
                    start = end > 1024 ? end - 1024 : 0;
            }
            float *ts = score + (size_t)ti * (maxold + seq), mx = -FLT_MAX;
            const float *qq = q + (size_t)ti * qn + (size_t)h * hd;
            for (int j = start; j < end; j++) {
                const float *kk = j < old ? cache[s]->k + (size_t)j * kn + (size_t)kh * hd
                                          : k + ((size_t)s * seq + j - old) * kn + (size_t)kh * hd;
                ts[j] = (float)dg_dot(qq, kk, hd);
                if (ts[j] > mx)
                    mx = ts[j];
            }
            float den = 0;
            for (int j = start; j < end; j++)
                den += expf(ts[j] - mx);
            float *oo = a + (size_t)ti * qn + (size_t)h * hd;
            for (int j = start; j < end; j++) {
                float p = expf(ts[j] - mx) / den;
                const float *vv = j < old ? cache[s]->v + (size_t)j * kn + (size_t)kh * hd
                                          : v + ((size_t)s * seq + j - old) * kn + (size_t)kh * hd;
                for (int d = 0; d < hd; d++)
                    oo[d] += p * vv[d];
            }
        }
    JB_TO(attention_core, core_start);
    JB_TICK(output_start);
    DGTensor *ow = dg_layer_tensor(m, l, "self_attn.o_proj.weight");
    dg_mm(ow, a, x, n, DG_H, qn);
    JB_TO(attention_output, output_start);
    if (out)
        for (int s = 0; s < segments; s++) {
            int old = cache[s]->n, keep = lens[s];
            out[s].n = old + keep;
            out[s].k = xmalloc((size_t)(old + keep) * kn * 4);
            out[s].v = xmalloc((size_t)(old + keep) * kn * 4);
            memcpy(out[s].k, cache[s]->k, (size_t)old * kn * 4);
            memcpy(out[s].v, cache[s]->v, (size_t)old * kn * 4);
            memcpy(out[s].k + (size_t)old * kn, k + (size_t)s * seq * kn, (size_t)keep * kn * 4);
            memcpy(out[s].v + (size_t)old * kn, v + (size_t)s * seq * kn, (size_t)keep * kn * 4);
        }
    free(q);
    free(k);
    free(v);
    free(a);
    free(score);
}
static void dg_ff(DGModel *m, int l, float *x, int n) {
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
    float *z1 = xcalloc((size_t)n * DG_H, 4), *g = xmalloc((size_t)n * DG_DENSE * 4),
          *u = xmalloc((size_t)n * DG_DENSE * 4), *d = xmalloc((size_t)n * DG_H * 4),
          *z2 = xmalloc((size_t)n * DG_H * 4), *rin = xcalloc((size_t)n * DG_H, 4),
          *route = xmalloc((size_t)n * 128 * 4);
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n; t++)
        dg_rms(z1 + (size_t)t * DG_H, x + (size_t)t * DG_H, pre, DG_H);
    dg_mm(gw, z1, g, n, DG_DENSE, DG_H);
    dg_mm(uw, z1, u, n, DG_DENSE, DG_H);
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < (size_t)n * DG_DENSE; i++)
        g[i] = dg_gelu(g[i]) * u[i];
    dg_mm(dw, g, d, n, DG_H, DG_DENSE);
    JB_TO(dense, ff_start);
    JB_TICK(router_start);
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n; t++) {
        float *r = x + (size_t)t * DG_H;
        dg_rms(d + (size_t)t * DG_H, d + (size_t)t * DG_H, p1, DG_H);
        dg_rms(z2 + (size_t)t * DG_H, r, pre2, DG_H);
        dg_rms(rin + (size_t)t * DG_H, r, NULL, DG_H);
        for (int i = 0; i < DG_H; i++)
            rin[(size_t)t * DG_H + i] *= dg_at(rs, i) / sqrtf(DG_H);
    }
    dg_mm(rw, rin, route, n, 128, DG_H);
    for (size_t i = 0; i < (size_t)n * 128; i++)
        if (!jb_finitef(route[i]))
            die("non-finite DiffusionGemma router logits");
    int *top = xmalloc((size_t)n * DG_TOPK * sizeof *top);
    float *tw = xmalloc((size_t)n * DG_TOPK * 4);
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n; t++) {
        float *rt = route + (size_t)t * 128;
        int ix[DG_TOPK];
        float ev[DG_TOPK];
        for (int k = 0; k < DG_TOPK; k++) {
            ix[k] = k;
            ev[k] = -FLT_MAX;
        }
        for (int e = 0; e < 128; e++)
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
        for (int e = 1; e < 128; e++)
            if (rt[e] > mx)
                mx = rt[e];
        double all = 0, sel = 0;
        for (int e = 0; e < 128; e++)
            all += exp((double)rt[e] - mx);
        for (int k = 0; k < DG_TOPK; k++) {
            ev[k] = (float)(exp((double)ev[k] - mx) / all);
            sel += ev[k];
        }
        for (int k = 0; k < DG_TOPK; k++) {
            top[(size_t)t * DG_TOPK + k] = ix[k];
            tw[(size_t)t * DG_TOPK + k] = ev[k] / (float)sel * dg_at(re, ix[k]);
        }
    }
    JB_TO(router, router_start);
    JB_TICK(expert_start);
    float *contrib = xcalloc((size_t)n * DG_TOPK * DG_H, 4),
          *gather = xmalloc((size_t)n * DG_H * 4), *gu = xmalloc((size_t)n * 1408 * 4),
          *hid = xmalloc((size_t)n * DG_MOE * 4), *eo = xmalloc((size_t)n * DG_H * 4),
          *qz2 = m->nvfp4 ? xmalloc((size_t)n * DG_H * 4) : NULL;
    int *owner = xmalloc((size_t)n * 2 * sizeof *owner);
    if (qz2) {
        JB_TICK(input_qdq_start);
        dg_nvfp4_qdq(qz2, z2, n, DG_H, m->nv_a13[l]);
        JB_TO(moe_input_qdq, input_qdq_start);
    }
    for (int e = 0; e < 128; e++) {
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
            DGNvExpert *v = &m->nvexpert[l * 128 + e];
            float *qh = xmalloc((size_t)ne * DG_MOE * 4);
            JB_TICK(gate_start);
            dg_nvfp4_mm(v->wg, v->sg, v->gg, gather, gu, ne, DG_MOE, DG_H);
            JB_TO(moe_gate, gate_start);
            JB_TICK(up_start);
            dg_nvfp4_mm(v->wu, v->su, v->gu, gather, hid, ne, DG_MOE, DG_H);
            JB_TO(moe_up, up_start);
            JB_TICK(act_start);
#ifdef _OPENMP
            JB_OMP();
#pragma omp parallel for schedule(static) if (ne * DG_MOE >= 4096)
#endif
            for (int qi = 0; qi < ne * DG_MOE; qi++)
                hid[qi] = dg_gelu(gu[qi]) * hid[qi];
            JB_TO(moe_activation, act_start);
            JB_TICK(hidden_qdq_start);
            dg_nvfp4_qdq(qh, hid, ne, DG_MOE, m->nv_a2[l]);
            JB_TO(moe_hidden_qdq, hidden_qdq_start);
            JB_TICK(down_start);
            dg_nvfp4_mm(v->wd, v->sd, v->gd, qh, eo, ne, DG_H, DG_MOE);
            JB_TO(moe_down, down_start);
            free(qh);
        } else {
            const uint8_t *gp = eg->data + (uint64_t)e * 1408 * DG_H * 2,
                          *dp = ed->data + (uint64_t)e * DG_H * DG_MOE * 2;
            dg_mm_data(gp, gather, gu, ne, 1408, DG_H);
            for (int q = 0; q < ne; q++)
                for (int i = 0; i < DG_MOE; i++)
                    hid[(size_t)q * DG_MOE + i] =
                        dg_gelu(gu[(size_t)q * 1408 + i]) * gu[(size_t)q * 1408 + DG_MOE + i];
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
    float *mo = xmalloc(DG_H * 4), *sum = xmalloc(DG_H * 4);
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
    free(z1);
    free(g);
    free(u);
    free(d);
    free(z2);
    free(rin);
    free(route);
    free(top);
    free(tw);
    free(contrib);
    free(gather);
    free(gu);
    free(hid);
    free(eo);
    free(qz2);
    free(owner);
    free(mo);
    free(sum);
    JB_TO(ff_other, ff_tail_start);
}
static void dg_layer(DGModel *m, int l, float *x, int n, int pos0, const DGKV *cache, DGKV *out,
                     int mode, int batch) {
    JB_TICK(layer_start);
    DGTensor *in = dg_layer_tensor(m, l, "input_layernorm.weight"),
             *pa = dg_layer_tensor(m, l, "post_attention_layernorm.weight");
    float *res = xmalloc((size_t)n * DG_H * 4), *z = xmalloc((size_t)n * DG_H * 4);
    memcpy(res, x, (size_t)n * DG_H * 4);
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n; t++)
        dg_rms(z + (size_t)t * DG_H, x + (size_t)t * DG_H, in, DG_H);
    JB_TICK(attention_start);
    dg_attention(m, l, z, n, pos0, cache, out, mode, batch);
    JB_TO(attention, attention_start);
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n; t++) {
        dg_rms(x + (size_t)t * DG_H, z + (size_t)t * DG_H, pa, DG_H);
        for (int i = 0; i < DG_H; i++)
            x[(size_t)t * DG_H + i] += res[(size_t)t * DG_H + i];
    }
    dg_ff(m, l, x, n);
    float sc = dg_at(dg_layer_tensor(m, l, "layer_scalar"), 0);
    for (size_t i = 0; i < (size_t)n * DG_H; i++)
        x[i] *= sc;
    free(res);
    free(z);
    JB_TO(layers, layer_start);
}
static void dg_layer_multi(DGModel *m, int l, float *x, int segments, int seq,
                           const DGKV *const *cache, const int *lens, DGKV *out, int mode) {
    int n = segments * seq;
    JB_TICK(layer_start);
    DGTensor *in = dg_layer_tensor(m, l, "input_layernorm.weight"),
             *pa = dg_layer_tensor(m, l, "post_attention_layernorm.weight");
    float *res = xmalloc((size_t)n * DG_H * 4), *z = xmalloc((size_t)n * DG_H * 4);
    memcpy(res, x, (size_t)n * DG_H * 4);
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n; t++)
        dg_rms(z + (size_t)t * DG_H, x + (size_t)t * DG_H, in, DG_H);
    JB_TICK(attention_start);
    dg_attention_multi(m, l, z, segments, seq, cache, lens, out, mode);
    JB_TO(attention, attention_start);
#ifdef _OPENMP
    JB_OMP();
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n; t++) {
        dg_rms(x + (size_t)t * DG_H, z + (size_t)t * DG_H, pa, DG_H);
        for (int i = 0; i < DG_H; i++)
            x[(size_t)t * DG_H + i] += res[(size_t)t * DG_H + i];
    }
    dg_ff(m, l, x, n);
    float sc = dg_at(dg_layer_tensor(m, l, "layer_scalar"), 0);
    for (size_t i = 0; i < (size_t)n * DG_H; i++)
        x[i] *= sc;
    free(res);
    free(z);
    JB_TO(layers, layer_start);
}
static float *dg_embed(DGModel *m, const int *ids, int n, int decoder) {
    DGTensor *w = dg_tensor(m, "model.decoder.embed_tokens.weight");
    float *x = xmalloc((size_t)n * DG_H * 4);
    float scale = sqrtf(DG_H);
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < DG_H; i++)
            x[(size_t)t * DG_H + i] = dg_at(w, (uint64_t)ids[t] * DG_H + i) * scale;
        if (decoder)
            dg_rms(x + (size_t)t * DG_H, x + (size_t)t * DG_H, NULL, DG_H);
    }
    return x;
}
static void dg_final_norm(DGModel *m, float *x, int n) {
    DGTensor *w = dg_tensor(m, "model.decoder.norm.weight");
    for (int t = 0; t < n; t++)
        dg_rms(x + (size_t)t * DG_H, x + (size_t)t * DG_H, w, DG_H);
}
static void dg_prefill(DGModel *m, const int *prompt, int np, DGKV kv[DG_L]) {
    JB_TICK(embed_start);
    float *enc = dg_embed(m, prompt, np, 0);
    JB_TO(embedding, embed_start);
    for (int l = 0; l < DG_L; l++)
        dg_layer(m, l, enc, np, 0, NULL, &kv[l], DG_CAUSAL, 1);
    free(enc);
}
static void dg_prefill_suffix(DGModel *m, const int *token, int n, const DGKV prefix[DG_L],
                              DGKV kv[DG_L]) {
    JB_TICK(embed_start);
    float *x = dg_embed(m, token, n, 0);
    JB_TO(embedding, embed_start);
    for (int l = 0; l < DG_L; l++)
        dg_layer(m, l, x, n, prefix[l].n, &prefix[l], &kv[l], DG_SUFFIX, 1);
    free(x);
}
static float *dg_decode(DGModel *m, DGKV kv[DG_L], int np, const int *canvas, int nc, int batch) {
    JB_TICK(embed_start);
    float *dec = dg_embed(m, canvas, nc * batch, 1);
    JB_TO(embedding, embed_start);
    for (int l = 0; l < DG_L; l++)
        dg_layer(m, l, dec, nc * batch, np, &kv[l], NULL, DG_CANVAS, batch);
    JB_TICK(norm_start);
    dg_final_norm(m, dec, nc * batch);
    JB_TO(final_norm, norm_start);
    return dec;
}
static void dg_prefill_suffix_multi(DGModel *m, const int *token, const int *lens, int batch,
                                    int seq, const DGKV prefix[DG_L], DGKV *kv) {
    JB_TICK(embed_start);
    float *x = dg_embed(m, token, batch * seq, 0);
    JB_TO(embedding, embed_start);
    const DGKV **cache = xmalloc((size_t)batch * sizeof *cache);
    DGKV *out = xmalloc((size_t)batch * sizeof *out);
    for (int l = 0; l < DG_L; l++) {
        for (int b = 0; b < batch; b++)
            cache[b] = &prefix[l];
        dg_layer_multi(m, l, x, batch, seq, cache, lens, out, DG_SUFFIX);
        for (int b = 0; b < batch; b++)
            kv[(size_t)b * DG_L + l] = out[b];
    }
    free(out);
    free(cache);
    free(x);
}
static float *dg_decode_multi(DGModel *m, DGKV *kv, const int *doc, int segments, int seq,
                              const int *canvas) {
    int n = segments * seq;
    JB_TICK(embed_start);
    float *x = dg_embed(m, canvas, n, 1);
    JB_TO(embedding, embed_start);
    const DGKV **cache = xmalloc((size_t)segments * sizeof *cache);
    int *lens = xmalloc((size_t)segments * sizeof *lens);
    for (int s = 0; s < segments; s++)
        lens[s] = seq;
    for (int l = 0; l < DG_L; l++) {
        for (int s = 0; s < segments; s++)
            cache[s] = &kv[(size_t)doc[s] * DG_L + l];
        dg_layer_multi(m, l, x, segments, seq, cache, lens, NULL, DG_CANVAS);
    }
    free(lens);
    free(cache);
    JB_TICK(norm_start);
    dg_final_norm(m, x, n);
    JB_TO(final_norm, norm_start);
    return x;
}
static void dg_free_kv(DGKV kv[DG_L]) {
    JB_TICK(kv_free_start);
    for (int l = 0; l < DG_L; l++) {
        free(kv[l].k);
        free(kv[l].v);
    }
    JB_TO(kv_free, kv_free_start);
}
static void dg_prefix_free(DGPrefixCache *c) {
    if (!c)
        return;
    dg_free_kv(c->kv);
    free(c->schema);
    free(c->ids);
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
static void db_fmt(DGBuf *b, const char *fmt, ...) {
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
        free(s);
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
        free(k[i].key);
    }
    db_ch(b, '}');
    free(k);
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
    free(msg);
    for (int i = 0; i < 8; i++)
        for (int q = 0; q < 4; q++)
            out[i * 4 + q] = (uint8_t)(h[i] >> (24 - 8 * q));
}
static void json_print_string(const char *s) {
    putchar('\"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
        case '\"':
            fputs("\\\"", stdout);
            break;
        case '\\':
            fputs("\\\\", stdout);
            break;
        case '\n':
            fputs("\\n", stdout);
            break;
        case '\r':
            fputs("\\r", stdout);
            break;
        case '\t':
            fputs("\\t", stdout);
            break;
        default:
            if (c < 32)
                printf("\\u%04x", c);
            else
                putchar(c);
        }
    }
    putchar('\"');
}
static void json_print_token(const char *j, const JTok *t) {
    if (t->type == JT_STRING) {
        char *s = jt_string(j, t);
        json_print_string(s);
        free(s);
    } else
        fwrite(j + t->start, 1, (size_t)(t->end - t->start), stdout);
}

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
static int dg_choice_inventory(DGTokenizer *tok, char *out[128]) {
    Tokens base = dgt_tokenize(tok, "q1: A");
    int seen[128], n = 0;
    for (int ci = 0; ci < 728 && n < 128; ci++) {
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
        free(e.v);
    }
    free(base.v);
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
        free(ins);
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
            free(desc);
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
static char *dg_answer_text(DecisionWork *w, int nq, const int *pick) {
    return dg_answer_text_range(w, nq, 0, nq, pick);
}
static void dg_print_answers(const char *qj, JTok *qt, int qnt, DecisionWork *w, int nq,
                             const char *id, uint32_t tokens, double ms, double prefill_ms,
                             double cand_ms, int reads, int canvases, const char *cache_state,
                             int cache_tokens, int processed_tokens, int microbatch) {
    printf(
        "{\"model\":\"jev-bush-diffusion-%s\",\"math\":\"%s\",\"kernels\":\"%s\",\"threads\":%d,",
        JB_VERSION, JB_MATH_MODE, dg_kernels()->name, jb_threads());
    if (id) {
        fputs("\"id\":", stdout);
        json_print_string(id);
        putchar(',');
    }
    fputs("\"answers\":{", stdout);
    for (int x = 0; x < nq; x++) {
        DecisionWork *d = &w[x];
        if (x)
            putchar(',');
        char *k = jt_string(qj, &qt[d->key]);
        json_print_string(k);
        free(k);
        fputs(":{\"type\":", stdout);
        json_print_string(d->kind);
        if (!strcmp(d->kind, "noul"))
            printf(",\"noul\":%.17g,\"probabilities\":{\"true\":%.17g,\"false\":%.17g},"
                   "\"confidence\":%.17g",
                   d->prob[0], d->prob[0], d->prob[1], confidence(d->prob, d->nc));
        else if (!strcmp(d->kind, "choice")) {
            int top = 0;
            for (int i = 1; i < d->nc; i++)
                if (d->prob[i] > d->prob[top])
                    top = i;
            fputs(",\"choice\":", stdout);
            json_print_string(d->cand[top]);
            fputs(",\"probabilities\":{", stdout);
            for (int i = 0; i < d->nc; i++) {
                if (i)
                    putchar(',');
                json_print_string(d->cand[i]);
                printf(":%.17g", d->prob[i]);
            }
            printf("},\"confidence\":%.17g", confidence(d->prob, d->nc));
        } else {
            double ev = 0;
            for (int i = 0; i < d->nc; i++)
                ev += i * d->prob[i];
            printf(",\"score\":%.17g,\"legend\":{", ev);
            int zc = 0;
            for (int z = d->criteria + 1; z < qnt; z++)
                if (qt[z].parent == d->criteria) {
                    printf("%s\"%d\":", zc ? "," : "", zc);
                    zc++;
                    json_print_token(qj, &qt[z]);
                }
            fputs("},\"probabilities\":{", stdout);
            for (int i = 0; i < d->nc; i++)
                printf("%s\"%d\":%.17g", i ? "," : "", i, d->prob[i]);
            printf("},\"confidence\":%.17g", confidence(d->prob, d->nc));
        }
        putchar('}');
    }
    printf("},\"usage\":{\"input_tokens\":%u,\"output_tokens\":0,\"prefill_tokens\":%d},\"timing_"
           "ms\":{\"total\":%.3f,\"prefill\":%.3f,\"decode\":%.3f,\"candidates\":%.3f,\"reads\":%d,"
           "\"canvases\":%d,\"microbatch\":%d,\"prefix_cache\":\"%s\",\"cache_tokens\":%d}}\n",
           tokens, processed_tokens, ms, prefill_ms, ms - prefill_ms - cand_ms, cand_ms, reads,
           canvases, microbatch, cache_state, cache_tokens);
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
    free(r->state);
    free(r->reqid);
    if (r->qowned) {
        free(r->qt);
        free(r->qowned);
    }
    free(r->t);
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
    free(seed_json.p);
    return (uint32_t)digest[0] << 24 | (uint32_t)digest[1] << 16 | (uint32_t)digest[2] << 8 |
           digest[3];
}
static DecisionWork *dg_questions(const DGRequest *r, char **choice_label, int *nq_out) {
    const char *qj = r->qj;
    JTok *qt = r->qt;
    int qnt = r->qnt;
    int keys[1024], nq = direct_keys(qt, qnt, r->qroot, keys, 1024);
    if (nq < 1 || nq > 1024)
        die("questions must contain 1..1024 entries");
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
                    if (d->nc >= 10)
                        die("decision candidate count out of range");
                    char b[16];
                    snprintf(b, sizeof b, "%d", d->nc);
                    d->cand[d->nc++] = xstrdup(b);
                }
        } else
            die("unknown question type");
        if (d->nc < 2 || d->nc > 128 || (!strcmp(d->kind, "score") && d->nc > 10))
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
            free(w[x].cand[i]);
            free(w[x].label[i]);
        }
        free(w[x].cand);
        free(w[x].label);
        free(w[x].kind);
        free(w[x].prob);
    }
    free(w);
}
static char *dg_answer_template(DecisionWork *w, int nq, const int *pick) {
    char *a = dg_answer_text(w, nq, pick);
    DGBuf b = {0};
    db_fmt(&b, "<|channel>thought\n<channel|>%s", a);
    free(a);
    return b.p;
}
static char *dg_answer_template_range(DecisionWork *w, int total, int start, int count,
                                      const int *pick) {
    char *a = dg_answer_text_range(w, total, start, count, pick);
    DGBuf b = {0};
    db_fmt(&b, "<|channel>thought\n<channel|>%s", a);
    free(a);
    return b.p;
}

typedef struct {
    DGRequest rq;
    DecisionWork *w;
    int nq;
    uint32_t seed;
    char *sys, *pre, *prompt;
    Tokens pt;
    int group_cap, groups, width, *slot, **ids, *canvas;
    DGCanvas *cv;
    int requested, reads, max_reads, active;
    uint64_t candidate_ns;
} DGJob;
static void dg_job_prepare(DGJob *g, DGTokenizer *tok, const char *j, size_t len) {
    memset(g, 0, sizeof *g);
    dg_request_parse(&g->rq, j, len);
    g->requested = g->rq.requested;
    g->seed = dg_request_seed(&g->rq);
    char *choice[128];
    int nc = dg_choice_inventory(tok, choice);
    if (nc != 128)
        die("cannot construct OpenJev choice labels");
    g->w = dg_questions(&g->rq, choice, &g->nq);
    for (int i = 0; i < nc; i++)
        free(choice[i]);
    g->sys = dg_system_prompt(g->rq.qj, g->rq.qt, g->rq.qnt, g->w, g->nq);
    DGBuf pre = {0}, pb = {0};
    db_fmt(&pre, "<bos><|turn>system\n%s<turn|>\n<|turn>user\n", g->sys);
    g->pre = pre.p;
    db_fmt(&pb, "%s%s<turn|>\n<|turn>model\n", g->pre, g->rq.state);
    g->prompt = pb.p;
    if (strlen(g->prompt) > (size_t)JB_MAX_CTX * tok->maxlen)
        die("prompt exceeds DiffusionGemma context");
    g->pt = dgt_tokenize(tok, g->prompt);
    if (g->pt.n > JB_MAX_CTX)
        die("prompt exceeds DiffusionGemma context");
    int *zero = xcalloc((size_t)g->nq, sizeof *zero);
    g->group_cap = g->nq <= 16 ? g->nq : 8;
    g->groups = (g->nq + g->group_cap - 1) / g->group_cap;
    g->cv = xcalloc((size_t)g->groups, sizeof *g->cv);
    g->slot = xmalloc((size_t)g->nq * sizeof *g->slot);
    g->ids = xcalloc((size_t)g->nq, sizeof *g->ids);
    for (int b = 0; b < g->groups; b++) {
        DGCanvas *c = &g->cv[b];
        c->start = b * g->group_cap;
        c->count = g->nq - c->start < g->group_cap ? g->nq - c->start : g->group_cap;
        char *text = dg_answer_template_range(g->w, g->nq, c->start, c->count, zero);
        c->base = dgt_tokenize(tok, text);
        free(text);
        if (c->base.n + 1 > 64)
            die("OpenJev answer template exceeds 64-token canvas");
        int cw = (int)(((c->base.n + 1 + 15) / 16) * 16);
        if (cw > g->width)
            g->width = cw;
        for (int x = c->start; x < c->start + c->count; x++) {
            g->ids[x] = xmalloc((size_t)g->w[x].nc * sizeof **g->ids);
            g->slot[x] = -1;
            for (int q = 1; q < g->w[x].nc; q++) {
                int old = zero[x];
                zero[x] = q;
                char *a = dg_answer_template_range(g->w, g->nq, c->start, c->count, zero);
                Tokens v = dgt_tokenize(tok, a);
                free(a);
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
                free(v.v);
            }
            if (g->slot[x] < 0)
                die("cannot resolve label slot");
            g->ids[x][0] = c->base.v[g->slot[x]];
        }
    }
    if ((uint64_t)g->width * g->groups > JB_MAX_CTX)
        die("batched answer canvases exceed DiffusionGemma context");
    g->canvas = xcalloc((size_t)g->width * g->groups, sizeof *g->canvas);
    for (int x = 0; x < g->nq; x++)
        memset(g->w[x].prob, 0, (size_t)g->w[x].nc * sizeof *g->w[x].prob);
    g->max_reads = g->requested ? g->requested : 4;
    g->active = 1;
    free(zero);
}
static void dg_job_free(DGJob *g) {
    for (int x = 0; x < g->nq; x++)
        free(g->ids[x]);
    for (int b = 0; b < g->groups; b++)
        free(g->cv[b].base.v);
    free(g->cv);
    free(g->ids);
    free(g->slot);
    free(g->canvas);
    free(g->pt.v);
    free(g->prompt);
    free(g->pre);
    free(g->sys);
    dg_questions_free(g->w, g->nq);
    dg_request_free(&g->rq);
    memset(g, 0, sizeof *g);
}

static int dg_systemone(DGModel *m, DGTokenizer *tok, const char *j, size_t len,
                        const char *line_id, DGPrefixCache *prefix_cache) {
#ifdef JB_PROFILE
    memset(&jb_profile, 0, sizeof jb_profile);
#endif
    DGRequest rq;
    dg_request_parse(&rq, j, len);
    const char *qj = rq.qj;
    JTok *qt = rq.qt;
    int qnt = rq.qnt, requested = rq.requested;
    uint32_t seed0 = dg_request_seed(&rq);
    char *choice_label[128];
    int nchoice = dg_choice_inventory(tok, choice_label);
    if (nchoice != 128)
        die("cannot construct OpenJev choice labels");
    int nq;
    DecisionWork *w = dg_questions(&rq, choice_label, &nq);
    for (int i = 0; i < nchoice; i++)
        free(choice_label[i]);
    /* Request text is spliced into the chat template and tokenized as one
     * string, so special-token spellings inside state, instructions, or
     * criteria are encoded as control tokens. This mirrors OpenJev, whose
     * rendered chat template is tokenized the same way, and is kept for
     * parity: callers must not treat request text as a trust boundary. */
    char *sys = dg_system_prompt(qj, qt, qnt, w, nq);
    DGBuf pre = {0}, pb = {0};
    db_fmt(&pre, "<bos><|turn>system\n%s<turn|>\n<|turn>user\n", sys);
    db_fmt(&pb, "%s%s<turn|>\n<|turn>model\n", pre.p, rq.state);
    char *prompt = pb.p;
    /* A token covers at most maxlen bytes, so a longer prompt cannot fit the
     * context; reject it before spending time on tokenization. */
    if (strlen(prompt) > (size_t)JB_MAX_CTX * tok->maxlen)
        die("prompt exceeds DiffusionGemma context");
    Tokens pt = dgt_tokenize(tok, prompt);
    if (pt.n > JB_MAX_CTX)
        die("prompt exceeds DiffusionGemma context");
#ifdef JB_PROFILE
    Tokens system_part = dgt_tokenize(tok, sys), state_part = dgt_tokenize(tok, rq.state);
    jb_profile.prompt_tokens = pt.n;
    jb_profile.system_tokens = system_part.n;
    jb_profile.state_tokens = state_part.n;
    free(system_part.v);
    free(state_part.v);
#endif
    int *zero = xcalloc((size_t)nq, sizeof *zero), group_cap = nq <= 16 ? nq : 8,
        groups = (nq + group_cap - 1) / group_cap, width = 0;
    DGCanvas *cv = xcalloc((size_t)groups, sizeof *cv);
    int *slot = xmalloc((size_t)nq * sizeof *slot);
    int **ids = xcalloc((size_t)nq, sizeof *ids);
    for (int b = 0; b < groups; b++) {
        DGCanvas *c = &cv[b];
        c->start = b * group_cap;
        c->count = nq - c->start < group_cap ? nq - c->start : group_cap;
        char *text = dg_answer_template_range(w, nq, c->start, c->count, zero);
        c->base = dgt_tokenize(tok, text);
        free(text);
        if (c->base.n + 1 > 64)
            die("OpenJev answer template exceeds 64-token canvas");
        int cw = (int)(((c->base.n + 1 + 15) / 16) * 16);
        if (cw > width)
            width = cw;
        for (int x = c->start; x < c->start + c->count; x++) {
            ids[x] = xmalloc((size_t)w[x].nc * sizeof **ids);
            slot[x] = -1;
            for (int q = 1; q < w[x].nc; q++) {
                int old = zero[x];
                zero[x] = q;
                char *a = dg_answer_template_range(w, nq, c->start, c->count, zero);
                Tokens v = dgt_tokenize(tok, a);
                free(a);
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
                if (slot[x] >= 0 && diff != slot[x])
                    die("labels do not share one template slot");
                slot[x] = diff;
                ids[x][q] = v.v[diff];
                free(v.v);
            }
            if (slot[x] < 0)
                die("cannot resolve label slot");
            ids[x][0] = c->base.v[slot[x]];
        }
    }
    if ((uint64_t)width * groups > JB_MAX_CTX)
        die("batched answer canvases exceed DiffusionGemma context");
    int *canvas = xcalloc((size_t)width * groups, sizeof *canvas);
    DGTensor *emb = dg_tensor(m, "model.decoder.embed_tokens.weight");
    uint64_t start = now_ns(), candidate_ns = 0;
    DGKV kv[DG_L] = {0};
    const char *cache_state = "off";
    int cache_tokens = 0, processed_tokens = (int)pt.n;
    if (prefix_cache) {
        Tokens prefix = dgt_tokenize(tok, pre.p);
        uint32_t common = 0, lim = prefix.n < pt.n ? prefix.n : pt.n;
        while (common < lim && prefix.v[common] == pt.v[common])
            common++;
        int hit = prefix_cache->schema && !strcmp(prefix_cache->schema, sys) &&
                  prefix_cache->n <= (int)pt.n &&
                  !memcmp(prefix_cache->ids, pt.v, (size_t)prefix_cache->n * sizeof *pt.v);
        if (!hit) {
            if (!common || common >= pt.n)
                die("cannot construct reusable schema prefix");
            dg_prefix_free(prefix_cache);
            prefix_cache->schema = xstrdup(sys);
            prefix_cache->n = (int)common;
            prefix_cache->ids = xmalloc((size_t)common * sizeof *prefix_cache->ids);
            memcpy(prefix_cache->ids, pt.v, (size_t)common * sizeof *prefix_cache->ids);
            dg_prefill(m, pt.v, (int)pt.n, kv);
            for (int l = 0; l < DG_L; l++) {
                int full = l % 6 == 5, kn = (full ? 2 : 8) * (full ? 512 : 256);
                prefix_cache->kv[l].n = (int)common;
                prefix_cache->kv[l].k = xmalloc((size_t)common * kn * 4);
                prefix_cache->kv[l].v = xmalloc((size_t)common * kn * 4);
                memcpy(prefix_cache->kv[l].k, kv[l].k, (size_t)common * kn * 4);
                memcpy(prefix_cache->kv[l].v, kv[l].v, (size_t)common * kn * 4);
            }
            cache_state = "miss";
        } else {
            cache_state = "hit";
            processed_tokens = (int)pt.n - prefix_cache->n;
            int suffix = (int)pt.n - prefix_cache->n;
            if (suffix <= 0)
                die("schema prefix consumes complete prompt");
            dg_prefill_suffix(m, pt.v + prefix_cache->n, suffix, prefix_cache->kv, kv);
        }
        cache_tokens = prefix_cache->n;
        free(prefix.v);
    } else
        dg_prefill(m, pt.v, (int)pt.n, kv);
    uint64_t prefill_ns = now_ns() - start;
    for (int x = 0; x < nq; x++)
        memset(w[x].prob, 0, (size_t)w[x].nc * sizeof *w[x].prob);
    int reads = 0, max_reads = requested ? requested : 4, auto_all = 0;
    for (int r = 0; r < max_reads; r++) {
        memset(canvas, 0, (size_t)width * groups * sizeof *canvas);
        for (int b = 0; b < groups; b++) {
            memcpy(canvas + (size_t)b * width, cv[b].base.v, (size_t)cv[b].base.n * sizeof *canvas);
            canvas[(size_t)b * width + cv[b].base.n] = 106;
        }
        DGMT rng;
        dg_mt_seed(&rng, seed0 + (uint32_t)r * 7919u);
        for (int x = 0; x < nq; x++)
            canvas[(size_t)(x / group_cap) * width + slot[x]] = (int)dg_mt_vocab(&rng);
#ifdef JB_CANVAS_SEQUENTIAL
        float *h = xmalloc((size_t)groups * width * DG_H * sizeof *h);
        for (int b = 0; b < groups; b++) {
            float *one = dg_decode(m, kv, (int)pt.n, canvas + (size_t)b * width, width, 1);
            memcpy(h + (size_t)b * width * DG_H, one, (size_t)width * DG_H * sizeof *h);
            free(one);
        }
#else
        float *h = dg_decode(m, kv, (int)pt.n, canvas, width, groups);
#endif
        uint64_t cs = now_ns();
        double max_entropy = 0;
        double **sc = xmalloc((size_t)nq * sizeof *sc);
        for (int x = 0; x < nq; x++)
            sc[x] = xmalloc((size_t)w[x].nc * sizeof **sc);
        if (!requested && r == 0) {
            float *answer_h = xmalloc((size_t)nq * DG_H * sizeof *answer_h);
            int *label_n = xmalloc((size_t)nq * sizeof *label_n);
            for (int x = 0; x < nq; x++) {
                memcpy(answer_h + (size_t)x * DG_H,
                       h + ((size_t)(x / group_cap) * width + slot[x]) * DG_H,
                       DG_H * sizeof *answer_h);
                label_n[x] = w[x].nc;
            }
            max_entropy = dg_slot_logits_entropy(emb, answer_h, nq, ids, label_n, sc);
            free(label_n);
            free(answer_h);
        } else
            for (int x = 0; x < nq; x++)
                for (int c = 0; c < w[x].nc; c++) {
                    float z;
                    dg_mv_slice(emb, (uint64_t)ids[x][c] * DG_H,
                                h + ((size_t)(x / group_cap) * width + slot[x]) * DG_H, &z, 1,
                                DG_H);
                    sc[x][c] = 30 * tanh(z / 30);
                }
        for (int x = 0; x < nq; x++) {
            double *pr = xmalloc((size_t)w[x].nc * sizeof *pr);
            normalize_scores(sc[x], w[x].nc, pr);
            for (int c = 0; c < w[x].nc; c++)
                w[x].prob[c] += pr[c];
            free(pr);
            free(sc[x]);
        }
        free(sc);
        candidate_ns += now_ns() - cs;
        free(h);
        reads++;
        if (!requested && r == 0) {
            auto_all = max_entropy > .1;
            if (!auto_all)
                break;
        }
    }
    dg_free_kv(kv);
    for (int x = 0; x < nq; x++)
        for (int c = 0; c < w[x].nc; c++)
            w[x].prob[c] /= reads;
    double cand_ms = candidate_ns / 1e6, ms = (now_ns() - start) / 1e6;
    uint32_t billed = pt.n * (requested ? reads : 1);
    dg_print_answers(qj, qt, qnt, w, nq, line_id ? line_id : rq.reqid, billed, ms, prefill_ns / 1e6,
                     cand_ms, reads, groups, cache_state, cache_tokens, processed_tokens, 1);
#ifdef JB_PROFILE
    uint64_t detailed = jb_profile.moe_input_qdq + jb_profile.moe_gate + jb_profile.moe_up +
                        jb_profile.moe_activation + jb_profile.moe_hidden_qdq + jb_profile.moe_down;
    double moe_misc =
        (double)(jb_profile.experts >= detailed ? jb_profile.experts - detailed : 0) / 1e6;
    uint64_t ad = jb_profile.attention_qkv + jb_profile.attention_prepare +
                  jb_profile.attention_kv + jb_profile.attention_core + jb_profile.attention_output;
    double attention_misc =
        (double)(jb_profile.attention >= ad ? jb_profile.attention - ad : 0) / 1e6;
    uint64_t ld = jb_profile.attention + jb_profile.dense + jb_profile.router + jb_profile.experts +
                  jb_profile.ff_other;
    double layer_other = (double)(jb_profile.layers >= ld ? jb_profile.layers - ld : 0) / 1e6;
    double accounted = jb_profile.layers / 1e6 + jb_profile.embedding / 1e6 +
                       jb_profile.final_norm / 1e6 + jb_profile.kv_free / 1e6 + cand_ms;
    double unaccounted = ms > accounted ? ms - accounted : 0;
    fprintf(
        stderr,
        "JB_PROFILE prompt_tokens=%llu system_tokens=%llu state_tokens=%llu omp_regions=%llu "
        "alloc_calls=%llu alloc_ms=%.3f attention=%.3f attention_qkv=%.3f attention_prepare=%.3f "
        "attention_kv=%.3f attention_core=%.3f attention_output=%.3f attention_misc=%.3f "
        "dense=%.3f router=%.3f experts=%.3f moe_input_qdq=%.3f moe_gate=%.3f moe_up=%.3f "
        "moe_activation=%.3f moe_hidden_qdq=%.3f moe_down=%.3f moe_misc=%.3f ff_other=%.3f "
        "layer_other=%.3f embedding=%.3f final_norm=%.3f kv_free=%.3f candidates=%.3f "
        "unaccounted=%.3f total=%.3f\n",
        (unsigned long long)jb_profile.prompt_tokens, (unsigned long long)jb_profile.system_tokens,
        (unsigned long long)jb_profile.state_tokens, (unsigned long long)jb_profile.omp_regions,
        (unsigned long long)jb_profile.alloc_calls, jb_profile.alloc_ns / 1e6,
        jb_profile.attention / 1e6, jb_profile.attention_qkv / 1e6,
        jb_profile.attention_prepare / 1e6, jb_profile.attention_kv / 1e6,
        jb_profile.attention_core / 1e6, jb_profile.attention_output / 1e6, attention_misc,
        jb_profile.dense / 1e6, jb_profile.router / 1e6, jb_profile.experts / 1e6,
        jb_profile.moe_input_qdq / 1e6, jb_profile.moe_gate / 1e6, jb_profile.moe_up / 1e6,
        jb_profile.moe_activation / 1e6, jb_profile.moe_hidden_qdq / 1e6, jb_profile.moe_down / 1e6,
        moe_misc, jb_profile.ff_other / 1e6, layer_other, jb_profile.embedding / 1e6,
        jb_profile.final_norm / 1e6, jb_profile.kv_free / 1e6, cand_ms, unaccounted, ms);
#endif
    fflush(stdout);
    for (int x = 0; x < nq; x++)
        free(ids[x]);
    for (int b = 0; b < groups; b++)
        free(cv[b].base.v);
    free(cv);
    free(ids);
    free(slot);
    free(canvas);
    free(pt.v);
    free(zero);
    free(prompt);
    free(pre.p);
    free(sys);
    dg_questions_free(w, nq);
    dg_request_free(&rq);
    return 0;
}
/* Returns zero when the rows do not all hit the current exact schema entry;
 * the caller then executes them sequentially, allowing normal cache replace. */
static int dg_system_batch(DGModel *m, DGTokenizer *tok, char **row, size_t *len, int batch,
                           DGPrefixCache *prefix) {
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
    if (!ok) {
        for (int b = 0; b < batch; b++)
            dg_job_free(&g[b]);
        free(g);
        return 0;
    }
    int seq = 0, *sl = xmalloc((size_t)batch * sizeof *sl);
    for (int b = 0; b < batch; b++) {
        sl[b] = (int)g[b].pt.n - prefix->n;
        if (sl[b] < 1)
            die("schema prefix consumes complete prompt");
        if (sl[b] > seq)
            seq = sl[b];
    }
    if ((uint64_t)batch * seq > JB_MAX_CTX)
        die("microbatch suffix exceeds DiffusionGemma work limit");
    int *st = xcalloc((size_t)batch * seq, sizeof *st);
    for (int b = 0; b < batch; b++)
        memcpy(st + (size_t)b * seq, g[b].pt.v + prefix->n, (size_t)sl[b] * sizeof *st);
    DGKV *kv = xcalloc((size_t)batch * DG_L, sizeof *kv);
    uint64_t start = now_ns();
    dg_prefill_suffix_multi(m, st, sl, batch, seq, prefix->kv, kv);
    uint64_t prefill_ns = now_ns() - start;
    free(st);
    free(sl);
    DGTensor *emb = dg_tensor(m, "model.decoder.embed_tokens.weight");
    int width = g[0].width, maxr = 0;
    for (int b = 0; b < batch; b++)
        if (g[b].max_reads > maxr)
            maxr = g[b].max_reads;
    for (int r = 0; r < maxr; r++) {
        int segments = 0;
        for (int b = 0; b < batch; b++)
            if (g[b].active && r < g[b].max_reads)
                segments += g[b].groups;
        if (!segments)
            break;
        int *canvas = xcalloc((size_t)segments * width, sizeof *canvas),
            *doc = xmalloc((size_t)segments * sizeof *doc),
            *base = xmalloc((size_t)batch * sizeof *base);
        int at = 0;
        for (int b = 0; b < batch; b++) {
            base[b] = -1;
            if (!g[b].active || r >= g[b].max_reads)
                continue;
            base[b] = at;
            memset(g[b].canvas, 0, (size_t)width * g[b].groups * sizeof *g[b].canvas);
            for (int q = 0; q < g[b].groups; q++) {
                memcpy(g[b].canvas + (size_t)q * width, g[b].cv[q].base.v,
                       (size_t)g[b].cv[q].base.n * sizeof *g[b].canvas);
                g[b].canvas[(size_t)q * width + g[b].cv[q].base.n] = 106;
            }
            DGMT rng;
            dg_mt_seed(&rng, g[b].seed + (uint32_t)r * 7919u);
            for (int x = 0; x < g[b].nq; x++)
                g[b].canvas[(size_t)(x / g[b].group_cap) * width + g[b].slot[x]] =
                    (int)dg_mt_vocab(&rng);
            for (int q = 0; q < g[b].groups; q++) {
                memcpy(canvas + (size_t)at * width, g[b].canvas + (size_t)q * width,
                       (size_t)width * sizeof *canvas);
                doc[at++] = b;
            }
        }
        float *h = dg_decode_multi(m, kv, doc, segments, width, canvas);
        free(canvas);
        free(doc);
        for (int b = 0; b < batch; b++)
            if (base[b] >= 0) {
                uint64_t cs = now_ns();
                double entropy = 0;
                double **sc = xmalloc((size_t)g[b].nq * sizeof *sc);
                for (int x = 0; x < g[b].nq; x++)
                    sc[x] = xmalloc((size_t)g[b].w[x].nc * sizeof **sc);
                if (!g[b].requested && r == 0) {
                    float *ah = xmalloc((size_t)g[b].nq * DG_H * sizeof *ah);
                    int *ln = xmalloc((size_t)g[b].nq * sizeof *ln);
                    for (int x = 0; x < g[b].nq; x++) {
                        memcpy(ah + (size_t)x * DG_H,
                               h + ((size_t)(base[b] + x / g[b].group_cap) * width + g[b].slot[x]) *
                                       DG_H,
                               DG_H * sizeof *ah);
                        ln[x] = g[b].w[x].nc;
                    }
                    entropy = dg_slot_logits_entropy(emb, ah, g[b].nq, g[b].ids, ln, sc);
                    free(ln);
                    free(ah);
                } else
                    for (int x = 0; x < g[b].nq; x++)
                        for (int c = 0; c < g[b].w[x].nc; c++) {
                            float z;
                            dg_mv_slice(emb, (uint64_t)g[b].ids[x][c] * DG_H,
                                        h + ((size_t)(base[b] + x / g[b].group_cap) * width +
                                             g[b].slot[x]) *
                                                DG_H,
                                        &z, 1, DG_H);
                            sc[x][c] = 30 * tanh(z / 30);
                        }
                for (int x = 0; x < g[b].nq; x++) {
                    double *pr = xmalloc((size_t)g[b].w[x].nc * sizeof *pr);
                    normalize_scores(sc[x], g[b].w[x].nc, pr);
                    for (int c = 0; c < g[b].w[x].nc; c++)
                        g[b].w[x].prob[c] += pr[c];
                    free(pr);
                    free(sc[x]);
                }
                free(sc);
                g[b].candidate_ns += now_ns() - cs;
                g[b].reads++;
                if (!g[b].requested && r == 0 && !((entropy > .1)))
                    g[b].active = 0;
            }
        free(base);
        free(h);
    }
    uint64_t total_ns = now_ns() - start;
    for (int b = 0; b < batch; b++) {
        dg_free_kv(kv + (size_t)b * DG_L);
        for (int x = 0; x < g[b].nq; x++)
            for (int c = 0; c < g[b].w[x].nc; c++)
                g[b].w[x].prob[c] /= g[b].reads;
        uint32_t billed = g[b].pt.n * (g[b].requested ? g[b].reads : 1);
        dg_print_answers(g[b].rq.qj, g[b].rq.qt, g[b].rq.qnt, g[b].w, g[b].nq, g[b].rq.reqid,
                         billed, total_ns / 1e6, prefill_ns / 1e6, g[b].candidate_ns / 1e6,
                         g[b].reads, g[b].groups, "hit", prefix->n, (int)g[b].pt.n - prefix->n,
                         batch);
        dg_job_free(&g[b]);
    }
    fflush(stdout);
    free(kv);
    free(g);
    return 1;
}
static int dg_decide_file(const char *dir, const char *path) {
    size_t n;
    char *j = read_all(path, &n);
    DGModel m;
    DGTokenizer t;
    dg_load(&m, dir);
    dgt_load(&t, dir);
    int rc = dg_systemone(&m, &t, j, n, NULL, NULL);
    dgt_free(&t);
    dg_free(&m);
    free(j);
    return rc;
}
static void dg_eval_group(DGModel *m, DGTokenizer *t, char **row, size_t *len, int n,
                          DGPrefixCache *cache) {
    if (n > 1 && dg_system_batch(m, t, row, len, n, cache))
        return;
    for (int i = 0; i < n; i++)
        dg_systemone(m, t, row[i], len[i], NULL, cache);
}
static int dg_eval_file(const char *dir, const char *path) {
    FILE *f = !strcmp(path, "-") ? stdin : fopen(path, "rb");
    if (!f)
        die2("cannot open", path);
#if defined(_WIN32)
    if (f == stdin && _setmode(_fileno(stdin), _O_BINARY) < 0)
        die("cannot set binary stdin");
#endif
    DGModel m;
    DGTokenizer t;
    DGPrefixCache cache = {0};
    int mb = 1;
    const char *me = getenv("JB_MICROBATCH");
    if (me) {
        char *e;
        long v = strtol(me, &e, 10);
        if (*e || v < 1 || v > 16)
            die("JB_MICROBATCH must be 1..16");
        mb = (int)v;
    }
#ifdef __FAST_MATH__
    DGPrefixCache *cachep = NULL;
#else
    DGPrefixCache *cachep = &cache;
#endif
    dg_load(&m, dir);
    dgt_load(&t, dir);
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
                    dg_eval_group(&m, &t, rows, lens, nr, cachep);
                    for (int i = 0; i < nr; i++)
                        free(rows[i]);
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
        dg_eval_group(&m, &t, rows, lens, nr, cachep);
        for (int i = 0; i < nr; i++)
            free(rows[i]);
    }
    free(lens);
    free(rows);
    free(line);
    if (f != stdin)
        fclose(f);
    dg_prefix_free(&cache);
    dgt_free(&t);
    dg_free(&m);
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
    free(s);
    DGRequest rq;
    dg_request_parse(&rq, j, n);
    (void)dg_request_seed(&rq);
    char *labels[128];
    for (int i = 0; i < 128; i++) {
        char l[3];
        dg_choice_candidate(i, l);
        labels[i] = xstrdup(l);
    }
    int nq;
    DecisionWork *w = dg_questions(&rq, labels, &nq);
    for (int i = 0; i < 128; i++)
        free(labels[i]);
    char *sys = dg_system_prompt(rq.qj, rq.qt, rq.qnt, w, nq);
    int *zero = xcalloc((size_t)nq, sizeof *zero);
    char *text = dg_answer_template(w, nq, zero);
    printf("{\"check\":\"ok\",\"questions\":%d}\n", nq);
    free(text);
    free(zero);
    free(sys);
    dg_questions_free(w, nq);
    dg_request_free(&rq);
    free(j);
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
static void jb_check_close(const char *what, double got, double want, double bound) {
    if (!jb_finite(got) || fabs(got - want) > bound) {
        char z[160];
        snprintf(z, sizeof z, "%s (got %.9g, want %.9g, bound %.3g)", what, got, want, bound);
        die2("kernel self-test failed", z);
    }
}
static void dg_test_mm(uint64_t *rs) {
    static const int shape[][3] = {{2, 16, 1}, {2, 7, 1},  {2, 15, 8},
                                   {4, 40, 3}, {6, 33, 9}, {8, 2816, 17}};
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
        for (int pass = 0; pass < 2; pass++) {
            if (pass)
                dg_mm_data(w, x, y, tokens, rows, cols);
            else
                dg_mm_data_ref(w, x, y, tokens, rows, cols);
            for (int i = 0; i < tokens * rows; i++)
                jb_check_close(pass ? dg_kernels()->name : "BF16 matmul (ref)", y[i], want[i],
                               1e-4 * mag[i]);
        }
        free(w);
        free(x);
        free(y);
        free(want);
        free(mag);
    }
}
static void dg_test_nvfp4(uint64_t *rs) {
    static const int shape[][3] = {{2, 32, 1}, {4, 64, 9}, {8, 704, 17}, {2, 2816, 3}};
    static const float mag4[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    const float global = 0.01f, base = 0.02f;
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
        dg_nvfp4_mm_ref(wd, sd, global, xq, y, tokens, rows, cols);
        for (int i = 0; i < tokens * rows; i++)
            jb_check_close("NVFP4 matmul (ref)", y[i], want[i], 1e-4 * mag[i]);
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
        dg_nvfp4_mm(&tw, &ts, &tg, xs, y, tokens, rows, cols);
        for (int i = 0; i < tokens * rows; i++)
            jb_check_close(dg_kernels()->name, y[i], want[i], 1e-4 * mag[i]);
        free(wd);
        free(sd);
        free(x);
        free(xq);
        free(xs);
        free(y);
        free(want);
        free(mag);
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
        for (int pass = 0; pass < 4; pass++) {
            const DGTensor *sc = pass & 1 ? &scale : NULL;
            if (pass & 2)
                dg_rms(y, x, sc, n);
            else
                dg_rms_ref(y, x, sc, n);
            for (int i = 0; i < n; i++) {
                double want = x[i] * q * (sc ? dg_at(sc, i) : 1.0);
                jb_check_close(pass & 2 ? dg_kernels()->name : "RMS norm (ref)", y[i], want,
                               1e-5 * fabs(want) + 1e-12);
            }
        }
        jb_check_close("dot (ref)", dg_dot_ref(x, b, n), dot, 1e-12 * dmag);
        jb_check_close(dg_kernels()->name, dg_dot(x, b, n), dot, 1e-12 * dmag);
        free(x);
        free(b);
        free(y);
        free(sw);
    }
}
static void dg_kernel_selftest(void) {
    uint64_t rs = 0x9e3779b97f4a7c15ull;
    dg_test_mm(&rs);
    dg_test_nvfp4(&rs);
    dg_test_rms_dot(&rs);
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
    printf("{\"bench\":\"info\",\"kernels\":\"%s\",\"math\":\"%s\",\"threads\":%d}\n",
           dg_kernels()->name, JB_MATH_MODE, jb_threads());
    uint64_t rs = 0x243f6a8885a308d3ull, ns[3];
    size_t nw = (size_t)1 << 26;
    uint64_t *buf = xmalloc(nw * sizeof *buf), acc = 0;
    for (size_t i = 0; i < nw; i++)
        buf[i] = i;
    for (int rep = 0; rep < 3; rep++) {
        uint64_t t0 = now_ns(), sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum) schedule(static)
#endif
        for (size_t i = 0; i < nw; i++)
            sum += buf[i];
        ns[rep] = now_ns() - t0;
        acc += sum;
    }
    free(buf);
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
    for (size_t k = 0; k < sizeof mm_tokens / sizeof *mm_tokens; k++) {
        int tokens = mm_tokens[k];
        for (int kernel = 0; kernel < 2; kernel++) {
            if (kernel && dg_kernels()->bf16_mm == dg_mm_data_ref)
                break;
            dg_mm_data_ref(w, x, y, tokens, rows, cols);
            for (int rep = 0; rep < 3; rep++) {
                uint64_t t0 = now_ns();
                if (kernel)
                    dg_mm_data(w, x, y, tokens, rows, cols);
                else
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
    free(w);
    free(x);
    free(y);

    /* All 128 experts of one gate projection, so weights stream from memory
     * as in the model rather than staying cache-resident. */
    static const int nv_tokens[] = {1, 4, 16, 64};
    int experts = 128, nr = DG_MOE, nc = DG_H;
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
    for (size_t k = 0; k < sizeof nv_tokens / sizeof *nv_tokens; k++) {
        int tokens = nv_tokens[k];
        for (int kernel = 0; kernel < 2; kernel++) {
            if (kernel && dg_kernels()->nvfp4_mm == dg_nvfp4_mm_ref)
                break;
            for (int rep = 0; rep < 3; rep++) {
                uint64_t t0 = now_ns();
                for (int e = 0; e < experts; e++) {
                    if (kernel)
                        dg_kernels()->nvfp4_mm(wd + wbytes * e, sd + sbytes * e, 0.01f, xs, yo,
                                               tokens, nr, nc);
                    else
                        dg_nvfp4_mm_ref(wd + wbytes * e, sd + sbytes * e, 0.01f, xq, yo, tokens, nr,
                                        nc);
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
    free(wd);
    free(sd);
    free(xr);
    free(xs);
    free(xq);
    free(yo);
    return 0;
}

static int selftest(void) {
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
    free(cb.p);
    free(ct);
    const char *tx = "{\"s\":\" \\u2003x\\u00a0 \",\"o\":{\"b\":2,\"a\":1},\"n\":null}";
    int txn;
    JTok *tt = json_tokens(tx, strlen(tx), &txn);
    int ts = jt_obj_get(tx, tt, txn, 0, "s"), to = jt_obj_get(tx, tt, txn, 0, "o"),
        tn = jt_obj_get(tx, tt, txn, 0, "n");
    char *sv = dg_text_of(tx, tt, txn, ts), *ov = dg_text_of(tx, tt, txn, to),
         *nv = dg_text_of(tx, tt, txn, tn);
    if (strcmp(sv, "x") || strcmp(ov, "{\"b\": 2, \"a\": 1}") || *nv)
        die("OpenJev text_of self-test failed");
    free(sv);
    free(ov);
    free(nv);
    free(tt);
    uint32_t nan_bits = 0x7fc00000u, inf_bits = 0x7f800000u;
    float fnan, finf;
    memcpy(&fnan, &nan_bits, 4);
    memcpy(&finf, &inf_bits, 4);
    if (jb_finitef(fnan) || jb_finitef(finf) || !jb_finitef(FLT_MAX) || jb_finite((double)finf) ||
        !jb_finite(0.0))
        die("finiteness self-test failed");
    const char *ek = "{\"st\\u0061te\":1}";
    int ekn;
    JTok *ekt = json_tokens(ek, strlen(ek), &ekn);
    if (jt_obj_get(ek, ekt, ekn, 0, "state") != 2)
        die("escaped JSON key self-test failed");
    free(ekt);
    uint8_t one[2] = {0x80, 0x3f};
    if (dg_bf(one) != 1.0f)
        die("BF16 conversion self-test failed");
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
    free(tokens);
    dg_kernel_selftest();
    puts("{\"selftest\":\"ok\"}");
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
int main(int ac, char **av) {
    if (ac == 2 && !strcmp(av[1], "--selftest"))
        return selftest();
    if (ac == 2 && !strcmp(av[1], "--bench-kernels"))
        return bench_kernels();
    if (ac == 3 && !strcmp(av[1], "--check-request"))
        return dg_check_request(av[2]);
    if (ac < 3) {
        usage();
        return 2;
    }
    if (!strcmp(av[2], "decide") && ac == 4)
        return dg_decide_file(av[1], av[3]);
    if (!strcmp(av[2], "eval") && ac == 4)
        return dg_eval_file(av[1], av[3]);
    usage();
    return 2;
}
