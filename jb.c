/*
 * Jev Bush -- CPU-first bounded decisions with DiffusionGemma.
 *
 * Build: cc -O3 -march=native -ffast-math -std=c11 -Wall -Wextra
 *        -pedantic -fopenmp jb.c -lm -o jb
 * The engine reads the exact public DiffusionGemma safetensors layout directly.
 */
#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(_WIN32)
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

#if defined(__FAST_MATH__)
#define JB_MATH_MODE "fast"
#else
#define JB_MATH_MODE "strict"
#endif

#ifdef JB_PROFILE
typedef struct {
    uint64_t attention, dense, router, experts, ff_other;
    uint64_t moe_input_qdq, moe_gate, moe_up, moe_activation;
    uint64_t moe_hidden_qdq, moe_down;
} JBProfile;
static JBProfile jb_profile;
#define JB_TICK(name) uint64_t name = now_ns()
#define JB_TO(field, name) (jb_profile.field += now_ns() - (name))
#else
#define JB_TICK(name)
#define JB_TO(field, name)
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
static void die(const char *s) {
    fprintf(stderr, "jb: %s\n", s);
    exit(2);
}
static void die2(const char *a, const char *b) {
    fprintf(stderr, "jb: %s: %s\n", a, b);
    exit(2);
}
static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p)
        die("out of memory");
    return p;
}
static void *xcalloc(size_t n, size_t z) {
    void *p = calloc(n ? n : 1, z);
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
    p = realloc(p, n ? n : 1);
    if (!p)
        die("out of memory");
    return p;
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
    if (c <= 0x7f) { z[0] = (char)c; return 1; }
    if (c <= 0x7ff) { z[0] = (char)(0xc0 | c >> 6); z[1] = (char)(0x80 | (c & 63)); return 2; }
    if (c <= 0xffff) { z[0] = (char)(0xe0 | c >> 12); z[1] = (char)(0x80 | ((c >> 6) & 63)); z[2] = (char)(0x80 | (c & 63)); return 3; }
    z[0] = (char)(0xf0 | c >> 18); z[1] = (char)(0x80 | ((c >> 12) & 63)); z[2] = (char)(0x80 | ((c >> 6) & 63)); z[3] = (char)(0x80 | (c & 63)); return 4;
}
static uint32_t next_cp(const unsigned char *s, size_t n, size_t *i) {
    size_t p = (*i)++; uint32_t c = s[p];
    if (c < 0x80) return c;
    int more = c >= 0xf0 ? 3 : c >= 0xe0 ? 2 : c >= 0xc2 ? 1 : -1;
    if (more < 0 || p + (size_t)more >= n) die("invalid UTF-8");
    uint32_t v = c & (0x7f >> more);
    for (int k = 0; k < more; k++) { unsigned q = s[(*i)++]; if ((q & 0xc0) != 0x80) die("invalid UTF-8"); v = (v << 6) | (q & 63); }
    if ((more == 1 && v < 0x80) || (more == 2 && v < 0x800) || (more == 3 && v < 0x10000) || v > 0x10ffff || (v >= 0xd800 && v <= 0xdfff)) die("invalid UTF-8");
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
    int nt, cap;
} JParser;
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
            } else if (!strchr("\"\\/bfnrt", c))
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
        while (1) {
            while (p->pos < p->n && isspace((unsigned char)p->s[p->pos]))
                p->pos++;
            if (p->pos >= p->n)
                die("unterminated JSON container");
            if (p->s[p->pos] == (ty == JT_OBJECT ? '}' : ']')) {
                p->t[i].end = (int)++p->pos;
                return i;
            }
            if (p->t[i].size) {
                if (p->s[p->pos++] != ',')
                    die("expected JSON comma");
                while (p->pos < p->n && isspace((unsigned char)p->s[p->pos]))
                    p->pos++;
            }
            if (ty == JT_OBJECT) {
                if (p->s[p->pos] != '\"')
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
static JTok *json_tokens(const char *s, size_t n, int *nt) {
    JParser p = {s, n, 0, 0, 0, 0};
    json_parse_value(&p, -1);
    while (p.pos < n && isspace((unsigned char)s[p.pos]))
        p.pos++;
    if (p.pos != n)
        die("trailing JSON data");
    *nt = p.nt;
    return p.t;
}
static int jt_eq(const char *j, const JTok *t, const char *z) {
    return t->type == JT_STRING && t->end - t->start == (int)strlen(z) &&
           !memcmp(j + t->start, z, strlen(z));
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
static char *jt_string(const char *j, const JTok *t);

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
typedef struct{
    DGTensor *wg,*sg,*gg,*ag,*wu,*su,*gu,*wd,*sd,*gd,*ad;
} DGNvExpert;
typedef struct {
    DGShard shard[11];
    DGTensor *tensor;
    size_t nt, cap;
    int nshard, nvfp4;
    DGNvExpert *nvexpert;
    float nv_a13[30],nv_a2[30];
} DGModel;

enum { DG_BF16, DG_U8, DG_F8E4M3, DG_F32 };

typedef struct {
    Vocab *vocab, **by_id;
    Merge *merge;
    Vocab **special;
    uint32_t nv, nm, ns;
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
static int dg_tensor_cmp(const void*a,const void*b){
    const DGTensor*x=a,*y=b;return strcmp(x->name,y->name);
}
static int dg_is_text(const char *name) {
    return !strncmp(name, "model.decoder.", 14) ||
           (!strncmp(name, "model.encoder.language_model.", 29) && strstr(name, ".layer_scalar"));
}
static void dg_parse_shard(DGModel *m, int si, const char *dir) {
    DGShard *s = &m->shard[si];
    snprintf(s->path, sizeof s->path, "%s/model-%05d-of-%05d.safetensors", dir, si + 1,m->nshard);
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
        int dtype = jt_obj_get(j, tok, nt, obj, "dtype"), shape = jt_obj_get(j, tok, nt, obj, "shape"),
            offsets = jt_obj_get(j, tok, nt, obj, "data_offsets");
        if (dtype < 0 || shape < 0 || offsets < 0 || tok[shape].type != JT_ARRAY ||
            tok[offsets].type != JT_ARRAY)
            die2("incomplete safetensors descriptor", name);
        char *dt = jt_string(j, &tok[dtype]);
        DGTensor t = {0};
        t.name = name;
        if(!strcmp(dt,"BF16"))t.dtype=DG_BF16;
        else if(!strcmp(dt,"U8"))t.dtype=DG_U8;
        else if(!strcmp(dt,"F8_E4M3"))t.dtype=DG_F8E4M3;
        else if(!strcmp(dt,"F32"))t.dtype=DG_F32;
        else if(dg_is_text(name))die2("unsupported DiffusionGemma text dtype",name);
        free(dt);
        if(dg_is_text(name)&&t.dtype!=DG_BF16&&
           !(m->nvfp4&&strstr(name,".experts.")&&
             (t.dtype==DG_U8||t.dtype==DG_F8E4M3||t.dtype==DG_F32)))
            die2("unsupported DiffusionGemma text tensor encoding",name);
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
        dg_push_tensor(m, t);
    }
    free(tok);
}
static DGTensor *dg_tensor(DGModel *m, const char *name) {
    DGTensor key={0};key.name=(char*)name;DGTensor*t=bsearch(&key,m->tensor,m->nt,sizeof*m->tensor,dg_tensor_cmp);
    if(t)return t;
    die2("missing DiffusionGemma tensor", name);
    return NULL;
}
static DGTensor*dg_expert_tensor(DGModel*m,int l,int e,const char*tail){
    char n[192];snprintf(n,sizeof n,"model.decoder.layers.%d.experts.%d.%s",l,e,tail);return dg_tensor(m,n);
}
static void dg_expect(DGModel *m, const char *name, int dtype,int nd, uint64_t a, uint64_t b, uint64_t c) {
    DGTensor *t = dg_tensor(m, name);
    if (t->dtype!=dtype||t->nd != nd || (nd>0&&t->shape[0] != a) || (nd > 1 && t->shape[1] != b) ||
        (nd > 2 && t->shape[2] != c))
        die2("unexpected DiffusionGemma tensor shape", name);
    uint64_t n = 1;
    for (int i = 0; i < nd; i++) {
        if (t->shape[i] && n > UINT64_MAX / t->shape[i])
            die2("DiffusionGemma tensor size overflow", name);
        n *= t->shape[i];
    }
    uint64_t item=dtype==DG_BF16?2:dtype==DG_F32?4:1;
    if (t->bytes != n * item)
        die2("unexpected DiffusionGemma tensor byte size", name);
}
static void dg_load(DGModel *m, const char *dir) {
    memset(m, 0, sizeof *m);
    char probe[768];snprintf(probe,sizeof probe,"%s/model-00001-of-00002.safetensors",dir);FILE*pf=fopen(probe,"rb");
    if(pf){fclose(pf);m->nshard=2;m->nvfp4=1;}else m->nshard=11;
    for (int i = 0; i < m->nshard; i++)
        dg_parse_shard(m, i, dir);
    qsort(m->tensor,m->nt,sizeof*m->tensor,dg_tensor_cmp);
    dg_expect(m, "model.decoder.embed_tokens.weight",DG_BF16, 2, 262144, 2816, 0);
    dg_expect(m, "model.decoder.layers.0.self_attn.q_proj.weight",DG_BF16, 2, 4096, 2816, 0);
    dg_expect(m, "model.decoder.layers.0.self_attn.k_proj.weight",DG_BF16, 2, 2048, 2816, 0);
    dg_expect(m, "model.decoder.layers.0.self_attn.v_proj.weight",DG_BF16, 2, 2048, 2816, 0);
    if(m->nvfp4){
        dg_expect(m,"model.decoder.layers.0.experts.0.gate_proj.weight",DG_U8,2,704,1408,0);
        dg_expect(m,"model.decoder.layers.0.experts.0.gate_proj.weight_scale",DG_F8E4M3,2,704,176,0);
        dg_expect(m,"model.decoder.layers.0.experts.0.gate_proj.weight_scale_2",DG_F32,0,0,0,0);
        dg_expect(m,"model.decoder.layers.0.experts.0.down_proj.weight",DG_U8,2,2816,352,0);
        m->nvexpert=xcalloc(30u*128,sizeof*m->nvexpert);
        for(int l=0;l<30;l++)for(int e=0;e<128;e++){
            DGNvExpert*v=&m->nvexpert[l*128+e];
            v->wg=dg_expert_tensor(m,l,e,"gate_proj.weight");v->sg=dg_expert_tensor(m,l,e,"gate_proj.weight_scale");v->gg=dg_expert_tensor(m,l,e,"gate_proj.weight_scale_2");v->ag=dg_expert_tensor(m,l,e,"gate_proj.input_scale");
            v->wu=dg_expert_tensor(m,l,e,"up_proj.weight");v->su=dg_expert_tensor(m,l,e,"up_proj.weight_scale");v->gu=dg_expert_tensor(m,l,e,"up_proj.weight_scale_2");
            v->wd=dg_expert_tensor(m,l,e,"down_proj.weight");v->sd=dg_expert_tensor(m,l,e,"down_proj.weight_scale");v->gd=dg_expert_tensor(m,l,e,"down_proj.weight_scale_2");v->ad=dg_expert_tensor(m,l,e,"down_proj.input_scale");
            float a13,a2;memcpy(&a13,v->ag->data,4);memcpy(&a2,v->ad->data,4);if(a13>m->nv_a13[l])m->nv_a13[l]=a13;if(a2>m->nv_a2[l])m->nv_a2[l]=a2;
        }
    }else{
        dg_expect(m, "model.decoder.layers.0.experts.gate_up_proj",DG_BF16, 3, 128, 1408, 2816);
        dg_expect(m, "model.decoder.layers.0.experts.down_proj",DG_BF16, 3, 128, 2816, 704);
    }
    dg_expect(m, "model.decoder.layers.0.router.proj.weight",DG_BF16, 2, 128, 2816, 0);
    for (int l = 0; l < 30; l++) {
        char n[128];
        snprintf(n, sizeof n, "model.decoder.layers.%d.layer_scalar", l);
        dg_expect(m, n,DG_BF16, 1, 1, 0, 0);
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
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    size_t nj;
    char *j = read_whole(path, &nj);
    int nt;
    JTok *t = json_tokens(j, nj, &nt);
    int model = jt_obj_get(j, t, nt, 0, "model");
    int vocab = jt_obj_get(j, t, nt, model, "vocab");
    int merges = jt_obj_get(j, t, nt, model, "merges");
    int added = jt_obj_get(j, t, nt, 0, "added_tokens");
    if (model < 0 || vocab < 0 || merges < 0 || added < 0 ||
        t[vocab].type != JT_OBJECT || t[merges].type != JT_ARRAY || t[added].type != JT_ARRAY)
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
                if (a < 0) a = k; else if (b < 0) b = k; else die("bad tokenizer merge");
            }
        if (t[i].type != JT_ARRAY || a < 0 || b < 0)
            die("bad tokenizer merge");
        char *sa = jt_string(j, &t[a]), *sb = jt_string(j, &t[b]);
        Vocab *va = dgt_vfind(d, sa, strlen(sa)), *vb = dgt_vfind(d, sb, strlen(sb));
        free(sa); free(sb);
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
        if (t[i].parent < added) break;
        if (t[i].parent != added) continue;
        int content = jt_obj_get(j, t, nt, i, "content");
        int special = jt_obj_get(j, t, nt, i, "special");
        if (content < 0 || special < 0 || !jt_literal(j, &t[special], "true"))
            die("unsupported added token");
        char *s = jt_string(j, &t[content]);
        Vocab *v = dgt_vfind(d, s, strlen(s));
        free(s);
        if (!v) die("added token absent from vocabulary");
        d->special[si++] = v;
    }
    d->ns = si;
    free(t); free(j);
}
static void dgt_free(DGTokenizer *d) {
    for (uint32_t i = 0; i < d->nv; i++) free(d->vocab[i].s);
    free(d->vocab); free(d->by_id); free(d->merge); free(d->special);
}
static void dgt_piece(DGTokenizer *d, const char *s, size_t n, Tokens *out) {
    uint32_t *ids = xmalloc((n ? n : 1) * sizeof *ids);
    uint32_t k = 0;
    for (size_t p = 0; p < n;) {
        size_t q = p;
        next_cp((const unsigned char *)s, n, &q);
        Vocab *v = dgt_vfind(d, s + p, q - p);
        if (v) ids[k++] = (uint32_t)v->id;
        else {
            for (; p < q; p++) {
                char z[7];
                snprintf(z, sizeof z, "<0x%02X>", (unsigned char)s[p]);
                v = dgt_vfind(d, z, 6);
                if (!v) die("DiffusionGemma byte fallback token absent");
                ids[k++] = (uint32_t)v->id;
            }
            continue;
        }
        p = q;
    }
    while (k > 1) {
        uint32_t rank = UINT32_MAX, at = 0;
        for (uint32_t i = 0; i + 1 < k; i++) {
            Merge *m = dgt_mfind(d, ids[i], ids[i + 1]);
            if (m && m->rank < rank) rank = m->rank, at = i;
        }
        if (rank == UINT32_MAX) break;
        Vocab *a = d->by_id[ids[at]], *b = d->by_id[ids[at + 1]];
        char *z = xmalloc((size_t)a->n + b->n);
        memcpy(z, a->s, a->n); memcpy(z + a->n, b->s, b->n);
        Vocab *v = dgt_vfind(d, z, (size_t)a->n + b->n);
        free(z);
        if (!v) die("DiffusionGemma merged token absent");
        ids[at] = (uint32_t)v->id;
        memmove(ids + at + 1, ids + at + 2, (k - at - 2) * sizeof *ids);
        k--;
    }
    for (uint32_t i = 0; i < k; i++) push(out, (int)ids[i]);
    free(ids);
}
static Tokens dgt_tokenize(DGTokenizer *d, const char *s) {
    Tokens out = {0};
    size_t n = strlen(s), plain = 0, i = 0;
    while (i < n) {
        Vocab *hit = NULL;
        for (uint32_t q = 0; q < d->ns; q++) {
            Vocab *v = d->special[q];
            if (v->n <= n - i && !memcmp(s + i, v->s, v->n) && (!hit || v->n > hit->n)) hit = v;
        }
        if (!hit) { i++; continue; }
        if (i > plain) {
            size_t zcap = (i - plain) * 3 + 1, zlen = 0;
            char *z = xmalloc(zcap);
            for (size_t q = plain; q < i; q++) {
                if (s[q] == ' ') memcpy(z + zlen, "\xE2\x96\x81", 3), zlen += 3;
                else z[zlen++] = s[q];
            }
            dgt_piece(d, z, zlen, &out); free(z);
        }
        push(&out, hit->id); i += hit->n; plain = i;
    }
    if (plain < n) {
        size_t zcap = (n - plain) * 3 + 1, zlen = 0;
        char *z = xmalloc(zcap);
        for (size_t q = plain; q < n; q++) {
            if (s[q] == ' ') memcpy(z + zlen, "\xE2\x96\x81", 3), zlen += 3;
            else z[zlen++] = s[q];
        }
        dgt_piece(d, z, zlen, &out); free(z);
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
typedef struct { float *k, *v; int n, kvh, hd; } DGKV;
typedef struct{uint32_t mt[624];int at;} DGMT;
static void dg_mt_seed(DGMT*r,uint32_t seed){r->mt[0]=19650218u;for(int i=1;i<624;i++)r->mt[i]=1812433253u*(r->mt[i-1]^(r->mt[i-1]>>30))+i;int i=1,j=0;for(int q=624;q;q--){r->mt[i]=(r->mt[i]^((r->mt[i-1]^(r->mt[i-1]>>30))*1664525u))+seed+j;i++;j++;if(i>=624){r->mt[0]=r->mt[623];i=1;}if(j>=1)j=0;}for(int q=623;q;q--){r->mt[i]=(r->mt[i]^((r->mt[i-1]^(r->mt[i-1]>>30))*1566083941u))-i;i++;if(i>=624){r->mt[0]=r->mt[623];i=1;}}r->mt[0]=0x80000000u;r->at=624;}
static uint32_t dg_mt_u32(DGMT*r){if(r->at>=624){for(int i=0;i<624;i++){uint32_t y=(r->mt[i]&0x80000000u)|(r->mt[(i+1)%624]&0x7fffffffu);r->mt[i]=r->mt[(i+397)%624]^(y>>1)^((y&1)?0x9908b0dfu:0);}r->at=0;}uint32_t y=r->mt[r->at++];y^=y>>11;y^=(y<<7)&0x9d2c5680u;y^=(y<<15)&0xefc60000u;y^=y>>18;return y;}
static uint32_t dg_mt_vocab(DGMT*r){uint32_t x;do{x=dg_mt_u32(r)>>13;}while(x>=DG_VOCAB);return x;}

static float dg_bf(const uint8_t *p) {
    uint32_t u = (uint32_t)p[0] << 16 | (uint32_t)p[1] << 24;
    float f; memcpy(&f, &u, 4); return f;
}
static float dg_at(const DGTensor *t, uint64_t i) { return dg_bf(t->data + i * 2); }
static float dg_f8e4m3(uint8_t u){
    int sign=u>>7,e=(u>>3)&15,m=u&7;
    float x=e?ldexpf(1.0f+m/8.0f,e-7):ldexpf((float)m,-9);
    return sign?-x:x;
}
static float dg_f8e4m3_round(float x){
    if(!(x>0))return 0;
    if(x>=448)return 448;
    int lo=0,hi=126;while(lo+1<hi){int m=(lo+hi)/2;if(dg_f8e4m3((uint8_t)m)<x)lo=m;else hi=m;}
    float a=dg_f8e4m3((uint8_t)lo),b=dg_f8e4m3((uint8_t)hi),da=x-a,db=b-x;
    return da<db||(da==db&&!(lo&1))?a:b;
}
static float dg_e2m1_round(float x){
    static const float q[8]={0,.5f,1,1.5f,2,3,4,6};float a=fabsf(x);int best=0;
    for(int i=1;i<8;i++){float d=fabsf(a-q[i]),old=fabsf(a-q[best]);if(d<old||(d==old&&!(i&1)))best=i;}
    return signbit(x)?-q[best]:q[best];
}
static void dg_nvfp4_qdq(float*out,const float*in,int tokens,int cols,float base){
    if(!(base>0)||!isfinite(base)||cols%16)die("invalid NVFP4 activation scale");
    for(int t=0;t<tokens;t++)for(int b=0;b<cols/16;b++){
        const float*x=in+(size_t)t*cols+(size_t)b*16;float*y=out+(size_t)t*cols+(size_t)b*16,amax=0;
        for(int k=0;k<16;k++)if(fabsf(x[k])>amax)amax=fabsf(x[k]);
        float s=dg_f8e4m3_round((amax/6)/base)*base;
        if(s==0){memset(y,0,16*sizeof*y);continue;}for(int k=0;k<16;k++)y[k]=dg_e2m1_round(x[k]/s)*s;
    }
#if defined(__AVX512F__)
    /* Match each packed weight byte: low-nibble activations, then high. This
     * one-time swizzle removes two activation permutes per expert row/tile. */
    for(int t=0;t<tokens;t++)for(int c=0;c<cols;c+=32){
        float *y=out+(size_t)t*cols+c,tmp[32];memcpy(tmp,y,sizeof tmp);
        for(int k=0;k<16;k++){y[k]=tmp[k*2];y[16+k]=tmp[k*2+1];}
    }
#endif
}
static void dg_nvfp4_mm(const DGTensor*w,const DGTensor*s,const DGTensor*g,
                        const float*x,float*y,int tokens,int rows,int cols){
    if(w->dtype!=DG_U8||w->nd!=2||w->shape[0]!=(uint64_t)rows||w->shape[1]!=(uint64_t)cols/2||
       s->dtype!=DG_F8E4M3||s->nd!=2||s->shape[0]!=(uint64_t)rows||s->shape[1]!=(uint64_t)cols/16||
       g->dtype!=DG_F32||g->nd!=0||g->bytes!=4||cols%16)die2("bad NVFP4 expert tensor",w->name);
    float global;memcpy(&global,g->data,4);
#if defined(__AVX512F__)
    static const float lut[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
    __m512 table=_mm512_loadu_ps(lut);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int r=0;r<rows;r+=2){
        const uint8_t*wp0=w->data+(uint64_t)r*cols/2,*wp1=wp0+cols/2;
        const uint8_t*sp0=s->data+(uint64_t)r*cols/16,*sp1=sp0+cols/16;
        for(int tb=0;tb<tokens;tb+=8){int nb=tokens-tb<8?tokens-tb:8;__m512 a[8],b[8];for(int q=0;q<nb;q++)a[q]=b[q]=_mm512_setzero_ps();
            for(int c=0;c<cols;c+=32){
                __m512i z=_mm512_set1_epi32(15),raw0=_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(wp0+c/2))),raw1=_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(wp1+c/2)));
                __m512 wl0=_mm512_permutexvar_ps(_mm512_and_si512(raw0,z),table),wh0=_mm512_permutexvar_ps(_mm512_srli_epi32(raw0,4),table);
                __m512 wl1=_mm512_permutexvar_ps(_mm512_and_si512(raw1,z),table),wh1=_mm512_permutexvar_ps(_mm512_srli_epi32(raw1,4),table);
                float a0=dg_f8e4m3(sp0[c/16])*global,a1=dg_f8e4m3(sp0[c/16+1])*global,b0=dg_f8e4m3(sp1[c/16])*global,b1=dg_f8e4m3(sp1[c/16+1])*global;
                __m512 sv0=_mm512_mask_blend_ps(0xff00,_mm512_set1_ps(a0),_mm512_set1_ps(a1)),sv1=_mm512_mask_blend_ps(0xff00,_mm512_set1_ps(b0),_mm512_set1_ps(b1));
                wl0=_mm512_mul_ps(wl0,sv0);wh0=_mm512_mul_ps(wh0,sv0);wl1=_mm512_mul_ps(wl1,sv1);wh1=_mm512_mul_ps(wh1,sv1);
                for(int q=0;q<nb;q++){const float*xp=x+(size_t)(tb+q)*cols+c;__m512 xe=_mm512_loadu_ps(xp),xo=_mm512_loadu_ps(xp+16);a[q]=_mm512_fmadd_ps(wl0,xe,a[q]);a[q]=_mm512_fmadd_ps(wh0,xo,a[q]);b[q]=_mm512_fmadd_ps(wl1,xe,b[q]);b[q]=_mm512_fmadd_ps(wh1,xo,b[q]);}
            }
            for(int q=0;q<nb;q++){y[(size_t)(tb+q)*rows+r]=_mm512_reduce_add_ps(a[q]);y[(size_t)(tb+q)*rows+r+1]=_mm512_reduce_add_ps(b[q]);}
        }
    }
    return;
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int r=0;r<rows;r++)for(int t=0;t<tokens;t++){
        const uint8_t*wp=w->data+(uint64_t)r*cols/2,*sp=s->data+(uint64_t)r*cols/16;
        const float*xp=x+(size_t)t*cols;float sum=0;
        for(int b=0;b<cols/16;b++){
            float scale=dg_f8e4m3(sp[b])*global;const uint8_t*q=wp+(size_t)b*8;const float*a=xp+(size_t)b*16;
            for(int k=0;k<8;k++){uint8_t v=q[k];int lo=v&7,hi=(v>>4)&7;float wl=(float[]){0,.5f,1,1.5f,2,3,4,6}[lo],wh=(float[]){0,.5f,1,1.5f,2,3,4,6}[hi];if(v&8)wl=-wl;if(v&128)wh=-wh;sum+=scale*(wl*a[k*2]+wh*a[k*2+1]);}
        }
        y[(size_t)t*rows+r]=sum;
    }
#endif
}
#if defined(__AVX512F__)
static __m512 dg_bf16x16(const uint8_t *p){
    __m256i h=_mm256_loadu_si256((const __m256i*)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h),16));
}
static void dg_mm_data_avx512(const uint8_t*data,const float*x,float*y,int tokens,int rows,int cols){
    if(rows%2)die("AVX-512 matrix row count must be even");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int r=0;r<rows;r+=2){
        const uint8_t*p0=data+(uint64_t)r*cols*2,*p1=p0+(uint64_t)cols*2;int t=0;
        for(;t+7<tokens;t+=8){__m512 a[8],b[8];for(int q=0;q<8;q++)a[q]=b[q]=_mm512_setzero_ps();int c=0;
            for(;c+15<cols;c+=16){__m512 w0=dg_bf16x16(p0+c*2),w1=dg_bf16x16(p1+c*2);for(int q=0;q<8;q++){__m512 v=_mm512_loadu_ps(x+(size_t)(t+q)*cols+c);a[q]=_mm512_fmadd_ps(w0,v,a[q]);b[q]=_mm512_fmadd_ps(w1,v,b[q]);}}
            for(int q=0;q<8;q++){float z0=_mm512_reduce_add_ps(a[q]),z1=_mm512_reduce_add_ps(b[q]);for(int k=(cols&~15);k<cols;k++){float v=x[(size_t)(t+q)*cols+k];z0+=dg_bf(p0+k*2)*v;z1+=dg_bf(p1+k*2)*v;}y[(size_t)(t+q)*rows+r]=z0;y[(size_t)(t+q)*rows+r+1]=z1;}}
        for(;t<tokens;t++){__m512 s0=_mm512_setzero_ps(),s1=s0;int c=0;for(;c+15<cols;c+=16){__m512 v=_mm512_loadu_ps(x+(size_t)t*cols+c);s0=_mm512_fmadd_ps(dg_bf16x16(p0+c*2),v,s0);s1=_mm512_fmadd_ps(dg_bf16x16(p1+c*2),v,s1);}float z0=_mm512_reduce_add_ps(s0),z1=_mm512_reduce_add_ps(s1);for(;c<cols;c++){float v=x[(size_t)t*cols+c];z0+=dg_bf(p0+c*2)*v;z1+=dg_bf(p1+c*2)*v;}y[(size_t)t*rows+r]=z0;y[(size_t)t*rows+r+1]=z1;}
    }
}
#endif
static void dg_mm_data(const uint8_t*data,const float*x,float*y,int tokens,int rows,int cols){
    if(tokens<1||tokens>JB_MAX_CTX)die("DiffusionGemma sequence exceeds context limit");
#if defined(__AVX512F__)
    dg_mm_data_avx512(data,x,y,tokens,rows,cols);return;
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int r=0;r<rows;r++){
        float sum[JB_MAX_CTX];for(int t=0;t<tokens;t++)sum[t]=0;
        const uint8_t*p=data+(uint64_t)r*cols*2;
        for(int c=0;c<cols;c++){float a=dg_bf(p+c*2);for(int t=0;t<tokens;t++)sum[t]+=a*x[(size_t)t*cols+c];}
        for(int t=0;t<tokens;t++)y[(size_t)t*rows+r]=sum[t];
    }
}
static void dg_mm(const DGTensor*w,const float*x,float*y,int tokens,int rows,int cols){if(w->nd!=2||w->shape[0]!=(uint64_t)rows||w->shape[1]!=(uint64_t)cols)die2("bad matrix shape",w->name);dg_mm_data(w->data,x,y,tokens,rows,cols);}
static void dg_mv_slice(const DGTensor *w, uint64_t base, const float *x, float *y, int rows, int cols) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r++) {
        const uint8_t *p = w->data + (base + (uint64_t)r * cols) * 2;
        float s = 0.0f;
        for (int c = 0; c < cols; c++) s += dg_bf(p + c * 2) * x[c];
        y[r] = s;
    }
}
/* Project all answer slots together so the tied LM head is streamed once.
 * OpenJev's automatic reread test is entropy over the union of the
 * full-vocabulary top 20 and the explicitly requested label ids. */
static double dg_slot_logits_entropy(const DGTensor *w,const float *hidden,int n,
                                     int **label_ids,const int *label_n,
                                     double **label_score) {
    if(n<1||n>128)die("bad DiffusionGemma answer slot count");
    /* vLLM returns top-20 plus the sorted union of requested label ids at
     * every slot, capped at its per-request 128-id limit. OpenJev computes
     * entropy over that entire returned set. Keep the 128 smallest unique
     * ids without allocating a vocabulary-sized bitmap. */
    int requested[128],nr=0;
    for(int t=0;t<n;t++)for(int c=0;c<label_n[t];c++){
        int id=label_ids[t][c],at=0;
        while(at<nr&&requested[at]<id)at++;
        if(at<nr&&requested[at]==id)continue;
        if(nr<128){memmove(requested+at+1,requested+at,(size_t)(nr-at)*sizeof*requested);requested[at]=id;nr++;}
        else if(at<128){memmove(requested+at+1,requested+at,(size_t)(127-at)*sizeof*requested);requested[at]=id;}
    }
    float *logits=xmalloc((size_t)n*DG_VOCAB*sizeof*logits);
    dg_mm_data(w->data,hidden,logits,n,DG_VOCAB,DG_H);
    double max_entropy=0;
    for(int t=0;t<n;t++){
        float *row=logits+(size_t)t*DG_VOCAB,topv[20];int topi[20];
        for(int k=0;k<20;k++){topv[k]=-INFINITY;topi[k]=-1;}
        float mx=-INFINITY;
        for(int v=0;v<DG_VOCAB;v++){
            float z=30.0f*tanhf(row[v]/30.0f);row[v]=z;if(z>mx)mx=z;
            for(int k=0;k<20;k++)if(z>topv[k]){for(int q=19;q>k;q--){topv[q]=topv[q-1];topi[q]=topi[q-1];}topv[k]=z;topi[k]=v;break;}
        }
        double den=0;for(int v=0;v<DG_VOCAB;v++)den+=exp((double)row[v]-mx);
        double entropy=0;
        for(int k=0;k<20;k++){double p=exp((double)topv[k]-mx)/den;if(p>0)entropy-=p*log(p);}
        for(int c=0;c<label_n[t];c++)label_score[t][c]=row[label_ids[t][c]];
        for(int c=0;c<nr;c++){
            int id=requested[c],seen=0;
            for(int k=0;k<20;k++)if(topi[k]==id){seen=1;break;}
            if(!seen){double p=exp((double)row[id]-mx)/den;if(p>0)entropy-=p*log(p);}
        }
        if(entropy>max_entropy)max_entropy=entropy;
    }
    free(logits);return max_entropy;
}
static void dg_rms(float *y, const float *x, const DGTensor *scale, int n) {
#if defined(__AVX512F__)
    __m512d a=_mm512_setzero_pd(),b=a,c=a,d=a;int i=0;
    for(;i+31<n;i+=32){
        __m512 x0=_mm512_loadu_ps(x+i),x1=_mm512_loadu_ps(x+i+16);
        __m512d q0=_mm512_cvtps_pd(_mm512_castps512_ps256(x0));
        __m512d q1=_mm512_cvtps_pd(_mm512_extractf32x8_ps(x0,1));
        __m512d q2=_mm512_cvtps_pd(_mm512_castps512_ps256(x1));
        __m512d q3=_mm512_cvtps_pd(_mm512_extractf32x8_ps(x1,1));
        a=_mm512_fmadd_pd(q0,q0,a);b=_mm512_fmadd_pd(q1,q1,b);
        c=_mm512_fmadd_pd(q2,q2,c);d=_mm512_fmadd_pd(q3,q3,d);
    }
    double ss=_mm512_reduce_add_pd(_mm512_add_pd(_mm512_add_pd(a,b),_mm512_add_pd(c,d)));
    for(;i<n;i++)ss+=(double)x[i]*x[i];
    float q=1.0f/sqrtf((float)(ss/n)+1e-6f);i=0;
    if(scale)for(;i+15<n;i+=16)_mm512_storeu_ps(y+i,_mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x+i),_mm512_set1_ps(q)),dg_bf16x16(scale->data+(size_t)i*2)));
    else for(;i+15<n;i+=16)_mm512_storeu_ps(y+i,_mm512_mul_ps(_mm512_loadu_ps(x+i),_mm512_set1_ps(q)));
    for(;i<n;i++)y[i]=x[i]*q*(scale?dg_at(scale,i):1.0f);
#else
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float q = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    for (int i = 0; i < n; i++) y[i] = x[i] * q * (scale ? dg_at(scale, i) : 1.0f);
#endif
}
static float dg_gelu(float x) {
    return .5f * x * (1.0f + tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
}
static double dg_dot(const float*a,const float*b,int n){
#if defined(__AVX512F__)
    __m512d s0=_mm512_setzero_pd(),s1=s0;int i=0;for(;i+15<n;i+=16){__m512 x=_mm512_loadu_ps(a+i),y=_mm512_loadu_ps(b+i);s0=_mm512_fmadd_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(x)),_mm512_cvtps_pd(_mm512_castps512_ps256(y)),s0);s1=_mm512_fmadd_pd(_mm512_cvtps_pd(_mm512_extractf32x8_ps(x,1)),_mm512_cvtps_pd(_mm512_extractf32x8_ps(y,1)),s1);}double s=_mm512_reduce_add_pd(_mm512_add_pd(s0,s1));for(;i<n;i++)s+=(double)a[i]*b[i];return s;
#else
    double s=0;for(int i=0;i<n;i++)s+=(double)a[i]*b[i];return s;
#endif
}
static DGTensor *dg_layer_tensor(DGModel *m, int l, const char *tail) {
    char n[192]; snprintf(n, sizeof n, "model.decoder.layers.%d.%s", l, tail); return dg_tensor(m, n);
}
static void dg_rope(float *x,int heads,int hd,const float *cv,const float *sv) {
    int half=hd/2;
    for(int h=0;h<heads;h++){float*v=x+(size_t)h*hd;for(int i=0;i<half;i++){float a=v[i],b=v[i+half];v[i]=a*cv[i]-b*sv[i];v[i+half]=b*cv[i]+a*sv[i];}}
}
static void dg_norm_heads(float *x, int heads, int hd, DGTensor *scale) {
    for(int h=0;h<heads;h++)dg_rms(x+(size_t)h*hd,x+(size_t)h*hd,scale,hd);
}
static void dg_attention(DGModel *m, int l, float *x, int n, int pos0, DGKV *cache, int decoder) {
    int full = l % 6 == 5, hd = full ? 512 : 256, kvh = full ? 2 : 8;
    int qn = DG_HEADS * hd, kn = kvh * hd;
    DGTensor *qw = dg_layer_tensor(m,l,"self_attn.q_proj.weight");
    DGTensor *kw = dg_layer_tensor(m,l,"self_attn.k_proj.weight");
    DGTensor *vw = full ? NULL : dg_layer_tensor(m,l,"self_attn.v_proj.weight");
    DGTensor *qnrm = dg_layer_tensor(m,l,"self_attn.q_norm.weight");
    DGTensor *knrm = dg_layer_tensor(m,l,"self_attn.k_norm.weight");
    float *q = xmalloc((size_t)n * qn * 4), *k = xmalloc((size_t)n * kn * 4), *v = xmalloc((size_t)n * kn * 4);
    dg_mm(qw,x,q,n,qn,DG_H);
    dg_mm(kw,x,k,n,kn,DG_H);
    if(vw)dg_mm(vw,x,v,n,kn,DG_H);else memcpy(v,k,(size_t)n*kn*4);
    int half=hd/2,rotated=full?64:half;float inv[256];
    for(int i=0;i<half;i++)inv[i]=i<rotated?powf(full?1000000.0f:10000.0f,-(float)(2*i)/hd):0.0f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n; t++) {
        float cv[256],sv[256];for(int i=0;i<half;i++){cv[i]=cosf((pos0+t)*inv[i]);sv[i]=sinf((pos0+t)*inv[i]);}
        dg_norm_heads(q+(size_t)t*qn,DG_HEADS,hd,qnrm);
        dg_norm_heads(k+(size_t)t*kn,kvh,hd,knrm);
        dg_norm_heads(v+(size_t)t*kn,kvh,hd,NULL);
        dg_rope(q+(size_t)t*qn,DG_HEADS,hd,cv,sv);
        dg_rope(k+(size_t)t*kn,kvh,hd,cv,sv);
    }
    int old = decoder && cache ? cache->n : 0;
    float *catk = k, *catv = v;
    if (old) {
        catk=xmalloc((size_t)(old+n)*kn*4); catv=xmalloc((size_t)(old+n)*kn*4);
        memcpy(catk,cache->k,(size_t)old*kn*4); memcpy(catv,cache->v,(size_t)old*kn*4);
        memcpy(catk+(size_t)old*kn,k,(size_t)n*kn*4); memcpy(catv+(size_t)old*kn,v,(size_t)n*kn*4);
    }
    float *a = xcalloc((size_t)n * qn, 4), *score = xmalloc((size_t)n*(old+n)*4);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int t=0;t<n;t++) for(int h=0;h<DG_HEADS;h++) {
        float *tscore=score+(size_t)t*(old+n);
        int kh=h/(DG_HEADS/kvh),end=decoder?old+n:old+t+1,start=0;
        if (!full) {
            if (decoder) {
                /* OpenJev's non-causal decoder symmetrizes Gemma's 1024-token
                 * window around each absolute query position. */
                int pos=old+t;
                start=pos>=1024?pos-1024+1:0;
                if(end>pos+1024)end=pos+1024;
            } else start=end>1024?end-1024:0;
        }
        float mx=-INFINITY;
        const float *qq=q+(size_t)t*qn+(size_t)h*hd;
        for(int j=start;j<end;j++) {const float *kk=catk+(size_t)j*kn+(size_t)kh*hd;tscore[j]=(float)dg_dot(qq,kk,hd);if(tscore[j]>mx)mx=tscore[j];}
        float den=0;for(int j=start;j<end;j++)den+=expf(tscore[j]-mx);
        float *oo=a+(size_t)t*qn+(size_t)h*hd;
        for(int j=start;j<end;j++){float p=expf(tscore[j]-mx)/den;const float *vv=catv+(size_t)j*kn+(size_t)kh*hd;for(int d=0;d<hd;d++)oo[d]+=p*vv[d];}
    }
    DGTensor *ow=dg_layer_tensor(m,l,"self_attn.o_proj.weight");
    dg_mm(ow,a,x,n,DG_H,qn);
    if (!decoder && cache) { cache->k=k;cache->v=v;cache->n=n;cache->kvh=kvh;cache->hd=hd;k=v=NULL; }
    if(old){free(catk);free(catv);} free(q);free(k);free(v);free(a);free(score);
}
static void dg_ff(DGModel *m, int l, float *x, int n) {
    JB_TICK(ff_start);
    DGTensor *pre=dg_layer_tensor(m,l,"pre_feedforward_layernorm.weight");
    DGTensor *gw=dg_layer_tensor(m,l,"mlp.gate_proj.weight"),*uw=dg_layer_tensor(m,l,"mlp.up_proj.weight"),*dw=dg_layer_tensor(m,l,"mlp.down_proj.weight");
    DGTensor *p1=dg_layer_tensor(m,l,"post_feedforward_layernorm_1.weight");
    DGTensor *pre2=dg_layer_tensor(m,l,"pre_feedforward_layernorm_2.weight");
    DGTensor *rw=dg_layer_tensor(m,l,"router.proj.weight"),*rs=dg_layer_tensor(m,l,"router.scale"),*re=dg_layer_tensor(m,l,"router.per_expert_scale");
    DGTensor *eg=m->nvfp4?NULL:dg_layer_tensor(m,l,"experts.gate_up_proj"),*ed=m->nvfp4?NULL:dg_layer_tensor(m,l,"experts.down_proj");
    DGTensor *p2=dg_layer_tensor(m,l,"post_feedforward_layernorm_2.weight"),*post=dg_layer_tensor(m,l,"post_feedforward_layernorm.weight");
    float *z1=xmalloc((size_t)n*DG_H*4),*g=xmalloc((size_t)n*DG_DENSE*4),*u=xmalloc((size_t)n*DG_DENSE*4),*d=xmalloc((size_t)n*DG_H*4),*z2=xmalloc((size_t)n*DG_H*4),*rin=xmalloc((size_t)n*DG_H*4),*route=xmalloc((size_t)n*128*4);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int t=0;t<n;t++)dg_rms(z1+(size_t)t*DG_H,x+(size_t)t*DG_H,pre,DG_H);
    dg_mm(gw,z1,g,n,DG_DENSE,DG_H);dg_mm(uw,z1,u,n,DG_DENSE,DG_H);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t i=0;i<(size_t)n*DG_DENSE;i++)g[i]=dg_gelu(g[i])*u[i];
    dg_mm(dw,g,d,n,DG_H,DG_DENSE);
    JB_TO(dense,ff_start); JB_TICK(router_start);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int t=0;t<n;t++){float*r=x+(size_t)t*DG_H;dg_rms(d+(size_t)t*DG_H,d+(size_t)t*DG_H,p1,DG_H);dg_rms(z2+(size_t)t*DG_H,r,pre2,DG_H);dg_rms(rin+(size_t)t*DG_H,r,NULL,DG_H);for(int i=0;i<DG_H;i++)rin[(size_t)t*DG_H+i]*=dg_at(rs,i)/sqrtf(DG_H);}
    dg_mm(rw,rin,route,n,128,DG_H);
    int *top=xmalloc((size_t)n*DG_TOPK*sizeof*top);float*tw=xmalloc((size_t)n*DG_TOPK*4);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int t=0;t<n;t++){
        float *rt=route+(size_t)t*128;
        int ix[DG_TOPK];float ev[DG_TOPK];for(int k=0;k<DG_TOPK;k++){ix[k]=-1;ev[k]=-INFINITY;}for(int e=0;e<128;e++)for(int k=0;k<DG_TOPK;k++)if(rt[e]>ev[k]){for(int q=DG_TOPK-1;q>k;q--){ev[q]=ev[q-1];ix[q]=ix[q-1];}ev[k]=rt[e];ix[k]=e;break;}
        float mx=rt[0];for(int e=1;e<128;e++)if(rt[e]>mx)mx=rt[e];double all=0,sel=0;for(int e=0;e<128;e++)all+=exp((double)rt[e]-mx);for(int k=0;k<DG_TOPK;k++){ev[k]=(float)(exp((double)ev[k]-mx)/all);sel+=ev[k];}
        for(int k=0;k<DG_TOPK;k++){top[(size_t)t*DG_TOPK+k]=ix[k];tw[(size_t)t*DG_TOPK+k]=ev[k]/(float)sel*dg_at(re,ix[k]);}
    }
    JB_TO(router,router_start); JB_TICK(expert_start);
    float*contrib=xcalloc((size_t)n*DG_TOPK*DG_H,4),*gather=xmalloc((size_t)n*DG_H*4),*gu=xmalloc((size_t)n*1408*4),*hid=xmalloc((size_t)n*DG_MOE*4),*eo=xmalloc((size_t)n*DG_H*4),*qz2=m->nvfp4?xmalloc((size_t)n*DG_H*4):NULL;int*owner=xmalloc((size_t)n*2*sizeof*owner);if(qz2){JB_TICK(input_qdq_start);dg_nvfp4_qdq(qz2,z2,n,DG_H,m->nv_a13[l]);JB_TO(moe_input_qdq,input_qdq_start);}
    for(int e=0;e<128;e++){int ne=0;for(int t=0;t<n;t++)for(int k=0;k<DG_TOPK;k++)if(top[(size_t)t*DG_TOPK+k]==e){owner[ne*2]=t;owner[ne*2+1]=k;memcpy(gather+(size_t)ne*DG_H,(qz2?qz2:z2)+(size_t)t*DG_H,DG_H*4);ne++;}if(!ne)continue;
        if(m->nvfp4){DGNvExpert*v=&m->nvexpert[l*128+e];float*qh=xmalloc((size_t)ne*DG_MOE*4);
            JB_TICK(gate_start);dg_nvfp4_mm(v->wg,v->sg,v->gg,gather,gu,ne,DG_MOE,DG_H);JB_TO(moe_gate,gate_start);
            JB_TICK(up_start);dg_nvfp4_mm(v->wu,v->su,v->gu,gather,hid,ne,DG_MOE,DG_H);JB_TO(moe_up,up_start);
            JB_TICK(act_start);for(int q=0;q<ne;q++)for(int i=0;i<DG_MOE;i++)hid[(size_t)q*DG_MOE+i]=dg_gelu(gu[(size_t)q*DG_MOE+i])*hid[(size_t)q*DG_MOE+i];JB_TO(moe_activation,act_start);
            JB_TICK(hidden_qdq_start);dg_nvfp4_qdq(qh,hid,ne,DG_MOE,m->nv_a2[l]);JB_TO(moe_hidden_qdq,hidden_qdq_start);
            JB_TICK(down_start);dg_nvfp4_mm(v->wd,v->sd,v->gd,qh,eo,ne,DG_H,DG_MOE);JB_TO(moe_down,down_start);free(qh);
        }else{const uint8_t*gp=eg->data+(uint64_t)e*1408*DG_H*2,*dp=ed->data+(uint64_t)e*DG_H*DG_MOE*2;dg_mm_data(gp,gather,gu,ne,1408,DG_H);for(int q=0;q<ne;q++)for(int i=0;i<DG_MOE;i++)hid[(size_t)q*DG_MOE+i]=dg_gelu(gu[(size_t)q*1408+i])*gu[(size_t)q*1408+DG_MOE+i];dg_mm_data(dp,hid,eo,ne,DG_H,DG_MOE);}
        for(int q=0;q<ne;q++){int t=owner[q*2],k=owner[q*2+1];float wt=tw[(size_t)t*DG_TOPK+k],*dst=contrib+((size_t)t*DG_TOPK+k)*DG_H,*src=eo+(size_t)q*DG_H;for(int i=0;i<DG_H;i++)dst[i]=wt*src[i];}}
    JB_TO(experts,expert_start); JB_TICK(ff_tail_start);
    float*mo=xmalloc(DG_H*4),*sum=xmalloc(DG_H*4);for(int t=0;t<n;t++){float*r=x+(size_t)t*DG_H;memset(mo,0,DG_H*4);for(int k=0;k<DG_TOPK;k++){float*src=contrib+((size_t)t*DG_TOPK+k)*DG_H;for(int i=0;i<DG_H;i++)mo[i]+=src[i];}dg_rms(mo,mo,p2,DG_H);for(int i=0;i<DG_H;i++)sum[i]=d[(size_t)t*DG_H+i]+mo[i];dg_rms(sum,sum,post,DG_H);for(int i=0;i<DG_H;i++)r[i]+=sum[i];}
    free(z1);free(g);free(u);free(d);free(z2);free(rin);free(route);free(top);free(tw);free(contrib);free(gather);free(gu);free(hid);free(eo);free(qz2);free(owner);free(mo);free(sum);
    JB_TO(ff_other,ff_tail_start);
}
static void dg_layer(DGModel *m,int l,float *x,int n,int pos0,DGKV *cache,int decoder){
    DGTensor *in=dg_layer_tensor(m,l,"input_layernorm.weight"),*pa=dg_layer_tensor(m,l,"post_attention_layernorm.weight");
    float *res=xmalloc((size_t)n*DG_H*4),*z=xmalloc((size_t)n*DG_H*4);memcpy(res,x,(size_t)n*DG_H*4);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int t=0;t<n;t++) dg_rms(z+(size_t)t*DG_H,x+(size_t)t*DG_H,in,DG_H);
    JB_TICK(attention_start); dg_attention(m,l,z,n,pos0,cache,decoder); JB_TO(attention,attention_start);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int t=0;t<n;t++){dg_rms(x+(size_t)t*DG_H,z+(size_t)t*DG_H,pa,DG_H);for(int i=0;i<DG_H;i++)x[(size_t)t*DG_H+i]+=res[(size_t)t*DG_H+i];}
    dg_ff(m,l,x,n);float sc=dg_at(dg_layer_tensor(m,l,"layer_scalar"),0);for(size_t i=0;i<(size_t)n*DG_H;i++)x[i]*=sc;free(res);free(z);
}
static float *dg_embed(DGModel*m,const int*ids,int n,int decoder){DGTensor*w=dg_tensor(m,"model.decoder.embed_tokens.weight");float*x=xmalloc((size_t)n*DG_H*4);float scale=sqrtf(DG_H);for(int t=0;t<n;t++){for(int i=0;i<DG_H;i++)x[(size_t)t*DG_H+i]=dg_at(w,(uint64_t)ids[t]*DG_H+i)*scale;if(decoder)dg_rms(x+(size_t)t*DG_H,x+(size_t)t*DG_H,NULL,DG_H);}return x;}
static void dg_final_norm(DGModel*m,float*x,int n){DGTensor*w=dg_tensor(m,"model.decoder.norm.weight");for(int t=0;t<n;t++)dg_rms(x+(size_t)t*DG_H,x+(size_t)t*DG_H,w,DG_H);}
static void dg_prefill(DGModel*m,const int*prompt,int np,DGKV kv[DG_L]){float*enc=dg_embed(m,prompt,np,0);for(int l=0;l<DG_L;l++)dg_layer(m,l,enc,np,0,&kv[l],0);free(enc);}
static float *dg_decode(DGModel*m,DGKV kv[DG_L],int np,const int*canvas,int nc){float*dec=dg_embed(m,canvas,nc,1);for(int l=0;l<DG_L;l++)dg_layer(m,l,dec,nc,np,&kv[l],1);dg_final_norm(m,dec,nc);return dec;}
static void dg_free_kv(DGKV kv[DG_L]){for(int l=0;l<DG_L;l++){free(kv[l].k);free(kv[l].v);}}
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
typedef struct {char*p;size_t n,cap;} DGBuf;
static void db_need(DGBuf*b,size_t add){if(add>SIZE_MAX-b->n-1)die("string size overflow");size_t need=b->n+add+1;if(need>b->cap){size_t c=b->cap?b->cap:256;while(c<need){if(c>SIZE_MAX/2)die("string size overflow");c*=2;}b->p=xrealloc(b->p,c);b->cap=c;}}
static void db_mem(DGBuf*b,const char*p,size_t n){db_need(b,n);memcpy(b->p+b->n,p,n);b->n+=n;b->p[b->n]=0;}
static void db_ch(DGBuf*b,char c){db_need(b,1);b->p[b->n++]=c;b->p[b->n]=0;}
static void db_fmt(DGBuf*b,const char*fmt,...){va_list a,z;va_start(a,fmt);va_copy(z,a);int n=vsnprintf(NULL,0,fmt,z);va_end(z);if(n<0)die("formatting failed");db_need(b,(size_t)n);vsnprintf(b->p+b->n,(size_t)n+1,fmt,a);va_end(a);b->n+=(size_t)n;}
static void db_json_string(DGBuf*b,const char*s,int ascii){db_ch(b,'"');size_t n=strlen(s),i=0;while(i<n){size_t at=i;uint32_t cp=next_cp((const unsigned char*)s,n,&i);if(cp=='"'||cp=='\\'){db_ch(b,'\\');db_ch(b,(char)cp);}else if(cp=='\b')db_mem(b,"\\b",2);else if(cp=='\f')db_mem(b,"\\f",2);else if(cp=='\n')db_mem(b,"\\n",2);else if(cp=='\r')db_mem(b,"\\r",2);else if(cp=='\t')db_mem(b,"\\t",2);else if(cp<32||(ascii&&cp>127)){char z[16];if(cp<=0xffff)snprintf(z,sizeof z,"\\u%04x",cp);else{cp-=0x10000;snprintf(z,sizeof z,"\\u%04x\\u%04x",0xd800+(cp>>10),0xdc00+(cp&1023));}db_mem(b,z,strlen(z));}else db_mem(b,s+at,i-at);}db_ch(b,'"');}
typedef struct{int tok;char*key;} DGKey;
static int dg_key_cmp(const void*a,const void*b){return strcmp(((const DGKey*)a)->key,((const DGKey*)b)->key);}
static void dg_json_value(DGBuf*b,const char*j,JTok*t,int nt,int at,int sort_keys,int ascii){
    if(t[at].type==JT_STRING){char*s=jt_string(j,&t[at]);db_json_string(b,s,ascii);free(s);return;}
    if(t[at].type==JT_PRIMITIVE){db_mem(b,j+t[at].start,(size_t)(t[at].end-t[at].start));return;}
    if(t[at].type==JT_ARRAY){db_ch(b,'[');int first=1;for(int i=at+1;i<nt;i++){if(t[i].parent<at)break;if(t[i].parent==at){if(!first)db_mem(b,", ",2);dg_json_value(b,j,t,nt,i,sort_keys,ascii);first=0;}}db_ch(b,']');return;}
    int nk=t[at].size/2;DGKey*k=xcalloc((size_t)nk,sizeof*k);int z=0;for(int i=at+1;i+1<nt;i++){if(t[i].parent<at)break;if(t[i].parent==at){k[z].tok=i;k[z].key=jt_string(j,&t[i]);z++;i++;}}if(z!=nk)die("bad JSON object");if(sort_keys)qsort(k,(size_t)nk,sizeof*k,dg_key_cmp);db_ch(b,'{');for(int i=0;i<nk;i++){if(i)db_mem(b,", ",2);db_json_string(b,k[i].key,ascii);db_mem(b,": ",2);dg_json_value(b,j,t,nt,k[i].tok+1,sort_keys,ascii);free(k[i].key);}db_ch(b,'}');free(k);
}
static char *dg_json_canonical(const char*j,JTok*t,int nt,int at,int sort_keys,int ascii){DGBuf b={0};dg_json_value(&b,j,t,nt,at,sort_keys,ascii);if(!b.p)return xstrdup("");return b.p;}
static int dg_py_space(uint32_t c){return(c>=9&&c<=13)||(c>=0x1c&&c<=0x20)||c==0x85||c==0xa0||c==0x1680||(c>=0x2000&&c<=0x200a)||c==0x2028||c==0x2029||c==0x202f||c==0x205f||c==0x3000;}
static char *dg_text_of(const char*j,JTok*t,int nt,int at){
    if(t[at].type!=JT_STRING){if(jt_literal(j,&t[at],"null"))return xstrdup("");return dg_json_canonical(j,t,nt,at,0,0);}
    char*s=jt_string(j,&t[at]);size_t n=strlen(s),p=0,first=n,last=0;
    while(p<n){size_t start=p;uint32_t c=next_cp((const unsigned char*)s,n,&p);if(!dg_py_space(c)){if(first==n)first=start;last=p;}}
    if(first==n){s[0]=0;return s;}memmove(s,s+first,last-first);s[last-first]=0;return s;
}

static uint32_t dg_rotr(uint32_t x,int n){return(x>>n)|(x<<(32-n));}
static void dg_sha256(const uint8_t*p,size_t n,uint8_t out[32]){static const uint32_t k[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};uint64_t bits=(uint64_t)n*8;size_t total=(n+9+63)&~(size_t)63;uint8_t*msg=xcalloc(total,1);memcpy(msg,p,n);msg[n]=0x80;for(int i=0;i<8;i++)msg[total-1-i]=(uint8_t)(bits>>(8*i));for(size_t off=0;off<total;off+=64){uint32_t w[64];for(int i=0;i<16;i++)w[i]=(uint32_t)msg[off+4*i]<<24|(uint32_t)msg[off+4*i+1]<<16|(uint32_t)msg[off+4*i+2]<<8|msg[off+4*i+3];for(int i=16;i<64;i++){uint32_t a=w[i-15],c=w[i-2];w[i]=(dg_rotr(c,17)^dg_rotr(c,19)^(c>>10))+w[i-7]+(dg_rotr(a,7)^dg_rotr(a,18)^(a>>3))+w[i-16];}uint32_t a=h[0],bb=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];for(int i=0;i<64;i++){uint32_t t1=hh+(dg_rotr(e,6)^dg_rotr(e,11)^dg_rotr(e,25))+((e&f)^((~e)&g))+k[i]+w[i],t2=(dg_rotr(a,2)^dg_rotr(a,13)^dg_rotr(a,22))+((a&bb)^(a&c)^(bb&c));hh=g;g=f;f=e;e=d+t1;d=c;c=bb;bb=a;a=t1+t2;}h[0]+=a;h[1]+=bb;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;}free(msg);for(int i=0;i<8;i++)for(int q=0;q<4;q++)out[i*4+q]=(uint8_t)(h[i]>>(24-8*q));}
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
    if (!isfinite(mx))
        die("non-finite candidate score");
    for (int i = 1; i < n; i++) {
        if (!isfinite(score[i]))
            die("non-finite candidate score");
        if (score[i] > mx)
            mx = score[i];
    }
    double den = 0;
    for (int i = 0; i < n; i++)
        den += exp(score[i] - mx);
    if (!isfinite(den) || den <= 0)
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

static int dg_choice_inventory(DGTokenizer *tok,char *out[128]){
    Tokens base=dgt_tokenize(tok,"q1: A");int seen[128],n=0;
    for(int ci=0;ci<728&&n<128;ci++){
        char label[3]={0},text[16];
        if(ci<26)label[0]=(char)('A'+ci);else if(ci<52)label[0]=(char)('a'+ci-26);
        else{int j=ci-52;label[0]=(char)('A'+j/26);label[1]=(char)('A'+j%26);}
        snprintf(text,sizeof text,"q1: %s",label);Tokens e=dgt_tokenize(tok,text);int ok=e.n==base.n&&e.n>0;
        for(uint32_t i=0;ok&&i+1<e.n;i++)if(e.v[i]!=base.v[i])ok=0;
        for(int i=0;ok&&i<n;i++)if(seen[i]==e.v[e.n-1])ok=0;
        if(ok){seen[n]=e.v[e.n-1];out[n++]=xstrdup(label);}free(e.v);
    }
    free(base.v);return n;
}
static char *dg_decision_label(const char *kind,int i,char **choice){char z[16];if(!strcmp(kind,"noul"))return xstrdup(i?"no":"yes");if(!strcmp(kind,"score")){snprintf(z,sizeof z,"%d",i);return xstrdup(z);}return xstrdup(choice[i]);}
static char *dg_instruction(const char*j,JTok*t,int nt,int qo){int i=jt_obj_get(j,t,nt,qo,"instructions");return i<0?xstrdup("Answer about the state."):dg_text_of(j,t,nt,i);}
static char *dg_system_prompt(const char *qj,JTok *qt,int qnt,DecisionWork*w,int nq){
    static const char intro[]="Answer a fixed set of questions about the state the user provides. Each question lists its allowed answers; reply with exactly one label per question.\n";
    DGBuf b={0};db_mem(&b,intro,sizeof intro-1);
    for(int x=0;x<nq;x++){int qo=w[x].key+1;char*ins=dg_instruction(qj,qt,qnt,qo);db_fmt(&b,"\nQuestion q%d: %s\n",x+1,*ins?ins:"Answer about the state.");free(ins);int cr=w[x].criteria;
        for(int i=0;i<w[x].nc;i++){char*desc=NULL;if(cr>=0&&qt[cr].type==JT_OBJECT){int z=jt_obj_get(qj,qt,qnt,cr,w[x].cand[i]);if(z>=0)desc=dg_text_of(qj,qt,qnt,z);}else if(cr>=0&&qt[cr].type==JT_ARRAY){int zc=0;for(int z=cr+1;z<qnt;z++)if(qt[z].parent==cr&&zc++==i){desc=dg_text_of(qj,qt,qnt,z);break;}}
            if(!strcmp(w[x].kind,"noul")){if(desc&&*desc)db_fmt(&b,"  %s: %s\n",w[x].label[i],desc);else db_fmt(&b,"  %s\n",w[x].label[i]);}else if(!strcmp(w[x].kind,"score"))db_fmt(&b,"  %s: %s\n",w[x].label[i],desc?desc:"");else if(desc&&*desc)db_fmt(&b,"  %s: %s (%s)\n",w[x].label[i],w[x].cand[i],desc);else db_fmt(&b,"  %s: %s\n",w[x].label[i],w[x].cand[i]);free(desc);}}
    db_fmt(&b,"\n%s",nq<=10?"Reply with one line per question, in this order, formatted as \"id: label\".":"Reply on one line with each question's id immediately followed by its label, separated by single spaces.");return b.p;
}
static char *dg_answer_text(DecisionWork*w,int nq,const int*pick){size_t cap=64;for(int i=0;i<nq;i++)cap+=strlen(w[i].label[pick[i]])+24;char*s=xmalloc(cap),*p=s;for(int i=0;i<nq;i++){if(i)*p++=nq<=10?'\n':' ';p+=sprintf(p,nq<=10?"q%d: %s":"q%d%s",i+1,w[i].label[pick[i]]);}return s;}
static void dg_print_answers(const char*qj,JTok*qt,int qnt,DecisionWork*w,int nq,const char*id,uint32_t tokens,double ms,double prefill_ms,double cand_ms,int reads){printf("{\"model\":\"jev-bush-diffusion-%s\",\"math\":\"%s\",",JB_VERSION,JB_MATH_MODE);if(id){fputs("\"id\":",stdout);json_print_string(id);putchar(',');}fputs("\"answers\":{",stdout);for(int x=0;x<nq;x++){DecisionWork*d=&w[x];if(x)putchar(',');char*k=jt_string(qj,&qt[d->key]);json_print_string(k);free(k);fputs(":{\"type\":",stdout);json_print_string(d->kind);if(!strcmp(d->kind,"noul"))printf(",\"noul\":%.17g,\"probabilities\":{\"true\":%.17g,\"false\":%.17g},\"confidence\":%.17g",d->prob[0],d->prob[0],d->prob[1],confidence(d->prob,d->nc));else if(!strcmp(d->kind,"choice")){int top=0;for(int i=1;i<d->nc;i++)if(d->prob[i]>d->prob[top])top=i;fputs(",\"choice\":",stdout);json_print_string(d->cand[top]);fputs(",\"probabilities\":{",stdout);for(int i=0;i<d->nc;i++){if(i)putchar(',');json_print_string(d->cand[i]);printf(":%.17g",d->prob[i]);}printf("},\"confidence\":%.17g",confidence(d->prob,d->nc));}else{double ev=0;for(int i=0;i<d->nc;i++)ev+=i*d->prob[i];printf(",\"score\":%.17g,\"legend\":{",ev);int zc=0;for(int z=d->criteria+1;z<qnt;z++)if(qt[z].parent==d->criteria){printf("%s\"%d\":",zc?",":"",zc);zc++;json_print_token(qj,&qt[z]);}fputs("},\"probabilities\":{",stdout);for(int i=0;i<d->nc;i++)printf("%s\"%d\":%.17g",i?",":"",i,d->prob[i]);printf("},\"confidence\":%.17g",confidence(d->prob,d->nc));}putchar('}');}printf("},\"usage\":{\"input_tokens\":%u,\"output_tokens\":0},\"timing_ms\":{\"total\":%.3f,\"prefill\":%.3f,\"decode\":%.3f,\"candidates\":%.3f,\"reads\":%d}}\n",tokens,ms,prefill_ms,ms-prefill_ms-cand_ms,cand_ms,reads);}

static int dg_systemone(DGModel*m,DGTokenizer*tok,const char*j,size_t len,const char*line_id){
#ifdef JB_PROFILE
    memset(&jb_profile,0,sizeof jb_profile);
#endif
    int nt;JTok*t=json_tokens(j,len,&nt);if(!nt||t[0].type!=JT_OBJECT)die("System One request must be an object");int si=jt_obj_get(j,t,nt,0,"state"),qi=jt_obj_get(j,t,nt,0,"questions"),ii=jt_obj_get(j,t,nt,0,"id"),sti=jt_obj_get(j,t,nt,0,"samples"),dpi=jt_obj_get(j,t,nt,0,"steps");int has_samples=sti>=0&&!jt_literal(j,&t[sti],"null"),requested=has_samples?jt_nonnegative_int(j,&t[sti],"samples"):0,steps=dpi>=0&&!jt_literal(j,&t[dpi],"null")?jt_nonnegative_int(j,&t[dpi],"steps"):1;if(has_samples&&(requested<1||requested>32))die("samples must be 1..32");if(steps!=1)die("only one-step DiffusionGemma reads are supported");if(si<0||qi<0||(t[qi].type!=JT_OBJECT&&t[qi].type!=JT_STRING))die("request needs state and questions object");char*state=t[si].type==JT_STRING?jt_string(j,&t[si]):dg_json_canonical(j,t,nt,si,0,0),*reqid=ii>=0?jt_string(j,&t[ii]):NULL,*qowned=NULL;const char*qj=j;JTok*qt=t;int qnt=nt,qroot=qi;if(t[qi].type==JT_STRING){qowned=jt_string(j,&t[qi]);qt=json_tokens(qowned,strlen(qowned),&qnt);qj=qowned;qroot=0;if(!qnt||qt[0].type!=JT_OBJECT)die("questions string is not a JSON object");}
    DGBuf seed_json={0};db_ch(&seed_json,'[');dg_json_value(&seed_json,j,t,nt,si,1,1);db_mem(&seed_json,", ",2);dg_json_value(&seed_json,qj,qt,qnt,qroot,1,1);db_ch(&seed_json,']');uint8_t digest[32];dg_sha256((const uint8_t*)seed_json.p,seed_json.n,digest);uint32_t seed0=(uint32_t)digest[0]<<24|(uint32_t)digest[1]<<16|(uint32_t)digest[2]<<8|digest[3];free(seed_json.p);
    int keys[1024],nq=direct_keys(qt,qnt,qroot,keys,1024);if(nq<1||nq>1024)die("questions must contain 1..1024 entries");char*choice_label[128];int nchoice=dg_choice_inventory(tok,choice_label);if(nchoice!=128)die("cannot construct OpenJev choice labels");DecisionWork*w=xcalloc((size_t)nq,sizeof*w);
    for(int x=0;x<nq;x++){DecisionWork*d=&w[x];d->key=keys[x];int qo=keys[x]+1,ty=jt_obj_get(qj,qt,qnt,qo,"type");d->criteria=jt_obj_get(qj,qt,qnt,qo,"criteria");if(ty<0)die("question missing type");d->kind=jt_string(qj,&qt[ty]);d->cand=xcalloc(JB_MAX_CAND,sizeof*d->cand);
        if(!strcmp(d->kind,"noul")){d->cand[d->nc++]=xstrdup("true");d->cand[d->nc++]=xstrdup("false");}else if(!strcmp(d->kind,"choice")){if(d->criteria<0||qt[d->criteria].type!=JT_OBJECT)die("choice needs criteria object");int ck[JB_MAX_CAND];d->nc=direct_keys(qt,qnt,d->criteria,ck,JB_MAX_CAND);for(int i=0;i<d->nc;i++)d->cand[i]=jt_string(qj,&qt[ck[i]]);}else if(!strcmp(d->kind,"score")){if(d->criteria<0||qt[d->criteria].type!=JT_ARRAY)die("score needs criteria array");for(int z=d->criteria+1;z<qnt;z++)if(qt[z].parent==d->criteria){char b[16];snprintf(b,sizeof b,"%d",d->nc);d->cand[d->nc++]=xstrdup(b);}}else die("unknown question type");if(d->nc<2||d->nc>128||(!strcmp(d->kind,"score")&&d->nc>10))die("decision candidate count out of range");d->label=xcalloc((size_t)d->nc,sizeof*d->label);for(int i=0;i<d->nc;i++)d->label[i]=dg_decision_label(d->kind,i,choice_label);d->prob=xmalloc((size_t)d->nc*sizeof*d->prob);}
    for(int i=0;i<nchoice;i++)free(choice_label[i]);
    char*sys=dg_system_prompt(qj,qt,qnt,w,nq);size_t pc=strlen(sys)+strlen(state)+96;char*prompt=xmalloc(pc);snprintf(prompt,pc,"<bos><|turn>system\n%s<turn|>\n<|turn>user\n%s<turn|>\n<|turn>model\n",sys,state);Tokens pt=dgt_tokenize(tok,prompt);
    int*zero=xcalloc((size_t)nq,sizeof*zero);char*ans=dg_answer_text(w,nq,zero);size_t tc=strlen(ans)+64;char*text=xmalloc(tc);snprintf(text,tc,"<|channel>thought\n<channel|>%s",ans);Tokens base=dgt_tokenize(tok,text);if(base.n+1>64)die("OpenJev answer template exceeds 64-token canvas");int*slot=xmalloc((size_t)nq*sizeof*slot);int**ids=xcalloc((size_t)nq,sizeof*ids);
    for(int x=0;x<nq;x++){ids[x]=xmalloc((size_t)w[x].nc*sizeof**ids);slot[x]=-1;for(int c=1;c<w[x].nc;c++){int old=zero[x];zero[x]=c;char*a=dg_answer_text(w,nq,zero);snprintf(text,tc,"<|channel>thought\n<channel|>%s",a);Tokens v=dgt_tokenize(tok,text);free(a);zero[x]=old;if(v.n!=base.n)die("labels do not share one template slot");int diff=-1;for(uint32_t z=0;z<v.n;z++)if(v.v[z]!=base.v[z]){if(diff>=0)die("label changes more than one template token");diff=(int)z;}if(diff<0)die("duplicate label token");if(slot[x]>=0&&diff!=slot[x])die("labels do not share one template slot");slot[x]=diff;ids[x][c]=v.v[diff];free(v.v);}if(slot[x]<0)die("cannot resolve label slot");ids[x][0]=base.v[slot[x]];}
    int width=(int)(((base.n+1+15)/16)*16);int*canvas=xcalloc((size_t)width,sizeof*canvas);DGTensor*emb=dg_tensor(m,"model.decoder.embed_tokens.weight");uint64_t start=now_ns(),candidate_ns=0;DGKV kv[DG_L]={0};dg_prefill(m,pt.v,(int)pt.n,kv);uint64_t prefill_ns=now_ns()-start;for(int x=0;x<nq;x++)memset(w[x].prob,0,(size_t)w[x].nc*sizeof*w[x].prob);int reads=0,max_reads=requested?requested:4,auto_all=0;
    for(int r=0;r<max_reads;r++){memset(canvas,0,(size_t)width*sizeof*canvas);memcpy(canvas,base.v,(size_t)base.n*sizeof*canvas);canvas[base.n]=106;DGMT rng;dg_mt_seed(&rng,seed0+(uint32_t)r*7919u);for(int x=0;x<nq;x++)canvas[slot[x]]=(int)dg_mt_vocab(&rng);float*h=dg_decode(m,kv,(int)pt.n,canvas,width);uint64_t cs=now_ns();double max_entropy=0;double**sc=xmalloc((size_t)nq*sizeof*sc);for(int x=0;x<nq;x++)sc[x]=xmalloc((size_t)w[x].nc*sizeof**sc);
        if(!requested&&r==0){float*answer_h=xmalloc((size_t)nq*DG_H*sizeof*answer_h);int*label_n=xmalloc((size_t)nq*sizeof*label_n);for(int x=0;x<nq;x++){memcpy(answer_h+(size_t)x*DG_H,h+(size_t)slot[x]*DG_H,DG_H*sizeof*answer_h);label_n[x]=w[x].nc;}max_entropy=dg_slot_logits_entropy(emb,answer_h,nq,ids,label_n,sc);free(label_n);free(answer_h);}else for(int x=0;x<nq;x++)for(int c=0;c<w[x].nc;c++){float z;dg_mv_slice(emb,(uint64_t)ids[x][c]*DG_H,h+(size_t)slot[x]*DG_H,&z,1,DG_H);sc[x][c]=30*tanh(z/30);}
        for(int x=0;x<nq;x++){double*pr=xmalloc((size_t)w[x].nc*sizeof*pr);normalize_scores(sc[x],w[x].nc,pr);for(int c=0;c<w[x].nc;c++)w[x].prob[c]+=pr[c];free(pr);free(sc[x]);}free(sc);candidate_ns+=now_ns()-cs;free(h);reads++;if(!requested&&r==0){auto_all=max_entropy>.1;if(!auto_all)break;}}
    dg_free_kv(kv);for(int x=0;x<nq;x++)for(int c=0;c<w[x].nc;c++)w[x].prob[c]/=reads;double cand_ms=candidate_ns/1e6,ms=(now_ns()-start)/1e6;uint32_t billed=pt.n*(requested?reads:1);dg_print_answers(qj,qt,qnt,w,nq,line_id?line_id:reqid,billed,ms,prefill_ns/1e6,cand_ms,reads);
#ifdef JB_PROFILE
    uint64_t detailed=jb_profile.moe_input_qdq+jb_profile.moe_gate+jb_profile.moe_up+jb_profile.moe_activation+jb_profile.moe_hidden_qdq+jb_profile.moe_down;
    double moe_misc=(double)(jb_profile.experts>=detailed?jb_profile.experts-detailed:0)/1e6;
    fprintf(stderr,"JB_PROFILE attention=%.3f dense=%.3f router=%.3f experts=%.3f moe_input_qdq=%.3f moe_gate=%.3f moe_up=%.3f moe_activation=%.3f moe_hidden_qdq=%.3f moe_down=%.3f moe_misc=%.3f ff_other=%.3f total=%.3f\n",jb_profile.attention/1e6,jb_profile.dense/1e6,jb_profile.router/1e6,jb_profile.experts/1e6,jb_profile.moe_input_qdq/1e6,jb_profile.moe_gate/1e6,jb_profile.moe_up/1e6,jb_profile.moe_activation/1e6,jb_profile.moe_hidden_qdq/1e6,jb_profile.moe_down/1e6,moe_misc,jb_profile.ff_other/1e6,ms);
#endif
    fflush(stdout);for(int x=0;x<nq;x++){for(int i=0;i<w[x].nc;i++){free(w[x].cand[i]);free(w[x].label[i]);}free(w[x].cand);free(w[x].label);free(w[x].kind);free(w[x].prob);free(ids[x]);}free(ids);free(slot);free(canvas);free(base.v);free(pt.v);free(text);free(ans);free(zero);free(prompt);free(sys);free(w);free(state);free(reqid);if(qowned){free(qt);free(qowned);}free(t);return 0;
}
static int dg_decide_file(const char*dir,const char*path){size_t n;char*j=read_all(path,&n);DGModel m;DGTokenizer t;dg_load(&m,dir);dgt_load(&t,dir);int rc=dg_systemone(&m,&t,j,n,NULL);dgt_free(&t);dg_free(&m);free(j);return rc;}
static int dg_eval_file(const char*dir,const char*path){FILE*f=!strcmp(path,"-")?stdin:fopen(path,"rb");if(!f)die2("cannot open",path);DGModel m;DGTokenizer t;dg_load(&m,dir);dgt_load(&t,dir);char*line=NULL;size_t cap=0,n=0;int ch;while((ch=fgetc(f))!=EOF){if(ch=='\n'){while(n&&line[n-1]=='\r')n--;if(n)dg_systemone(&m,&t,line,n,NULL);n=0;continue;}if(n==cap){cap=cap?cap*2:4096;if(cap>JB_MAX_JSON)die("JSONL row too large");line=xrealloc(line,cap);}line[n++]=(char)ch;}if(ferror(f))die2("cannot read",path);while(n&&line[n-1]=='\r')n--;if(n)dg_systemone(&m,&t,line,n,NULL);free(line);if(f!=stdin)fclose(f);dgt_free(&t);dg_free(&m);return 0;}

static int selftest(void) {
    uint8_t sh[32],sha_abc[32]={0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};dg_sha256((const uint8_t*)"abc",3,sh);if(memcmp(sh,sha_abc,32))die("SHA-256 self-test failed");DGMT mt;dg_mt_seed(&mt,0);uint32_t py_mt[5]={201979,220500,21225,135746,254766};for(int i=0;i<5;i++)if(dg_mt_vocab(&mt)!=py_mt[i])die("MT19937 self-test failed");const char*cj="{\"state\":{\"b\":2,\"a\":\"é\"},\"questions\":{\"q\":{\"type\":\"noul\",\"instructions\":\"x\"}}}";int cnt;JTok*ct=json_tokens(cj,strlen(cj),&cnt);int cs=jt_obj_get(cj,ct,cnt,0,"state"),cq=jt_obj_get(cj,ct,cnt,0,"questions");DGBuf cb={0};db_ch(&cb,'[');dg_json_value(&cb,cj,ct,cnt,cs,1,1);db_mem(&cb,", ",2);dg_json_value(&cb,cj,ct,cnt,cq,1,1);db_ch(&cb,']');dg_sha256((const uint8_t*)cb.p,cb.n,sh);uint32_t cseed=(uint32_t)sh[0]<<24|(uint32_t)sh[1]<<16|(uint32_t)sh[2]<<8|sh[3];if(cseed!=1123096946u)die("OpenJev canonical seed self-test failed");dg_mt_seed(&mt,cseed);if(dg_mt_vocab(&mt)!=238957u)die("OpenJev canvas RNG self-test failed");free(cb.p);free(ct);
    const char*tx="{\"s\":\" \\u2003x\\u00a0 \",\"o\":{\"b\":2,\"a\":1},\"n\":null}";int txn;JTok*tt=json_tokens(tx,strlen(tx),&txn);int ts=jt_obj_get(tx,tt,txn,0,"s"),to=jt_obj_get(tx,tt,txn,0,"o"),tn=jt_obj_get(tx,tt,txn,0,"n");char*sv=dg_text_of(tx,tt,txn,ts),*ov=dg_text_of(tx,tt,txn,to),*nv=dg_text_of(tx,tt,txn,tn);if(strcmp(sv,"x")||strcmp(ov,"{\"b\": 2, \"a\": 1}")||*nv)die("OpenJev text_of self-test failed");free(sv);free(ov);free(nv);free(tt);
    uint8_t one[2] = {0x80, 0x3f};
    if (dg_bf(one) != 1.0f)
        die("BF16 conversion self-test failed");
    if(dg_f8e4m3(0x01)!=0.001953125f||dg_f8e4m3(0x7e)!=448.0f||
       dg_f8e4m3(0xfe)!=-448.0f||dg_f8e4m3_round(5.25f)!=5.0f||
       dg_e2m1_round(2.5f)!=2.0f||dg_e2m1_round(3.5f)!=4.0f)
        die("NVFP4 conversion self-test failed");
    double score[2] = {1000.0, 999.0}, prob[2];
    normalize_scores(score, 2, prob);
    if (fabs(prob[0] - 0.7310585786300049) > 1e-12 ||
        fabs(prob[0] + prob[1] - 1.0) > 1e-15)
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
    puts("{\"selftest\":\"ok\"}");
    return 0;
}

static void usage(void) {
    fprintf(stderr,
            "Jev Bush %s -- Please clap.\nusage:\n"
            "  jb MODEL_DIR decide REQUEST.json\n"
            "  jb MODEL_DIR eval OPENJEV.jsonl\n"
            "  jb --selftest\n",
            JB_VERSION);
}
int main(int ac, char **av) {
    if (ac == 2 && !strcmp(av[1], "--selftest"))
        return selftest();
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
