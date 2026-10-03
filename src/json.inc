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
static size_t jt_string_copy(const char *j, const JTok *t, char *out);

static int jt_new(JParser *p, int type, int start, int parent) {
    if (p->nt == p->cap) {
        if (p->cap > INT32_MAX / 2)
            die("JSON token array size overflow");
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
        size_t slab_bytes = 0;
        for (int i = obj + 1; i < nt && t[i].parent >= obj; i++)
            if (t[i].parent == obj) {
                if (nk == cap) {
                    if (cap > INT32_MAX / 2)
                        die("JSON key array size overflow");
                    cap = cap ? cap * 2 : 64;
                    keys = xrealloc(keys, (size_t)cap * sizeof *keys);
                }
                slab_bytes =
                    jb_size_add(slab_bytes, jb_size_add((size_t)(t[i].end - t[i].start) * 3, 1));
                nk++;
                i++;
            }
        char local_slab[4096];
        char *slab = slab_bytes <= sizeof local_slab ? local_slab : xmalloc(slab_bytes);
        char *cursor = slab;
        nk = 0;
        for (int i = obj + 1; i < nt && t[i].parent >= obj; i++)
            if (t[i].parent == obj) {
                keys[nk++] = cursor;
                cursor += jt_string_copy(s, &t[i], cursor) + 1;
                i++;
            }
        if (nk > 1)
            qsort(keys, (size_t)nk, sizeof *keys, cmp_str);
        for (int i = 0; i < nk; i++) {
            if (i && !strcmp(keys[i - 1], keys[i])) {
                char message[256];
                snprintf(message, sizeof message, "duplicate JSON object key: %s", keys[i]);
                if (slab != local_slab)
                    jb_release(slab);
                jb_release(keys);
                die(message);
            }
        }
        if (slab != local_slab)
            jb_release(slab);
    }
    jb_release(keys);
}

static JTok *json_tokens(const char *s, size_t n, int *nt) {
    /* Token offsets are int; every caller also enforces this limit earlier. */
    if (n > JB_MAX_JSON)
        die("JSON input is too large");
    JParser parser = {s, n, 0, 0, 0, 0, 0};
    JParser *p = &parser;
    json_parse_value(p, -1);
    while (p->pos < n && isspace((unsigned char)s[p->pos]))
        p->pos++;
    if (p->pos != n)
        die("trailing JSON data");
    json_check_keys(s, p->t, p->nt);
    *nt = p->nt;
    return p->t;
}

static int jt_eq(const char *j, const JTok *t, const char *z) {
    if (t->type != JT_STRING)
        return 0;
    size_t n = (size_t)(t->end - t->start);
    if (!memchr(j + t->start, '\\', n))
        return n == strlen(z) && !memcmp(j + t->start, z, n);
    char *s = jt_string(j, t);
    int eq = !strcmp(s, z);
    jb_release(s);
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

static unsigned hex4(const char *s) {
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        v <<= 4;
        v += (unsigned)(s[i] <= '9' ? s[i] - '0' : (tolower((unsigned char)s[i]) - 'a' + 10));
    }
    return v;
}

static size_t jt_string_copy(const char *j, const JTok *t, char *out) {
    char *o = out;
    if (t->type != JT_STRING) {
        size_t n = (size_t)(t->end - t->start);
        memcpy(out, j + t->start, n);
        out[n] = 0;
        return n;
    }
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
    size_t at = 0, zn = (size_t)(o - out);
    while (at < zn)
        (void)next_cp((const unsigned char *)out, zn, &at);
    return zn;
}

static char *jt_string(const char *j, const JTok *t) {
    size_t capacity = (size_t)(t->end - t->start) * 3 + 1;
    char *z = xmalloc(capacity);
    (void)jt_string_copy(j, t, z);
    return z;
}
