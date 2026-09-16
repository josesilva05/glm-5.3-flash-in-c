/* glm53f_load.c - see glm53f_load.h for the on-disk layout this is built on. */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glm53f.h"
#include "glm53f_load.h"

#define EXPERT_FMT "model.language_model.layers.%d.mlp.experts.%d.%s_proj.%s"

static int64_t align_up(int64_t x) { return (x + GLM53F_ST_ALIGN - 1) & ~(int64_t)(GLM53F_ST_ALIGN - 1); }

int glm53f_expert_ref(const Glm53fSt *s, const Glm53fCfg *c, int layer, int expert,
                      Glm53fExpertRef *r)
{
    static const char *P[3] = { "gate", "up", "down" };
    const int want_rows[3] = { c->moe_inter, c->moe_inter, c->hidden };
    const int want_cols[3] = { c->hidden, c->hidden, c->moe_inter };
    char name[256];

    memset(r, 0, sizeof *r);
    r->layer = layer; r->expert = expert;

    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof name, EXPERT_FMT, layer, expert, P[i], "weight");
        const Glm53fTensor *w = glm53f_st_find(s, name);
        if (!w) { fprintf(stderr, "glm53f_load: missing %s\n", name); return -1; }
        if (w->ndim != 2 || w->shape[0] != want_rows[i] || w->shape[1] != want_cols[i]) {
            fprintf(stderr, "glm53f_load: %s is not [%d,%d]\n", name, want_rows[i], want_cols[i]);
            return -1;
        }
        snprintf(name, sizeof name, EXPERT_FMT, layer, expert, P[i], "weight_scale_inv");
        const Glm53fTensor *sc = glm53f_st_find(s, name);
        Glm53fQMat *m = &r->m[i];
        m->rows = want_rows[i]; m->cols = want_cols[i];
        if (w->dtype == GLM53F_DT_F8_E4M3) {
            if (!sc || sc->dtype != GLM53F_DT_F32 || sc->ndim != 2) {
                fprintf(stderr, "glm53f_load: %s missing or not a 2-D F32 scale grid\n", name);
                return -1;
            }
            const int br = c->fp8_block_r, bc = c->fp8_block_c;
            if (br < 1 || bc < 1 ||
                sc->shape[0] != (m->rows + br - 1) / br || sc->shape[1] != (m->cols + bc - 1) / bc) {
                fprintf(stderr, "glm53f_load: %s is [%lld,%lld], block %dx%d implies [%d,%d]\n",
                        name, (long long)sc->shape[0], (long long)sc->shape[1], br, bc,
                        br ? (m->rows + br - 1) / br : 0, bc ? (m->cols + bc - 1) / bc : 0);
                return -1;
            }
            m->dt = GLM53F_WF8;
            m->srows = (int)sc->shape[0]; m->scols = (int)sc->shape[1];
            m->s_bytes = sc->nbytes;
            r->ts[i] = sc;
        } else if (w->dtype == GLM53F_DT_BF16 || w->dtype == GLM53F_DT_F32) {
            m->dt = w->dtype == GLM53F_DT_BF16 ? GLM53F_WBF16 : GLM53F_WF32;
        } else {
            fprintf(stderr, "glm53f_load: L%d expert %d %s has unsupported dtype\n", layer, expert, P[i]);
            return -1;
        }
        m->w_bytes = w->nbytes;
        r->tw[i] = w;
    }

    /* Are the three weights one run, and the three scales another, all in one shard? */
    int shard = r->tw[0]->shard, same = 1;
    int64_t wlo = r->tw[0]->off, whi = 0, wown = 0, slo = -1, shi = 0, sown = 0;
    for (int i = 0; i < 3; i++) {
        if (r->tw[i]->shard != shard) same = 0;
        if (r->tw[i]->off < wlo) wlo = r->tw[i]->off;
        if (r->tw[i]->off + r->tw[i]->nbytes > whi) whi = r->tw[i]->off + r->tw[i]->nbytes;
        wown += r->tw[i]->nbytes;
        if (r->ts[i]) {
            if (r->ts[i]->shard != shard) same = 0;
            if (slo < 0 || r->ts[i]->off < slo) slo = r->ts[i]->off;
            if (r->ts[i]->off + r->ts[i]->nbytes > shi) shi = r->ts[i]->off + r->ts[i]->nbytes;
            sown += r->ts[i]->nbytes;
        }
    }
    r->contiguous = same && (whi - wlo == wown) && (sown == 0 || shi - slo == sown);
    r->w_bytes = wown;
    r->s_bytes = sown;
    if (r->contiguous) {
        r->shard = shard;
        r->w_abs = wlo;
        r->s_abs = slo;
        for (int i = 0; i < 3; i++) {
            r->m[i].w_off = r->tw[i]->off - wlo;
            r->m[i].s_off = r->ts[i] ? r->ts[i]->off - slo : 0;
        }
    } else {
        int64_t wo = 0, so = 0;
        for (int i = 0; i < 3; i++) {
            r->m[i].w_off = wo; wo += r->m[i].w_bytes;
            r->m[i].s_off = so; so += r->m[i].s_bytes;
        }
    }
    return 0;
}

int64_t glm53f_expert_weight_area(const Glm53fExpertRef *r)
{
    return align_up(r->w_bytes + 2 * GLM53F_ST_ALIGN);
}

int64_t glm53f_expert_slot_bytes(const Glm53fExpertRef *r)
{
    return align_up(glm53f_expert_weight_area(r) + r->s_bytes);
}

int glm53f_expert_load(const Glm53fSt *s, const Glm53fExpertRef *r, unsigned char *slot,
                       int64_t slot_bytes, int64_t *pad)
{
    const int64_t warea = glm53f_expert_weight_area(r);
    if (glm53f_expert_slot_bytes(r) > slot_bytes) return -1;

    if (r->contiguous) {
        const int64_t got = glm53f_st_read_aligned(s, r->shard, r->w_abs, r->w_bytes,
                                                   slot, warea, pad);
        if (got != r->w_bytes) return -1;
        if (r->s_bytes) {
            Glm53fTensor t;
            memset(&t, 0, sizeof t);
            t.name = (char *)"expert scales";
            t.shard = r->shard; t.off = r->s_abs; t.nbytes = r->s_bytes;
            t.dtype = GLM53F_DT_U8; t.ndim = 1; t.shape[0] = r->s_bytes;
            if (glm53f_st_read(s, &t, slot + warea) != r->s_bytes) return -1;
        }
        return 0;
    }
    *pad = 0;
    for (int i = 0; i < 3; i++) {
        if (glm53f_st_read(s, r->tw[i], slot + r->m[i].w_off) != r->tw[i]->nbytes) return -1;
        if (r->ts[i] &&
            glm53f_st_read(s, r->ts[i], slot + warea + r->m[i].s_off) != r->ts[i]->nbytes) return -1;
    }
    return 0;
}

/* Offsets of matrix i inside an int4 slot: nibbles, then steps. */
static void i4_offsets(const Glm53fExpertRef *r, int i, int64_t *q_off, int64_t *s_off, int64_t *end)
{
    int64_t off = 0;
    for (int k = 0; k <= i; k++) {
        const int64_t rows = r->m[k].rows, cols = r->m[k].cols;
        const int64_t qb = rows * (cols / 2), sb = rows * (cols / GLM53F_I4_GROUP) * 4;
        off = (off + 63) & ~(int64_t)63;
        if (k == i) *q_off = off;
        off += qb;
        off = (off + 63) & ~(int64_t)63;
        if (k == i) { *s_off = off; }
        off += sb;
    }
    *end = off;
}

int64_t glm53f_expert_i4_slot_bytes(const Glm53fExpertRef *r)
{
    int64_t q = 0, s = 0, end = 0;
    i4_offsets(r, 2, &q, &s, &end);
    return (end + GLM53F_ST_ALIGN - 1) / GLM53F_ST_ALIGN * GLM53F_ST_ALIGN;
}

void glm53f_expert_i4_pack(const Glm53fExpertRef *r, const Glm53fCfg *c, const unsigned char *src,
                           int64_t pad, unsigned char *dst)
{
    Glm53fExpertQ q;
    glm53f_expert_view(r, c, src, pad, glm53f_expert_weight_area(r), &q);
    const Glm53fMat *in[3] = { &q.gate, &q.up, &q.down };
    for (int i = 0; i < 3; i++) {
        int64_t qo = 0, so = 0, end = 0;
        i4_offsets(r, i, &qo, &so, &end);
        glm53f_i4_from_f8(dst + qo, (float *)(void *)(dst + so), in[i]);
    }
}

void glm53f_expert_i4_view(const Glm53fExpertRef *r, const unsigned char *slot, Glm53fExpertQ *q)
{
    Glm53fMat *out[3] = { &q->gate, &q->up, &q->down };
    for (int i = 0; i < 3; i++) {
        int64_t qo = 0, so = 0, end = 0;
        i4_offsets(r, i, &qo, &so, &end);
        Glm53fMat *o = out[i];
        memset(o, 0, sizeof *o);
        o->w = slot + qo;
        o->s = (const float *)(const void *)(slot + so);
        o->dt = GLM53F_WI4;
        o->rows = r->m[i].rows; o->cols = r->m[i].cols;
        o->br = 1; o->bc = GLM53F_I4_GROUP; o->scols = r->m[i].cols / GLM53F_I4_GROUP;
    }
}

void glm53f_expert_view(const Glm53fExpertRef *r, const Glm53fCfg *c, const unsigned char *slot,
                        int64_t pad, int64_t weight_area, Glm53fExpertQ *q)
{
    Glm53fMat *out[3] = { &q->gate, &q->up, &q->down };
    for (int i = 0; i < 3; i++) {
        const Glm53fQMat *m = &r->m[i];
        Glm53fMat *o = out[i];
        memset(o, 0, sizeof *o);
        o->w = slot + pad + m->w_off;
        o->dt = m->dt;
        o->rows = m->rows; o->cols = m->cols;
        if (m->dt == GLM53F_WF8) {
            o->s = (const float *)(void *)(slot + weight_area + m->s_off);
            o->br = c->fp8_block_r; o->bc = c->fp8_block_c; o->scols = m->scols;
        }
    }
}
