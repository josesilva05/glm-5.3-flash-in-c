/* glm53f_mtp.c - see glm53f_mtp.h. */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glm53f_mtp.h"

struct Glm53fMtp {
    Glm53fMtpBind b;
    float *kv;                 /* [cap][n_heads*(qk_nope+v_head)] */
    int    cap, cached, maxT;
    float *x, *cat, *y, *lg;   /* [maxT][hidden], [maxT][2*hidden], [maxT][hidden], [vocab] */
    float *scratch;
    long   drafts, right;
};

static size_t work_floats(const Glm53fCfg *c, int T, int cap)
{
    size_t n = glm53f_mla_scratch(c, T, cap);
    const size_t moe = glm53f_moe_scratch(c, T);
    if (moe > n) n = moe;
    return n;
}

int glm53f_mtp_open(Glm53fModel *m, int cap)
{
    const Glm53fCfg *c = &m->cfg;
    if (m->mtp) return 0;
    if (!m->mb.lm_head.w) {
        fprintf(stderr, "glm53f_mtp: the lm_head is not resident on the host (--gpu frees it)\n");
        return -1;
    }
    Glm53fMtp *t = (Glm53fMtp *)calloc(1, sizeof *t);
    if (!t) return -1;
    if (glm53f_bind_mtp(&m->st, c, &t->b) != 0) {
        fprintf(stderr, "glm53f_mtp: this checkpoint has no MTP layer\n");
        free(t);
        return -1;
    }
    t->b.w.w.moe.src = NULL;                     /* the same streaming cache as the trunk */
    for (int L = 0; L < m->n_bound; L++)
        if (!m->lay[L].w.is_dense) { t->b.w.w.moe.src = m->lay[L].w.moe.src; break; }
    if (!t->b.w.w.moe.src) {
        fprintf(stderr, "glm53f_mtp: no MoE layer to share the expert cache with\n");
        glm53f_bind_mtp_free(&t->b);
        free(t);
        return -1;
    }
    t->b.w.w.moe.prefetch_n = 0;
    t->cap = cap;
    t->maxT = cap;
    const size_t E = (size_t)c->hidden;
    t->kv = (float *)malloc((size_t)cap * glm53f_kv_floats_per_pos(c) * sizeof(float));
    t->x = (float *)malloc((size_t)t->maxT * E * sizeof(float));
    t->y = (float *)malloc((size_t)t->maxT * E * sizeof(float));
    t->cat = (float *)malloc((size_t)t->maxT * 2 * E * sizeof(float));
    t->lg = (float *)malloc((size_t)c->vocab * sizeof(float));
    t->scratch = (float *)malloc(work_floats(c, t->maxT, cap) * sizeof(float));
    m->mtp_h = (float *)malloc((size_t)cap * E * sizeof(float));
    if (!t->kv || !t->x || !t->y || !t->cat || !t->lg || !t->scratch || !m->mtp_h) {
        glm53f_bind_mtp_free(&t->b);
        free(t->kv); free(t->x); free(t->y); free(t->cat); free(t->lg); free(t->scratch);
        free(t);
        return -1;
    }
    printf("mtp: layer %d bound (%.2f GB resident, 288 experts streamed like the trunk)\n",
           c->n_layers, (double)t->b.nbytes / 1e9);
    m->mtp = t;
    return 0;
}

void glm53f_mtp_close(Glm53fModel *m)
{
    Glm53fMtp *t = m->mtp;
    if (!t) return;
    glm53f_bind_mtp_free(&t->b);
    free(t->kv); free(t->x); free(t->y); free(t->cat); free(t->lg); free(t->scratch);
    free(t);
    m->mtp = NULL;
}

void glm53f_mtp_reset(Glm53fModel *m)
{
    if (m->mtp) m->mtp->cached = 0;
}

int glm53f_mtp_feed(Glm53fModel *m, const float *h, const int *next_ids, int T, int *draft)
{
    Glm53fMtp *t = m->mtp;
    const Glm53fCfg *c = &m->cfg;
    const int E = c->hidden;
    if (!t || T < 1) return -1;
    if (T > t->maxT || t->cached + T > t->cap) {
        fprintf(stderr, "glm53f_mtp: %d + %d positions exceed the capacity %d\n", t->cached, T, t->cap);
        return -1;
    }
    const Glm53fMtpW *w = &t->b.w;

    for (int i = 0; i < T; i++) {
        float *cat = t->cat + (size_t)i * 2 * E;
        glm53f_embed_row(cat, &m->mb.embed, next_ids[i]);
        glm53f_rmsnorm(cat, cat, w->enorm, E, c->rms_eps);
        glm53f_rmsnorm(cat + E, h + (size_t)i * E, w->hnorm, E, c->rms_eps);
        glm53f_mm(t->x + (size_t)i * E, cat, &w->eh);
    }

    const long drops = glm53f_expert_drops;
    for (int i = 0; i < T; i++)
        glm53f_rmsnorm(t->y + (size_t)i * E, t->x + (size_t)i * E, w->w.in_norm, E, c->rms_eps);
    glm53f_mla(t->cat, t->y, &w->w.mla, c, T, t->scratch, t->kv, t->cached, t->cap);
    for (size_t i = 0; i < (size_t)T * E; i++) t->x[i] += t->cat[i];

    for (int i = 0; i < T; i++)
        glm53f_rmsnorm(t->y + (size_t)i * E, t->x + (size_t)i * E, w->w.post_norm, E, c->rms_eps);
    glm53f_moe(t->cat, t->y, &w->w.moe, c, T, t->scratch);
    for (size_t i = 0; i < (size_t)T * E; i++) t->x[i] += t->cat[i];

    t->cached += T;
    if (glm53f_expert_drops != drops) return -1;

    if (draft) {
        float *last = t->x + (size_t)(T - 1) * E;
        glm53f_rmsnorm(t->y, last, w->head_norm, E, c->rms_eps);
        glm53f_mm(t->lg, t->y, &m->mb.lm_head);
        int b = 0;
        for (int i = 1; i < c->vocab; i++) if (t->lg[i] > t->lg[b]) b = i;
        *draft = b;
    }
    return 0;
}

void glm53f_mtp_score(Glm53fModel *m, int draft, int actual)
{
    if (!m->mtp) return;
    m->mtp->drafts++;
    m->mtp->right += draft == actual;
}

void glm53f_mtp_report(const Glm53fModel *m)
{
    if (!m->mtp || !m->mtp->drafts) return;
    printf("mtp drafts: %ld, accepted %ld (%.1f%%)\n", m->mtp->drafts, m->mtp->right,
           100.0 * (double)m->mtp->right / (double)m->mtp->drafts);
}
