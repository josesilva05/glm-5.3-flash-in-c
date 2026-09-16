/* glm53f_bind.c - see glm53f_bind.h. */
#define _POSIX_C_SOURCE 200809L

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glm53f_bind.h"

#define PRE "model.language_model."
#define MAXR 80

typedef struct {
    char               name[200];
    int                is_mat;
    int64_t            rows, cols;     /* matrix shape, or n in rows for a vector */
    int                conv_k;         /* >0: vector stored as [n][1][conv_k]      */
    const Glm53fTensor *t, *ts;        /* tensor and FP8 scale grid                */
    Glm53fMat         *mat;
    const float      **vec;
    size_t             off, soff;
} Req;

typedef struct {
    Req r[MAXR];
    int n, bad;
} Plan;

static void req_mat(Plan *p, Glm53fMat *dst, int64_t rows, int64_t cols, const char *fmt, ...)
{
    if (p->n >= MAXR) { p->bad++; return; }
    Req *q = &p->r[p->n++];
    memset(q, 0, sizeof *q);
    va_list ap; va_start(ap, fmt); vsnprintf(q->name, sizeof q->name, fmt, ap); va_end(ap);
    q->is_mat = 1; q->rows = rows; q->cols = cols; q->mat = dst;
}

static void req_vec(Plan *p, const float **dst, int64_t n, int conv_k, const char *fmt, ...)
{
    if (p->n >= MAXR) { p->bad++; return; }
    Req *q = &p->r[p->n++];
    memset(q, 0, sizeof *q);
    va_list ap; va_start(ap, fmt); vsnprintf(q->name, sizeof q->name, fmt, ap); va_end(ap);
    q->rows = n; q->cols = conv_k ? conv_k : 1; q->conv_k = conv_k; q->vec = dst;
}

static size_t align8(size_t x) { return (x + 7u) & ~(size_t)7u; }

static int64_t plan_resolve(Plan *p, const Glm53fSt *s, const Glm53fCfg *c)
{
    size_t off = 0;
    for (int i = 0; i < p->n; i++) {
        Req *q = &p->r[i];
        q->t = glm53f_st_find(s, q->name);
        if (!q->t) { fprintf(stderr, "glm53f_bind: missing tensor %s\n", q->name); p->bad++; continue; }
        const Glm53fTensor *t = q->t;
        int shape_ok;
        if (q->is_mat)
            shape_ok = t->ndim == 2 && t->shape[0] == q->rows && t->shape[1] == q->cols;
        else if (q->conv_k)
            shape_ok = t->ndim == 3 && t->shape[0] == q->rows && t->shape[1] == 1 && t->shape[2] == q->conv_k;
        else
            shape_ok = glm53f_st_numel(t) == q->rows && t->ndim <= 2;
        if (!shape_ok) {
            fprintf(stderr, "glm53f_bind: %s has shape [", q->name);
            for (int d = 0; d < t->ndim; d++) fprintf(stderr, "%s%lld", d ? "," : "", (long long)t->shape[d]);
            fprintf(stderr, "], engine expects %lld x %lld\n", (long long)q->rows, (long long)q->cols);
            p->bad++;
            continue;
        }
        off = align8(off);
        q->off = off;
        if (q->is_mat) {
            if (t->dtype == GLM53F_DT_F8_E4M3) {
                char sn[220];
                snprintf(sn, sizeof sn, "%s_scale_inv", q->name);
                q->ts = glm53f_st_find(s, sn);
                const int br = c->fp8_block_r, bc = c->fp8_block_c;
                if (br < 1 || bc < 1) {
                    fprintf(stderr, "glm53f_bind: %s is FP8 but the config has no weight_block_size\n", q->name);
                    p->bad++; continue;
                }
                if (!q->ts || q->ts->dtype != GLM53F_DT_F32 || q->ts->ndim != 2 ||
                    q->ts->shape[0] != (q->rows + br - 1) / br || q->ts->shape[1] != (q->cols + bc - 1) / bc) {
                    fprintf(stderr, "glm53f_bind: %s has no matching F32 scale grid %s\n", q->name, sn);
                    p->bad++; continue;
                }
                off += (size_t)t->nbytes;
                off = align8(off);
                q->soff = off;
                off += (size_t)q->ts->nbytes;
            } else if (t->dtype == GLM53F_DT_BF16 || t->dtype == GLM53F_DT_F32) {
                off += (size_t)t->nbytes;
            } else {
                fprintf(stderr, "glm53f_bind: %s has a dtype this engine cannot multiply\n", q->name);
                p->bad++; continue;
            }
        } else {
            if (t->dtype != GLM53F_DT_BF16 && t->dtype != GLM53F_DT_F32 && t->dtype != GLM53F_DT_F16) {
                fprintf(stderr, "glm53f_bind: %s must be a float vector\n", q->name);
                p->bad++; continue;
            }
            off += (size_t)glm53f_st_numel(t) * sizeof(float);
        }
    }
    return p->bad ? -1 : (int64_t)align8(off);
}

static int plan_load(Plan *p, const Glm53fSt *s, const Glm53fCfg *c, unsigned char *blob)
{
    for (int i = 0; i < p->n; i++) {
        Req *q = &p->r[i];
        const Glm53fTensor *t = q->t;
        if (q->is_mat) {
            if (glm53f_st_read(s, t, blob + q->off) != t->nbytes) return -1;
            Glm53fMat *m = q->mat;
            memset(m, 0, sizeof *m);
            m->w = blob + q->off;
            m->rows = (int)q->rows; m->cols = (int)q->cols;
            if (t->dtype == GLM53F_DT_F8_E4M3) {
                if (glm53f_st_read(s, q->ts, blob + q->soff) != q->ts->nbytes) return -1;
                m->dt = GLM53F_WF8;
                m->s = (const float *)(void *)(blob + q->soff);
                m->br = c->fp8_block_r; m->bc = c->fp8_block_c;
                m->scols = (int)q->ts->shape[1];
            } else {
                m->dt = t->dtype == GLM53F_DT_BF16 ? GLM53F_WBF16 : GLM53F_WF32;
            }
        } else {
            const int64_t n = glm53f_st_numel(t);
            if (glm53f_st_read_f32(s, t, (float *)(void *)(blob + q->off)) != n) return -1;
            *q->vec = (const float *)(void *)(blob + q->off);
        }
    }
    return 0;
}

static void plan_layer(Plan *p, const Glm53fCfg *c, int L, Glm53fLayerW *w)
{
    const int64_t E = c->hidden, M = c->hc_mult, MIX = (2 + M) * M;
    req_vec(p, &w->in_norm,   E, 0, PRE "layers.%d.input_layernorm.weight", L);
    req_vec(p, &w->post_norm, E, 0, PRE "layers.%d.post_attention_layernorm.weight", L);
    req_mat(p, &w->attn_hc.fn, MIX, M * E, PRE "layers.%d.hc_attn_fn", L);
    req_vec(p, &w->attn_hc.base, MIX, 0, PRE "layers.%d.hc_attn_base", L);
    req_vec(p, &w->attn_hc.scale, 3, 0, PRE "layers.%d.hc_attn_scale", L);
    req_mat(p, &w->ffn_hc.fn, MIX, M * E, PRE "layers.%d.hc_ffn_fn", L);
    req_vec(p, &w->ffn_hc.base, MIX, 0, PRE "layers.%d.hc_ffn_base", L);
    req_vec(p, &w->ffn_hc.scale, 3, 0, PRE "layers.%d.hc_ffn_scale", L);

    w->is_mla = glm53f_is_mla(c, L);
    w->is_dense = glm53f_is_dense(c, L);
    if (w->is_mla) {
        const int64_t H = c->n_heads;
        Glm53fMlaW *a = &w->mla;
        req_mat(p, &a->q_a,  c->q_lora, E, PRE "layers.%d.self_attn.q_a_proj.weight", L);
        req_vec(p, &a->q_a_norm, c->q_lora, 0, PRE "layers.%d.self_attn.q_a_layernorm.weight", L);
        req_mat(p, &a->q_b,  H * c->qk_nope, c->q_lora, PRE "layers.%d.self_attn.q_b_proj.weight", L);
        req_mat(p, &a->kv_a, c->kv_lora, E, PRE "layers.%d.self_attn.kv_a_proj_with_mqa.weight", L);
        req_vec(p, &a->kv_a_norm, c->kv_lora, 0, PRE "layers.%d.self_attn.kv_a_layernorm.weight", L);
        req_mat(p, &a->kv_b, H * (c->qk_nope + c->v_head), c->kv_lora,
                PRE "layers.%d.self_attn.kv_b_proj.weight", L);
        req_mat(p, &a->o, E, H * c->v_head, PRE "layers.%d.self_attn.o_proj.weight", L);
        Glm53fIdxW *ix = &a->idx;
        const int64_t IH = c->index_heads, ID = c->index_dim;
        req_mat(p, &ix->wq_b, IH * ID, c->q_lora, PRE "layers.%d.self_attn.indexer.wq_b.weight", L);
        req_mat(p, &ix->wk, ID, E, PRE "layers.%d.self_attn.indexer.wk.weight", L);
        req_vec(p, &ix->k_norm_w, ID, 0, PRE "layers.%d.self_attn.indexer.k_norm.weight", L);
        req_vec(p, &ix->k_norm_b, ID, 0, PRE "layers.%d.self_attn.indexer.k_norm.bias", L);
        req_mat(p, &ix->wproj, IH, E, PRE "layers.%d.self_attn.indexer.weights_proj.weight", L);
        req_mat(p, &ix->gate, ID, E, PRE "layers.%d.self_attn.indexer.index_kpool_compress_gate", L);
        req_vec(p, &ix->ape, (int64_t)c->index_kpool * ID, 0,
                PRE "layers.%d.self_attn.indexer.index_kpool_compress_ape", L);
    } else {
        const int64_t H = c->kda_heads, D = c->kda_head_dim, P = H * D;
        Glm53fKdaW *k = &w->kda;
        req_mat(p, &k->q, P, E, PRE "layers.%d.self_attn.q_proj.weight", L);
        req_mat(p, &k->k, P, E, PRE "layers.%d.self_attn.k_proj.weight", L);
        req_mat(p, &k->v, P, E, PRE "layers.%d.self_attn.v_proj.weight", L);
        req_vec(p, &k->q_conv, P, c->conv_k, PRE "layers.%d.self_attn.q_conv1d.weight", L);
        req_vec(p, &k->k_conv, P, c->conv_k, PRE "layers.%d.self_attn.k_conv1d.weight", L);
        req_vec(p, &k->v_conv, P, c->conv_k, PRE "layers.%d.self_attn.v_conv1d.weight", L);
        req_mat(p, &k->f_a, D, E, PRE "layers.%d.self_attn.f_a_proj.weight", L);
        req_mat(p, &k->f_b, P, D, PRE "layers.%d.self_attn.f_b_proj.weight", L);
        req_vec(p, &k->A_log, H, 0, PRE "layers.%d.self_attn.A_log", L);
        req_vec(p, &k->dt_bias, P, 0, PRE "layers.%d.self_attn.dt_bias", L);
        req_mat(p, &k->b, H, E, PRE "layers.%d.self_attn.b_proj.weight", L);
        req_mat(p, &k->g_a, D, E, PRE "layers.%d.self_attn.g_a_proj.weight", L);
        req_mat(p, &k->g_b, P, D, PRE "layers.%d.self_attn.g_b_proj.weight", L);
        req_vec(p, &k->o_norm, D, 0, PRE "layers.%d.self_attn.o_norm.weight", L);
        req_mat(p, &k->o, E, P, PRE "layers.%d.self_attn.o_proj.weight", L);
    }
    if (w->is_dense) {
        req_mat(p, &w->d_gate, c->dense_inter, E, PRE "layers.%d.mlp.gate_proj.weight", L);
        req_mat(p, &w->d_up,   c->dense_inter, E, PRE "layers.%d.mlp.up_proj.weight", L);
        req_mat(p, &w->d_down, E, c->dense_inter, PRE "layers.%d.mlp.down_proj.weight", L);
    } else {
        const int64_t SI = (int64_t)c->moe_inter * c->n_shared;
        req_vec(p, &w->moe.gate, (int64_t)c->n_experts * E, 0, PRE "layers.%d.mlp.gate.weight", L);
        req_vec(p, &w->moe.bias, c->n_experts, 0, PRE "layers.%d.mlp.gate.e_score_correction_bias", L);
        req_mat(p, &w->moe.sh_gate, SI, E, PRE "layers.%d.mlp.shared_experts.gate_proj.weight", L);
        req_mat(p, &w->moe.sh_up,   SI, E, PRE "layers.%d.mlp.shared_experts.up_proj.weight", L);
        req_mat(p, &w->moe.sh_down, E, SI, PRE "layers.%d.mlp.shared_experts.down_proj.weight", L);
        w->moe.layer = L;
    }
}

/* The MTP layer: an MLA + MoE layer of the same shapes, plus the two input norms, the
 * projection of [norm(embedding), norm(hidden)] and the norm before the shared lm_head.
 * It carries no mHC mixers. */
static void plan_mtp(Plan *p, const Glm53fCfg *c, Glm53fMtpW *w)
{
    const int L = c->n_layers, E = c->hidden, H = c->n_heads;
    Glm53fLayerW *lw = &w->w;
    lw->is_mla = 1;
    lw->is_dense = 0;
    req_vec(p, &lw->in_norm,   E, 0, PRE "layers.%d.input_layernorm.weight", L);
    req_vec(p, &lw->post_norm, E, 0, PRE "layers.%d.post_attention_layernorm.weight", L);
    req_vec(p, &w->enorm, E, 0, PRE "layers.%d.enorm.weight", L);
    req_vec(p, &w->hnorm, E, 0, PRE "layers.%d.hnorm.weight", L);
    req_vec(p, &w->head_norm, E, 0, PRE "layers.%d.shared_head.norm.weight", L);
    req_mat(p, &w->eh, E, 2 * E, PRE "layers.%d.eh_proj.weight", L);
    Glm53fMlaW *a = &lw->mla;
    req_mat(p, &a->q_a,  c->q_lora, E, PRE "layers.%d.self_attn.q_a_proj.weight", L);
    req_vec(p, &a->q_a_norm, c->q_lora, 0, PRE "layers.%d.self_attn.q_a_layernorm.weight", L);
    req_mat(p, &a->q_b,  (int64_t)H * c->qk_nope, c->q_lora, PRE "layers.%d.self_attn.q_b_proj.weight", L);
    req_mat(p, &a->kv_a, c->kv_lora, E, PRE "layers.%d.self_attn.kv_a_proj_with_mqa.weight", L);
    req_vec(p, &a->kv_a_norm, c->kv_lora, 0, PRE "layers.%d.self_attn.kv_a_layernorm.weight", L);
    req_mat(p, &a->kv_b, (int64_t)H * (c->qk_nope + c->v_head), c->kv_lora,
            PRE "layers.%d.self_attn.kv_b_proj.weight", L);
    req_mat(p, &a->o, E, (int64_t)H * c->v_head, PRE "layers.%d.self_attn.o_proj.weight", L);
        Glm53fIdxW *ix = &a->idx;
        const int64_t IH = c->index_heads, ID = c->index_dim;
        req_mat(p, &ix->wq_b, IH * ID, c->q_lora, PRE "layers.%d.self_attn.indexer.wq_b.weight", L);
        req_mat(p, &ix->wk, ID, E, PRE "layers.%d.self_attn.indexer.wk.weight", L);
        req_vec(p, &ix->k_norm_w, ID, 0, PRE "layers.%d.self_attn.indexer.k_norm.weight", L);
        req_vec(p, &ix->k_norm_b, ID, 0, PRE "layers.%d.self_attn.indexer.k_norm.bias", L);
        req_mat(p, &ix->wproj, IH, E, PRE "layers.%d.self_attn.indexer.weights_proj.weight", L);
        req_mat(p, &ix->gate, ID, E, PRE "layers.%d.self_attn.indexer.index_kpool_compress_gate", L);
        req_vec(p, &ix->ape, (int64_t)c->index_kpool * ID, 0,
                PRE "layers.%d.self_attn.indexer.index_kpool_compress_ape", L);
    const int64_t SI = (int64_t)c->moe_inter * c->n_shared;
    req_vec(p, &lw->moe.gate, (int64_t)c->n_experts * E, 0, PRE "layers.%d.mlp.gate.weight", L);
    req_vec(p, &lw->moe.bias, c->n_experts, 0, PRE "layers.%d.mlp.gate.e_score_correction_bias", L);
    req_mat(p, &lw->moe.sh_gate, SI, E, PRE "layers.%d.mlp.shared_experts.gate_proj.weight", L);
    req_mat(p, &lw->moe.sh_up,   SI, E, PRE "layers.%d.mlp.shared_experts.up_proj.weight", L);
    req_mat(p, &lw->moe.sh_down, E, SI, PRE "layers.%d.mlp.shared_experts.down_proj.weight", L);
    lw->moe.layer = L;
}

int glm53f_bind_mtp(const Glm53fSt *s, const Glm53fCfg *c, Glm53fMtpBind *b)
{
    memset(b, 0, sizeof *b);
    Plan *p = (Plan *)calloc(1, sizeof *p);
    if (!p) return -1;
    plan_mtp(p, c, &b->w);
    const int64_t need = plan_resolve(p, s, c);
    if (need < 0) { free(p); return -1; }
    b->blob = malloc((size_t)need);
    if (!b->blob) { free(p); return -1; }
    b->nbytes = (size_t)need;
    if (plan_load(p, s, c, (unsigned char *)b->blob) != 0) {
        free(b->blob); b->blob = NULL; free(p); return -1;
    }
    free(p);
    return 0;
}

void glm53f_bind_mtp_free(Glm53fMtpBind *b)
{
    free(b->blob);
    memset(b, 0, sizeof *b);
}

int64_t glm53f_bind_layer_bytes(const Glm53fSt *s, const Glm53fCfg *c, int L)
{
    Glm53fLayerW tmp; memset(&tmp, 0, sizeof tmp);
    Plan p; memset(&p, 0, sizeof p);
    plan_layer(&p, c, L, &tmp);
    return plan_resolve(&p, s, c);
}

int glm53f_bind_layer(const Glm53fSt *s, const Glm53fCfg *c, int L, Glm53fLayerBind *b)
{
    memset(b, 0, sizeof *b);
    Plan *p = (Plan *)calloc(1, sizeof *p);
    if (!p) return -1;
    plan_layer(p, c, L, &b->w);
    const int64_t need = plan_resolve(p, s, c);
    if (need < 0) { fprintf(stderr, "glm53f_bind: layer %d cannot be bound\n", L); free(p); return -1; }
    b->blob = malloc((size_t)need);
    if (!b->blob) {
        fprintf(stderr, "glm53f_bind: cannot allocate %.2f GB for layer %d\n", (double)need / 1e9, L);
        free(p); return -1;
    }
    b->nbytes = (size_t)need;
    if (plan_load(p, s, c, (unsigned char *)b->blob) != 0) {
        fprintf(stderr, "glm53f_bind: short read binding layer %d\n", L);
        free(b->blob); b->blob = NULL; free(p); return -1;
    }
    free(p);
    return 0;
}

void glm53f_bind_free(Glm53fLayerBind *b)
{
    free(b->blob);
    memset(b, 0, sizeof *b);
}

int glm53f_bind_model(const Glm53fSt *s, const Glm53fCfg *c, Glm53fModelBind *m)
{
    memset(m, 0, sizeof *m);
    Plan p; memset(&p, 0, sizeof p);
    req_mat(&p, &m->embed, c->vocab, c->hidden, PRE "embed_tokens.weight");
    req_vec(&p, &m->norm, c->hidden, 0, PRE "norm.weight");
    req_mat(&p, &m->lm_head, c->vocab, c->hidden, "lm_head.weight");
    const int64_t need = plan_resolve(&p, s, c);
    if (need < 0) return -1;
    if (p.r[0].t->dtype == GLM53F_DT_F8_E4M3) {
        fprintf(stderr, "glm53f_bind: an FP8 embedding table is not supported\n");
        return -1;
    }
    m->blob = malloc((size_t)need);
    if (!m->blob) return -1;
    m->nbytes = (size_t)need;
    if (plan_load(&p, s, c, (unsigned char *)m->blob) != 0) {
        free(m->blob); m->blob = NULL; return -1;
    }
    return 0;
}

void glm53f_bind_model_free(Glm53fModelBind *m)
{
    free(m->blob);
    memset(m, 0, sizeof *m);
}
