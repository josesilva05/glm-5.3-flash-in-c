/* glm53f_dsa.c - DeepSeek Sparse Attention indexer (Glm5NextTextIndexer).
 *
 * For every query position the indexer decides WHICH earlier positions the MLA attends to.
 * It never changes the attention arithmetic, only the set of keys:
 *
 *   k_j    = LayerNorm(wk . x_j, k_norm, eps 1e-6)                    [index_dim]
 *   g_j    = index_kpool_compress_gate . x_j                          [index_dim]
 *   pool p = positions [kpool*p, kpool*p + kpool), complete pools only
 *   P_p[c] = sum_j softmax_j(g_[kpool*p+j][c] + ape[j][c]) * k_[kpool*p+j][c]
 *   q_t    = wq_b . q_resid_t                                         [heads][index_dim]
 *   w_t    = (weights_proj . x_t) / sqrt(heads)                       [heads]
 *   s_t[p] = sum_h w_t[h] * relu(q_t[h] . P_p / sqrt(index_dim))
 *
 * The index_topk / index_kpool best-scoring pools are kept, and the incomplete tail pool
 * (the positions after the last complete pool, up to the query) is always appended. While
 * the complete pools fit in that budget every visible position is selected, which is the
 * dense case the engine used to be limited to; the caller then skips the scoring entirely.
 *
 * Reference: Glm5NextTextIndexer in transformers 5.17 (batch 1, no padding, NoPE).
 */
#include "glm53f.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void dsa_fatal(const char *what, long v, long limit)
{
    fprintf(stderr, "glm53f: FATAL, %s is %ld, above the limit of %ld.\n", what, v, limit);
    abort();
}

static void *dsa_alloc(size_t bytes)
{
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "glm53f: FATAL, could not allocate %zu bytes for the DSA selection.\n", bytes);
        abort();
    }
    return p;
}

/* state layout: k [cap][D] | gate [cap][D] | pool keys [cap/kpool][D] */
static float *idx_k(float *st, const Glm53fCfg *c, int pos)    { return st + (size_t)pos * c->index_dim; }
static float *idx_g(float *st, const Glm53fCfg *c, int cap, int pos)
{
    return st + (size_t)cap * c->index_dim + (size_t)pos * c->index_dim;
}
static float *idx_pool(float *st, const Glm53fCfg *c, int cap, int pool)
{
    return st + (size_t)2 * cap * c->index_dim + (size_t)pool * c->index_dim;
}

size_t glm53f_dsa_scratch(const Glm53fCfg *c, int cap)
{
    const size_t D = (size_t)c->index_dim, H = (size_t)c->index_heads;
    return 2 * D + H * D + H + (size_t)(cap / c->index_kpool + 1);   /* k, g, q, w, scores */
}

/* nn.LayerNorm over n values: biased variance, eps 1e-6, then weight and bias. */
static void layernorm(float *y, const float *x, const float *w, const float *b, int n)
{
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += (double)x[i];
    mean /= (double)n;
    double var = 0.0;
    for (int i = 0; i < n; i++) { const double d = (double)x[i] - mean; var += d * d; }
    var /= (double)n;
    const double inv = 1.0 / sqrt(var + 1e-6);
    for (int i = 0; i < n; i++) y[i] = (float)(((double)x[i] - mean) * inv) * w[i] + b[i];
}

void glm53f_dsa_append(float *st, const float *x, const Glm53fIdxW *w, const Glm53fCfg *c,
                       int T, int cached, int cap, float *scratch)
{
    const int D = c->index_dim, K = c->index_kpool, E = c->hidden;
    float *tmp = scratch;
    for (int t = 0; t < T; t++) {
        const int pos = cached + t;
        if (pos >= cap) dsa_fatal("DSA indexer position", pos, cap - 1);
        const float *xt = x + (size_t)t * E;
        glm53f_mm(tmp, xt, &w->wk);
        layernorm(idx_k(st, c, pos), tmp, w->k_norm_w, w->k_norm_b, D);
        glm53f_mm(idx_g(st, c, cap, pos), xt, &w->gate);
        if ((pos + 1) % K == 0) {                       /* the pool ending here is complete */
            const int pool = (pos + 1) / K - 1;
            float *pk = idx_pool(st, c, cap, pool);
            for (int i = 0; i < D; i++) {
                double m = -INFINITY, z = 0.0, acc = 0.0;
                for (int j = 0; j < K; j++) {
                    const double l = (double)idx_g(st, c, cap, pool * K + j)[i] + (double)w->ape[j * D + i];
                    if (l > m) m = l;
                }
                for (int j = 0; j < K; j++)
                    z += exp((double)idx_g(st, c, cap, pool * K + j)[i] + (double)w->ape[j * D + i] - m);
                for (int j = 0; j < K; j++) {
                    const double p = exp((double)idx_g(st, c, cap, pool * K + j)[i] +
                                         (double)w->ape[j * D + i] - m) / z;
                    acc += p * (double)idx_k(st, c, pool * K + j)[i];
                }
                pk[i] = (float)acc;
            }
        }
    }
}

/* Pools ranked by score, ties going to the lower index (torch.topk's order on CPU). */
typedef struct { float s; int i; } Cand;

static int cmp_cand(const void *a, const void *b)
{
    const Cand *x = (const Cand *)a, *y = (const Cand *)b;
    if (x->s != y->s) return x->s > y->s ? -1 : 1;
    return x->i < y->i ? -1 : x->i > y->i;
}

static int cmp_int(const void *a, const void *b)
{
    const int x = *(const int *)a, y = *(const int *)b;
    return x < y ? -1 : x > y;
}

int glm53f_dsa_select(int *sel, float *st, const float *x_t, const float *q_resid,
                      const Glm53fIdxW *w, const Glm53fCfg *c, int p, int cap, float *scratch)
{
    const int D = c->index_dim, HI = c->index_heads, K = c->index_kpool;
    const int np = (p + 1) / K;                      /* complete pools visible to this query */
    int select_k = c->index_topk / K;
    if (select_k > np) select_k = np;
    if (np <= c->index_topk / K) return -1;          /* everything fits: the caller goes dense */

    float *q = scratch + 2 * (size_t)D;
    float *wt = q + (size_t)HI * D;
    float *score = wt + HI;
    glm53f_mm(q, q_resid, &w->wq_b);
    glm53f_mm(wt, x_t, &w->wproj);
    const float hscale = 1.0f / sqrtf((float)HI), dscale = 1.0f / sqrtf((float)D);

    int pp;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (np > 64)
#endif
    for (pp = 0; pp < np; pp++) {
        const float *pk = idx_pool(st, c, cap, pp);
        double acc = 0.0;
        for (int h = 0; h < HI; h++) {
            const float *qh = q + (size_t)h * D;
            double d = 0.0;
            for (int i = 0; i < D; i++) d += (double)qh[i] * (double)pk[i];
            d *= (double)dscale;
            if (d > 0.0) acc += (double)wt[h] * (double)hscale * d;   /* relu, then weight */
        }
        score[pp] = (float)acc;
    }

    Cand *rank = (Cand *)dsa_alloc((size_t)np * sizeof *rank);
    int *pools = (int *)dsa_alloc((size_t)select_k * sizeof *pools);
    for (pp = 0; pp < np; pp++) { rank[pp].s = score[pp]; rank[pp].i = pp; }
    qsort(rank, (size_t)np, sizeof *rank, cmp_cand);
    for (int i = 0; i < select_k; i++) pools[i] = rank[i].i;
    free(rank);
    qsort(pools, (size_t)select_k, sizeof(int), cmp_int);

    int ns = 0;
    for (int i = 0; i < select_k; i++)
        for (int j = 0; j < K; j++) sel[ns++] = pools[i] * K + j;
    free(pools);
    for (int j = np * K; j <= p; j++) sel[ns++] = j;     /* the incomplete tail pool */
    return ns;
}
