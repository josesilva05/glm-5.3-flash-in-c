/* glm53f_gpu.c - host side of the CUDA backend: upload, placement, forward. See glm53f_gpu.h. */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glm53f_gpu.h"
#include "glm53f_gpu_kernels.h"
#include "glm53f_model.h"

#define GPU_MAX_DEV   8
#define GPU_CHUNK     256          /* prefill runs in chunks of this many tokens */
#define GPU_MARGIN    ((size_t)768 << 20)
#define GPU_WIDEN_MAX ((size_t)64 << 20)   /* floats in the fp32 widening buffer (256 MB) */
#define GPU_MLA_SCORES ((size_t)8 << 20)   /* doubles of MLA attention scores (64 MB) */

typedef struct {
    void  *w;
    float *s;
    int    dt, rows, cols, br, bc, scols;
} GMat;

typedef struct {
    GMat   fn;
    float *base, *scale;
} GHc;

typedef struct {
    int    dev;
    int    is_mla, is_dense;
    GHc    attn_hc, ffn_hc;
    float *in_norm, *post_norm;
    /* KDA */
    GMat   q, k, v, f_a, f_b, b, g_a, g_b, o;
    float *q_conv, *k_conv, *v_conv, *A_log, *dt_bias, *o_norm;
    float *state;                          /* [H][D][D] recurrent + 3 x [P][K-1] conv */
    size_t state_bytes;
    /* MLA */
    GMat   q_a, q_b, kv_a, kv_b, mo;
    float *q_a_norm, *kv_a_norm;
    float *kv;                             /* [cap][H][qk_nope + v_head] */
    /* dense MLP or shared expert */
    GMat   gate, up, down;
    size_t bytes;
} GLayer;

typedef struct {
    int     id;
    int     used;
    float  *lut;
    float  *h, *h2, *flat, *mix, *post, *comb, *xc, *xn, *y, *r;
    double *pre, *dtmp;
    float  *q, *k, *v, *z, *bt, *o, *gb, *lr;
    float  *ql, *mq, *ct, *acc;
    float  *g1, *g2, *act;
    float  *xm, *xm2, *logits;
    double *nbs;                           /* rmsnorm block sums  [T][ceil(M*E/256)] */
    double *msc, *mtz;                     /* MLA scores [tsub][H][cap], normalisers [tsub][H][2] */
    int     mla_tsub;
    float  *ninv, *ku;                     /* rmsnorm row factors [T], KDA u [H*D]   */
    float  *wf;                            /* fp32 rows of the matrix being multiplied */
    size_t  wf_floats;
    cublasHandle_t blas;
    int     exact;                         /* 1: fused-multiply-add kernels, bit-matching the CPU */
    size_t  scratch_bytes;
} GDev;

struct Glm53fGpu {
    int     ndev;
    GDev    dev[GPU_MAX_DEV];
    GLayer *lay;
    int     nl, head;                      /* head: device index holding norm + lm_head */
    float  *norm;
    GMat    lm_head;
    int     maxT;
    float  *host_h, *host_x, *host_r, *host_logits;
    float  *moe_scratch;
    /* keep-warm: busy the devices while the CPU computes routed experts (see warm_main) */
    struct {
        int            id;
        cublasHandle_t blas;
        cudaEvent_t    ev;
        float         *A, *x, *y;
    }       warm[GPU_MAX_DEV];
    int     nwarm, warm_started;
    pthread_t warm_thread;
    volatile int warm_want, warm_stop;
};

int glm53f_gpu_device_count(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

/* GLM53F_GPU_PROFILE=1: wall time per forward split into GPU trunk (up to each MoE hand-off,
 * which synchronises), CPU routed experts, and the final norm + lm_head; printed to stderr at close. */
static int    prof_on = -1;
static double prof_gpu, prof_cpu, prof_head;
static long   prof_tokens;

/* GLM53F_GPU_PROFILE=2 additionally synchronises at every section of single-token forwards
 * (which perturbs the GPU/CPU overlap) and reports where a decode step goes. */
enum { PS_MOVE, PS_ATTN_HC, PS_MLA, PS_KDA, PS_HC_POST, PS_FFN_HC, PS_DENSE, PS_D2H, PS_SHARED, PS_ROUTED,
       PS_H2D, PS_HEAD, PS_N,
       /* sub-sections, reported separately (not part of the step total) */
       PS_HC_NORM = PS_N, PS_HC_MV, PS_HC_MIX, PS_HC_COLL, PS_KDA_MV6, PS_KDA_CONV, PS_KDA_PREP, PS_KDA_RECUR,
       PS_KDA_GATEMV, PS_KDA_ONORM, PS_KDA_OMV, PS_ALL };
static const char *prof_names[PS_ALL] = { "stream moves", "attn mHC pre+norm", "MLA", "KDA", "mHC post",
                                        "ffn mHC pre+norm", "dense MLP", "D2H x", "shared expert",
                                        "routed experts (CPU)", "H2D r + add", "final norm + lm_head",
                                        "  mHC rmsnorm", "  mHC fn matvec", "  mHC mix+sinkhorn", "  mHC collapse",
                                        "  KDA 6 in matvecs", "  KDA shortconv x3", "  KDA prep", "  KDA recurrence",
                                        "  KDA gate matvecs", "  KDA out norm+gate", "  KDA out matvec" };
static double prof_sec[PS_ALL], prof_sub;          /* prof_sub: time of the last section mark */
static long   prof_steps;
static double prof_t;

static void prof_mark(int bucket, int T)
{
    if (prof_on != 2 || T != 1) return;
    gk_sync();
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const double t = (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
    /* sub-sections time from the previous mark; sections from the previous section mark,
     * so a section's figure includes its sub-sections */
    if (bucket >= PS_N) { prof_sec[bucket] += t - prof_t; prof_t = t; return; }
    if (bucket >= 0) prof_sec[bucket] += t - prof_sub;
    prof_t = prof_sub = t;
}

/* ---------------------------------------------------------------- memory ---- */
static void *dmalloc(size_t bytes, int *fail)
{
    void *p = NULL;
    if (bytes == 0) bytes = 1;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { *fail = 1; return NULL; }
    return p;
}

static void *dupload(const void *src, size_t bytes, int *fail)
{
    void *p = dmalloc(bytes, fail);
    if (p && cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) *fail = 1;
    return p;
}

static size_t mat_w_bytes(const Glm53fMat *m)
{
    const size_t n = (size_t)m->rows * (size_t)m->cols;
    return m->dt == GLM53F_WBF16 ? 2 * n : m->dt == GLM53F_WF8 ? n : 4 * n;
}

static size_t mat_s_bytes(const Glm53fMat *m)
{
    return m->dt == GLM53F_WF8 ? (size_t)((m->rows + m->br - 1) / m->br) * (size_t)m->scols * 4 : 0;
}

static size_t mat_bytes(const Glm53fMat *m) { return mat_w_bytes(m) + mat_s_bytes(m); }

static GMat upload_mat(const Glm53fMat *m, int *fail)
{
    GMat g;
    memset(&g, 0, sizeof g);
    g.dt = m->dt; g.rows = m->rows; g.cols = m->cols; g.br = m->br; g.bc = m->bc; g.scols = m->scols;
    g.w = dupload(m->w, mat_w_bytes(m), fail);
    if (m->dt == GLM53F_WF8) g.s = (float *)dupload(m->s, mat_s_bytes(m), fail);
    return g;
}

static void free_mat(GMat *g) { if (g->w) cudaFree(g->w); if (g->s) cudaFree(g->s); memset(g, 0, sizeof *g); }

static float *upload_vec(const float *v, size_t n, int *fail) { return (float *)dupload(v, n * sizeof(float), fail); }

/* y[T][R] = W[R][C] . x[T][C].
 *
 * Default: cuBLAS SGEMM on fp32 values. BF16 rows are widened exactly (a bit shift) and FP8
 * rows are dequantised as transformers does (code value times its block scale) into d->wf,
 * a block of rows at a time when the matrix is larger than the buffer (the lm_head). The
 * row-major [R][C] buffer is the column-major [C][R] matrix, so op(A) = A^T.
 *
 * GLM53F_GPU_EXACT=1 selects the fused-multiply-add kernels instead, which repeat the CPU
 * reduction order bit for bit at roughly half the speed. */
static void mv(float *y, const float *x, const GMat *m, const GDev *d, int T)
{
    const int R = m->rows, C = m->cols;
    if (d->exact) {
        if (m->dt == GLM53F_WBF16)    gk_mv_bf16(y, x, (const uint16_t *)m->w, T, R, C);
        else if (m->dt == GLM53F_WF8) gk_mv_f8(y, x, (const uint8_t *)m->w, m->s, d->lut, T, R, C,
                                              m->br, m->bc, m->scols);
        else                          gk_mv_f32(y, x, (const float *)m->w, T, R, C);
        return;
    }
    const float one = 1.0f, zero = 0.0f;
    if (m->dt == GLM53F_WF32) {
        cublasSgemm(d->blas, CUBLAS_OP_T, CUBLAS_OP_N, R, T, C, &one, (const float *)m->w, C, x, C, &zero, y, R);
        return;
    }
    int step = (int)(d->wf_floats / (size_t)C);
    if (step < 1) step = 1;
    if (m->dt == GLM53F_WF8 && step > m->br) step -= step % m->br;
    for (int r0 = 0; r0 < R; r0 += step) {
        const int n = R - r0 < step ? R - r0 : step;
        if (m->dt == GLM53F_WBF16)
            gk_widen_bf16(d->wf, (const uint16_t *)m->w + (size_t)r0 * C, (int64_t)n * C);
        else
            gk_dequant_f8(d->wf, (const uint8_t *)m->w, m->s, d->lut, r0, n, C, m->br, m->bc, m->scols);
        if (n == R)
            cublasSgemm(d->blas, CUBLAS_OP_T, CUBLAS_OP_N, R, T, C, &one, d->wf, C, x, C, &zero, y, R);
        else
            for (int t = 0; t < T; t++)
                cublasSgemv(d->blas, CUBLAS_OP_T, C, n, &one, d->wf, C, x + (size_t)t * C, 1, &zero,
                            y + (size_t)t * R + r0, 1);
    }
}

/* RMSNorm over `rows` rows of n (in place allowed): launched per block and per element by
 * default, the CPU's sequential sum under GLM53F_GPU_EXACT. */
static void norm(const GDev *d, float *y, const float *x, const float *w, int rows, int n, float eps)
{
    if (d->exact) gk_rmsnorm(y, x, w, rows, n, eps);
    else          gk_rmsnorm_par(y, x, w, rows, n, eps, d->nbs, d->ninv);
}

/* ------------------------------------------------------------- sizes ---- */
static size_t layer_weight_bytes(const Glm53fCfg *c, const Glm53fLayerW *w)
{
    const size_t E = (size_t)c->hidden, M = (size_t)c->hc_mult, NM = (2 + M) * M;
    size_t b = 2 * E * 4 + mat_bytes(&w->attn_hc.fn) + mat_bytes(&w->ffn_hc.fn) + 2 * (NM + 3) * 4;
    if (w->is_mla) {
        const Glm53fMlaW *a = &w->mla;
        b += mat_bytes(&a->q_a) + mat_bytes(&a->q_b) + mat_bytes(&a->kv_a) + mat_bytes(&a->kv_b) +
             mat_bytes(&a->o) + ((size_t)c->q_lora + (size_t)c->kv_lora) * 4;
    } else {
        const Glm53fKdaW *k = &w->kda;
        const size_t H = (size_t)c->kda_heads, D = (size_t)c->kda_head_dim, P = H * D;
        b += mat_bytes(&k->q) + mat_bytes(&k->k) + mat_bytes(&k->v) + mat_bytes(&k->f_a) +
             mat_bytes(&k->f_b) + mat_bytes(&k->b) + mat_bytes(&k->g_a) + mat_bytes(&k->g_b) +
             mat_bytes(&k->o) + (3 * P * (size_t)c->conv_k + H + P + D) * 4;
    }
    if (w->is_dense) b += mat_bytes(&w->d_gate) + mat_bytes(&w->d_up) + mat_bytes(&w->d_down);
    else             b += mat_bytes(&w->moe.sh_gate) + mat_bytes(&w->moe.sh_up) + mat_bytes(&w->moe.sh_down);
    return b;
}

static size_t layer_state_bytes(const Glm53fCfg *c, int is_mla, int cap)
{
    if (is_mla) return (size_t)cap * glm53f_kv_floats_per_pos(c) * 4;
    return glm53f_kda_state_floats(c) * 4;
}

static int mla_tsub(const Glm53fCfg *c, int T, int cap)
{
    const size_t per = (size_t)c->n_heads * (size_t)cap;
    size_t n = GPU_MLA_SCORES / (per ? per : 1);
    if (n < 1) n = 1;
    return (int)(n < (size_t)T ? n : (size_t)T);
}

static size_t dev_scratch_bytes(const Glm53fCfg *c, int T, int cap, size_t wf_floats)
{
    const size_t E = (size_t)c->hidden, M = (size_t)c->hc_mult, NM = (2 + M) * M;
    const size_t P = (size_t)c->kda_heads * c->kda_head_dim, H = (size_t)c->kda_heads;
    const size_t HA = (size_t)c->n_heads, qn = (size_t)c->qk_nope, vh = (size_t)c->v_head;
    size_t I = (size_t)c->dense_inter;
    if ((size_t)c->moe_inter * c->n_shared > I) I = (size_t)c->moe_inter * c->n_shared;
    size_t n = 0;
    n += 3 * T * M * E * 4;                       /* h, h2, flat              */
    n += T * NM * 4 + T * M * 4 + T * M * M * 4;  /* mix, post, comb         */
    n += T * M * 8;                               /* pre                      */
    n += 4 * T * E * 4;                           /* xc, xn, y, r             */
    n += 6 * T * P * 4 + T * H * 4 + T * (size_t)c->kda_head_dim * 4;  /* q k v z o gb, bt, lr */
    n += T * (size_t)c->q_lora * 4 + T * HA * qn * 4 + T * (size_t)c->kv_lora * 4 + T * HA * vh * 4;
    n += T * HA * vh * 8;                         /* dtmp                     */
    n += 3 * T * I * 4;                           /* g1, g2, act              */
    n += 2 * T * E * 4 + (size_t)c->vocab * 4;    /* xm, xm2, logits          */
    n += 256 * 4;                                 /* lut                      */
    n += T * ((M * E + 255) / 256) * 8 + T * 4 + P * 4;  /* nbs, ninv, ku     */
    n += (size_t)mla_tsub(c, T, cap) * HA * ((size_t)cap * 8 + 16);  /* msc, mtz */
    n += wf_floats * 4;                           /* wf                       */
    return n;
}

static int alloc_scratch(GDev *d, const Glm53fCfg *c, int T, int cap, size_t wf_floats)
{
    const size_t E = (size_t)c->hidden, M = (size_t)c->hc_mult, NM = (2 + M) * M;
    const size_t P = (size_t)c->kda_heads * c->kda_head_dim, H = (size_t)c->kda_heads;
    const size_t HA = (size_t)c->n_heads, qn = (size_t)c->qk_nope, vh = (size_t)c->v_head;
    size_t I = (size_t)c->dense_inter;
    if ((size_t)c->moe_inter * c->n_shared > I) I = (size_t)c->moe_inter * c->n_shared;
    int fail = 0;
    const size_t t = (size_t)T;
    d->h = (float *)dmalloc(t * M * E * 4, &fail);    d->h2 = (float *)dmalloc(t * M * E * 4, &fail);
    d->flat = (float *)dmalloc(t * M * E * 4, &fail);
    d->mix = (float *)dmalloc(t * NM * 4, &fail);     d->post = (float *)dmalloc(t * M * 4, &fail);
    d->comb = (float *)dmalloc(t * M * M * 4, &fail); d->pre = (double *)dmalloc(t * M * 8, &fail);
    d->xc = (float *)dmalloc(t * E * 4, &fail);       d->xn = (float *)dmalloc(t * E * 4, &fail);
    d->y = (float *)dmalloc(t * E * 4, &fail);        d->r = (float *)dmalloc(t * E * 4, &fail);
    d->q = (float *)dmalloc(t * P * 4, &fail);        d->k = (float *)dmalloc(t * P * 4, &fail);
    d->v = (float *)dmalloc(t * P * 4, &fail);        d->z = (float *)dmalloc(t * P * 4, &fail);
    d->o = (float *)dmalloc(t * P * 4, &fail);        d->gb = (float *)dmalloc(t * P * 4, &fail);
    d->bt = (float *)dmalloc(t * H * 4, &fail);
    d->lr = (float *)dmalloc(t * (size_t)c->kda_head_dim * 4, &fail);
    d->ql = (float *)dmalloc(t * (size_t)c->q_lora * 4, &fail);
    d->mq = (float *)dmalloc(t * HA * qn * 4, &fail);
    d->ct = (float *)dmalloc(t * (size_t)c->kv_lora * 4, &fail);
    d->acc = (float *)dmalloc(t * HA * vh * 4, &fail);
    d->dtmp = (double *)dmalloc(t * HA * vh * 8, &fail);
    d->g1 = (float *)dmalloc(t * I * 4, &fail);       d->g2 = (float *)dmalloc(t * I * 4, &fail);
    d->act = (float *)dmalloc(t * I * 4, &fail);
    d->xm = (float *)dmalloc(t * E * 4, &fail);       d->xm2 = (float *)dmalloc(t * E * 4, &fail);
    d->logits = (float *)dmalloc((size_t)c->vocab * 4, &fail);
    d->nbs = (double *)dmalloc(t * ((M * E + 255) / 256) * 8, &fail);
    d->mla_tsub = mla_tsub(c, T, cap);
    d->msc = (double *)dmalloc((size_t)d->mla_tsub * HA * (size_t)cap * 8, &fail);
    d->mtz = (double *)dmalloc((size_t)d->mla_tsub * HA * 16, &fail);
    d->ninv = (float *)dmalloc(t * 4, &fail);
    d->ku = (float *)dmalloc(P * 4, &fail);
    float lut[256];
    for (int b = 0; b < 256; b++) lut[b] = glm53f_e4m3f((uint8_t)b);
    d->lut = (float *)dupload(lut, sizeof lut, &fail);
    d->wf_floats = wf_floats;
    d->wf = (float *)dmalloc(wf_floats * 4, &fail);
    const char *ex = getenv("GLM53F_GPU_EXACT");
    d->exact = ex && atoi(ex) != 0;
    if (cublasCreate(&d->blas) != CUBLAS_STATUS_SUCCESS) { d->blas = NULL; fail = 1; }
    d->scratch_bytes = dev_scratch_bytes(c, T, cap, wf_floats);
    return fail ? -1 : 0;
}

static void free_scratch(GDev *d)
{
    void *all[] = { d->h, d->h2, d->flat, d->mix, d->post, d->comb, d->pre, d->dtmp, d->xc, d->xn, d->y,
                    d->r, d->q, d->k, d->v, d->z, d->bt, d->o, d->gb, d->lr, d->ql, d->mq, d->ct,
                    d->acc, d->g1, d->g2, d->act, d->xm, d->xm2, d->logits, d->lut, d->wf, d->nbs, d->ninv, d->ku, d->msc, d->mtz };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) if (all[i]) cudaFree(all[i]);
    if (d->blas) cublasDestroy(d->blas);
}

/* ------------------------------------------------------------- keep-warm ---- */
/* A decode step alternates ~0.25 s of GPU trunk work with ~1.6 s of routed experts on the
 * CPU. Over such a duty cycle the driver lowers the devices' performance state (P3: memory
 * at 5 GHz instead of 7.3), and the trunk then takes ~0.6 s. While the CPU computes experts
 * this thread keeps every device busy with a throwaway 4096 x 4096 SGEMV at a time, waiting
 * for each on a blocking event (no CPU spin). The model's kernels queue behind at most one
 * such product (~0.2 ms). Costs power while decoding; GLM53F_GPU_WARM=0 turns it off. */
#define GPU_WARM_N 4096

static void *warm_main(void *arg)
{
    Glm53fGpu *g = (Glm53fGpu *)arg;
    const float one = 1.0f, zero = 0.0f;
    while (!g->warm_stop) {
        if (!g->warm_want) {
#ifdef _WIN32
            Sleep(1);
#else
            usleep(1000);
#endif
            continue;
        }
        for (int i = 0; i < g->nwarm; i++) {
            cudaSetDevice(g->warm[i].id);
            cublasSgemv(g->warm[i].blas, CUBLAS_OP_T, GPU_WARM_N, GPU_WARM_N, &one, g->warm[i].A, GPU_WARM_N,
                        g->warm[i].x, 1, &zero, g->warm[i].y, 1);
            cudaEventRecord(g->warm[i].ev, NULL);
        }
        for (int i = 0; i < g->nwarm; i++) {
            cudaSetDevice(g->warm[i].id);
            cudaEventSynchronize(g->warm[i].ev);
        }
    }
    return NULL;
}

static void warm_start(Glm53fGpu *g)
{
    const char *e = getenv("GLM53F_GPU_WARM");
    if (e && atoi(e) == 0) return;
    const size_t n = (size_t)GPU_WARM_N;
    for (int i = 0; i < g->ndev; i++) {
        if (!g->dev[i].used) continue;
        int fail = 0;
        const int k = g->nwarm;
        g->warm[k].id = g->dev[i].id;
        cudaSetDevice(g->dev[i].id);
        g->warm[k].A = (float *)dmalloc(n * n * 4, &fail);
        g->warm[k].x = (float *)dmalloc(n * 4, &fail);
        g->warm[k].y = (float *)dmalloc(n * 4, &fail);
        if (!fail && (cudaMemset(g->warm[k].A, 0, n * n * 4) != cudaSuccess ||
                      cudaMemset(g->warm[k].x, 0, n * 4) != cudaSuccess)) fail = 1;
        if (!fail && cublasCreate(&g->warm[k].blas) != CUBLAS_STATUS_SUCCESS) { g->warm[k].blas = NULL; fail = 1; }
        if (!fail && cudaEventCreateWithFlags(&g->warm[k].ev, cudaEventBlockingSync | cudaEventDisableTiming) != cudaSuccess) {
            g->warm[k].ev = NULL;
            fail = 1;
        }
        g->nwarm++;
        if (fail) { fprintf(stderr, "glm53f_gpu: keep-warm unavailable on cuda:%d\n", g->dev[i].id); return; }
    }
    if (g->nwarm > 0 && pthread_create(&g->warm_thread, NULL, warm_main, g) == 0) g->warm_started = 1;
}

static void warm_free(Glm53fGpu *g)
{
    if (g->warm_started) {
        g->warm_stop = 1;
        pthread_join(g->warm_thread, NULL);
    }
    for (int k = 0; k < g->nwarm; k++) {
        cudaSetDevice(g->warm[k].id);
        if (g->warm[k].ev) cudaEventDestroy(g->warm[k].ev);
        if (g->warm[k].blas) cublasDestroy(g->warm[k].blas);
        if (g->warm[k].A) cudaFree(g->warm[k].A);
        if (g->warm[k].x) cudaFree(g->warm[k].x);
        if (g->warm[k].y) cudaFree(g->warm[k].y);
    }
}

/* ------------------------------------------------------------- lifecycle ---- */
/* Replace a BF16 or FP8 matrix by its fp32 values, computed on the device by the same kernels
 * mv() uses on the fly, so the product is unchanged; only the per-token widening goes away.
 * Needs the fp32 size free while both copies exist. Returns the extra bytes, 0 if skipped. */
static size_t widen_mat(GMat *g, const GDev *d, size_t *budget)
{
    if (!g->w || g->dt == GLM53F_WF32) return 0;
    const size_t n = (size_t)g->rows * (size_t)g->cols;
    const size_t old = g->dt == GLM53F_WBF16 ? 2 * n : n + (g->s ? (size_t)((g->rows + g->br - 1) / g->br) * g->scols * 4 : 0);
    if (4 * n > *budget) return 0;
    int fail = 0;
    float *w = (float *)dmalloc(4 * n, &fail);
    if (fail) return 0;
    if (g->dt == GLM53F_WBF16) gk_widen_bf16(w, (const uint16_t *)g->w, (int64_t)n);
    else                       gk_dequant_f8(w, (const uint8_t *)g->w, g->s, d->lut, 0, g->rows, g->cols, g->br, g->bc, g->scols);
    if (gk_sync() != 0) { cudaFree(w); return 0; }
    cudaFree(g->w);
    if (g->s) cudaFree(g->s);
    g->w = w; g->s = NULL; g->dt = GLM53F_WF32;
    *budget -= 4 * n;
    *budget += old;
    return 4 * n - old;
}

/* Spend each device's free memory beyond GPU_MARGIN on fp32 copies: the lm_head first (it is
 * otherwise widened in blocks), then the layers in order. Off under GLM53F_GPU_EXACT, whose
 * kernels read the stored formats, and under GLM53F_GPU_WIDEN=0. */
static size_t widen_resident(Glm53fGpu *g)
{
    const char *e = getenv("GLM53F_GPU_WIDEN");
    if (e && atoi(e) == 0) return 0;
    size_t extra = 0;
    for (int i = 0; i < g->ndev; i++) {
        GDev *d = &g->dev[i];
        if (!d->used || d->exact) continue;
        size_t fr = 0, tot = 0;
        if (cudaSetDevice(d->id) != cudaSuccess || cudaMemGetInfo(&fr, &tot) != cudaSuccess) continue;
        size_t budget = fr > GPU_MARGIN ? fr - GPU_MARGIN : 0;
        if (i == g->head) extra += widen_mat(&g->lm_head, d, &budget);
        for (int L = 0; L < g->nl; L++) {
            GLayer *l = &g->lay[L];
            if (l->dev != i) continue;
            GMat *mats[] = { &l->attn_hc.fn, &l->ffn_hc.fn, &l->q, &l->k, &l->v, &l->f_a, &l->f_b, &l->b, &l->g_a,
                             &l->g_b, &l->o, &l->q_a, &l->q_b, &l->kv_a, &l->kv_b, &l->mo, &l->gate, &l->up, &l->down };
            for (size_t k = 0; k < sizeof mats / sizeof mats[0]; k++) extra += widen_mat(mats[k], d, &budget);
        }
    }
    return extra;
}

static int upload_layer(GLayer *g, const Glm53fCfg *c, const Glm53fLayerW *w, int cap)
{
    int fail = 0;
    const size_t E = (size_t)c->hidden, M = (size_t)c->hc_mult, NM = (2 + M) * M;
    g->is_mla = w->is_mla; g->is_dense = w->is_dense;
    g->attn_hc.fn = upload_mat(&w->attn_hc.fn, &fail);
    g->attn_hc.base = upload_vec(w->attn_hc.base, NM, &fail);
    g->attn_hc.scale = upload_vec(w->attn_hc.scale, 3, &fail);
    g->ffn_hc.fn = upload_mat(&w->ffn_hc.fn, &fail);
    g->ffn_hc.base = upload_vec(w->ffn_hc.base, NM, &fail);
    g->ffn_hc.scale = upload_vec(w->ffn_hc.scale, 3, &fail);
    g->in_norm = upload_vec(w->in_norm, E, &fail);
    g->post_norm = upload_vec(w->post_norm, E, &fail);
    if (w->is_mla) {
        const Glm53fMlaW *a = &w->mla;
        g->q_a = upload_mat(&a->q_a, &fail); g->q_b = upload_mat(&a->q_b, &fail);
        g->kv_a = upload_mat(&a->kv_a, &fail); g->kv_b = upload_mat(&a->kv_b, &fail);
        g->mo = upload_mat(&a->o, &fail);
        g->q_a_norm = upload_vec(a->q_a_norm, (size_t)c->q_lora, &fail);
        g->kv_a_norm = upload_vec(a->kv_a_norm, (size_t)c->kv_lora, &fail);
        g->kv = (float *)dmalloc((size_t)cap * glm53f_kv_floats_per_pos(c) * 4, &fail);
    } else {
        const Glm53fKdaW *k = &w->kda;
        const size_t H = (size_t)c->kda_heads, D = (size_t)c->kda_head_dim, P = H * D;
        g->q = upload_mat(&k->q, &fail); g->k = upload_mat(&k->k, &fail); g->v = upload_mat(&k->v, &fail);
        g->f_a = upload_mat(&k->f_a, &fail); g->f_b = upload_mat(&k->f_b, &fail);
        g->b = upload_mat(&k->b, &fail); g->g_a = upload_mat(&k->g_a, &fail);
        g->g_b = upload_mat(&k->g_b, &fail); g->o = upload_mat(&k->o, &fail);
        g->q_conv = upload_vec(k->q_conv, P * (size_t)c->conv_k, &fail);
        g->k_conv = upload_vec(k->k_conv, P * (size_t)c->conv_k, &fail);
        g->v_conv = upload_vec(k->v_conv, P * (size_t)c->conv_k, &fail);
        g->A_log = upload_vec(k->A_log, H, &fail);
        g->dt_bias = upload_vec(k->dt_bias, P, &fail);
        g->o_norm = upload_vec(k->o_norm, D, &fail);
        g->state_bytes = glm53f_kda_state_floats(c) * 4;
        g->state = (float *)dmalloc(g->state_bytes, &fail);
        if (g->state && cudaMemset(g->state, 0, g->state_bytes) != cudaSuccess) fail = 1;
    }
    if (w->is_dense) {
        g->gate = upload_mat(&w->d_gate, &fail); g->up = upload_mat(&w->d_up, &fail);
        g->down = upload_mat(&w->d_down, &fail);
    } else {
        g->gate = upload_mat(&w->moe.sh_gate, &fail); g->up = upload_mat(&w->moe.sh_up, &fail);
        g->down = upload_mat(&w->moe.sh_down, &fail);
    }
    return fail ? -1 : 0;
}

static void free_layer(GLayer *g)
{
    GMat *mats[] = { &g->attn_hc.fn, &g->ffn_hc.fn, &g->q, &g->k, &g->v, &g->f_a, &g->f_b, &g->b, &g->g_a,
                     &g->g_b, &g->o, &g->q_a, &g->q_b, &g->kv_a, &g->kv_b, &g->mo, &g->gate, &g->up, &g->down };
    for (size_t i = 0; i < sizeof mats / sizeof mats[0]; i++) free_mat(mats[i]);
    void *vecs[] = { g->attn_hc.base, g->attn_hc.scale, g->ffn_hc.base, g->ffn_hc.scale, g->in_norm,
                     g->post_norm, g->q_conv, g->k_conv, g->v_conv, g->A_log, g->dt_bias, g->o_norm,
                     g->state, g->q_a_norm, g->kv_a_norm, g->kv };
    for (size_t i = 0; i < sizeof vecs / sizeof vecs[0]; i++) if (vecs[i]) cudaFree(vecs[i]);
}

void glm53f_gpu_free(Glm53fGpu *g)
{
    if (!g) return;
    warm_free(g);
    if (prof_on > 0 && prof_tokens > 0)
        fprintf(stderr, "gpu profile: %ld decode steps | trunk on GPU %.2f s | routed experts on CPU %.2f s | "
                        "final norm + lm_head %.2f s\n", prof_tokens, prof_gpu, prof_cpu, prof_head);
    if (prof_on == 2 && prof_steps > 0) {
        double tot = 0.0;
        for (int i = 0; i < PS_N; i++) tot += prof_sec[i];
        fprintf(stderr, "gpu profile per decode step (%ld steps, synchronised): %.1f ms\n", prof_steps,
                1e3 * tot / prof_steps);
        for (int i = 0; i < PS_ALL; i++)
            fprintf(stderr, "  %-22s %8.1f ms\n", prof_names[i], 1e3 * prof_sec[i] / prof_steps);
    }
    for (int L = 0; L < g->nl; L++) {
        if (!g->lay) break;
        cudaSetDevice(g->dev[g->lay[L].dev].id);
        free_layer(&g->lay[L]);
    }
    for (int i = 0; i < g->ndev; i++) {
        if (!g->dev[i].used) continue;
        cudaSetDevice(g->dev[i].id);
        if (i == g->head) { free_mat(&g->lm_head); if (g->norm) cudaFree(g->norm); }
        free_scratch(&g->dev[i]);
    }
    if (g->host_h) cudaFreeHost(g->host_h);
    if (g->host_x) cudaFreeHost(g->host_x);
    if (g->host_r) cudaFreeHost(g->host_r);
    if (g->host_logits) cudaFreeHost(g->host_logits);
    free(g->moe_scratch);
    free(g->lay);
    free(g);
}

Glm53fGpu *glm53f_gpu_create(Glm53fModel *m, const int *devices, int ndev)
{
    const Glm53fCfg *c = &m->cfg;
    Glm53fGpu *g = (Glm53fGpu *)calloc(1, sizeof *g);
    if (!g) return NULL;
    const int count = glm53f_gpu_device_count();
    if (ndev <= 0) {
        g->ndev = count < GPU_MAX_DEV ? count : GPU_MAX_DEV;
        for (int i = 0; i < g->ndev; i++) g->dev[i].id = i;
    } else {
        g->ndev = ndev < GPU_MAX_DEV ? ndev : GPU_MAX_DEV;
        for (int i = 0; i < g->ndev; i++) g->dev[i].id = devices[i];
    }
    g->nl = m->n_bound;
    g->maxT = m->cap < GPU_CHUNK ? m->cap : GPU_CHUNK;
    g->head = -1;
    g->lay = (GLayer *)calloc((size_t)g->nl, sizeof(GLayer));
    if (!g->lay || g->ndev <= 0) { free(g->lay); free(g); return NULL; }

    /* ---- placement: layers in order, each device taking a share of the trunk proportional to
     * its free memory, so every device keeps spare room for fp32 copies (widen_resident) ---- */
    size_t freeb[GPU_MAX_DEV], left[GPU_MAX_DEV];
    /* The widening buffer holds the largest non-fp32 trunk matrix, up to GPU_WIDEN_MAX floats;
     * the lm_head is widened a block of rows at a time. */
    size_t wf_floats = (size_t)m->mb.lm_head.cols * 64;
    for (int L = 0; L < g->nl; L++) {
        const Glm53fLayerW *w = &m->lay[L].w;
        const Glm53fMat *mats[] = { &w->attn_hc.fn, &w->ffn_hc.fn, &w->kda.q, &w->kda.k, &w->kda.v, &w->kda.f_a,
                                    &w->kda.f_b, &w->kda.b, &w->kda.g_a, &w->kda.g_b, &w->kda.o, &w->mla.q_a,
                                    &w->mla.q_b, &w->mla.kv_a, &w->mla.kv_b, &w->mla.o, &w->d_gate, &w->d_up,
                                    &w->d_down, &w->moe.sh_gate, &w->moe.sh_up, &w->moe.sh_down };
        for (size_t i = 0; i < sizeof mats / sizeof mats[0]; i++) {
            const size_t n = (size_t)mats[i]->rows * (size_t)mats[i]->cols;
            if (mats[i]->w && mats[i]->dt != GLM53F_WF32 && n > wf_floats) wf_floats = n;
        }
    }
    if (m->mb.lm_head.dt != GLM53F_WF32) {
        const size_t n = (size_t)m->mb.lm_head.rows * (size_t)m->mb.lm_head.cols;
        if (n > wf_floats) wf_floats = n;
    }
    if (wf_floats > GPU_WIDEN_MAX) wf_floats = GPU_WIDEN_MAX;
    const size_t scratch = dev_scratch_bytes(c, g->maxT, m->cap, wf_floats);
    const size_t head_bytes = mat_bytes(&m->mb.lm_head) + (size_t)c->hidden * 4;
    for (int i = 0; i < g->ndev; i++) {
        size_t fr = 0, tot = 0;
        if (cudaSetDevice(g->dev[i].id) != cudaSuccess || cudaMemGetInfo(&fr, &tot) != cudaSuccess) fr = 0;
        freeb[i] = fr;
        left[i] = fr > scratch + GPU_MARGIN ? fr - scratch - GPU_MARGIN : 0;
    }
    /* GLM53F_GPU_LAYERS_PER_DEV caps layers per device: a diagnostic that forces a
     * multi-device placement (and its stream transfers) even for a model that fits on one. */
    const char *lpd = getenv("GLM53F_GPU_LAYERS_PER_DEV");
    const int max_per_dev = lpd ? atoi(lpd) : 0;
    double total = (double)head_bytes, room = 0.0;
    for (int L = 0; L < g->nl; L++)
        total += (double)(layer_weight_bytes(c, &m->lay[L].w) + layer_state_bytes(c, m->lay[L].w.is_mla, m->cap));
    for (int i = 0; i < g->ndev; i++) room += (double)left[i];
    const double share = room > 0.0 && total < room ? total / room : 1.0;
    int di = 0, on_dev = 0;
    size_t placed = 0;
    for (int L = 0; L < g->nl; L++) {
        const size_t need = layer_weight_bytes(c, &m->lay[L].w) + layer_state_bytes(c, m->lay[L].w.is_mla, m->cap);
        const int last = L == g->nl - 1;
        if (max_per_dev > 0 && on_dev >= max_per_dev && di + 1 < g->ndev) { di++; on_dev = 0; placed = 0; }
        if (!max_per_dev && di + 1 < g->ndev && on_dev > 0 &&
            (double)(placed + need) > share * (double)(left[di] + placed) + (double)need / 2)
            { di++; on_dev = 0; placed = 0; }
        while (di < g->ndev && left[di] < need + (last ? head_bytes : 0)) { di++; on_dev = 0; placed = 0; }
        if (di >= g->ndev) {
            fprintf(stderr, "glm53f_gpu: the trunk does not fit: layer %d needs %.2f GB and no device has room "
                            "(free:", L, (double)need / 1e9);
            for (int i = 0; i < g->ndev; i++) fprintf(stderr, " dev%d %.2f GB", g->dev[i].id, (double)freeb[i] / 1e9);
            fprintf(stderr, "). Running on the CPU.\n");
            free(g->lay); free(g);
            return NULL;
        }
        g->lay[L].dev = di;
        g->lay[L].bytes = need;
        left[di] -= need;
        placed += need;
        g->dev[di].used = 1;
        on_dev++;
    }
    g->head = g->lay[g->nl - 1].dev;

    /* ---- upload ---- */
    for (int L = 0; L < g->nl; L++) {
        if (cudaSetDevice(g->dev[g->lay[L].dev].id) != cudaSuccess ||
            upload_layer(&g->lay[L], c, &m->lay[L].w, m->cap) != 0) {
            fprintf(stderr, "glm53f_gpu: upload of layer %d failed; running on the CPU\n", L);
            g->nl = L + 1;
            glm53f_gpu_free(g);
            return NULL;
        }
    }
    int fail = 0;
    for (int i = 0; i < g->ndev; i++) {
        if (!g->dev[i].used) continue;
        cudaSetDevice(g->dev[i].id);
        if (alloc_scratch(&g->dev[i], c, g->maxT, m->cap, wf_floats) != 0) fail = 1;
        if (i == g->head) {
            g->lm_head = upload_mat(&m->mb.lm_head, &fail);
            g->norm = upload_vec(m->mb.norm, (size_t)c->hidden, &fail);
        }
    }
    const size_t E = (size_t)c->hidden, M = (size_t)c->hc_mult;
    if (cudaMallocHost((void **)&g->host_h, (size_t)g->maxT * M * E * 4) != cudaSuccess) fail = 1;
    if (cudaMallocHost((void **)&g->host_x, (size_t)g->maxT * E * 4) != cudaSuccess) fail = 1;
    if (cudaMallocHost((void **)&g->host_r, (size_t)g->maxT * E * 4) != cudaSuccess) fail = 1;
    if (cudaMallocHost((void **)&g->host_logits, (size_t)c->vocab * 4) != cudaSuccess) fail = 1;
    g->moe_scratch = (float *)malloc(glm53f_moe_scratch(c, g->maxT) * sizeof(float));
    if (fail || !g->moe_scratch) {
        fprintf(stderr, "glm53f_gpu: allocating GPU scratch or pinned buffers failed; running on the CPU\n");
        glm53f_gpu_free(g);
        return NULL;
    }

    const size_t widened = widen_resident(g);
    warm_start(g);

    if (!glm53f_quiet)
    printf("gpu: trunk placed on %d device(s):", g->ndev);
    for (int i = 0; i < g->ndev; i++) {
        if (!g->dev[i].used) continue;
        int first = -1, lastl = -1;
        size_t bytes = 0;
        for (int L = 0; L < g->nl; L++) if (g->lay[L].dev == i) { if (first < 0) first = L; lastl = L; bytes += g->lay[L].bytes; }
        if (!glm53f_quiet)
        printf(" [cuda:%d layers %d-%d, %.2f GB + %.2f GB scratch%s]", g->dev[i].id, first, lastl,
               (double)bytes / 1e9, (double)g->dev[i].scratch_bytes / 1e9, i == g->head ? " + lm_head" : "");
    }
    if (!glm53f_quiet)
    printf("\n");
    if (widened && !glm53f_quiet) printf("gpu: %.2f GB of weights kept as fp32 copies in spare device memory\n", (double)widened / 1e9);
    return g;
}

void glm53f_gpu_reset(Glm53fGpu *g)
{
    for (int L = 0; L < g->nl; L++) {
        GLayer *l = &g->lay[L];
        if (!l->state) continue;
        cudaSetDevice(g->dev[l->dev].id);
        cudaMemset(l->state, 0, l->state_bytes);
    }
}

/* ------------------------------------------------------------------ forward ---- */
/* mHC collapse at one site: flat norm, mix, pre/post/comb, collapse into d->xc. */
static void hc_pre(GDev *d, const GHc *hc, const Glm53fCfg *c, int T)
{
    const int M = c->hc_mult, E = c->hidden;
    norm(d, d->flat, d->h, NULL, T, M * E, c->rms_eps);
    prof_mark(PS_HC_NORM, T);
    mv(d->mix, d->flat, &hc->fn, d, T);
    prof_mark(PS_HC_MV, T);
    gk_hc_mix(d->pre, d->post, d->comb, d->mix, hc->base, hc->scale, T, M, c->hc_iters, c->hc_eps);
    prof_mark(PS_HC_MIX, T);
    gk_hc_collapse(d->xc, d->pre, d->h, T, M, E);
    prof_mark(PS_HC_COLL, T);
}

static void hc_post(GDev *d, const Glm53fCfg *c, int T)
{
    gk_hc_post(d->h2, d->h, d->y, d->post, d->comb, T, c->hc_mult, c->hidden);
    float *t = d->h; d->h = d->h2; d->h2 = t;
}

static void mlp(GDev *d, const GLayer *l, const Glm53fCfg *c, int T)
{
    const int I = l->gate.rows, E = c->hidden;
    mv(d->g1, d->xn, &l->gate, d, T);
    mv(d->g2, d->xn, &l->up, d, T);
    gk_swiglu(d->act, d->g1, d->g2, T, I, c->swiglu_limit);
    mv(d->y, d->act, &l->down, d, T);
    (void)E;
}

static void kda(GDev *d, const GLayer *l, const Glm53fCfg *c, int T)
{
    const int H = c->kda_heads, D = c->kda_head_dim, P = H * D, K = c->conv_k, hist = K - 1;
    mv(d->q, d->xn, &l->q, d, T);
    mv(d->k, d->xn, &l->k, d, T);
    mv(d->v, d->xn, &l->v, d, T);
    mv(d->bt, d->xn, &l->b, d, T);
    mv(d->lr, d->xn, &l->f_a, d, T);
    mv(d->z, d->lr, &l->f_b, d, T);
    prof_mark(PS_KDA_MV6, T);
    float *cs = l->state + (size_t)H * D * D;
    gk_shortconv(d->q, d->q, l->q_conv, cs, P, K, T);
    gk_shortconv(d->k, d->k, l->k_conv, cs + (size_t)P * hist, P, K, T);
    gk_shortconv(d->v, d->v, l->v_conv, cs + (size_t)2 * P * hist, P, K, T);
    prof_mark(PS_KDA_CONV, T);
    gk_kda_prep(d->q, d->k, d->z, d->bt, l->A_log, l->dt_bias, T, H, D, c->gate_lb);
    prof_mark(PS_KDA_PREP, T);
    if (d->exact) gk_kda_recur(l->state, d->o, d->q, d->k, d->v, d->z, d->bt, T, H, D);
    else          gk_kda_recur_par(l->state, d->o, d->ku, d->q, d->k, d->v, d->z, d->bt, T, H, D);
    prof_mark(PS_KDA_RECUR, T);
    mv(d->lr, d->xn, &l->g_a, d, T);
    mv(d->gb, d->lr, &l->g_b, d, T);
    prof_mark(PS_KDA_GATEMV, T);
    gk_onorm_gate(d->o, d->gb, l->o_norm, T, H, D, c->rms_eps);
    prof_mark(PS_KDA_ONORM, T);
    mv(d->y, d->o, &l->o, d, T);
    prof_mark(PS_KDA_OMV, T);
}

static void mla(GDev *d, const GLayer *l, const Glm53fCfg *c, int T, int cached)
{
    const int H = c->n_heads, qn = c->qk_nope, vh = c->v_head, kvd = qn + vh;
    mv(d->ql, d->xn, &l->q_a, d, T);
    norm(d, d->ql, d->ql, l->q_a_norm, T, c->q_lora, c->rms_eps);
    mv(d->mq, d->ql, &l->q_b, d, T);
    mv(d->ct, d->xn, &l->kv_a, d, T);
    norm(d, d->ct, d->ct, l->kv_a_norm, T, c->kv_lora, c->rms_eps);
    mv(l->kv + (size_t)cached * H * kvd, d->ct, &l->kv_b, d, T);
    if (d->exact) gk_mla_attn(d->acc, d->dtmp, d->mq, l->kv, T, H, qn, vh, cached, 1.0f / sqrtf((float)qn));
    else          gk_mla_attn_par(d->acc, d->msc, d->mtz, d->mq, l->kv, T, H, qn, vh, cached,
                                  1.0f / sqrtf((float)qn), d->mla_tsub);
    mv(d->y, d->acc, &l->mo, d, T);
}

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Move the residual streams from device a to device b through pinned host memory. */
static int move_streams(Glm53fGpu *g, int a, int b, size_t bytes)
{
    if (cudaSetDevice(g->dev[a].id) != cudaSuccess ||
        cudaMemcpy(g->host_h, g->dev[a].h, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    if (cudaSetDevice(g->dev[b].id) != cudaSuccess ||
        cudaMemcpy(g->dev[b].h, g->host_h, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    return 0;
}

static int forward_chunk(Glm53fGpu *g, Glm53fModel *m, const int *ids, int T, float *logits,
                         int *argmax_all, int want_last)
{
    const Glm53fCfg *c = &m->cfg;
    const int E = c->hidden, M = c->hc_mult;
    const size_t hbytes = (size_t)T * M * E * 4;
    if (prof_on < 0) { const char *e = getenv("GLM53F_GPU_PROFILE"); prof_on = e ? atoi(e) : 0; }
    double tp = prof_on ? now_s() : 0.0;
    prof_mark(-1, T);
    if (prof_on == 2 && T == 1) prof_steps++;

    for (int t = 0; t < T; t++) {
        float *ht = g->host_h + (size_t)t * M * E;
        glm53f_embed_row(ht, &m->mb.embed, ids[t]);
        for (int i = 1; i < M; i++) memcpy(ht + (size_t)i * E, ht, (size_t)E * sizeof(float));
    }
    int cur = g->lay[0].dev;
    if (cudaSetDevice(g->dev[cur].id) != cudaSuccess ||
        cudaMemcpy(g->dev[cur].h, g->host_h, hbytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;

    m->layers_completed = 0;
    for (int L = 0; L < g->nl; L++) {
        GLayer *l = &g->lay[L];
        if (l->dev != cur) {
            if (move_streams(g, cur, l->dev, hbytes) != 0) return -1;
            cur = l->dev;
            prof_mark(PS_MOVE, T);
        }
        GDev *d = &g->dev[cur];

        hc_pre(d, &l->attn_hc, c, T);
        norm(d, d->xn, d->xc, l->in_norm, T, E, c->rms_eps);
        prof_mark(PS_ATTN_HC, T);
        if (l->is_mla) mla(d, l, c, T, m->cached);
        else           kda(d, l, c, T);
        prof_mark(l->is_mla ? PS_MLA : PS_KDA, T);
        hc_post(d, c, T);
        prof_mark(PS_HC_POST, T);

        hc_pre(d, &l->ffn_hc, c, T);
        norm(d, d->xn, d->xc, l->post_norm, T, E, c->rms_eps);
        prof_mark(PS_FFN_HC, T);
        if (l->is_dense) {
            mlp(d, l, c, T);
            prof_mark(PS_DENSE, T);
        } else {
            /* The routed experts run on the CPU while the GPU computes the shared expert. */
            if (cudaMemcpy(g->host_x, d->xn, (size_t)T * E * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
            prof_mark(PS_D2H, T);
            if (prof_on && T == 1) { const double t = now_s(); prof_gpu += t - tp; tp = t; }
            mlp(d, l, c, T);
            prof_mark(PS_SHARED, T);
            const long drops = glm53f_expert_drops;
            g->warm_want = 1;
            glm53f_moe_routed(g->host_r, g->host_x, &m->lay[L].w.moe, c, T, g->moe_scratch);
            g->warm_want = 0;
            prof_mark(PS_ROUTED, T);
            if (prof_on && T == 1) { const double t = now_s(); prof_cpu += t - tp; tp = t; }
            if (glm53f_expert_drops != drops) {
                fprintf(stderr, "glm53f_gpu: routed expert load failed at layer %d; refusing partial output\n", L);
                return -1;
            }
            if (cudaMemcpy(d->r, g->host_r, (size_t)T * E * 4, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
            gk_add(d->y, d->r, (int64_t)T * E);
            prof_mark(PS_H2D, T);
        }
        hc_post(d, c, T);
        prof_mark(PS_HC_POST, T);
        if (gk_check("layer") != 0) return -1;
        m->layers_completed = L + 1;
    }

    if (cur != g->head) {
        if (move_streams(g, cur, g->head, hbytes) != 0) return -1;
        cur = g->head;
        prof_mark(PS_MOVE, T);
    }
    GDev *d = &g->dev[cur];
    if (prof_on && T == 1) { gk_sync(); const double t = now_s(); prof_gpu += t - tp; tp = t; }
    gk_mean_streams(d->xm, d->h, T, M, E);
    norm(d, d->xm2, d->xm, g->norm, T, E, c->rms_eps);
    const int V = c->vocab;
    for (int t = (argmax_all ? 0 : T - 1); t < T; t++) {
        if (!argmax_all && !want_last) break;
        mv(d->logits, d->xm2 + (size_t)t * E, &g->lm_head, d, 1);
        if (cudaMemcpy(g->host_logits, d->logits, (size_t)V * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
        if (argmax_all) {
            int b = 0;
            for (int i = 1; i < V; i++) if (g->host_logits[i] > g->host_logits[b]) b = i;
            argmax_all[t] = b;
        }
        if (t == T - 1 && want_last && logits) memcpy(logits, g->host_logits, (size_t)V * sizeof(float));
    }
    if (gk_check("head") != 0) return -1;
    prof_mark(PS_HEAD, T);
    if (prof_on && T == 1) {
        const double t = now_s();
        prof_head += t - tp;
        prof_tokens += T;
    }
    m->cached += T;
    return 0;
}

int glm53f_gpu_forward(Glm53fGpu *g, Glm53fModel *m, const int *ids, int T, float *logits, int *argmax_all)
{
    for (int t0 = 0; t0 < T; t0 += g->maxT) {
        const int n = (T - t0) < g->maxT ? (T - t0) : g->maxT;
        const int last = t0 + n == T;
        if (forward_chunk(g, m, ids + t0, n, last ? logits : NULL, argmax_all ? argmax_all + t0 : NULL,
                          last) != 0)
            return -1;
    }
    return 0;
}
