/* SPDX-License-Identifier: Apache-2.0 */
/*
 * glm53f.h, GLM-5.3-Flash inference engine: public configuration, weight types and kernels.
 *
 * ARCHITECTURE (text model only; verified against the released config.json and against
 * transformers/models/glm5_next/modeling_glm5_next.py, which is the reference for every
 * kernel below)
 *
 *   45 decoder layers, hidden 4096, vocab 154880.
 *   Attention: 34 KDA layers (gated delta-rule linear attention, recurrent) + 11 MLA layers
 *              (NoPE, DeepSeek Sparse Attention indexer). layer_types in the config says which.
 *   Feed-forward: 3 dense SwiGLU layers, then 42 MoE layers: 288 routed experts, top-8,
 *              sigmoid router with e_score_correction_bias, 1 shared expert.
 *   Residual: Manifold-Constrained Hyper-Connections (mHC). The residual stream is
 *              hc_mult = 4 copies of the hidden state; every sub-block collapses them with
 *              learned weights, and writes back through a Sinkhorn-normalised 4x4 mixer.
 *   Activation: SwiGLU with clamping, silu(min(gate, 10)) * clamp(up, -10, 10).
 *   Weights: BF16 for KDA, norms, embeddings and lm_head; FP8 E4M3 with 128x128 block
 *              scales (weight_scale_inv) for MLA projections, dense/shared MLPs and every
 *              routed expert.
 *
 * WHAT IS NOT IMPLEMENTED, AND HOW THAT IS KEPT HONEST
 *   - The DSA indexer is not run. Its top-k selection keeps index_topk / index_kpool
 *     compressed pools plus the incomplete tail, so while the sequence has at most
 *     index_topk + index_kpool - 1 positions (2051 for the released config) every visible
 *     token is selected and sparse attention IS dense causal attention, exactly. The
 *     engine refuses sequences longer than that rather than silently computing a
 *     different model. glm53f_dense_attn_limit() returns the bound.
 *   - The multi-token-prediction layer (layer 45 in the checkpoint) and the vision
 *     encoder are ignored; neither participates in text generation.
 */
#ifndef GLM53F_H
#define GLM53F_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLM53F_MAX_LAYERS 256
#define GLM53F_MAX_TOPK   64
#define GLM53F_MAX_HC     8
#define GLM53F_MAX_EOS    8

/* ---------------------------------------------------------------- config ---- */
typedef struct {
    int   hidden;            /* 4096   */
    int   n_layers;          /* 45     */
    int   vocab;             /* 154880 */
    float rms_eps;           /* 1e-5   */

    /* mHC residual */
    int   hc_mult;           /* 4      */
    int   hc_iters;          /* 20     */
    float hc_eps;            /* 1e-6   */

    /* KDA */
    int   kda_heads;         /* 64     */
    int   kda_head_dim;      /* 128    */
    int   conv_k;            /* 4      */
    float gate_lb;           /* -5.0   */

    /* MLA (NoPE) */
    int   n_heads;           /* 64     */
    int   q_lora;            /* 1536   */
    int   kv_lora;           /* 512    */
    int   qk_nope;           /* 256    */
    int   v_head;            /* 256    */
    int   index_topk;        /* 2048   */
    int   index_kpool;       /* 4      */

    /* feed-forward */
    int   n_experts;         /* 288    */
    int   topk;              /* 8      */
    int   n_shared;          /* 1      */
    int   moe_inter;         /* 2048   */
    int   dense_inter;       /* 12288  */
    float routed_scale;      /* 2.5    */
    int   norm_topk;         /* 1      */
    float swiglu_limit;      /* 10.0   */

    /* FP8 block size from quantization_config.weight_block_size, 0 if unquantised */
    int   fp8_block_r, fp8_block_c;

    unsigned char is_mla[GLM53F_MAX_LAYERS];     /* 1 = MLA, 0 = KDA   */
    unsigned char is_dense[GLM53F_MAX_LAYERS];   /* 1 = dense MLP      */

    int   eos[GLM53F_MAX_EOS];
    int   n_eos;
} Glm53fCfg;

static inline int glm53f_is_mla(const Glm53fCfg *c, int layer)   { return c->is_mla[layer]; }
static inline int glm53f_is_dense(const Glm53fCfg *c, int layer) { return c->is_dense[layer]; }

/* Longest sequence for which dense causal attention equals DSA exactly (see above). */
static inline int glm53f_dense_attn_limit(const Glm53fCfg *c)
{
    return c->index_topk + c->index_kpool - 1;
}

/* ---------------------------------------------------------------- matrices ---- */
/* Every weight matrix is self-describing. A layer mixes three storage formats (BF16 KDA
 * projections beside FP8 MLA projections beside an fp32 router), so the dtype lives on
 * the matrix, not on the struct that holds it.
 *
 *   GLM53F_WF32   w -> float[rows*cols]
 *   GLM53F_WBF16  w -> uint16[rows*cols], bf16 bit patterns
 *   GLM53F_WF8    w -> uint8[rows*cols] E4M3 codes; s -> float[srows*scols], the checkpoint's
 *                 weight_scale_inv. Element (i,j) is E4M3(w[i*cols+j]) * s[i/br][j/bc].
 *   GLM53F_WI4    w -> uint8[rows*cols/2] signed 4-bit levels, two per byte (even column in
 *                 the low nibble); s -> float[rows*scols] one step per group of `bc`
 *                 columns. Element (i,j) is (nibble - 8) * s[i*scols + j/bc]. This is not a
 *                 checkpoint format: the expert cache produces it from FP8 when asked to
 *                 hold more experts in the same RAM (--experts int4), and it is an
 *                 approximation, not the released weights.
 */
enum { GLM53F_WF32 = 0, GLM53F_WBF16 = 1, GLM53F_WF8 = 2, GLM53F_WI4 = 3 };

/* Group size of GLM53F_WI4 matrices produced by the engine. */
#define GLM53F_I4_GROUP 64

typedef struct {
    const void  *w;
    const float *s;
    int dt;
    int rows, cols;
    int br, bc;              /* FP8 block size */
    int scols;               /* FP8 scale columns = ceil(cols / bc) */
} Glm53fMat;

/* y[rows] = M . x[cols]. Parallel over output rows; deterministic at any thread count. */
void glm53f_mm(float *y, const float *x, const Glm53fMat *m);

/* Quantise an FP8 matrix into GLM53F_WI4: q holds rows*(cols/2) bytes, steps rows*groups
 * floats (groups = cols / GLM53F_I4_GROUP). Symmetric, one step per group: step = max|w|/7,
 * level = lrintf(w/step) clamped to [-8, 7]. cols must be a multiple of the group. */
void glm53f_i4_from_f8(unsigned char *q, float *steps, const Glm53fMat *src);

/* bf16 -> f32 is a pure left shift: bf16 IS the top 16 bits of an f32. */
static inline float glm53f_bf16f(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

/* E4M3FN code -> float, exact (256-entry table built once). */
float glm53f_e4m3f(uint8_t b);

/* Gather one embedding row (BF16 or F32 table). */
void glm53f_embed_row(float *dst, const Glm53fMat *table, int64_t row);

/* ------------------------------------------------------------------ ops ---- */
/* y = w * x / sqrt(mean(x^2) + eps); w may be NULL (unweighted). */
void glm53f_rmsnorm(float *y, const float *x, const float *w, int n, float eps);

/* SwiGLU with clamping over a [gate | up] input of 2n: y = silu(min(g,lim)) * clamp(u,±lim) */
void glm53f_swiglu_clamp(float *y, const float *gu, int n, float limit);

/* Causal depthwise conv with fused SiLU; state holds k-1 previous inputs per channel. */
void glm53f_shortconv(float *y, const float *x, const float *w, float *state,
                      int channels, int k, int T);

/* ------------------------------------------------------------ weight structs ---- */
/* mHC site: fn [(2+M)*M][M*hidden], base [(2+M)*M], scale [3]. */
typedef struct {
    Glm53fMat    fn;
    const float *base, *scale;
} Glm53fHcW;

typedef struct {
    Glm53fMat    q, k, v;              /* [H*D][hidden]                       */
    const float *q_conv, *k_conv, *v_conv; /* [H*D][conv_k]                   */
    Glm53fMat    f_a, f_b;             /* forget gate low rank [D][hidden], [H*D][D] */
    const float *A_log;                /* [H]                                 */
    const float *dt_bias;              /* [H*D]                               */
    Glm53fMat    b;                    /* [H][hidden]                         */
    Glm53fMat    g_a, g_b;             /* output gate low rank                */
    const float *o_norm;               /* [D]                                 */
    Glm53fMat    o;                    /* [hidden][H*D]                       */
} Glm53fKdaW;

typedef struct {
    Glm53fMat    q_a, q_b, kv_a, kv_b, o;
    const float *q_a_norm, *kv_a_norm;
} Glm53fMlaW;

/* One routed expert as the cache serves it: three FP8 (or BF16/F32) matrices. */
typedef struct {
    Glm53fMat gate, up, down;
} Glm53fExpertQ;

typedef struct Glm53fExpertSrc {
    /* Pointers must stay valid until the caller finishes the current token. */
    int (*get)(struct Glm53fExpertSrc *self, int layer, int expert, Glm53fExpertQ *out);
    /* Optional: bring n experts resident with overlapping reads. May be NULL. */
    int (*getmany)(struct Glm53fExpertSrc *self, int layer, const int *experts, int n);
    /* Optional: the expert returned by get() is no longer in use and may be evicted.
     * Every successful get() must be matched by one release() when this is non-NULL. */
    void (*release)(struct Glm53fExpertSrc *self, int layer, int expert);
    /* Optional: these experts are likely to be requested soon; start reading them in the
     * background. Must not block and must not change what get() returns. May be NULL. */
    void (*hint)(struct Glm53fExpertSrc *self, int layer, const int *experts, int n);
    void *ctx;
} Glm53fExpertSrc;

typedef struct {
    const float *gate;                 /* router [n_experts][hidden], fp32    */
    const float *bias;                 /* e_score_correction_bias             */
    Glm53fMat    sh_gate, sh_up, sh_down;
    Glm53fExpertSrc *src;
    /* Router of the NEXT MoE layer, or NULL. Used to predict its experts for prefetch
     * (src->hint) and for GLM53F_PREDICT_STATS; never changes routing or output. */
    const float *next_gate, *next_bias;
    /* Routers of the MoE layers 2, 3 and 4 ahead (NULL past the end); prediction research
     * (GLM53F_PREDICT_STATS) only. */
    const float *ahead_gate[3], *ahead_bias[3];
    int          prefetch_n;          /* predicted experts to hint per decode step, 0 = off */
    int          layer;
} Glm53fMoeW;

typedef struct {
    Glm53fHcW    attn_hc, ffn_hc;
    const float *in_norm, *post_norm;
    int          is_mla, is_dense;
    Glm53fKdaW   kda;
    Glm53fMlaW   mla;
    Glm53fMoeW   moe;
    Glm53fMat    d_gate, d_up, d_down;  /* dense MLP */
} Glm53fLayerW;

/* Multi-token prediction head (checkpoint layer n_layers, "MTP"/nextn). One MLA + MoE
 * layer without mHC: from the hidden state of position i and the embedding of the token at
 * i+1 it predicts the token at i+2, which speculative decoding uses as a draft. */
typedef struct {
    Glm53fLayerW w;                    /* is_mla = 1, is_dense = 0; hc unused   */
    const float *enorm, *hnorm;        /* RMSNorm of the embedding and of h     */
    const float *head_norm;            /* shared_head.norm before the lm_head   */
    Glm53fMat    eh;                   /* [hidden][2*hidden]                    */
} Glm53fMtpW;

/* ------------------------------------------------------------------ kernels ---- */
/* Routed experts that failed to load. Non-zero means corrupt output; callers must fail. */
extern long glm53f_expert_drops;

/* mHC collapse. h is [M][hidden]. Writes pre-collapsed x[hidden], post[M], comb[M*M]. */
void glm53f_hc_pre(float *x, float *post, float *comb, const float *h,
                   const Glm53fHcW *w, const Glm53fCfg *c, float *scratch);
/* mHC expand in place: h[j] = post[j]*y + sum_i comb[i][j]*h[i]. scratch holds M*hidden. */
void glm53f_hc_post(float *h, const float *y, const float *post, const float *comb,
                    const Glm53fCfg *c, float *scratch);
size_t glm53f_hc_scratch(const Glm53fCfg *c);

/* KDA over T tokens. state = H*D*D recurrent + 3*H*D*(conv_k-1) conv floats, carried. */
size_t glm53f_kda_scratch(const Glm53fCfg *c, int T);
void   glm53f_kda_layer(float *out, const float *x, const Glm53fKdaW *w, const Glm53fCfg *c,
                        int T, float *state, float *scratch);

/* MLA (dense causal, NoPE). kvc is [cap][n_heads*(qk_nope+v_head)]; cached positions are
 * attended over and this call's T positions are appended at [cached, cached+T). */
size_t glm53f_mla_scratch(const Glm53fCfg *c, int T, int cap);
void   glm53f_mla(float *out, const float *x, const Glm53fMlaW *w, const Glm53fCfg *c,
                  int T, float *scratch, float *kvc, int cached, int cap);

/* Router: idx/w written with topk entries, weights from the UNBIASED sigmoid scores. */
void glm53f_router(int *idx, float *w, const float *x, const Glm53fMoeW *m, const Glm53fCfg *c);

size_t glm53f_moe_scratch(const Glm53fCfg *c, int T);
/* glm53f_moe without the shared expert (out receives only the weighted routed sum). */
void   glm53f_moe_routed(float *out, const float *x, const Glm53fMoeW *w, const Glm53fCfg *c,
                         int T, float *scratch);
/* Print expert-prediction statistics gathered when GLM53F_PREDICT_STATS is set. */
void   glm53f_predict_report(void);
void   glm53f_moe(float *out, const float *x, const Glm53fMoeW *w, const Glm53fCfg *c,
                  int T, float *scratch);

size_t glm53f_mlp_scratch(const Glm53fCfg *c);
void   glm53f_mlp(float *out, const float *x, const Glm53fMat *gate, const Glm53fMat *up,
                  const Glm53fMat *down, float limit, float *scratch);

/* One decoder layer over T tokens. h is [T][M][hidden], updated in place.
 * kvc/cached/cap are used by MLA layers only (kvc NULL on KDA layers). */
size_t glm53f_layer_scratch(const Glm53fCfg *c, int T, int cap);
void   glm53f_decoder_layer(float *h, const Glm53fLayerW *w, const Glm53fCfg *c, int T,
                            float *state, float *scratch, float *kvc, int cached, int cap);

/* Recurrent + conv state floats for one KDA layer. */
static inline size_t glm53f_kda_state_floats(const Glm53fCfg *c)
{
    const size_t P = (size_t)c->kda_heads * c->kda_head_dim;
    return P * c->kda_head_dim + 3 * P * (size_t)(c->conv_k - 1);
}

/* KV floats per position for one MLA layer. */
static inline size_t glm53f_kv_floats_per_pos(const Glm53fCfg *c)
{
    return (size_t)c->n_heads * (size_t)(c->qk_nope + c->v_head);
}

#define GLM53F_MAX_PROMPT 32768
#define GLM53F_MAX_GEN     8192

#ifdef __cplusplus
}
#endif

#endif /* GLM53F_H */
