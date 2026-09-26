/* glm53f_gpu_kernels.h - CUDA kernels of the GPU backend, callable from C.
 *
 * Every function launches on the CURRENT device (cudaSetDevice) and returns without waiting;
 * a synchronous copy or gk_sync() waits. All pointers are device pointers. Each kernel
 * mirrors a function of src/core/glm53f_ops.c and repeats its arithmetic: same accumulator
 * types, same reduction order, explicit fused multiply-adds. Batched forms take T rows.
 */
#ifndef GLM53F_GPU_KERNELS_H
#define GLM53F_GPU_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* y[T][R] = W[R][C] . x[T][C], one kernel per storage format (glm53f_mm). */
void gk_mv_f32 (float *y, const float *x, const float *W, int T, int R, int C);
void gk_mv_bf16(float *y, const float *x, const uint16_t *W, int T, int R, int C);
void gk_mv_f8  (float *y, const float *x, const uint8_t *W, const float *S, const float *lut,
                int T, int R, int C, int br, int bc, int scols);

/* Widen to fp32 for cuBLAS: n bf16 elements (a bit shift, exact), or FP8 rows [r0, r0+rows) of a
 * [*][C] matrix dequantised as transformers does: E4M3(code) * scale[row/br][col/bc]. */
void gk_widen_bf16(float *out, const uint16_t *W, int64_t n);
void gk_dequant_f8(float *out, const uint8_t *W, const float *S, const float *lut,
                   int r0, int rows, int C, int br, int bc, int scols);

/* glm53f_rmsnorm over `rows` rows of n; w may be NULL; in place allowed. */
void gk_rmsnorm(float *y, const float *x, const float *w, int rows, int n, float eps);

/* glm53f_swiglu_clamp with gate and up in separate arrays of rows x n. */
void gk_swiglu(float *y, const float *gate, const float *up, int rows, int n, float limit);

/* The same for one row of n, launched on `stream` (a cudaStream_t). */
void gk_swiglu_on(float *y, const float *gate, const float *up, int n, float limit, void *stream);

/* y[R] = W[R][C] . x[C] for a GLM53F_WI4 matrix (nibbles W, group steps S, groups of 64),
 * on `stream` (a cudaStream_t); bit-identical to the CPU's matmul_i4. -1 if C is not a
 * multiple of 64 or has more than 64 groups. */
int  gk_mv_i4_on(float *y, const float *x, const unsigned char *W, const float *S, int R, int C,
                 void *stream);

void gk_add(float *y, const float *x, int64_t n);

/* glm53f_shortconv, in place allowed, state [channels][k-1] carried. */
void gk_shortconv(float *y, const float *x, const float *w, float *state, int channels, int k, int T);

/* KDA per (t, head): l2norm q and k, beta = sigmoid(beta), z -> alpha = exp(lb*sigmoid(exp(A_log)*(z+dt_bias))). */
void gk_kda_prep(float *q, float *k, float *z, float *beta, const float *A_log, const float *dt_bias,
                 int T, int H, int D, float lb);

/* KDA delta-rule recurrence, all heads, T steps; S [H][D][D] carried. */
void gk_kda_recur(float *S, float *o, const float *q, const float *k, const float *v,
                  const float *alpha, const float *beta, int T, int H, int D);

/* The same recurrence launched per matrix element and per column; bit-identical, much faster
 * on a GPU. u is scratch of H*D floats. */
void gk_kda_recur_par(float *S, float *o, float *u, const float *q, const float *k, const float *v,
                      const float *alpha, const float *beta, int T, int H, int D);

/* RMSNorm launched per block of 256 and per element. bs: rows*ceil(n/256) doubles, inv: rows
 * floats (scratch). Equal to gk_rmsnorm except for the order of the double sum of squares. */
void gk_rmsnorm_par(float *y, const float *x, const float *w, int rows, int n, float eps,
                    double *bs, float *inv);

/* KDA output: per (t, head) RMSNorm with w, times sigmoid(gate). In place on o. */
void gk_onorm_gate(float *o, const float *gate, const float *w, int T, int H, int D, float eps);

/* mHC: from mix [T][(2+M)M] produce pre (double) [T][M], post [T][M], comb [T][M][M]. */
void gk_hc_mix(double *pre, float *post, float *comb, const float *mix, const float *base,
               const float *scale, int T, int M, int iters, float eps);
/* mHC collapse: x[t][d] = sum_i pre[t][i] * h[t][i][d]. */
void gk_hc_collapse(float *x, const double *pre, const float *h, int T, int M, int E);
/* mHC expand: out[t][j][d] = post[t][j]*y[t][d] + sum_i comb[t][i][j]*h[t][i][d]. */
void gk_hc_post(float *out, const float *h, const float *y, const float *post, const float *comb,
                int T, int M, int E);

/* MLA causal attention over a KV cache [cached+T][H][qn+vh]: acc [T][H][vh]; tmp double [T][H][vh]. */
void gk_mla_attn(float *acc, double *tmp, const float *q, const float *kv, int T, int H, int qn,
                 int vh, int cached, float scale);

/* The same attention in launches per score and per output element; bit-identical. Processes
 * tsub query positions at a time: sc holds tsub*H*(cached+T) doubles, tz tsub*H*2. */
void gk_mla_attn_par(float *acc, double *sc, double *tz, const float *q, const float *kv, int T, int H,
                     int qn, int vh, int cached, float scale, int tsub);

/* Mean of the M streams: xm[t][d] = (float)(sum_i h[t][i][d] / M). */
void gk_mean_streams(float *xm, const float *h, int T, int M, int E);

/* 0 if every launch so far on this device succeeded. Prints the error otherwise. */
int  gk_check(const char *where);
int  gk_sync(void);

#ifdef __cplusplus
}
#endif

#endif /* GLM53F_GPU_KERNELS_H */
