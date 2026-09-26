/* glm53f_gpu_kernels.cu - see glm53f_gpu_kernels.h.
 *
 * Built with automatic FMA contraction disabled so every fused multiply-add is the explicit
 * fma()/fmaf() the CPU kernel also uses, and every other product and sum rounds separately
 * exactly as the CPU code (compiled with -ffp-contract=off / /fp:precise) does. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "glm53f_gpu_kernels.h"

#define GK_BLOCK 256

static inline dim3 gk_grid(int64_t n)
{
    const int64_t g = (n + GK_BLOCK - 1) / GK_BLOCK;
    return dim3((unsigned int)(g > 0 ? g : 1), 1, 1);
}

#define GK_INDEX(n) \
    const int64_t gi_ = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x; \
    if (gi_ >= (n)) return;

static inline __device__ float gk_sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

static inline __device__ float gk_bf16f(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

/* ------------------------------------------------------------------ matvec ---- */
__global__ static void k_mv_f32(float *y, const float *x, const float *W, int T, int R, int C, int64_t n)
{
    GK_INDEX(n);
    const int64_t t = gi_ / R, o = gi_ % R;
    const float *row = W + o * C, *xt = x + t * C;
    double a[16] = {0};
    int i = 0;
    for (; i + 15 < C; i += 16)
        for (int l = 0; l < 16; l++) a[l] = fma((double)row[i + l], (double)xt[i + l], a[l]);
    double b0 = (a[0] + a[4]) + (a[8]  + a[12]);
    double b1 = (a[1] + a[5]) + (a[9]  + a[13]);
    double b2 = (a[2] + a[6]) + (a[10] + a[14]);
    double b3 = (a[3] + a[7]) + (a[11] + a[15]);
    double acc = (b0 + b1) + (b2 + b3);
    for (; i < C; i++) acc = fma((double)row[i], (double)xt[i], acc);
    y[gi_] = (float)acc;
}

__global__ static void k_mv_bf16(float *y, const float *x, const uint16_t *W, int T, int R, int C, int64_t n)
{
    GK_INDEX(n);
    const int64_t t = gi_ / R, o = gi_ % R;
    const uint16_t *row = W + o * C;
    const float *xt = x + t * C;
    double a[16] = {0};
    int i = 0;
    for (; i + 15 < C; i += 16)
        for (int l = 0; l < 16; l++) a[l] = fma((double)gk_bf16f(row[i + l]), (double)xt[i + l], a[l]);
    double b0 = (a[0] + a[4]) + (a[8]  + a[12]);
    double b1 = (a[1] + a[5]) + (a[9]  + a[13]);
    double b2 = (a[2] + a[6]) + (a[10] + a[14]);
    double b3 = (a[3] + a[7]) + (a[11] + a[15]);
    double acc = (b0 + b1) + (b2 + b3);
    for (; i < C; i++) acc = fma((double)gk_bf16f(row[i]), (double)xt[i], acc);
    y[gi_] = (float)acc;
}

__global__ static void k_mv_f8(float *y, const float *x, const uint8_t *W, const float *S,
                               const float *lut, int T, int R, int C, int br, int bc, int scols,
                               int64_t n)
{
    GK_INDEX(n);
    const int64_t t = gi_ / R, o = gi_ % R;
    const uint8_t *row = W + o * C;
    const float *xt = x + t * C;
    const float *srow = S + (o / br) * scols;
    double acc = 0.0;
    for (int b = 0, j0 = 0; j0 < C; b++, j0 += bc) {
        const int j1 = (j0 + bc < C) ? j0 + bc : C;
        int j = j0;
        float a[8] = {0};
        for (; j + 8 <= j1; j += 8)
            for (int l = 0; l < 8; l++) a[l] = fmaf(lut[row[j + l]], xt[j + l], a[l]);
        double bsum = (((double)a[0] + a[4]) + ((double)a[2] + a[6]))
                    + (((double)a[1] + a[5]) + ((double)a[3] + a[7]));
        for (; j < j1; j++) bsum += (double)lut[row[j]] * (double)xt[j];
        acc += bsum * (double)srow[b];
    }
    y[gi_] = (float)acc;
}

extern "C" void gk_mv_f32(float *y, const float *x, const float *W, int T, int R, int C)
{
    const int64_t n = (int64_t)T * R;
    k_mv_f32<<<gk_grid(n), GK_BLOCK>>>(y, x, W, T, R, C, n);
}

extern "C" void gk_mv_bf16(float *y, const float *x, const uint16_t *W, int T, int R, int C)
{
    const int64_t n = (int64_t)T * R;
    k_mv_bf16<<<gk_grid(n), GK_BLOCK>>>(y, x, W, T, R, C, n);
}

extern "C" void gk_mv_f8(float *y, const float *x, const uint8_t *W, const float *S, const float *lut,
                         int T, int R, int C, int br, int bc, int scols)
{
    const int64_t n = (int64_t)T * R;
    k_mv_f8<<<gk_grid(n), GK_BLOCK>>>(y, x, W, S, lut, T, R, C, br, bc, scols, n);
}

__global__ static void k_widen_bf16(float *out, const uint16_t *W, int64_t n)
{
    GK_INDEX(n);
    out[gi_] = gk_bf16f(W[gi_]);
}

extern "C" void gk_widen_bf16(float *out, const uint16_t *W, int64_t n)
{
    k_widen_bf16<<<gk_grid(n), GK_BLOCK>>>(out, W, n);
}

__global__ static void k_dequant_f8(float *out, const uint8_t *W, const float *S, const float *lut,
                                    int r0, int C, int br, int bc, int scols, int64_t n)
{
    GK_INDEX(n);
    const int64_t r = r0 + gi_ / C, j = gi_ % C;
    out[gi_] = lut[W[r * C + j]] * S[(r / br) * scols + j / bc];
}

extern "C" void gk_dequant_f8(float *out, const uint8_t *W, const float *S, const float *lut,
                              int r0, int rows, int C, int br, int bc, int scols)
{
    const int64_t n = (int64_t)rows * C;
    k_dequant_f8<<<gk_grid(n), GK_BLOCK>>>(out, W, S, lut, r0, C, br, bc, scols, n);
}

/* --------------------------------------------------------------- elementwise ---- */
__global__ static void k_rmsnorm(float *y, const float *x, const float *w, int n, float eps, int64_t rows)
{
    GK_INDEX(rows);
    const float *xr = x + gi_ * n;
    float *yr = y + gi_ * n;
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)xr[i] * (double)xr[i];
    const float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    if (w) for (int i = 0; i < n; i++) yr[i] = w[i] * (xr[i] * inv);
    else   for (int i = 0; i < n; i++) yr[i] = xr[i] * inv;
}

extern "C" void gk_rmsnorm(float *y, const float *x, const float *w, int rows, int n, float eps)
{
    k_rmsnorm<<<gk_grid(rows), GK_BLOCK>>>(y, x, w, n, eps, (int64_t)rows);
}

__global__ static void k_swiglu(float *y, const float *g, const float *u, float limit, int64_t n)
{
    GK_INDEX(n);
    float gi = g[gi_] > limit ? limit : g[gi_];
    float ui = u[gi_] > limit ? limit : (u[gi_] < -limit ? -limit : u[gi_]);
    y[gi_] = (gi * gk_sigmoidf(gi)) * ui;
}

extern "C" void gk_swiglu(float *y, const float *gate, const float *up, int rows, int n, float limit)
{
    const int64_t total = (int64_t)rows * n;
    k_swiglu<<<gk_grid(total), GK_BLOCK>>>(y, gate, up, limit, total);
}

extern "C" void gk_swiglu_on(float *y, const float *gate, const float *up, int n, float limit, void *stream)
{
    k_swiglu<<<gk_grid(n), GK_BLOCK, 0, (cudaStream_t)stream>>>(y, gate, up, limit, (int64_t)n);
}

/* ------------------------------------------------------------- int4 matvec ---- */
/* matmul_i4 (glm53f_ops.c) for one input vector. One warp per output row: each lane takes
 * whole groups and forms the group sum as the CPU does (eight float accumulators fed by
 * fmaf in column order, then the same double reduction tree), times the group step in
 * double; lane 0 then adds the group products in group order into the double accumulator.
 * Same operations in the same order, so y is bit-identical to the CPU kernel. */
#define GK_I4_WARPS 8
#define GK_I4_MAXG  64                       /* groups per row held in shared memory */

__global__ static void k_mv_i4(float *y, const float *x, const unsigned char *W, const float *S,
                               int R, int C)
{
    extern __shared__ float xs[];            /* C floats, then GK_I4_WARPS * GK_I4_MAXG doubles */
    double *prod = (double *)(xs + ((C + 1) & ~1));
    for (int i = threadIdx.x; i < C; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * GK_I4_WARPS + warp;
    if (r >= R) return;
    const int groups = C / 64;
    const unsigned char *row = W + (size_t)r * (C / 2);
    const float *srow = S + (size_t)r * groups;
    double *p = prod + warp * GK_I4_MAXG;
    for (int g = lane; g < groups; g += 32) {
        const int j0 = g * 64;
        float a[8] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        for (int j = j0; j < j0 + 64; j += 8) {
            const unsigned int w = *(const unsigned int *)(row + (j >> 1));   /* 8 levels, low nibble first */
#pragma unroll
            for (int l = 0; l < 8; l++)
                a[l] = fmaf((float)((int)((w >> (4 * l)) & 15u) - 8), xs[j + l], a[l]);
        }
        const double bsum = (((double)a[0] + a[4]) + ((double)a[2] + a[6]))
                          + (((double)a[1] + a[5]) + ((double)a[3] + a[7]));
        p[g] = bsum * (double)srow[g];
    }
    __syncwarp();
    if (lane == 0) {
        double acc = 0.0;
        for (int g = 0; g < groups; g++) acc += p[g];
        y[r] = (float)acc;
    }
}

extern "C" int gk_mv_i4_on(float *y, const float *x, const unsigned char *W, const float *S, int R, int C,
                           void *stream)
{
    if (C % 64 || C / 64 > GK_I4_MAXG) return -1;
    const size_t shm = (size_t)((C + 1) & ~1) * sizeof(float) + (size_t)GK_I4_WARPS * GK_I4_MAXG * sizeof(double);
    k_mv_i4<<<(R + GK_I4_WARPS - 1) / GK_I4_WARPS, GK_I4_WARPS * 32, shm, (cudaStream_t)stream>>>(y, x, W, S, R, C);
    return 0;
}

__global__ static void k_add(float *y, const float *x, int64_t n)
{
    GK_INDEX(n);
    y[gi_] += x[gi_];
}

extern "C" void gk_add(float *y, const float *x, int64_t n)
{
    k_add<<<gk_grid(n), GK_BLOCK>>>(y, x, n);
}

/* ------------------------------------------------------------------ ShortConv ---- */
__global__ static void k_shortconv(float *y, const float *x, const float *w, float *state,
                                   int channels, int k, int T)
{
    GK_INDEX(channels);
    const int c = (int)gi_, hist = k - 1;
    float buf[16];
    for (int j = 0; j < hist; j++) buf[j] = state[(int64_t)c * hist + j];
    for (int t = 0; t < T; t++) {
        const float cur = x[(int64_t)t * channels + c];
        float acc = w[(int64_t)c * k + hist] * cur;
        for (int j = 0; j < hist; j++) acc += w[(int64_t)c * k + j] * buf[j];
        for (int j = 0; j + 1 < hist; j++) buf[j] = buf[j + 1];
        if (hist > 0) buf[hist - 1] = cur;
        y[(int64_t)t * channels + c] = acc * gk_sigmoidf(acc);
    }
    for (int j = 0; j < hist; j++) state[(int64_t)c * hist + j] = buf[j];
}

extern "C" void gk_shortconv(float *y, const float *x, const float *w, float *state, int channels, int k, int T)
{
    k_shortconv<<<gk_grid(channels), GK_BLOCK>>>(y, x, w, state, channels, k, T);
}

/* ------------------------------------------------------------------------ KDA ---- */
__global__ static void k_kda_prep(float *q, float *k, float *z, float *beta, const float *A_log,
                                  const float *dt_bias, int T, int H, int D, float lb)
{
    GK_INDEX((int64_t)T * H);
    const int64_t t = gi_ / H, h = gi_ % H, P = (int64_t)H * D;
    const int64_t off = t * P + h * D;
    for (int pass = 0; pass < 2; pass++) {
        float *v = pass ? k + off : q + off;
        double ss = 0.0;
        for (int i = 0; i < D; i++) ss += (double)v[i] * (double)v[i];
        const float inv = (float)(1.0 / sqrt(ss + (double)1e-6f));
        for (int i = 0; i < D; i++) v[i] *= inv;
    }
    beta[t * H + h] = gk_sigmoidf(beta[t * H + h]);
    const float a = expf(A_log[h]);
    for (int d = 0; d < D; d++) {
        const int64_t i = off + d;
        const float g = lb * gk_sigmoidf(a * (z[i] + dt_bias[h * D + d]));
        z[i] = expf(g);
    }
}

extern "C" void gk_kda_prep(float *q, float *k, float *z, float *beta, const float *A_log, const float *dt_bias,
                            int T, int H, int D, float lb)
{
    k_kda_prep<<<gk_grid((int64_t)T * H), GK_BLOCK>>>(q, k, z, beta, A_log, dt_bias, T, H, D, lb);
}

__global__ static void k_kda_recur(float *S, float *o, const float *q, const float *k, const float *v,
                                   const float *alpha, const float *beta, int T, int H, int D)
{
    GK_INDEX(H);
    const int64_t h = gi_, P = (int64_t)H * D;
    const float qscale = 1.0f / sqrtf((float)D);
    float *Sh = S + h * D * D;
    float wh[512], ubuf[512];
    for (int t = 0; t < T; t++) {
        const int64_t off = (int64_t)t * P + h * D;
        for (int i = 0; i < D; i++) wh[i] = q[off + i] * qscale;
        const float *kt = k + off, *vt = v + off, *al = alpha + off;
        const float bt = beta[(int64_t)t * H + h];
        float *ot = o + off;
        for (int i = 0; i < D; i++) {
            float *row = Sh + (int64_t)i * D;
            const float a = al[i];
            for (int j = 0; j < D; j++) row[j] *= a;
        }
        for (int j = 0; j < D; j++) ubuf[j] = 0.0f;
        for (int i = 0; i < D; i++) {
            const float ki = kt[i];
            const float *row = Sh + (int64_t)i * D;
            for (int j = 0; j < D; j++) ubuf[j] += row[j] * ki;
        }
        for (int j = 0; j < D; j++) ubuf[j] = (vt[j] - ubuf[j]) * bt;
        for (int i = 0; i < D; i++) {
            const float ki = kt[i];
            float *row = Sh + (int64_t)i * D;
            for (int j = 0; j < D; j++) row[j] += ki * ubuf[j];
        }
        for (int j = 0; j < D; j++) ot[j] = 0.0f;
        for (int i = 0; i < D; i++) {
            const float qi = wh[i];
            const float *row = Sh + (int64_t)i * D;
            for (int j = 0; j < D; j++) ot[j] += row[j] * qi;
        }
    }
}

extern "C" void gk_kda_recur(float *S, float *o, const float *q, const float *k, const float *v,
                             const float *alpha, const float *beta, int T, int H, int D)
{
    k_kda_recur<<<gk_grid(H), GK_BLOCK>>>(S, o, q, k, v, alpha, beta, T, H, D);
}

/* The same recurrence with one GPU thread per matrix element (decay, write) or per column
 * (read, output) instead of per head: the per-head kernel above runs 64 000 scalar steps in
 * each thread, which a GPU executes slowly. Every sum keeps its order (over i, for each j),
 * so the result is bit-identical. Four launches per token. */
__global__ static void k_kda_decay(float *S, const float *al, int D, int64_t n)
{
    GK_INDEX(n);
    const int64_t h = gi_ / ((int64_t)D * D), i = (gi_ / D) % D;
    S[gi_] *= al[h * D + i];
}

__global__ static void k_kda_read(float *u, const float *S, const float *kt, const float *vt, const float *beta,
                                  int D, int64_t n)
{
    GK_INDEX(n);
    const int64_t h = gi_ / D, j = gi_ % D;
    const float *Sh = S + h * D * D, *kh = kt + h * D;
    float acc = 0.0f;
    for (int i = 0; i < D; i++) acc += Sh[(int64_t)i * D + j] * kh[i];
    u[gi_] = (vt[gi_] - acc) * beta[h];
}

__global__ static void k_kda_write(float *S, const float *kt, const float *u, int D, int64_t n)
{
    GK_INDEX(n);
    const int64_t h = gi_ / ((int64_t)D * D), i = (gi_ / D) % D, j = gi_ % D;
    S[gi_] += kt[h * D + i] * u[h * D + j];
}

__global__ static void k_kda_out(float *o, const float *S, const float *qt, int D, float qscale, int64_t n)
{
    GK_INDEX(n);
    const int64_t h = gi_ / D, j = gi_ % D;
    const float *Sh = S + h * D * D, *qh = qt + h * D;
    float acc = 0.0f;
    for (int i = 0; i < D; i++) acc += Sh[(int64_t)i * D + j] * (qh[i] * qscale);
    o[gi_] = acc;
}

extern "C" void gk_kda_recur_par(float *S, float *o, float *u, const float *q, const float *k, const float *v,
                                 const float *alpha, const float *beta, int T, int H, int D)
{
    const int64_t P = (int64_t)H * D, nS = P * D;
    const float qscale = 1.0f / sqrtf((float)D);
    for (int t = 0; t < T; t++) {
        const int64_t off = (int64_t)t * P;
        k_kda_decay<<<gk_grid(nS), GK_BLOCK>>>(S, alpha + off, D, nS);
        k_kda_read<<<gk_grid(P), GK_BLOCK>>>(u, S, k + off, v + off, beta + (int64_t)t * H, D, P);
        k_kda_write<<<gk_grid(nS), GK_BLOCK>>>(S, k + off, u, D, nS);
        k_kda_out<<<gk_grid(P), GK_BLOCK>>>(o + off, S, q + off, D, qscale, P);
    }
}

/* RMSNorm without a long loop in any thread: squares are summed in double over blocks of
 * GK_NORM_BLOCK elements (one thread per block), the block sums in double per row, then
 * every element is scaled by its row's factor. The double sum is taken in a different order
 * than glm53f_rmsnorm, so the factor can differ from the CPU's in the last bits. */
#define GK_NORM_BLOCK 256

__global__ static void k_norm_blocks(double *bs, const float *x, int n, int nb, int64_t cnt)
{
    GK_INDEX(cnt);
    const int64_t r = gi_ / nb, b = gi_ % nb;
    const int64_t i0 = r * n + b * GK_NORM_BLOCK;
    const int64_t i1 = b * GK_NORM_BLOCK + GK_NORM_BLOCK < n ? i0 + GK_NORM_BLOCK : r * n + n;
    double ss = 0.0;
    for (int64_t i = i0; i < i1; i++) ss += (double)x[i] * (double)x[i];
    bs[gi_] = ss;
}

__global__ static void k_norm_inv(float *inv, const double *bs, int n, int nb, float eps, int64_t rows)
{
    GK_INDEX(rows);
    double ss = 0.0;
    for (int b = 0; b < nb; b++) ss += bs[gi_ * nb + b];
    inv[gi_] = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
}

__global__ static void k_norm_scale(float *y, const float *x, const float *w, const float *inv, int n, int64_t cnt)
{
    GK_INDEX(cnt);
    const float s = inv[gi_ / n];
    y[gi_] = w ? w[gi_ % n] * (x[gi_] * s) : x[gi_] * s;
}

extern "C" void gk_rmsnorm_par(float *y, const float *x, const float *w, int rows, int n, float eps,
                               double *bs, float *inv)
{
    const int nb = (n + GK_NORM_BLOCK - 1) / GK_NORM_BLOCK;
    const int64_t nblk = (int64_t)rows * nb, cnt = (int64_t)rows * n;
    k_norm_blocks<<<gk_grid(nblk), GK_BLOCK>>>(bs, x, n, nb, nblk);
    k_norm_inv<<<gk_grid(rows), GK_BLOCK>>>(inv, bs, n, nb, eps, (int64_t)rows);
    k_norm_scale<<<gk_grid(cnt), GK_BLOCK>>>(y, x, w, inv, n, cnt);
}

__global__ static void k_onorm_gate(float *o, const float *gate, const float *w, int T, int H, int D, float eps)
{
    GK_INDEX((int64_t)T * H);
    const int64_t t = gi_ / H, h = gi_ % H, P = (int64_t)H * D;
    float *oh = o + t * P + h * D;
    const float *gh = gate + t * P + h * D;
    double ss = 0.0;
    for (int i = 0; i < D; i++) ss += (double)oh[i] * (double)oh[i];
    const float inv = (float)(1.0 / sqrt(ss / (double)D + (double)eps));
    for (int i = 0; i < D; i++) oh[i] = w[i] * (oh[i] * inv);
    for (int i = 0; i < D; i++) oh[i] *= gk_sigmoidf(gh[i]);
}

extern "C" void gk_onorm_gate(float *o, const float *gate, const float *w, int T, int H, int D, float eps)
{
    k_onorm_gate<<<gk_grid((int64_t)T * H), GK_BLOCK>>>(o, gate, w, T, H, D, eps);
}

/* ------------------------------------------------------------------------ mHC ---- */
__global__ static void k_hc_mix(double *pre, float *post, float *comb, const float *mix, const float *base,
                                const float *scale, int T, int M, int iters, float epsf)
{
    GK_INDEX(T);
    const int64_t t = gi_;
    const int NM = (2 + M) * M;
    const float *mx = mix + t * NM;
    for (int i = 0; i < M; i++) {
        pre[t * M + i]  = (double)gk_sigmoidf(mx[i] * scale[0] + base[i]) + (double)epsf;
        post[t * M + i] = 2.0f * gk_sigmoidf(mx[M + i] * scale[1] + base[M + i]);
    }
    double cm[64];
    for (int i = 0; i < M; i++) {
        double top = -INFINITY;
        double lg[8];
        for (int j = 0; j < M; j++) {
            const int k = 2 * M + i * M + j;
            lg[j] = (double)(mx[k] * scale[2] + base[k]);
            if (lg[j] > top) top = lg[j];
        }
        double z = 0.0;
        for (int j = 0; j < M; j++) z += exp(lg[j] - top);
        for (int j = 0; j < M; j++) cm[i * M + j] = exp(lg[j] - top) / z + (double)epsf;
    }
    const double eps = (double)epsf;
    for (int j = 0; j < M; j++) {
        double s = 0.0;
        for (int i = 0; i < M; i++) s += cm[i * M + j];
        for (int i = 0; i < M; i++) cm[i * M + j] /= (s + eps);
    }
    for (int it = 0; it < iters - 1; it++) {
        for (int i = 0; i < M; i++) {
            double s = 0.0;
            for (int j = 0; j < M; j++) s += cm[i * M + j];
            for (int j = 0; j < M; j++) cm[i * M + j] /= (s + eps);
        }
        for (int j = 0; j < M; j++) {
            double s = 0.0;
            for (int i = 0; i < M; i++) s += cm[i * M + j];
            for (int i = 0; i < M; i++) cm[i * M + j] /= (s + eps);
        }
    }
    for (int k = 0; k < M * M; k++) comb[t * M * M + k] = (float)cm[k];
}

extern "C" void gk_hc_mix(double *pre, float *post, float *comb, const float *mix, const float *base,
                          const float *scale, int T, int M, int iters, float eps)
{
    k_hc_mix<<<gk_grid(T), GK_BLOCK>>>(pre, post, comb, mix, base, scale, T, M, iters, eps);
}

__global__ static void k_hc_collapse(float *x, const double *pre, const float *h, int M, int E, int64_t n)
{
    GK_INDEX(n);
    const int64_t t = gi_ / E, d = gi_ % E;
    double s = 0.0;
    for (int i = 0; i < M; i++) s += pre[t * M + i] * (double)h[(t * M + i) * E + d];
    x[gi_] = (float)s;
}

extern "C" void gk_hc_collapse(float *x, const double *pre, const float *h, int T, int M, int E)
{
    const int64_t n = (int64_t)T * E;
    k_hc_collapse<<<gk_grid(n), GK_BLOCK>>>(x, pre, h, M, E, n);
}

__global__ static void k_hc_post(float *out, const float *h, const float *y, const float *post,
                                 const float *comb, int M, int E, int64_t n)
{
    GK_INDEX(n);
    const int64_t per = (int64_t)M * E;
    const int64_t t = gi_ / per, j = (gi_ / E) % M, d = gi_ % E;
    double s = (double)post[t * M + j] * (double)y[t * E + d];
    for (int i = 0; i < M; i++) s += (double)comb[(t * M + i) * M + j] * (double)h[(t * M + i) * E + d];
    out[gi_] = (float)s;
}

extern "C" void gk_hc_post(float *out, const float *h, const float *y, const float *post, const float *comb,
                           int T, int M, int E)
{
    const int64_t n = (int64_t)T * M * E;
    k_hc_post<<<gk_grid(n), GK_BLOCK>>>(out, h, y, post, comb, M, E, n);
}

/* ------------------------------------------------------------------------ MLA ---- */
__global__ static void k_mla_attn(float *acc, double *tmp, const float *q, const float *kv, int T, int H,
                                  int qn, int vh, int cached, float scale)
{
    GK_INDEX((int64_t)T * H);
    const int64_t t = gi_ / H, h = gi_ % H, kvd = qn + vh;
    const int64_t p = cached + t;
    const float *qt = q + (t * H + h) * qn;
    /* The CPU kernel keeps the scores in a buffer; here they are recomputed per pass, with
     * the same double sum and the same float rounding, to avoid a [cap] buffer per thread. */
    double top = -INFINITY;
    for (int64_t j = 0; j <= p; j++) {
        const float *kj = kv + (j * H + h) * kvd;
        double d = 0.0;
        for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)kj[i];
        d *= scale;
        if (d > top) top = d;
    }
    double z = 0.0;
    for (int64_t j = 0; j <= p; j++) {
        const float *kj = kv + (j * H + h) * kvd;
        double d = 0.0;
        for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)kj[i];
        d *= scale;
        z += exp((double)(float)d - top);
    }
    double *tt = tmp + (t * H + h) * vh;
    for (int i = 0; i < vh; i++) tt[i] = 0.0;
    for (int64_t j = 0; j <= p; j++) {
        const float *kj = kv + (j * H + h) * kvd;
        double d = 0.0;
        for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)kj[i];
        d *= scale;
        const double pr = exp((double)(float)d - top) / z;
        const float *vj = kj + qn;
        for (int i = 0; i < vh; i++) tt[i] += pr * (double)vj[i];
    }
    float *o = acc + (t * H + h) * vh;
    for (int i = 0; i < vh; i++) o[i] = (float)tt[i];
}

extern "C" void gk_mla_attn(float *acc, double *tmp, const float *q, const float *kv, int T, int H, int qn,
                            int vh, int cached, float scale)
{
    k_mla_attn<<<gk_grid((int64_t)T * H), GK_BLOCK>>>(acc, tmp, q, kv, T, H, qn, vh, cached, scale);
}

/* The same attention in four launches with short loops per thread: scores per (t, head,
 * position), softmax normaliser per (t, head), probabilities per (t, head, position), output
 * per (t, head, value dim). Sums and exponentials are taken in the order of k_mla_attn, so
 * the result is bit-identical. sc holds Tn x H x (cached+T) doubles, tz Tn x H x 2. */
__global__ static void k_mla_scores(double *sc, const float *q, const float *kv, int t0, int H, int qn, int kvd,
                                    int cached, int64_t P, float scale, int64_t n)
{
    GK_INDEX(n);
    const int64_t tt = gi_ / ((int64_t)H * P), h = (gi_ / P) % H, j = gi_ % P;
    const int64_t t = t0 + tt;
    if (j > cached + t) { sc[gi_] = 0.0; return; }
    const float *qt = q + (t * H + h) * qn, *kj = kv + (j * H + h) * kvd;
    double d = 0.0;
    for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)kj[i];
    d *= scale;
    sc[gi_] = d;
}

__global__ static void k_mla_norm(double *tz, const double *sc, int t0, int H, int cached, int64_t P, int64_t n)
{
    GK_INDEX(n);
    const int64_t tt = gi_ / H, p = cached + t0 + tt;
    const double *s = sc + gi_ * P;
    double top = -INFINITY;
    for (int64_t j = 0; j <= p; j++) if (s[j] > top) top = s[j];
    double z = 0.0;
    for (int64_t j = 0; j <= p; j++) z += exp((double)(float)s[j] - top);
    tz[2 * gi_] = top;
    tz[2 * gi_ + 1] = z;
}

__global__ static void k_mla_prob(double *sc, const double *tz, int t0, int H, int cached, int64_t P, int64_t n)
{
    GK_INDEX(n);
    const int64_t th = gi_ / P, j = gi_ % P, p = cached + t0 + th / H;
    if (j > p) return;
    sc[gi_] = exp((double)(float)sc[gi_] - tz[2 * th]) / tz[2 * th + 1];
}

__global__ static void k_mla_out(float *acc, const double *pr, const float *kv, int t0, int H, int qn, int vh,
                                 int cached, int64_t P, int64_t n)
{
    GK_INDEX(n);
    const int64_t tt = gi_ / ((int64_t)H * vh), h = (gi_ / vh) % H, i = gi_ % vh;
    const int64_t t = t0 + tt, p = cached + t, kvd = (int64_t)qn + vh;
    const double *ph = pr + (tt * H + h) * P;
    double s = 0.0;
    for (int64_t j = 0; j <= p; j++) s += ph[j] * (double)kv[(j * H + h) * kvd + qn + i];
    acc[(t * H + h) * vh + i] = (float)s;
}

extern "C" void gk_mla_attn_par(float *acc, double *sc, double *tz, const float *q, const float *kv, int T, int H,
                                int qn, int vh, int cached, float scale, int tsub)
{
    const int64_t P = (int64_t)cached + T;
    if (tsub < 1) tsub = 1;
    for (int t0 = 0; t0 < T; t0 += tsub) {
        const int Tn = T - t0 < tsub ? T - t0 : tsub;
        const int64_t nth = (int64_t)Tn * H, ns = nth * P, no = nth * vh;
        k_mla_scores<<<gk_grid(ns), GK_BLOCK>>>(sc, q, kv, t0, H, qn, qn + vh, cached, P, scale, ns);
        k_mla_norm<<<gk_grid(nth), GK_BLOCK>>>(tz, sc, t0, H, cached, P, nth);
        k_mla_prob<<<gk_grid(ns), GK_BLOCK>>>(sc, tz, t0, H, cached, P, ns);
        k_mla_out<<<gk_grid(no), GK_BLOCK>>>(acc, sc, kv, t0, H, qn, vh, cached, P, no);
    }
}

/* ---------------------------------------------------------------------- head ---- */
__global__ static void k_mean_streams(float *xm, const float *h, int M, int E, int64_t n)
{
    GK_INDEX(n);
    const int64_t t = gi_ / E, d = gi_ % E;
    double s = 0.0;
    for (int i = 0; i < M; i++) s += (double)h[(t * M + i) * E + d];
    xm[gi_] = (float)(s / M);
}

extern "C" void gk_mean_streams(float *xm, const float *h, int T, int M, int E)
{
    const int64_t n = (int64_t)T * E;
    k_mean_streams<<<gk_grid(n), GK_BLOCK>>>(xm, h, M, E, n);
}

extern "C" int gk_check(const char *where)
{
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "glm53f_gpu: CUDA error in %s: %s\n", where, cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

extern "C" int gk_sync(void)
{
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "glm53f_gpu: CUDA error: %s\n", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}
