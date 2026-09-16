/* glm53f_ops.c - the numeric core of the GLM-5.3-Flash engine.
 *
 * Every function here mirrors a module of transformers' modeling_glm5_next.py, named at
 * its definition. The reference for correctness is tools/glm_reference.py (real
 * checkpoint, upstream modules) and tools/make_glm_tiny.py (tiny random model, upstream
 * modules), both compared against this engine by the tests.
 *
 * FLOATING POINT. Builds use -ffp-contract=off (/fp:precise on MSVC). Long sums use
 * double accumulators; OpenMP only ever splits over independent output rows or
 * independent heads, so results do not depend on the thread count.
 */
#include "glm53f.h"
#include "glm53f_portable_io.h"   /* clock_gettime on Windows (C11's timespec_get is not in gnu99) */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* --------------------------------------------------------- fatal errors ---- */
/* Kernels return void; a failed allocation cannot be reported without leaving the output
 * buffer holding stale values that the caller would consume as a result. Abort instead. */
static void glm53f_fatal_oom(const char *what, size_t bytes)
{
    fprintf(stderr, "glm53f: FATAL, could not allocate %zu bytes for %s; aborting rather "
                    "than continuing with an uninitialised buffer.\n", bytes, what);
    abort();
}

static void glm53f_fatal_bound(const char *what, long value, long limit)
{
    fprintf(stderr, "glm53f: FATAL, %s is %ld, above the limit of %ld.\n", what, value, limit);
    abort();
}

long glm53f_expert_drops = 0;
int  glm53f_quiet = 0;

static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ------------------------------------------------------------------ E4M3 ---- */
/* float8_e4m3fn: 1 sign, 4 exponent (bias 7), 3 mantissa; no infinities; only
 * exponent 15 with mantissa 7 is NaN; exponent 0 is subnormal (m/8 * 2^-6). */
static float GLM53F_E4M3[256];
static int   glm53f_e4m3_ready = 0;

static void e4m3_init(void)
{
    if (glm53f_e4m3_ready) return;
    for (int b = 0; b < 256; b++) {
        const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
        double v;
        if (e == 15 && m == 7) v = NAN;
        else if (e == 0)       v = ldexp((double)m / 8.0, -6);
        else                   v = ldexp(1.0 + (double)m / 8.0, e - 7);
        GLM53F_E4M3[b] = (float)(s ? -v : v);
    }
    glm53f_e4m3_ready = 1;
}

float glm53f_e4m3f(uint8_t b)
{
    e4m3_init();
    return GLM53F_E4M3[b];
}

/* ---------------------------------------------------------------- matmul ---- */
/* fp32: sixteen double accumulators with explicitly fused products, reduced in a fixed
 * tree, so the result is independent of vectorisation and thread count. */
static void matmul_f32(float *y, const float *x, const float *W, int in, int out)
{
    int o;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (o = 0; o < out; o++) {
        const float *row = W + (size_t)o * in;
        double a[16] = {0};
        int i = 0;
        for (; i + 15 < in; i += 16)
            for (int l = 0; l < 16; l++)
                a[l] = fma((double)row[i + l], (double)x[i + l], a[l]);
        double b0 = (a[0] + a[4]) + (a[8]  + a[12]);
        double b1 = (a[1] + a[5]) + (a[9]  + a[13]);
        double b2 = (a[2] + a[6]) + (a[10] + a[14]);
        double b3 = (a[3] + a[7]) + (a[11] + a[15]);
        double acc = (b0 + b1) + (b2 + b3);
        for (; i < in; i++) acc = fma((double)row[i], (double)x[i], acc);
        y[o] = (float)acc;
    }
}

/* bf16: the checkpoint's own bytes, widened inside the dot product (a pure bit shift). */
static void matmul_bf16(float *y, const float *x, const uint16_t *W, int in, int out)
{
    int o;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (o = 0; o < out; o++) {
        const uint16_t *row = W + (size_t)o * in;
        int i = 0;
        double acc;
#if defined(__AVX2__)
        {
            __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
            __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();
            for (; i + 15 < in; i += 16) {
                const __m128i h0 = _mm_loadl_epi64((const __m128i *)(row + i));
                const __m128i h1 = _mm_loadl_epi64((const __m128i *)(row + i + 4));
                const __m128i h2 = _mm_loadl_epi64((const __m128i *)(row + i + 8));
                const __m128i h3 = _mm_loadl_epi64((const __m128i *)(row + i + 12));
                v0 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h0), 16))),
                    _mm256_cvtps_pd(_mm_loadu_ps(x + i)), v0);
                v1 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h1), 16))),
                    _mm256_cvtps_pd(_mm_loadu_ps(x + i + 4)), v1);
                v2 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h2), 16))),
                    _mm256_cvtps_pd(_mm_loadu_ps(x + i + 8)), v2);
                v3 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h3), 16))),
                    _mm256_cvtps_pd(_mm_loadu_ps(x + i + 12)), v3);
            }
            const __m256d vt = _mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3));
            double a[4];
            _mm256_storeu_pd(a, vt);
            acc = (a[0] + a[1]) + (a[2] + a[3]);
        }
#else
        {
            double a[16] = {0};
            for (; i + 15 < in; i += 16)
                for (int l = 0; l < 16; l++)
                    a[l] = fma((double)glm53f_bf16f(row[i + l]), (double)x[i + l], a[l]);
            double b0 = (a[0] + a[4]) + (a[8]  + a[12]);
            double b1 = (a[1] + a[5]) + (a[9]  + a[13]);
            double b2 = (a[2] + a[6]) + (a[10] + a[14]);
            double b3 = (a[3] + a[7]) + (a[11] + a[15]);
            acc = (b0 + b1) + (b2 + b3);
        }
#endif
        for (; i < in; i++) acc = fma((double)glm53f_bf16f(row[i]), (double)x[i], acc);
        y[o] = (float)acc;
    }
}

/* FP8 E4M3 with block scales, never dequantised to a matrix.
 *
 * transformers dequantises weight[i][j] = E4M3(code) * scale_inv[i/br][j/bc] and then runs
 * a float32 linear. Here each (row, column-block) product sum is taken first and scaled
 * once, which is the same value to within float rounding and reads a quarter of the
 * bytes an fp32 copy would. One expert matrix is 8.4 MB of codes plus 2 KB of scales.
 *
 * Within a block the products accumulate in eight float lanes (AVX2: one __m256), and the
 * lanes are reduced in double before the scale is applied. */
#if defined(__AVX2__)
/* Eight E4M3 codes to their float values without a table gather (which costs ~2x more
 * here): |v| = mant << max(e, 1) times 2^-10, mant = m + 8 when e != 0, sign from bit 7,
 * NaN when the low seven bits are all ones. The integer is below 2^20 and 2^-10 is a
 * power of two, so every value equals GLM53F_E4M3 bit for bit (checked by test_fp8). */
static inline __m256 e4m3_decode8(const unsigned char *p)
{
    const __m256i c   = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p));
    const __m256i m   = _mm256_and_si256(c, _mm256_set1_epi32(7));
    const __m256i e   = _mm256_and_si256(_mm256_srli_epi32(c, 3), _mm256_set1_epi32(15));
    const __m256i nz  = _mm256_cmpgt_epi32(e, _mm256_setzero_si256());
    const __m256i man = _mm256_add_epi32(m, _mm256_and_si256(nz, _mm256_set1_epi32(8)));
    const __m256i sh  = _mm256_max_epi32(e, _mm256_set1_epi32(1));
    __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sllv_epi32(man, sh)), _mm256_set1_ps(1.0f / 1024.0f));
    v = _mm256_or_ps(v, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_srli_epi32(c, 7), 31)));
    const __m256i nan = _mm256_cmpeq_epi32(_mm256_and_si256(c, _mm256_set1_epi32(0x7f)), _mm256_set1_epi32(0x7f));
    return _mm256_blendv_ps(v, _mm256_set1_ps(NAN), _mm256_castsi256_ps(nan));
}
#endif

static void matmul_f8(float *y, const float *x, const Glm53fMat *m)
{
    e4m3_init();
    const int in = m->cols, out = m->rows, br = m->br, bc = m->bc, scols = m->scols;
    const unsigned char *W = (const unsigned char *)m->w;
    const float *S = m->s;
    int o;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (o = 0; o < out; o++) {
        const unsigned char *row = W + (size_t)o * in;
        const float *srow = S + (size_t)(o / br) * scols;
        double acc = 0.0;
        for (int b = 0, j0 = 0; j0 < in; b++, j0 += bc) {
            const int j1 = (j0 + bc < in) ? j0 + bc : in;
            int j = j0;
            double bsum;
#if defined(__AVX2__)
            {
                __m256 v = _mm256_setzero_ps();
                for (; j + 8 <= j1; j += 8)
                    v = _mm256_fmadd_ps(e4m3_decode8(row + j), _mm256_loadu_ps(x + j), v);
                float a[8];
                _mm256_storeu_ps(a, v);
                bsum = (((double)a[0] + a[4]) + ((double)a[2] + a[6]))
                     + (((double)a[1] + a[5]) + ((double)a[3] + a[7]));
            }
#else
            {
                float a[8] = {0};
                for (; j + 8 <= j1; j += 8)
                    for (int l = 0; l < 8; l++)
                        a[l] = fmaf(GLM53F_E4M3[row[j + l]], x[j + l], a[l]);
                bsum = (((double)a[0] + a[4]) + ((double)a[2] + a[6]))
                     + (((double)a[1] + a[5]) + ((double)a[3] + a[7]));
            }
#endif
            for (; j < j1; j++) bsum += (double)GLM53F_E4M3[row[j]] * (double)x[j];
            acc += bsum * (double)srow[b];
        }
        y[o] = (float)acc;
    }
}

/* int4, symmetric, one step per group of GLM53F_I4_GROUP columns (see glm53f.h). The
 * engine quantises routed experts into this format to fit more of them in the cache; it is
 * an approximation of the checkpoint, never the checkpoint itself.
 *
 * Same shape of sum as the FP8 kernel: eight float lanes inside a group, reduced in double,
 * scaled once by the group step, accumulated in double over the row. */
static void matmul_i4(float *y, const float *x, const Glm53fMat *m)
{
    const int in = m->cols, out = m->rows, gs = m->bc, groups = m->scols;
    const unsigned char *W = (const unsigned char *)m->w;
    const float *S = m->s;
    int o;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (o = 0; o < out; o++) {
        const unsigned char *row = W + (size_t)o * (in / 2);
        const float *srow = S + (size_t)o * groups;
        double acc = 0.0;
        for (int g = 0; g < groups; g++) {
            const int j0 = g * gs, j1 = (j0 + gs < in) ? j0 + gs : in;
            int j = j0;
            double bsum;
#if defined(__AVX2__)
            {
                const __m128i dup = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1);
                const __m256i shift = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);
                const __m256i mask = _mm256_set1_epi32(0x0f);
                const __m256i eight = _mm256_set1_epi32(8);
                __m256 v = _mm256_setzero_ps();
                for (; j + 8 <= j1; j += 8) {
                    /* four bytes hold eight levels; lane l takes byte l/2, low nibble first */
                    const __m128i b = _mm_shuffle_epi8(
                        _mm_cvtsi32_si128(*(const int *)(const void *)(row + (j >> 1))), dup);
                    const __m256i lv = _mm256_sub_epi32(
                        _mm256_and_si256(_mm256_srlv_epi32(_mm256_cvtepu8_epi32(b), shift), mask), eight);
                    v = _mm256_fmadd_ps(_mm256_cvtepi32_ps(lv), _mm256_loadu_ps(x + j), v);
                }
                float a[8];
                _mm256_storeu_ps(a, v);
                bsum = (((double)a[0] + a[4]) + ((double)a[2] + a[6]))
                     + (((double)a[1] + a[5]) + ((double)a[3] + a[7]));
            }
#else
            {
                float a[8] = {0};
                for (; j + 8 <= j1; j += 8)
                    for (int l = 0; l < 8; l++) {
                        const unsigned char byte = row[(j + l) >> 1];
                        const int lv = (int)(((j + l) & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
                        a[l] = fmaf((float)lv, x[j + l], a[l]);
                    }
                bsum = (((double)a[0] + a[4]) + ((double)a[2] + a[6]))
                     + (((double)a[1] + a[5]) + ((double)a[3] + a[7]));
            }
#endif
            for (; j < j1; j++) {
                const unsigned char byte = row[j >> 1];
                const int lv = (int)((j & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
                bsum += (double)lv * (double)x[j];
            }
            acc += bsum * (double)srow[g];
        }
        y[o] = (float)acc;
    }
}

/* One group of GLM53F_I4_GROUP columns lies inside a single FP8 scale block whenever the
 * block width is a multiple of the group (128 and 64 in the released checkpoint), and then
 * the block scale cancels out of the level: with amax = blockscale * max|code| and
 * step = amax / 7, level = round(code * 7 / max|code|). Only the stored step carries the
 * block scale. */
void glm53f_i4_from_f8(unsigned char *q, float *steps, const Glm53fMat *src)
{
    e4m3_init();
    const int in = src->cols, out = src->rows, gs = GLM53F_I4_GROUP;
    const int groups = in / gs;
    const unsigned char *W = (const unsigned char *)src->w;
    int o;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (o = 0; o < out; o++) {
        const unsigned char *row = W + (size_t)o * in;
        const float *srow = src->s + (size_t)(o / src->br) * src->scols;
        unsigned char *dst = q + (size_t)o * (in / 2);
        float *st = steps + (size_t)o * groups;
        for (int g = 0; g < groups; g++) {
            const int j0 = g * gs;
            const int aligned = src->bc % gs == 0;          /* one FP8 scale for the group */
            const float bs = srow[j0 / src->bc];
            float vals[GLM53F_I4_GROUP];
            float amax = 0.0f;
#if defined(__AVX2__)
            if (aligned) {
                __m256 mx = _mm256_setzero_ps();
                const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
                for (int j = 0; j < gs; j += 8) {
                    const __m256 v = e4m3_decode8(row + j0 + j);
                    _mm256_storeu_ps(vals + j, v);
                    mx = _mm256_max_ps(mx, _mm256_and_ps(v, abs_mask));
                }
                float t[8];
                _mm256_storeu_ps(t, mx);
                for (int l = 0; l < 8; l++) if (t[l] > amax) amax = t[l];
            } else
#endif
            {
                for (int j = 0; j < gs; j++) {
                    const float w = GLM53F_E4M3[row[j0 + j]] * (aligned ? 1.0f : srow[(j0 + j) / src->bc]);
                    vals[j] = w;
                    const float a = w < 0.0f ? -w : w;
                    if (a > amax) amax = a;
                }
            }
            float step = (aligned ? amax * bs : amax) / 7.0f;
            if (!(step > 1e-30f)) step = 1e-30f;
            st[g] = step;
            const float inv = amax > 0.0f ? 7.0f / amax : 0.0f;
            int lv[GLM53F_I4_GROUP];
#if defined(__AVX2__)
            {
                const __m256 vinv = _mm256_set1_ps(aligned ? inv : (1.0f / step));
                const __m256i lo = _mm256_set1_epi32(-8), hi = _mm256_set1_epi32(7);
                for (int j = 0; j < gs; j += 8) {
                    __m256i l = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(vals + j), vinv));
                    l = _mm256_min_epi32(_mm256_max_epi32(l, lo), hi);
                    _mm256_storeu_si256((__m256i *)(void *)(lv + j), l);
                }
            }
#else
            for (int j = 0; j < gs; j++) {
                int l = (int)lrintf(vals[j] * (aligned ? inv : (1.0f / step)));
                if (l < -8) l = -8; else if (l > 7) l = 7;
                lv[j] = l;
            }
#endif
            for (int j = 0; j < gs; j += 2)
                dst[(j0 + j) >> 1] = (unsigned char)((lv[j] + 8) | ((lv[j + 1] + 8) << 4));
        }
    }
}

/* FP8 for T inputs at once: a column block's codes are decoded once and then used by every
 * input, so the weights cross the memory bus once instead of T times. Per input the lanes,
 * their order and the block scaling are exactly matmul_f8's. */
#define GLM53F_MM_BATCH 64

static void matmul_f8_batch(float *y, const float *x, int T, const Glm53fMat *m)
{
    e4m3_init();
    const int in = m->cols, out = m->rows, br = m->br, bc = m->bc, scols = m->scols;
    const unsigned char *W = (const unsigned char *)m->w;
    const float *S = m->s;
    int o;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (o = 0; o < out; o++) {
        const unsigned char *row = W + (size_t)o * in;
        const float *srow = S + (size_t)(o / br) * scols;
        double acc[GLM53F_MM_BATCH];
        float wb[1024];
        for (int t = 0; t < T; t++) acc[t] = 0.0;
        for (int b = 0, j0 = 0; j0 < in; b++, j0 += bc) {
            const int j1 = (j0 + bc < in) ? j0 + bc : in;
            const int nb = j1 - j0;
            for (int j = 0; j < nb; j++) wb[j] = GLM53F_E4M3[row[j0 + j]];
            for (int t = 0; t < T; t++) {
                const float *xt = x + (size_t)t * in;
                int j = 0;
                double bsum;
#if defined(__AVX2__)
                {
                    __m256 v = _mm256_setzero_ps();
                    for (; j + 8 <= nb; j += 8)
                        v = _mm256_fmadd_ps(_mm256_loadu_ps(wb + j), _mm256_loadu_ps(xt + j0 + j), v);
                    float a[8];
                    _mm256_storeu_ps(a, v);
                    bsum = (((double)a[0] + a[4]) + ((double)a[2] + a[6]))
                         + (((double)a[1] + a[5]) + ((double)a[3] + a[7]));
                }
#else
                {
                    float a[8] = {0};
                    for (; j + 8 <= nb; j += 8)
                        for (int l = 0; l < 8; l++) a[l] = fmaf(wb[j + l], xt[j0 + j + l], a[l]);
                    bsum = (((double)a[0] + a[4]) + ((double)a[2] + a[6]))
                         + (((double)a[1] + a[5]) + ((double)a[3] + a[7]));
                }
#endif
                for (; j < nb; j++) bsum += (double)wb[j] * (double)xt[j0 + j];
                acc[t] += bsum * (double)srow[b];
            }
        }
        for (int t = 0; t < T; t++) y[(size_t)t * out + o] = (float)acc[t];
    }
}

void glm53f_mm_batch(float *y, const float *x, int T, const Glm53fMat *m)
{
    if (T < 1) return;
    while (T > GLM53F_MM_BATCH) {                       /* the accumulator array is fixed */
        glm53f_mm_batch(y, x, GLM53F_MM_BATCH, m);
        y += (size_t)GLM53F_MM_BATCH * m->rows;
        x += (size_t)GLM53F_MM_BATCH * m->cols;
        T -= GLM53F_MM_BATCH;
    }
    if (m->dt == GLM53F_WF8 && m->cols <= 1024 * 1024 && m->bc <= 1024) {
        matmul_f8_batch(y, x, T, m);
        return;
    }
    for (int t = 0; t < T; t++)                          /* other formats: one at a time */
        glm53f_mm(y + (size_t)t * m->rows, x + (size_t)t * m->cols, m);
}

void glm53f_mm(float *y, const float *x, const Glm53fMat *m)
{
    if (!m->w) {
        fprintf(stderr, "glm53f: FATAL, multiply through an unbound matrix\n");
        abort();
    }
    switch (m->dt) {
    case GLM53F_WBF16: matmul_bf16(y, x, (const uint16_t *)m->w, m->cols, m->rows); break;
    case GLM53F_WF8:   matmul_f8(y, x, m); break;
    case GLM53F_WI4:   matmul_i4(y, x, m); break;
    default:           matmul_f32(y, x, (const float *)m->w, m->cols, m->rows); break;
    }
}

void glm53f_embed_row(float *dst, const Glm53fMat *t, int64_t row)
{
    if (t->dt == GLM53F_WBF16) {
        const uint16_t *p = (const uint16_t *)t->w + row * t->cols;
        for (int i = 0; i < t->cols; i++) dst[i] = glm53f_bf16f(p[i]);
    } else {
        memcpy(dst, (const float *)t->w + row * t->cols, (size_t)t->cols * sizeof(float));
    }
}

/* ------------------------------------------------------------- norms ---- */
/* Glm5NextTextRMSNorm: weight * x * rsqrt(mean(x^2) + eps). w == NULL is the unweighted
 * variant (Glm5NextTextUnweightedRMSNorm). */
void glm53f_rmsnorm(float *y, const float *x, const float *w, int n, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    if (w) for (int i = 0; i < n; i++) y[i] = w[i] * (x[i] * inv);
    else   for (int i = 0; i < n; i++) y[i] = x[i] * inv;
}

/* ------------------------------------------------------------ SwiGLU ---- */
/* Glm5NextTextMLP / Glm5NextTextExperts._apply_gate:
 *   gate = min(gate, limit); up = clamp(up, -limit, limit); silu(gate) * up
 * The gate is clamped from ABOVE only. */
void glm53f_swiglu_clamp(float *y, const float *gu, int n, float limit)
{
    const float *g = gu, *u = gu + n;
    for (int i = 0; i < n; i++) {
        float gi = g[i] > limit ? limit : g[i];
        float ui = u[i] > limit ? limit : (u[i] < -limit ? -limit : u[i]);
        y[i] = (gi * sigmoidf_(gi)) * ui;
    }
}

size_t glm53f_mlp_scratch(const Glm53fCfg *c)
{
    const int I = c->dense_inter > c->moe_inter * c->n_shared ? c->dense_inter
                                                              : c->moe_inter * c->n_shared;
    return (size_t)3 * I;
}

void glm53f_mlp(float *out, const float *x, const Glm53fMat *gate, const Glm53fMat *up,
                const Glm53fMat *down, float limit, float *scratch)
{
    const int I = gate->rows;
    glm53f_mm(scratch, x, gate);
    glm53f_mm(scratch + I, x, up);
    glm53f_swiglu_clamp(scratch + 2 * I, scratch, I, limit);
    glm53f_mm(out, scratch + 2 * I, down);
}

/* ------------------------------------------------------------ ShortConv ---- */
/* F.conv1d(groups=C, padding=k-1) truncated to T, then silu. Taps are ordered
 * oldest..newest: w[k-1] multiplies the current input. state holds the k-1 most recent
 * inputs per channel, oldest first, and is updated in place. */
void glm53f_shortconv(float *y, const float *x, const float *w, float *state,
                      int channels, int k, int T)
{
    const int hist = k - 1;
    int c;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (channels * T > 65536)
#endif
    for (c = 0; c < channels; c++) {
        float buf[16];
        if (hist > 16) glm53f_fatal_bound("conv kernel history", hist, 16);
        if (hist) {
            if (state) memcpy(buf, state + (size_t)c * hist, (size_t)hist * sizeof(float));
            else       memset(buf, 0, (size_t)hist * sizeof(float));
        }
        for (int t = 0; t < T; t++) {
            const float cur = x[(size_t)t * channels + c];
            float acc = w[(size_t)c * k + hist] * cur;
            for (int j = 0; j < hist; j++) acc += w[(size_t)c * k + j] * buf[j];
            for (int j = 0; j + 1 < hist; j++) buf[j] = buf[j + 1];
            if (hist > 0) buf[hist - 1] = cur;
            y[(size_t)t * channels + c] = acc * sigmoidf_(acc);
        }
        if (state && hist) memcpy(state + (size_t)c * hist, buf, (size_t)hist * sizeof(float));
    }
}

/* ------------------------------------------------------------------ mHC ---- */
/* Glm5NextTextHyperConnection.forward, for one token.
 *
 *   flat   = UnweightedRMSNorm(h.flatten())                  over M*hidden
 *   mix    = fn @ flat                                        [(2+M)*M]
 *   pre    = sigmoid(mix[:M]    * scale[0] + base[:M]) + eps
 *   post   = 2 * sigmoid(mix[M:2M] * scale[1] + base[M:2M])
 *   comb   = softmax_row(mix[2M:].view(M,M) * scale[2] + base[2M:]) + eps
 *            then column-normalise, then (iters-1) x (row-normalise, column-normalise)
 *   x      = sum_i pre[i] * h[i]
 */
size_t glm53f_hc_scratch(const Glm53fCfg *c)
{
    return (size_t)c->hc_mult * c->hidden + (size_t)(2 + c->hc_mult) * c->hc_mult;
}

void glm53f_hc_pre(float *x, float *post, float *comb, const float *h,
                   const Glm53fHcW *w, const Glm53fCfg *c, float *scratch)
{
    const int M = c->hc_mult, E = c->hidden, ME = M * E;
    const float *flat_in = h;
    float *flat = scratch;
    float *mix  = scratch + ME;

    glm53f_rmsnorm(flat, flat_in, NULL, ME, c->rms_eps);
    glm53f_mm(mix, flat, &w->fn);

    double pre[GLM53F_MAX_HC];
    for (int i = 0; i < M; i++) {
        pre[i]  = (double)sigmoidf_(mix[i] * w->scale[0] + w->base[i]) + (double)c->hc_eps;
        post[i] = 2.0f * sigmoidf_(mix[M + i] * w->scale[1] + w->base[M + i]);
    }

    double cm[GLM53F_MAX_HC * GLM53F_MAX_HC];
    for (int i = 0; i < M; i++) {
        double mx = -INFINITY;
        double lg[GLM53F_MAX_HC];
        for (int j = 0; j < M; j++) {
            const int k = 2 * M + i * M + j;
            lg[j] = (double)(mix[k] * w->scale[2] + w->base[k]);
            if (lg[j] > mx) mx = lg[j];
        }
        double z = 0.0;
        for (int j = 0; j < M; j++) z += exp(lg[j] - mx);
        for (int j = 0; j < M; j++) cm[i * M + j] = exp(lg[j] - mx) / z + (double)c->hc_eps;
    }
    const double eps = (double)c->hc_eps;
    #define COLNORM() do { for (int j = 0; j < M; j++) { double s = 0.0; \
        for (int i = 0; i < M; i++) s += cm[i * M + j]; \
        for (int i = 0; i < M; i++) cm[i * M + j] /= (s + eps); } } while (0)
    #define ROWNORM() do { for (int i = 0; i < M; i++) { double s = 0.0; \
        for (int j = 0; j < M; j++) s += cm[i * M + j]; \
        for (int j = 0; j < M; j++) cm[i * M + j] /= (s + eps); } } while (0)
    COLNORM();
    for (int it = 0; it < c->hc_iters - 1; it++) { ROWNORM(); COLNORM(); }
    #undef COLNORM
    #undef ROWNORM
    for (int k = 0; k < M * M; k++) comb[k] = (float)cm[k];

    for (int d = 0; d < E; d++) {
        double s = 0.0;
        for (int i = 0; i < M; i++) s += pre[i] * (double)h[(size_t)i * E + d];
        x[d] = (float)s;
    }
}

/* h'[j] = post[j] * y + sum_i comb[i][j] * h[i]
 * (modeling: post.unsqueeze(-1) * y.unsqueeze(-2) + matmul(comb.transpose(-1,-2), h)) */
void glm53f_hc_post(float *h, const float *y, const float *post, const float *comb,
                    const Glm53fCfg *c, float *scratch)
{
    const int M = c->hc_mult, E = c->hidden;
    for (int j = 0; j < M; j++) {
        float *o = scratch + (size_t)j * E;
        for (int d = 0; d < E; d++) {
            double s = (double)post[j] * (double)y[d];
            for (int i = 0; i < M; i++) s += (double)comb[i * M + j] * (double)h[(size_t)i * E + d];
            o[d] = (float)s;
        }
    }
    memcpy(h, scratch, (size_t)M * E * sizeof(float));
}

/* -------------------------------------------------------- KDA recurrence ---- */
/* The delta-rule recurrence of Glm5NextTextLinearAttention, one head, one step. S is [dk][dv].
 *   S *= alpha (per key channel); u = S^T k; S += k (beta (v - u))^T; o = S^T q */
static void kda_step(float *S, float *o, const float *q, const float *k,
                     const float *v, const float *alpha, float beta, int dk, int dv)
{
    for (int i = 0; i < dk; i++) {
        float *row = S + (size_t)i * dv;
        const float a = alpha[i];
        for (int j = 0; j < dv; j++) row[j] *= a;
    }
    float ubuf[512];
    if (dv > 512) glm53f_fatal_bound("KDA value head dim", dv, 512);
    for (int j = 0; j < dv; j++) ubuf[j] = 0.0f;
    for (int i = 0; i < dk; i++) {
        const float ki = k[i];
        const float *row = S + (size_t)i * dv;
        for (int j = 0; j < dv; j++) ubuf[j] += row[j] * ki;
    }
    for (int j = 0; j < dv; j++) ubuf[j] = (v[j] - ubuf[j]) * beta;
    for (int i = 0; i < dk; i++) {
        const float ki = k[i];
        float *row = S + (size_t)i * dv;
        for (int j = 0; j < dv; j++) row[j] += ki * ubuf[j];
    }
    for (int j = 0; j < dv; j++) o[j] = 0.0f;
    for (int i = 0; i < dk; i++) {
        const float qi = q[i];
        const float *row = S + (size_t)i * dv;
        for (int j = 0; j < dv; j++) o[j] += row[j] * qi;
    }
}

/* l2norm as FLA does it: x / sqrt(sum(x^2) + eps), SUM not mean. */
static void l2norm_(float *v, int n, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)v[i] * (double)v[i];
    const float inv = (float)(1.0 / sqrt(ss + (double)eps));
    for (int i = 0; i < n; i++) v[i] *= inv;
}

size_t glm53f_kda_scratch(const Glm53fCfg *c, int T)
{
    const size_t P = (size_t)c->kda_heads * c->kda_head_dim;
    return 3 * (size_t)T * P            /* q k v                      */
         + 2 * (size_t)T * P            /* forget gate, alpha         */
         + (size_t)T * c->kda_heads     /* beta                       */
         + (size_t)T * P                /* recurrence output          */
         + 2 * P                        /* output gate, work row      */
         + 2 * (size_t)c->kda_head_dim; /* low-rank intermediates     */
}

/* Glm5NextTextLinearAttention.forward.
 *   1. q,k,v = Linear(x); ShortConv with silu on each (one conv1d over [q|k|v])
 *   2. g = gate_lb * sigmoid(exp(A_log[h]) * (f_b(f_a(x)) + dt_bias))   (log decay)
 *   3. beta = sigmoid(b_proj(x))                                         per head
 *   4. l2norm q,k per head; q *= D^-0.5; delta-rule recurrence with alpha = exp(g)
 *   5. out = o_proj( RMSNorm_head(core) * w * sigmoid(g_b(g_a(x))) )
 */
void glm53f_kda_layer(float *out, const float *x, const Glm53fKdaW *w, const Glm53fCfg *c,
                      int T, float *state, float *scratch)
{
    const int E = c->hidden, H = c->kda_heads, D = c->kda_head_dim;
    const int P = H * D, K = c->conv_k, hist = K - 1;

    float *q  = scratch;                  float *k  = q  + (size_t)T * P;
    float *v  = k  + (size_t)T * P;       float *z  = v  + (size_t)T * P;
    float *al = z  + (size_t)T * P;       float *bt = al + (size_t)T * P;
    float *o  = bt + (size_t)T * H;       float *gb = o  + (size_t)T * P;
    float *wr = gb + P;                   float *lr = wr + P;

    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        glm53f_mm(q  + (size_t)t * P, xt, &w->q);
        glm53f_mm(k  + (size_t)t * P, xt, &w->k);
        glm53f_mm(v  + (size_t)t * P, xt, &w->v);
        glm53f_mm(bt + (size_t)t * H, xt, &w->b);
        glm53f_mm(lr, xt, &w->f_a);
        glm53f_mm(z  + (size_t)t * P, lr, &w->f_b);
    }

    float *cs = state + (size_t)H * D * D;
    glm53f_shortconv(q, q, w->q_conv, cs,                          P, K, T);
    glm53f_shortconv(k, k, w->k_conv, cs + (size_t)P * hist,       P, K, T);
    glm53f_shortconv(v, v, w->v_conv, cs + (size_t)2 * P * hist,   P, K, T);

    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++) {
            l2norm_(q + (size_t)t * P + (size_t)h * D, D, 1e-6f);
            l2norm_(k + (size_t)t * P + (size_t)h * D, D, 1e-6f);
            bt[(size_t)t * H + h] = sigmoidf_(bt[(size_t)t * H + h]);
            const float a = expf(w->A_log[h]);
            for (int d = 0; d < D; d++) {
                const size_t i = (size_t)t * P + (size_t)h * D + d;
                const float g = c->gate_lb * sigmoidf_(a * (z[i] + w->dt_bias[(size_t)h * D + d]));
                al[i] = expf(g);
            }
        }
    }

    const float qscale = 1.0f / sqrtf((float)D);
    int h;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (h = 0; h < H; h++) {
        float *wh = wr + (size_t)h * D;
        for (int t = 0; t < T; t++) {
            const size_t off = (size_t)t * P + (size_t)h * D;
            for (int i = 0; i < D; i++) wh[i] = q[off + i] * qscale;
            kda_step(state + (size_t)h * D * D, o + off, wh, k + off, v + off,
                     al + off, bt[(size_t)t * H + h], D, D);
        }
    }

    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        float *ot = o + (size_t)t * P;
        glm53f_mm(lr, xt, &w->g_a);
        glm53f_mm(gb, lr, &w->g_b);
        for (int hh = 0; hh < H; hh++) {
            float *oh = ot + (size_t)hh * D;
            glm53f_rmsnorm(oh, oh, w->o_norm, D, c->rms_eps);
            for (int d = 0; d < D; d++) oh[d] *= sigmoidf_(gb[(size_t)hh * D + d]);
        }
        glm53f_mm(out + (size_t)t * E, ot, &w->o);
    }
}

/* ------------------------------------------------------------------ MLA ---- */
/* Glm5NextTextAttention.forward with NoPE (qk_rope_head_dim = 0) and no output gate.
 *
 *   q      = q_b(RMSNorm(q_a(x)))                   [H][qk_nope]
 *   kv     = kv_b(RMSNorm(kv_a(x)))                 [H][qk_nope + v_head]
 *   attn   = softmax(q . k / sqrt(qk_nope)) causal  (dense: see glm53f_dense_attn_limit)
 *   out    = o_proj(sum attn * v)
 *
 * The EXPANDED per-head keys and values are cached: re-expanding a 512-float latent
 * through kv_b at every position of every step would cost far more than the memory.
 */
/* ---- compressed KV (MLA absorption) ----
 * The cache can hold the 512-float latent per position instead of the expanded keys and
 * values (32,768 floats), 64x less memory, by folding kv_b into the query and the output:
 *
 *   q . (W_k c)            = (W_k^T q) . c           scores against the latent
 *   sum_j p_j (W_v c_j)    = W_v (sum_j p_j c_j)     one expansion per query, not per key
 *
 * Same value in exact arithmetic, a different order of sums in floating point.
 */

/* A row slice [r0, r0+n) of a matrix, as a matrix. FP8 slices must start on a scale-block
 * row, which holds here: kv_b rows are sliced at multiples of qk_nope + v_head and of
 * qk_nope, both multiples of the block height in the released checkpoint. */
static Glm53fMat row_slice(const Glm53fMat *m, int r0, int n)
{
    Glm53fMat s = *m;
    s.rows = n;
    if (m->dt == GLM53F_WBF16)      s.w = (const uint16_t *)m->w + (size_t)r0 * m->cols;
    else if (m->dt == GLM53F_WF8) { s.w = (const unsigned char *)m->w + (size_t)r0 * m->cols;
                                    s.s = m->s + (size_t)(r0 / m->br) * m->scols; }
    else                            s.w = (const float *)m->w + (size_t)r0 * m->cols;
    return s;
}

/* y[c] = sum_i x[i] * M[r0+i][c], the transpose of a row slice applied to x. */
static void matmul_transposed(double *y, const float *x, const Glm53fMat *m, int r0, int n)
{
    const int C = m->cols;
    for (int c = 0; c < C; c++) y[c] = 0.0;
    for (int i = 0; i < n; i++) {
        const double xi = (double)x[i];
        if (xi == 0.0) continue;
        const int r = r0 + i;
        if (m->dt == GLM53F_WBF16) {
            const uint16_t *row = (const uint16_t *)m->w + (size_t)r * C;
            for (int c = 0; c < C; c++) y[c] += xi * (double)glm53f_bf16f(row[c]);
        } else if (m->dt == GLM53F_WF8) {
            const unsigned char *row = (const unsigned char *)m->w + (size_t)r * C;
            const float *srow = m->s + (size_t)(r / m->br) * m->scols;
            for (int c = 0; c < C; c++)
                y[c] += xi * (double)GLM53F_E4M3[row[c]] * (double)srow[c / m->bc];
        } else {
            const float *row = (const float *)m->w + (size_t)r * C;
            for (int c = 0; c < C; c++) y[c] += xi * (double)row[c];
        }
    }
}

size_t glm53f_mla_scratch(const Glm53fCfg *c, int T, int cap)
{
    const size_t H = (size_t)c->n_heads;
    return (size_t)T * H * c->qk_nope      /* q                             */
         + (size_t)T * c->q_lora + (size_t)c->kv_lora  /* q residual per query */
         + (size_t)T * H * c->v_head       /* attention output              */
         + H * (size_t)(cap > T ? cap : T) /* per-head score rows           */
         + glm53f_dsa_scratch(c, cap)      /* indexer                       */
         + (size_t)(c->index_topk + c->index_kpool)    /* selected positions */
         + 4 * (size_t)c->kv_lora * H;     /* absorbed query and latent sum, per head (double) */
}

void glm53f_mla(float *out, const float *x, const Glm53fMlaW *w, const Glm53fCfg *c,
                int T, float *scratch, float *kvc, int cached, int cap, float *istate,
                float *ckv)
{
    const int E = c->hidden, H = c->n_heads, qn = c->qk_nope, vh = c->v_head;
    const int kvd = qn + vh;
    const int last = cached + T - 1;
    if (last >= cap) glm53f_fatal_bound("MLA KV cache position", last, cap - 1);
    const float scale = 1.0f / sqrtf((float)qn);

    float *q   = scratch;
    float *ql  = q  + (size_t)T * H * qn;
    float *ct  = ql + (size_t)T * c->q_lora;
    float *acc = ct + c->kv_lora;
    float *sc  = acc + (size_t)T * H * vh;
    float *ds  = sc + (size_t)H * (size_t)(cap > T ? cap : T);
    int   *sel = (int *)(void *)(ds + glm53f_dsa_scratch(c, cap));

    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        float *qlt = ql + (size_t)t * c->q_lora;
        glm53f_mm(qlt, xt, &w->q_a);
        glm53f_rmsnorm(qlt, qlt, w->q_a_norm, c->q_lora, c->rms_eps);
        glm53f_mm(q + (size_t)t * H * qn, qlt, &w->q_b);
        glm53f_mm(ct, xt, &w->kv_a);
        glm53f_rmsnorm(ct, ct, w->kv_a_norm, c->kv_lora, c->rms_eps);
        if (ckv) memcpy(ckv + (size_t)(cached + t) * c->kv_lora, ct, (size_t)c->kv_lora * sizeof(float));
        else     glm53f_mm(kvc + (size_t)(cached + t) * H * kvd, ct, &w->kv_b);
    }

    /* The indexer sees every position, including the ones this call appends. */
    if (istate) glm53f_dsa_append(istate, x, &w->idx, c, T, cached, cap, ds);

    const int nb = cached + T;
    for (int t = 0; t < T; t++) {
        const int p = cached + t;
        /* Positions this query attends to: all of them (nsel < 0), or the indexer's
         * selection in ascending order, so the dense case keeps the same order of sums. */
        const int nsel = istate ? glm53f_dsa_select(sel, istate, x + (size_t)t * E,
                                                    ql + (size_t)t * c->q_lora, &w->idx, c,
                                                    p, cap, ds)
                                : -1;
        const int nvis = nsel < 0 ? p + 1 : nsel;
        int h;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (h = 0; h < H; h++) {
            const float *qt = q + ((size_t)t * H + h) * qn;
            float *s = sc + (size_t)h * nb;
            double m = -INFINITY;
            if (ckv) {                                   /* absorbed: score against latents */
                const int KL = c->kv_lora;
                /* one pair of double buffers per head: the head loop runs in parallel */
                double *qa = (double *)(void *)(sel + c->index_topk + c->index_kpool) +
                             (size_t)h * 2 * KL;
                double *cb = qa + KL;
                matmul_transposed(qa, qt, &w->kv_b, h * kvd, qn);
                for (int jj = 0; jj < nvis; jj++) {
                    const int j = nsel < 0 ? jj : sel[jj];
                    const float *cj = ckv + (size_t)j * KL;
                    double d = 0.0;
                    for (int i = 0; i < KL; i++) d += qa[i] * (double)cj[i];
                    d *= scale;
                    s[jj] = (float)d;
                    if (d > m) m = d;
                }
                double z = 0.0;
                for (int jj = 0; jj < nvis; jj++) z += exp((double)s[jj] - m);
                for (int i = 0; i < KL; i++) cb[i] = 0.0;
                for (int jj = 0; jj < nvis; jj++) {
                    const int j = nsel < 0 ? jj : sel[jj];
                    const double pr = exp((double)s[jj] - m) / z;
                    const float *cj = ckv + (size_t)j * KL;
                    for (int i = 0; i < KL; i++) cb[i] += pr * (double)cj[i];
                }
                float *o = acc + ((size_t)t * H + h) * vh;
                float cbf[1024];
                if (KL > 1024) glm53f_fatal_bound("MLA latent width", KL, 1024);
                for (int i = 0; i < KL; i++) cbf[i] = (float)cb[i];
                const Glm53fMat wv = row_slice(&w->kv_b, h * kvd + qn, vh);
                glm53f_mm(o, cbf, &wv);
                continue;
            }
            for (int jj = 0; jj < nvis; jj++) {
                const int j = nsel < 0 ? jj : sel[jj];
                const float *kj = kvc + ((size_t)j * H + h) * kvd;
                double d = 0.0;
                for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)kj[i];
                d *= scale;
                s[jj] = (float)d;
                if (d > m) m = d;
            }
            double z = 0.0;
            for (int jj = 0; jj < nvis; jj++) z += exp((double)s[jj] - m);
            float *o = acc + ((size_t)t * H + h) * vh;
            double tmp[1024];
            if (vh > 1024) glm53f_fatal_bound("MLA value head dim", vh, 1024);
            for (int i = 0; i < vh; i++) tmp[i] = 0.0;
            for (int jj = 0; jj < nvis; jj++) {
                const int j = nsel < 0 ? jj : sel[jj];
                const double pr = exp((double)s[jj] - m) / z;
                const float *vj = kvc + ((size_t)j * H + h) * kvd + qn;
                for (int i = 0; i < vh; i++) tmp[i] += pr * (double)vj[i];
            }
            for (int i = 0; i < vh; i++) o[i] = (float)tmp[i];
        }
        glm53f_mm(out + (size_t)t * E, acc + (size_t)t * H * vh, &w->o);
    }
}

/* ---------------------------------------------------------------- router ---- */
/* Glm5NextTextTopkRouter.forward with n_group = topk_group = 1 (group masking is then a
 * no-op, which glm53f_cfg.h checks):
 *   scores = sigmoid(W x)             float32
 *   pick   = topk(scores + bias)      the bias steers SELECTION only
 *   w      = scores[pick] / (sum + 1e-20) * routed_scaling_factor */
void glm53f_router(int *idx, float *w, const float *x, const Glm53fMoeW *m, const Glm53fCfg *c)
{
    const int NE = c->n_experts, E = c->hidden, K = c->topk;
    float score[1024], choice[1024];
    if (NE > 1024) glm53f_fatal_bound("routed experts", NE, 1024);
    int e;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (e = 0; e < NE; e++) {
        const float *row = m->gate + (size_t)e * E;
        double acc = 0.0;
        for (int i = 0; i < E; i++) acc += (double)row[i] * (double)x[i];
        score[e]  = sigmoidf_((float)acc);
        choice[e] = score[e] + m->bias[e];
    }
    for (int j = 0; j < K; j++) {
        int best = -1; float bv = -INFINITY;
        for (int ee = 0; ee < NE; ee++)
            if (choice[ee] > bv) { bv = choice[ee]; best = ee; }
        if (best < 0) glm53f_fatal_bound("router selection (NaN scores)", j, K);
        idx[j] = best;
        w[j] = score[best];
        choice[best] = -INFINITY;
    }
    if (c->norm_topk) {
        double s = 0.0;
        for (int j = 0; j < K; j++) s += (double)w[j];
        for (int j = 0; j < K; j++) w[j] = (float)((double)w[j] / (s + 1e-20));
    }
    for (int j = 0; j < K; j++) w[j] *= c->routed_scale;
}

/* ------------------------------------------------------------------ MoE ---- */
/* Glm5NextTextMoE.forward: sum_j w_j * expert_j(x) + shared_experts(x).
 *
 * Expert-major over a chunk of tokens: route every token, then fetch each distinct
 * expert ONCE and apply it to every (token, slot) that chose it. Each contribution is
 * kept per (token, slot) and summed in top-k order, so the result does not depend on the
 * order experts arrive from disk. */
/* Tokens routed together. More of them share one pass over an expert's 25 MB in a prefill
 * (at 64 an expert serves ~1.8 of them), but the union of experts per chunk grows with it,
 * and past 64 the cache thrashes: 256 measured twice as slow (docs/PERFORMANCE.md). */
#define GLM53F_MOE_CHUNK 64
#define GLM53F_MOE_BATCH 16

size_t glm53f_moe_scratch(const Glm53fCfg *c, int T)
{
    const size_t n = (size_t)(T < GLM53F_MOE_CHUNK ? T : GLM53F_MOE_CHUNK);
    const size_t E = (size_t)c->hidden, K = (size_t)c->topk;
    return n * K * E                          /* contributions          */
         + n * K * 2                          /* idx (as float slots) + weights */
         + (size_t)3 * c->moe_inter + E       /* one expert's activations */
         + glm53f_mlp_scratch(c) + E          /* shared expert          */
         + n * (2 * E + 3 * (size_t)c->moe_inter);  /* one expert's tokens, batched */
}

/* ---- expert-prediction statistics (GLM53F_PREDICT_STATS=1) ----
 * Measurement only: records how well a layer's routing can be predicted before that
 * layer runs, which decides whether prefetching experts ahead of compute can work. Two
 * predictors per decode step, scored against the actual top-k at layer L:
 *   P1  router of layer L applied to the MoE input of layer L-1 (top-k and top-2k)
 *   P2  the experts layer L used for the previous token
 * Nothing here influences routing or output. */
static struct {
    int  on;                                      /* -1 unknown, 0 off, 1 on */
    int  pred[GLM53F_MAX_LAYERS][2 * GLM53F_MAX_TOPK];
    int  have_pred[GLM53F_MAX_LAYERS];
    int  predk[3][GLM53F_MAX_LAYERS][2 * GLM53F_MAX_TOPK];   /* from 2, 3, 4 layers back */
    int  have_predk[3][GLM53F_MAX_LAYERS];
    long pk[3], p2kk[3], needk[3];
    int  prev[GLM53F_MAX_LAYERS][GLM53F_MAX_TOPK];
    int  have_prev[GLM53F_MAX_LAYERS];
    long steps, p1k, p12k, p2, both, need;
} glm53f_ps = { .on = -1 };

static void route_topn(int *idx, int n, const float *x, const float *gate, const float *bias,
                       const Glm53fCfg *c)
{
    const int NE = c->n_experts, E = c->hidden;
    float choice[1024];
    int e;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (e = 0; e < NE; e++) {
        const float *row = gate + (size_t)e * E;
        double acc = 0.0;
        for (int i = 0; i < E; i++) acc += (double)row[i] * (double)x[i];
        choice[e] = sigmoidf_((float)acc) + bias[e];
    }
    for (int j = 0; j < n; j++) {
        int best = 0;
        for (int ee = 1; ee < NE; ee++) if (choice[ee] > choice[best]) best = ee;
        idx[j] = best;
        choice[best] = -INFINITY;
    }
}

static int in_set(int v, const int *s, int n)
{
    for (int i = 0; i < n; i++) if (s[i] == v) return 1;
    return 0;
}

static void predict_stats(const Glm53fMoeW *w, const Glm53fCfg *c, const int *actual, const float *x)
{
    const int L = w->layer, K = c->topk;
    if (glm53f_ps.have_pred[L] && glm53f_ps.have_prev[L]) {
        glm53f_ps.steps++;
        glm53f_ps.need += K;
        for (int j = 0; j < K; j++) {
            const int a = actual[j];
            const int h1 = in_set(a, glm53f_ps.pred[L], K);
            const int h2 = in_set(a, glm53f_ps.prev[L], K);
            glm53f_ps.p1k += h1;
            glm53f_ps.p12k += in_set(a, glm53f_ps.pred[L], 2 * K);
            glm53f_ps.p2 += h2;
            glm53f_ps.both += (h1 || h2);
        }
    }
    for (int d = 0; d < 3; d++) {
        if (!glm53f_ps.have_predk[d][L]) continue;
        glm53f_ps.needk[d] += K;
        for (int j = 0; j < K; j++) {
            glm53f_ps.pk[d] += in_set(actual[j], glm53f_ps.predk[d][L], K);
            glm53f_ps.p2kk[d] += in_set(actual[j], glm53f_ps.predk[d][L], 2 * K);
        }
    }
    for (int d = 0; d < 3; d++) {
        const int A = L + 2 + d;
        if (w->ahead_gate[d] && A < GLM53F_MAX_LAYERS) {
            route_topn(glm53f_ps.predk[d][A], 2 * K, x, w->ahead_gate[d], w->ahead_bias[d], c);
            glm53f_ps.have_predk[d][A] = 1;
        }
    }
    memcpy(glm53f_ps.prev[L], actual, (size_t)K * sizeof(int));
    glm53f_ps.have_prev[L] = 1;
    if (w->next_gate && L + 1 < GLM53F_MAX_LAYERS && 2 * K <= 2 * GLM53F_MAX_TOPK) {
        route_topn(glm53f_ps.pred[L + 1], 2 * K, x, w->next_gate, w->next_bias, c);
        glm53f_ps.have_pred[L + 1] = 1;
    }
}

void glm53f_predict_report(void)
{
    if (glm53f_ps.on != 1 || !glm53f_ps.need) return;
    const double n = (double)glm53f_ps.need;
    printf("expert prediction over %ld layer-steps (share of actual experts known in advance):\n"
           "  P1 next-layer router, top-k          : %5.1f%%\n"
           "  P1 next-layer router, top-2k         : %5.1f%%\n"
           "  P2 previous token, same layer        : %5.1f%%\n"
           "  P1 top-k or P2                        : %5.1f%%\n",
           glm53f_ps.steps, 100.0 * glm53f_ps.p1k / n, 100.0 * glm53f_ps.p12k / n,
           100.0 * glm53f_ps.p2 / n, 100.0 * glm53f_ps.both / n);
    for (int d = 0; d < 3; d++)
        if (glm53f_ps.needk[d])
            printf("  router of layer L applied %d layers early: top-k %5.1f%%, top-2k %5.1f%%\n", d + 2,
                   100.0 * glm53f_ps.pk[d] / glm53f_ps.needk[d], 100.0 * glm53f_ps.p2kk[d] / glm53f_ps.needk[d]);
}

static void moe_chunk(float *out, const float *x, const Glm53fMoeW *w, const Glm53fCfg *c,
                      int T, float *scratch, int with_shared)
{
    const int E = c->hidden, K = c->topk, I = c->moe_inter, NE = c->n_experts;
    float *contrib = scratch;
    float *wts     = contrib + (size_t)T * K * E;
    int   *ids     = (int *)(void *)(wts + (size_t)T * K);   /* int and float are 4 bytes */
    float *gu      = (float *)(void *)ids + (size_t)T * K;
    float *act     = gu + 2 * I;
    float *eo      = act + I;
    float *shs     = eo + E;
    float *sho     = shs + glm53f_mlp_scratch(c);
    /* one expert's tokens gathered together, so its weights are read once for all of them */
    float *xb      = sho + E;                 /* strides follow T, as glm53f_moe_scratch does */
    float *g1b     = xb + (size_t)T * E;
    float *g2b     = g1b + (size_t)T * I;
    float *actb    = g2b + (size_t)T * I;
    float *outb    = actb + (size_t)T * I;

    int uniq[GLM53F_MOE_CHUNK * GLM53F_MAX_TOPK];
    unsigned char seen[1024];
    if (NE > 1024) glm53f_fatal_bound("routed experts", NE, 1024);
    if (T > GLM53F_MOE_CHUNK) glm53f_fatal_bound("MoE chunk", T, GLM53F_MOE_CHUNK);
    memset(seen, 0, (size_t)NE);
    int nu = 0;
    for (int t = 0; t < T; t++) {
        glm53f_router(ids + (size_t)t * K, wts + (size_t)t * K, x + (size_t)t * E, w, c);
        for (int j = 0; j < K; j++) {
            const int e = ids[(size_t)t * K + j];
            if (!seen[e]) { seen[e] = 1; uniq[nu++] = e; }
        }
    }
    if (glm53f_ps.on < 0) glm53f_ps.on = getenv("GLM53F_PREDICT_STATS") ? 1 : 0;
    if (glm53f_ps.on && T == 1) predict_stats(w, c, ids, x);

    /* With prefetch: hand this layer's experts and the predicted experts of the next layer
     * to the background readers, then take each routed expert as it lands (get() waits for
     * a read already in flight), so expert compute overlaps the reads still underway. In
     * decode the shared expert is computed first, while the readers start.
     * Without prefetch, read this layer's misses in batches and wait. Either way the same
     * experts are multiplied in the same order, so the output is identical.
     *
     * Prefill (T > 1) predicts the next layer per token and orders the union by how many
     * tokens voted for each expert. GLM53F_PREFILL_PRED overrides the per-token count and
     * GLM53F_NO_PREFILL_PREFETCH disables prefetch for prefill only (for A/B measurement). */
    static int prefill_off = -1, prefill_pred = -1;
    if (prefill_off < 0) {
        prefill_off = getenv("GLM53F_NO_PREFILL_PREFETCH") ? 1 : 0;
        const char *pp = getenv("GLM53F_PREFILL_PRED");
        prefill_pred = pp ? atoi(pp) : -1;
    }
    const int prefetch = w->prefetch_n > 0 && w->src->hint && (T == 1 || !prefill_off);
    if (prefetch) {
        w->src->hint(w->src, w->layer, uniq, nu);
        if (w->next_gate) {
            int pred[64 * 2 * GLM53F_MAX_TOPK];
            int per = (T > 1 && prefill_pred >= 0) ? prefill_pred : w->prefetch_n;
            if (per > 2 * GLM53F_MAX_TOPK) per = 2 * GLM53F_MAX_TOPK;
            if (T == 1) {
                route_topn(pred, per, x, w->next_gate, w->next_bias, c);
                w->src->hint(w->src, w->layer + 1, pred, per);
            } else if (per > 0) {
                int votes[1024], order[1024], no = 0, tmp[2 * GLM53F_MAX_TOPK];
                memset(votes, 0, (size_t)NE * sizeof(int));
                for (int t = 0; t < T; t++) {
                    route_topn(tmp, per, x + (size_t)t * E, w->next_gate, w->next_bias, c);
                    for (int j = 0; j < per; j++) {
                        if (!votes[tmp[j]]) order[no++] = tmp[j];
                        votes[tmp[j]]++;
                    }
                }
                /* most-voted first; ties keep first-seen order (insertion sort is stable) */
                for (int i = 1; i < no; i++) {
                    const int v = order[i];
                    int j = i - 1;
                    while (j >= 0 && votes[order[j]] < votes[v]) { order[j + 1] = order[j]; j--; }
                    order[j + 1] = v;
                }
                memcpy(pred, order, (size_t)no * sizeof(int));
                w->src->hint(w->src, w->layer + 1, pred, no);
            }
        }
        if (T == 1 && with_shared) glm53f_mlp(sho, x, &w->sh_gate, &w->sh_up, &w->sh_down, c->swiglu_limit, shs);
    }

    /* GLM53F_ROUTE_TRACE=path: append one line per decode MoE layer (layer, the routed
     * experts, microseconds waiting in get(), microseconds multiplying). Diagnostics only. */
    static FILE *trace = NULL;
    static int trace_on = -1;
    if (trace_on < 0) {
        const char *tp = getenv("GLM53F_ROUTE_TRACE");
        trace = tp ? fopen(tp, "a") : NULL;
        trace_on = trace != NULL;
    }
    double tr_wait = 0.0, tr_comp = 0.0;
    struct timespec tr_a, tr_b;   /* clock_gettime: C11's timespec_get is not in gnu99 */

    for (int b0 = 0; b0 < nu; b0 += GLM53F_MOE_BATCH) {
        const int bn = (nu - b0) < GLM53F_MOE_BATCH ? (nu - b0) : GLM53F_MOE_BATCH;
        if (!prefetch && w->src->getmany) w->src->getmany(w->src, w->layer, uniq + b0, bn);
        for (int u = b0; u < b0 + bn; u++) {
            const int e = uniq[u];
            Glm53fExpertQ q;
            if (trace_on) clock_gettime(CLOCK_MONOTONIC, &tr_a);
            const int got = w->src->get(w->src, w->layer, e, &q);
            if (trace_on) {
                clock_gettime(CLOCK_MONOTONIC, &tr_b);
                tr_wait += (double)(tr_b.tv_sec - tr_a.tv_sec) * 1e6 + (double)(tr_b.tv_nsec - tr_a.tv_nsec) / 1e3;
            }
            if (got != 0) {
                glm53f_expert_drops++;
                fprintf(stderr, "EXPERT DROP: layer %d expert %d failed to load; "
                                "this output is CORRUPT\n", w->layer, e);
                for (int t = 0; t < T; t++)
                    for (int j = 0; j < K; j++)
                        if (ids[(size_t)t * K + j] == e)
                            memset(contrib + ((size_t)t * K + j) * E, 0, (size_t)E * sizeof(float));
                continue;
            }
            /* every (token, slot) that chose this expert, so one pass over its weights
             * serves all of them; each token's own sums keep their order, bit for bit */
            int tok[GLM53F_MOE_CHUNK], slot[GLM53F_MOE_CHUNK], nt = 0;
            for (int t = 0; t < T; t++)
                for (int j = 0; j < K; j++)
                    if (ids[(size_t)t * K + j] == e) { tok[nt] = t; slot[nt] = j; nt++; }
            if (nt == 1) {                       /* decode: one token, no gathering */
                const float *xt = x + (size_t)tok[0] * E;
                glm53f_mm(gu, xt, &q.gate);
                glm53f_mm(gu + I, xt, &q.up);
                glm53f_swiglu_clamp(act, gu, I, c->swiglu_limit);
                glm53f_mm(contrib + ((size_t)tok[0] * K + slot[0]) * E, act, &q.down);
            } else if (nt > 1) {
                for (int i = 0; i < nt; i++)
                    memcpy(xb + (size_t)i * E, x + (size_t)tok[i] * E, (size_t)E * sizeof(float));
                glm53f_mm_batch(g1b, xb, nt, &q.gate);
                glm53f_mm_batch(g2b, xb, nt, &q.up);
                for (int i = 0; i < nt; i++) {
                    memcpy(gu, g1b + (size_t)i * I, (size_t)I * sizeof(float));
                    memcpy(gu + I, g2b + (size_t)i * I, (size_t)I * sizeof(float));
                    glm53f_swiglu_clamp(actb + (size_t)i * I, gu, I, c->swiglu_limit);
                }
                glm53f_mm_batch(outb, actb, nt, &q.down);
                for (int i = 0; i < nt; i++)
                    memcpy(contrib + ((size_t)tok[i] * K + slot[i]) * E, outb + (size_t)i * E,
                           (size_t)E * sizeof(float));
            }
            if (trace_on) {
                clock_gettime(CLOCK_MONOTONIC, &tr_a);
                tr_comp += (double)(tr_a.tv_sec - tr_b.tv_sec) * 1e6 + (double)(tr_a.tv_nsec - tr_b.tv_nsec) / 1e3;
            }
            if (w->src->release) w->src->release(w->src, w->layer, e);
        }
    }
    if (trace_on && T == 1) {
        fprintf(trace, "%d", w->layer);
        for (int j = 0; j < K; j++) fprintf(trace, " %d", ids[j]);
        fprintf(trace, " %.0f %.0f\n", tr_wait, tr_comp);
    }

    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        float *ot = out + (size_t)t * E;
        for (int d = 0; d < E; d++) {
            double s = 0.0;
            for (int j = 0; j < K; j++)
                s += (double)wts[(size_t)t * K + j] * (double)contrib[((size_t)t * K + j) * E + d];
            ot[d] = (float)s;
        }
        if (!with_shared) continue;
        if (!(prefetch && T == 1)) glm53f_mlp(sho, xt, &w->sh_gate, &w->sh_up, &w->sh_down, c->swiglu_limit, shs);
        for (int d = 0; d < E; d++) ot[d] += sho[d];
    }
}

void glm53f_moe(float *out, const float *x, const Glm53fMoeW *w, const Glm53fCfg *c,
                int T, float *scratch)
{
    for (int t0 = 0; t0 < T; t0 += GLM53F_MOE_CHUNK) {
        const int n = (T - t0) < GLM53F_MOE_CHUNK ? (T - t0) : GLM53F_MOE_CHUNK;
        moe_chunk(out + (size_t)t0 * c->hidden, x + (size_t)t0 * c->hidden, w, c, n, scratch, 1);
    }
}

/* The routed part of glm53f_moe only (no shared expert): for a backend that computes the
 * shared expert elsewhere and adds it. Same arithmetic as glm53f_moe for the routed sum. */
void glm53f_moe_routed(float *out, const float *x, const Glm53fMoeW *w, const Glm53fCfg *c,
                       int T, float *scratch)
{
    for (int t0 = 0; t0 < T; t0 += GLM53F_MOE_CHUNK) {
        const int n = (T - t0) < GLM53F_MOE_CHUNK ? (T - t0) : GLM53F_MOE_CHUNK;
        moe_chunk(out + (size_t)t0 * c->hidden, x + (size_t)t0 * c->hidden, w, c, n, scratch, 0);
    }
}

/* --------------------------------------------------------- decoder layer ---- */
/* Glm5NextTextDecoderLayer.forward, h = [T][M][hidden]:
 *   post, comb, x = attn_hc(h);  y = attn(input_layernorm(x));  h = post*y + comb^T h
 *   post, comb, x = ffn_hc(h);   y = mlp(post_attention_layernorm(x)); h = post*y + comb^T h
 */
size_t glm53f_layer_scratch(const Glm53fCfg *c, int T, int cap)
{
    const size_t E = (size_t)c->hidden, M = (size_t)c->hc_mult;
    size_t sub = glm53f_kda_scratch(c, T);
    size_t a = glm53f_mla_scratch(c, T, cap);      if (a > sub) sub = a;
    a = glm53f_moe_scratch(c, T);                  if (a > sub) sub = a;
    a = glm53f_mlp_scratch(c);                     if (a > sub) sub = a;
    return 3 * (size_t)T * E                       /* collapsed, normed, block out */
         + (size_t)T * M * (1 + M)                 /* post, comb per token         */
         + glm53f_hc_scratch(c) + M * E            /* hc_pre / hc_post work        */
         + sub;
}

static void hc_site_pre(float *xc, float *post, float *comb, float *h, const Glm53fHcW *hw,
                        const Glm53fCfg *c, int T, float *work)
{
    const int E = c->hidden, M = c->hc_mult;
    for (int t = 0; t < T; t++)
        glm53f_hc_pre(xc + (size_t)t * E, post + (size_t)t * M, comb + (size_t)t * M * M,
                      h + (size_t)t * M * E, hw, c, work);
}

static void hc_site_post(float *h, const float *y, const float *post, const float *comb,
                         const Glm53fCfg *c, int T, float *work)
{
    const int E = c->hidden, M = c->hc_mult;
    for (int t = 0; t < T; t++)
        glm53f_hc_post(h + (size_t)t * M * E, y + (size_t)t * E, post + (size_t)t * M,
                       comb + (size_t)t * M * M, c, work);
}

void glm53f_decoder_layer(float *h, const Glm53fLayerW *w, const Glm53fCfg *c, int T,
                          float *state, float *scratch, float *kvc, int cached, int cap,
                          float *istate, float *ckv)
{
    const size_t E = (size_t)c->hidden, M = (size_t)c->hc_mult;
    float *xc   = scratch;
    float *xn   = xc + (size_t)T * E;
    float *y    = xn + (size_t)T * E;
    float *post = y + (size_t)T * E;
    float *comb = post + (size_t)T * M;
    float *work = comb + (size_t)T * M * M;
    float *sub  = work + glm53f_hc_scratch(c) + M * E;

    /* attention site */
    hc_site_pre(xc, post, comb, h, &w->attn_hc, c, T, work);
    for (int t = 0; t < T; t++)
        glm53f_rmsnorm(xn + (size_t)t * E, xc + (size_t)t * E, w->in_norm, c->hidden, c->rms_eps);
    if (w->is_mla) glm53f_mla(y, xn, &w->mla, c, T, sub, kvc, cached, cap, istate, ckv);
    else           glm53f_kda_layer(y, xn, &w->kda, c, T, state, sub);
    hc_site_post(h, y, post, comb, c, T, work);

    /* feed-forward site */
    hc_site_pre(xc, post, comb, h, &w->ffn_hc, c, T, work);
    for (int t = 0; t < T; t++)
        glm53f_rmsnorm(xn + (size_t)t * E, xc + (size_t)t * E, w->post_norm, c->hidden, c->rms_eps);
    if (w->is_dense) {
        for (int t = 0; t < T; t++)
            glm53f_mlp(y + (size_t)t * E, xn + (size_t)t * E, &w->d_gate, &w->d_up, &w->d_down,
                       c->swiglu_limit, sub);
    } else {
        glm53f_moe(y, xn, &w->moe, c, T, sub);
    }
    hc_site_post(h, y, post, comb, c, T, work);
}
