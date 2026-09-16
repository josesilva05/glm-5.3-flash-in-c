/* test_fp8.c - the FP8 matvec kernel against a scalar reference with the same arithmetic.
 *
 * glm53f_mm on an FP8 matrix sums, per (row, column block), eight float lanes of fused
 * products of the decoded weight and x, reduces the lanes in double, adds the block tail
 * in double and scales by the block scale. The reference below does exactly that with the
 * 256-entry table (glm53f_e4m3f), so any vectorised decoding must match it bit for bit.
 * Shapes cover column counts that are not multiples of 8 or of the block size, partial
 * scale blocks, and every one of the 256 codes. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glm53f.h"

static void reference(float *y, const float *x, const Glm53fMat *m)
{
    const unsigned char *W = (const unsigned char *)m->w;
    for (int o = 0; o < m->rows; o++) {
        const unsigned char *row = W + (size_t)o * m->cols;
        const float *srow = m->s + (size_t)(o / m->br) * m->scols;
        double acc = 0.0;
        for (int b = 0, j0 = 0; j0 < m->cols; b++, j0 += m->bc) {
            const int j1 = (j0 + m->bc < m->cols) ? j0 + m->bc : m->cols;
            int j = j0;
            float a[8] = {0};
            for (; j + 8 <= j1; j += 8)
                for (int l = 0; l < 8; l++) a[l] = fmaf(glm53f_e4m3f(row[j + l]), x[j + l], a[l]);
            double bsum = (((double)a[0] + a[4]) + ((double)a[2] + a[6]))
                        + (((double)a[1] + a[5]) + ((double)a[3] + a[7]));
            for (; j < j1; j++) bsum += (double)glm53f_e4m3f(row[j]) * (double)x[j];
            acc += bsum * (double)srow[b];
        }
        y[o] = (float)acc;
    }
}

static uint32_t rng = 12345;
static uint32_t next(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

static int check(int rows, int cols, int br, int bc, int with_nan)
{
    const int srows = (rows + br - 1) / br, scols = (cols + bc - 1) / bc;
    unsigned char *w = (unsigned char *)malloc((size_t)rows * cols);
    float *s = (float *)malloc((size_t)srows * scols * sizeof(float));
    float *x = (float *)malloc((size_t)cols * sizeof(float));
    float *y = (float *)malloc((size_t)rows * sizeof(float)), *r = (float *)malloc((size_t)rows * sizeof(float));
    for (size_t i = 0; i < (size_t)rows * cols; i++) {
        unsigned char v = (unsigned char)(i < 256 ? i : next());
        if (!with_nan && (v & 0x7f) == 0x7f) v ^= 1;
        w[i] = v;
    }
    for (int i = 0; i < srows * scols; i++) s[i] = (float)(next() % 1000 + 1) * 1e-6f;
    for (int i = 0; i < cols; i++) x[i] = (float)((int)(next() % 20001) - 10000) * 1e-4f;
    Glm53fMat m;
    memset(&m, 0, sizeof m);
    m.w = w; m.s = s; m.dt = GLM53F_WF8; m.rows = rows; m.cols = cols; m.br = br; m.bc = bc; m.scols = scols;
    glm53f_mm(y, x, &m);
    reference(r, x, &m);
    int bad = 0;
    for (int o = 0; o < rows; o++) {
        const int same = memcmp(&y[o], &r[o], sizeof(float)) == 0 || (isnan(y[o]) && isnan(r[o]));
        if (!same) bad++;
    }
    printf("  %s  FP8 matvec %4d x %-4d block %3dx%-3d%s: %d/%d rows bit-identical to the reference\n",
           bad ? "FAIL" : "PASS", rows, cols, br, bc, with_nan ? " (NaN codes)" : "", rows - bad, rows);
    free(w); free(s); free(x); free(y); free(r);
    return bad == 0;
}

/* The int4 kernel against the same shape of sum, and the quantiser against its definition. */
static int check_i4(int rows, int cols)
{
    const int gs = GLM53F_I4_GROUP, groups = cols / gs;
    unsigned char *q = (unsigned char *)malloc((size_t)rows * (cols / 2));
    float *st = (float *)malloc((size_t)rows * groups * sizeof(float));
    float *x = (float *)malloc((size_t)cols * sizeof(float));
    float *y = (float *)malloc((size_t)rows * sizeof(float)), *r = (float *)malloc((size_t)rows * sizeof(float));
    for (size_t i = 0; i < (size_t)rows * (cols / 2); i++) q[i] = (unsigned char)next();
    for (int i = 0; i < rows * groups; i++) st[i] = (float)(next() % 1000 + 1) * 1e-6f;
    for (int i = 0; i < cols; i++) x[i] = (float)((int)(next() % 20001) - 10000) * 1e-4f;
    Glm53fMat m;
    memset(&m, 0, sizeof m);
    m.w = q; m.s = st; m.dt = GLM53F_WI4; m.rows = rows; m.cols = cols; m.br = 1; m.bc = gs; m.scols = groups;
    glm53f_mm(y, x, &m);
    for (int o = 0; o < rows; o++) {                       /* reference: same lane order */
        const unsigned char *row = q + (size_t)o * (cols / 2);
        double acc = 0.0;
        for (int g = 0; g < groups; g++) {
            float a[8] = {0};
            for (int j = g * gs; j < (g + 1) * gs; j += 8)
                for (int l = 0; l < 8; l++) {
                    const unsigned char byte = row[(j + l) >> 1];
                    const int lv = (int)(((j + l) & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
                    a[l] = fmaf((float)lv, x[j + l], a[l]);
                }
            double bsum = (((double)a[0] + a[4]) + ((double)a[2] + a[6]))
                        + (((double)a[1] + a[5]) + ((double)a[3] + a[7]));
            acc += bsum * (double)st[(size_t)o * groups + g];
        }
        r[o] = (float)acc;
    }
    int bad = 0;
    for (int o = 0; o < rows; o++) if (memcmp(&y[o], &r[o], sizeof(float)) != 0) bad++;
    printf("  %s  int4 matvec %4d x %-4d group %d          : %d/%d rows bit-identical to the reference\n",
           bad ? "FAIL" : "PASS", rows, cols, gs, rows - bad, rows);
    free(q); free(st); free(x); free(y); free(r);
    return bad == 0;
}

/* Quantising an FP8 matrix: every level within range, every value within half a step. */
static int check_quant(int rows, int cols)
{
    const int gs = GLM53F_I4_GROUP, groups = cols / gs, bc = 128, br = 128;
    const int scols = (cols + bc - 1) / bc;
    unsigned char *w = (unsigned char *)malloc((size_t)rows * cols);
    float *s = (float *)malloc((size_t)((rows + br - 1) / br) * scols * sizeof(float));
    unsigned char *q = (unsigned char *)malloc((size_t)rows * (cols / 2));
    float *st = (float *)malloc((size_t)rows * groups * sizeof(float));
    for (size_t i = 0; i < (size_t)rows * cols; i++) {
        unsigned char v = (unsigned char)next();
        if ((v & 0x7f) == 0x7f) v ^= 1;
        w[i] = v;
    }
    for (int i = 0; i < ((rows + br - 1) / br) * scols; i++) s[i] = (float)(next() % 500 + 1) * 1e-5f;
    Glm53fMat m;
    memset(&m, 0, sizeof m);
    m.w = w; m.s = s; m.dt = GLM53F_WF8; m.rows = rows; m.cols = cols; m.br = br; m.bc = bc; m.scols = scols;
    glm53f_i4_from_f8(q, st, &m);
    int bad = 0;
    double worst = 0.0;
    for (int o = 0; o < rows; o++)
        for (int g = 0; g < groups; g++) {
            const float step = st[(size_t)o * groups + g];
            for (int j = g * gs; j < (g + 1) * gs; j++) {
                const float ref = glm53f_e4m3f(w[(size_t)o * cols + j]) * s[(size_t)(o / br) * scols + j / bc];
                const unsigned char byte = q[(size_t)o * (cols / 2) + (j >> 1)];
                const int lv = (int)((j & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
                const double err = fabs((double)lv * step - ref);
                if (err > 0.5 * (double)step * 1.0001 + 1e-30) bad++;   /* half a step, plus float rounding */
                if (step > 0 && err / step > worst) worst = err / step;
            }
        }
    printf("  %s  int4 quantiser %4d x %-4d               : every weight within %.3f steps of FP8\n",
           bad ? "FAIL" : "PASS", rows, cols, worst);
    free(w); free(s); free(q); free(st);
    return bad == 0;
}

int main(void)
{
    int ok = 1;
    ok &= check(2048, 4096, 128, 128, 0);   /* expert gate/up shape */
    ok &= check(4096, 2048, 128, 128, 0);   /* expert down shape    */
    ok &= check(37, 1021, 16, 128, 0);      /* ragged columns and blocks */
    ok &= check(9, 13, 4, 4, 0);
    ok &= check(64, 256, 128, 64, 1);       /* every code, NaN included */
    ok &= check_i4(2048, 4096);
    ok &= check_i4(4096, 2048);
    ok &= check_i4(9, 128);
    ok &= check_quant(512, 4096);
    return ok ? 0 : 1;
}
