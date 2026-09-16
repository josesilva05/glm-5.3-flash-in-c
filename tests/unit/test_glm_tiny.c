/* test_glm_tiny.c - the full engine against the official transformers implementation, on a
 * tiny random model with the released architecture. No real weights needed.
 *
 * tests/fixtures/glm_tiny is written by tools/make_glm_tiny.py: config.json and
 * model.safetensors in the released checkpoint's names and dtypes (FP8 block-quantised
 * MLA/MLP/expert matrices, BF16 KDA and mHC matrices, F32 vectors), and ref.json with
 * the reference outputs computed by modeling_glm5_next.py from the same stored weights.
 *
 *   GATE 1  prefill logits      last-position logits match within tolerance
 *   GATE 2  greedy decode       incremental decode (carried KDA state + KV cache)
 *                               generates exactly the reference continuation
 *   GATE 3  teacher forcing     one full recompute over prompt + continuation gives the
 *                               reference argmax at EVERY position
 *   GATE 4  refusal             a session longer than the dense-attention limit is refused
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glm53f_model.h"
#include "json.h"

static int fails = 0;
#define CHECK(cond, ...) do { if (cond) { printf("  PASS  "); printf(__VA_ARGS__); printf("\n"); } \
                              else { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static char *slurp(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (!b) { fclose(f); return NULL; }
    b[fread(b, 1, (size_t)n, f)] = 0;
    fclose(f);
    return b;
}

static int ints(jval *a, int *out, int max)
{
    int n = 0;
    for (int i = 0; a && a->t == J_ARR && i < a->len && n < max; i++) out[n++] = (int)a->kids[i]->num;
    return n;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "tests/fixtures/glm_tiny";
    char path[4096];
    snprintf(path, sizeof path, "%s/ref.json", dir);
    char *txt = slurp(path);
    if (!txt) return 1;
    char *arena = NULL;
    jval *ref = json_parse(txt, &arena);

    int prompt[64], gen[64], full[128], tf[128];
    const int np = ints(json_get(ref, "prompt_ids"), prompt, 64);
    const int ng = ints(json_get(ref, "generated_ids"), gen, 64);
    const int nf = ints(json_get(ref, "full_ids"), full, 128);
    const int nt = ints(json_get(ref, "tf_argmax"), tf, 128);
    jval *rl = json_get(ref, "logits_last");
    if (!np || !ng || nf != np + ng || nt != nf || !rl || rl->t != J_ARR) {
        fprintf(stderr, "%s is malformed\n", path);
        return 1;
    }

    printf("GLM-5.3-Flash engine vs transformers glm5_next, tiny random model\n");
    Glm53fModel m;
    if (glm53f_model_open(&m, dir, NULL, 0.01, -1, nf, 0, 0, 0) != 0) { printf("  FAIL  model open\n"); return 1; }

    /* GATE 1 */
    float *lg = (float *)malloc((size_t)m.cfg.vocab * sizeof(float));
    if (glm53f_model_forward(&m, prompt, np, lg, NULL) != 0) { printf("  FAIL  prefill\n"); return 1; }
    double worst = 0.0, maxref = 0.0;
    for (int i = 0; i < m.cfg.vocab && i < rl->len; i++) {
        const double r = rl->kids[i]->num;
        const double d = fabs(r - lg[i]) / (1e-5 + 1e-4 * fabs(r));
        if (d > worst) worst = d;
        if (fabs(r) > maxref) maxref = fabs(r);
    }
    CHECK(rl->len == m.cfg.vocab && worst <= 1.0,
          "GATE 1 prefill logits    worst %.3fx tolerance (atol 1e-5, rtol 1e-4; |ref| max %.2f)",
          worst, maxref);

    /* GATE 2 */
    int ok = 1, next = 0;
    for (int i = 0; i < ng; i++) {
        int b = 0;
        for (int v = 1; v < m.cfg.vocab; v++) if (lg[v] > lg[b]) b = v;
        if (b != gen[i]) { ok = 0; printf("        step %d: engine %d, reference %d\n", i, b, gen[i]); break; }
        next = b;
        if (i + 1 < ng && glm53f_model_forward(&m, &next, 1, lg, NULL) != 0) { ok = 0; break; }
    }
    CHECK(ok, "GATE 2 greedy decode     %d/%d tokens, incremental (KDA state + KV cache carried)", ok ? ng : 0, ng);
    float *lg_plain = (float *)malloc((size_t)m.cfg.vocab * sizeof(float));
    memcpy(lg_plain, lg, (size_t)m.cfg.vocab * sizeof(float));

    /* GATE 3 */
    glm53f_model_reset(&m);
    int arg[128];
    int match = 0;
    if (glm53f_model_forward(&m, full, nf, NULL, arg) == 0)
        for (int i = 0; i < nf; i++) match += arg[i] == tf[i];
    CHECK(match == nf, "GATE 3 teacher forcing   %d/%d positions match the reference argmax", match, nf);
    glm53f_model_close(&m);

    /* GATE 4 */
    Glm53fModel m2;
    const int limit = 16 + 4 - 1;   /* index_topk + index_kpool - 1 in the tiny config */
    const int refused = glm53f_model_open(&m2, dir, NULL, 0.01, -1, limit + 1, 0, 0, 0) != 0;
    if (!refused) glm53f_model_close(&m2);
    CHECK(refused, "GATE 4 refusal           a %d-position session beyond the dense-attention limit %d is refused",
          limit + 1, limit);

    /* GATE 5: the same decode with predictive prefetch and a cache too small for the 24
     * experts, so background reads, waits on in-flight reads, pins and evictions all
     * happen. Prefetch may only change WHEN experts are read, so the tokens must match and
     * the final logits must be bit-identical to GATE 2's. Repeated to shake out races. */
    {
        int all_ok = 1, runs = 0;
        for (int rep = 0; rep < 20 && all_ok; rep++, runs++) {
            Glm53fModel m3;
            if (glm53f_model_open(&m3, dir, NULL, 0.0002, -1, nf, 4, 0, 0) != 0) { all_ok = 0; break; }
            int ok3 = glm53f_model_forward(&m3, prompt, np, lg, NULL) == 0;
            for (int i = 0; ok3 && i < ng; i++) {
                int b = 0;
                for (int v = 1; v < m3.cfg.vocab; v++) if (lg[v] > lg[b]) b = v;
                if (b != gen[i]) { ok3 = 0; break; }
                next = b;
                if (i + 1 < ng && glm53f_model_forward(&m3, &next, 1, lg, NULL) != 0) ok3 = 0;
            }
            if (ok3 && memcmp(lg, lg_plain, (size_t)m3.cfg.vocab * sizeof(float)) != 0) ok3 = 0;
            if (rep == 0)
                printf("        prefetch: %d slots, %d I/O threads, %llu experts read ahead, %llu used\n",
                       m3.cache.nslot, m3.cache.n_io, (unsigned long long)m3.cache.bg_reads,
                       (unsigned long long)m3.cache.bg_used);
            glm53f_model_close(&m3);
            all_ok = ok3;
        }
        CHECK(all_ok, "GATE 5 prefetch          %d/20 runs with a tiny cache: same tokens, bit-identical logits", runs);
    }

#ifdef GLM53F_CUDA
    /* GPU gates, when a CUDA device is present: the same reference checks with the trunk
     * on the GPU (routed experts on the CPU), plus the distance to the CPU path's logits.
     * GLM53F_GPU_LAYERS_PER_DEV=2 (set by the test registration) spreads the 5 layers over
     * the devices so stream transfers between GPUs are exercised too. */
    {
        Glm53fModel mg;
        if (glm53f_model_open(&mg, dir, NULL, 0.01, -1, nf, 4, 0, 1) != 0) { printf("  FAIL  GPU model open\n"); fails++; }
        else if (glm53f_model_use_gpu(&mg, NULL, 0) != 0) {
            printf("  SKIP  GPU gates: no usable CUDA device\n");
            glm53f_model_close(&mg);
        } else {
            int okg = glm53f_model_forward(&mg, prompt, np, lg, NULL) == 0;
            double wg = 0.0;
            for (int i = 0; okg && i < mg.cfg.vocab; i++) {
                const double r = rl->kids[i]->num;
                const double d = fabs(r - lg[i]) / (1e-5 + 1e-4 * fabs(r));
                if (d > wg) wg = d;
            }
            CHECK(okg && wg <= 1.0, "GATE G1 GPU prefill      worst %.3fx tolerance against the reference", wg);
            int gok = okg;
            for (int i = 0; gok && i < ng; i++) {
                int b = 0;
                for (int v = 1; v < mg.cfg.vocab; v++) if (lg[v] > lg[b]) b = v;
                if (b != gen[i]) { gok = 0; printf("        GPU step %d: %d vs reference %d\n", i, b, gen[i]); break; }
                next = b;
                if (i + 1 < ng && glm53f_model_forward(&mg, &next, 1, lg, NULL) != 0) gok = 0;
            }
            double dmax = 0.0;
            for (int i = 0; gok && i < mg.cfg.vocab; i++) {
                const double d = fabs((double)lg[i] - (double)lg_plain[i]);
                if (d > dmax) dmax = d;
            }
            CHECK(gok, "GATE G2 GPU greedy decode %d/%d tokens; final logits within %.2e of the CPU path",
                  gok ? ng : 0, ng, dmax);
            glm53f_model_reset(&mg);
            int argg[128], mg_match = 0;
            if (glm53f_model_forward(&mg, full, nf, NULL, argg) == 0)
                for (int i = 0; i < nf; i++) mg_match += argg[i] == tf[i];
            CHECK(mg_match == nf, "GATE G3 GPU teacher forcing %d/%d positions", mg_match, nf);
            glm53f_model_close(&mg);
        }
    }
#endif

    free(lg_plain);
    free(lg);
    free(txt);
    printf(fails ? "\nTINY ORACLE: %d FAILED\n" : "\nTINY ORACLE: ENGINE MATCHES THE REFERENCE\n", fails);
    return fails ? 1 : 0;
}
