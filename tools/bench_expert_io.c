/* bench_expert_io.c - routed-expert read throughput versus concurrency, on the real checkpoint.
 *
 *   bench_expert_io <model_dir> [experts_per_point]
 *
 * For each concurrency level it reads a fresh set of random (layer, expert) pairs through
 * glm53f_expert_load (O_DIRECT, the engine's own read path), with that many reads in flight
 * at once, and reports GB/s. Every point uses experts no earlier point touched, so the
 * operating system's cache cannot flatter a later point.
 */
#define _POSIX_C_SOURCE 200809L
#include "glm53f_portable_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "glm53f_cfg.h"
#include "glm53f_load.h"

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: bench_expert_io <model_dir> [experts_per_point]\n"); return 2; }
    const int per = argc > 2 ? atoi(argv[2]) : 96;
    char path[4096];
    snprintf(path, sizeof path, "%s/config.json", argv[1]);
    Glm53fCfg c;
    if (!glm53f_cfg_load_file(&c, path)) return 1;
    Glm53fSt st;
    if (glm53f_st_open(&st, argv[1]) != 0) return 1;

    int first_moe = 0;
    while (first_moe < c.n_layers && glm53f_is_dense(&c, first_moe)) first_moe++;
    const int nmoe = c.n_layers - first_moe;
    const int total = nmoe * c.n_experts;
    int *order = (int *)malloc((size_t)total * sizeof(int));
    for (int i = 0; i < total; i++) order[i] = i;
    srand(12345);
    for (int i = total - 1; i > 0; i--) { int j = rand() % (i + 1); int t = order[i]; order[i] = order[j]; order[j] = t; }

    Glm53fExpertRef r0;
    if (glm53f_expert_ref(&st, &c, first_moe, 0, &r0) != 0) return 1;
    const int64_t slot = glm53f_expert_slot_bytes(&r0) + GLM53F_ST_ALIGN;

    static const int QD[] = { 1, 4, 8, 16, 32 };
    int next = 0;
    printf("expert read throughput, %d experts per point, %.2f MB each (O_DIRECT)\n", per,
           (double)r0.w_bytes / 1e6);
    for (int qi = 0; qi < 5; qi++) {
        const int qd = QD[qi];
        unsigned char **buf = (unsigned char **)calloc((size_t)qd, sizeof *buf);
        for (int b = 0; b < qd; b++)
            if (posix_memalign((void **)&buf[b], GLM53F_ST_ALIGN, (size_t)slot) != 0) return 1;
        uint64_t bytes = 0;
        int fails = 0;
        const double t0 = now_s();
        for (int done = 0; done < per; done += qd) {
            const int n = per - done < qd ? per - done : qd;
            int k;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(qd) reduction(+:bytes, fails)
#endif
            for (k = 0; k < n; k++) {
                const int key = order[next + k];
                Glm53fExpertRef r;
                int64_t pad = 0;
                if (glm53f_expert_ref(&st, &c, first_moe + key / c.n_experts, key % c.n_experts, &r) != 0 ||
                    glm53f_expert_load(&st, &r, buf[k], slot, &pad) != 0) { fails++; continue; }
                bytes += (uint64_t)(r.w_bytes + r.s_bytes);
            }
            next += n;
        }
        const double dt = now_s() - t0;
        printf("  %2d in flight: %6.2f GB/s  (%.2f GB in %.2f s, %d failed)\n", qd,
               (double)bytes / 1e9 / dt, (double)bytes / 1e9, dt, fails);
        for (int b = 0; b < qd; b++) glm53f_aligned_free(buf[b]);
        free(buf);
    }
    glm53f_st_close(&st);
    return 0;
}
