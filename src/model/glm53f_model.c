/* glm53f_model.c - see glm53f_model.h. */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "glm53f_cfg.h"
#include "glm53f_model.h"
#include "glm53f_mtp.h"
#ifdef GLM53F_CUDA
#include "glm53f_gpu.h"
#endif

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

size_t glm53f_model_state_bytes(const Glm53fModel *m)
{
    return glm53f_kda_state_floats(&m->cfg) * (size_t)m->n_bound * sizeof(float);
}

static int forward_cpu(Glm53fModel *m, const int *ids, int T, float *logits, int *argmax_all);

static int n_mla_layers(const Glm53fCfg *c)
{
    int n = 0;
    for (int L = 0; L < c->n_layers; L++) n += glm53f_is_mla(c, L);
    return n;
}

/* Point every MoE layer at the routers of the MoE layers after it (prefetch, statistics). */
static void wire_routers(Glm53fModel *m)
{
    for (int L = 0; L < m->n_bound; L++) {
        Glm53fMoeW *w = &m->lay[L].w.moe;
        if (m->lay[L].w.is_dense) continue;
        for (int k = 1; k <= 4; k++) {
            const int A = L + k;
            const int ok = A < m->n_bound && !m->lay[A].w.is_dense;
            const float *g = ok ? m->lay[A].w.moe.gate : NULL, *b = ok ? m->lay[A].w.moe.bias : NULL;
            if (k == 1) { w->next_gate = g; w->next_bias = b; }
            else        { w->ahead_gate[k - 2] = g; w->ahead_bias[k - 2] = b; }
        }
    }
}

int glm53f_model_open(Glm53fModel *m, const char *dir, const char *cfg_path, double cache_gb,
                      int max_layers, int cap, int prefetch_n, int expert_i4, int gpu_planned,
                      int kv)
{
    memset(m, 0, sizeof *m);
    char guess[4096];
    if (!cfg_path) { snprintf(guess, sizeof guess, "%s/config.json", dir); cfg_path = guess; }
    if (!glm53f_cfg_load_file(&m->cfg, cfg_path)) return -1;
    Glm53fCfg *c = &m->cfg;

    if (cap < 1) cap = 1;
    if (cap > GLM53F_MAX_POSITIONS) {
        fprintf(stderr, "glm53f: %d positions requested, above this build's limit of %d\n",
                cap, GLM53F_MAX_POSITIONS);
        return -1;
    }
    m->cap = cap;

    const double t0 = now_s();
    if (glm53f_st_open(&m->st, dir) != 0) return -1;
    if (!glm53f_quiet)
    printf("indexed %d tensors from %d shards in %.2f s\n", m->st.nt, m->st.nshard, now_s() - t0);

    m->n_bound = (max_layers > 0 && max_layers < c->n_layers) ? max_layers : c->n_layers;

    /* Resolve and size everything before allocating anything. */
    int64_t total = 0;
    for (int L = 0; L < m->n_bound; L++) {
        const int64_t n = glm53f_bind_layer_bytes(&m->st, c, L);
        if (n < 0) {
            fprintf(stderr, "glm53f: layer %d is incomplete in %s; a partial checkpoint cannot run.\n", L, dir);
            return -1;
        }
        total += n;
    }
    m->trunk_bytes = (size_t)total;

    m->lay = (Glm53fLayerBind *)calloc((size_t)m->n_bound, sizeof *m->lay);
    if (!m->lay) return -1;

    /* Layers are independent, so their reads overlap; each thread owns its layer. */
    const double tb = now_s();
    int failed = 0, L;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(8)
#endif
    for (L = 0; L < m->n_bound; L++) {
        if (failed) continue;
        if (glm53f_bind_layer(&m->st, c, L, &m->lay[L]) != 0) {
#ifdef _OPENMP
#pragma omp critical
#endif
            failed = 1;
        }
    }
    if (failed) { glm53f_model_close(m); return -1; }
    if (glm53f_bind_model(&m->st, c, &m->mb) != 0) { glm53f_model_close(m); return -1; }
    m->load_seconds = now_s() - tb;
    if (!glm53f_quiet)
    printf("trunk: %d layers, %.2f GB resident + %.2f GB embeddings/lm_head, loaded in %.1f s\n",
           m->n_bound, (double)total / 1e9, (double)m->mb.nbytes / 1e9, m->load_seconds);

    int n_io = 0;
    if (prefetch_n > 0) {
        const char *io = getenv("GLM53F_IO_THREADS");
        n_io = io ? atoi(io) : 2;
        if (n_io < 1) n_io = 1;
    }
    /* Keep the cache inside the RAM that is actually free right now (the trunk is already
     * resident at this point). Paging an expert cache is far worse than a smaller one. */
    int64_t budget = (int64_t)(cache_gb * 1e9);
    uint64_t avail = glm53f_avail_ram_bytes();
    /* cache_gb <= 0 asks for the cache to be sized from the free memory. The system keeps a
     * fifth of the installed RAM on top of the usual headroom, for whatever is opened while
     * the session runs: a cache that fits only the moment the model opens is how a machine
     * ends up paging. */
    const int auto_size = cache_gb <= 0.0;
    const uint64_t total_ram = glm53f_total_ram_bytes();
    if (auto_size) budget = avail > 0 ? INT64_MAX : (int64_t)16e9;
    /* --gpu frees the trunk right after the upload, so that RAM counts as free here. */
    if (avail > 0 && gpu_planned) avail += (uint64_t)m->trunk_bytes + (uint64_t)m->mb.nbytes / 2;
    /* What this session will take besides the expert cache: the attention caches, the
     * indexer state and the buffers one chunk of a forward pass needs. */
    const int mla = n_mla_layers(c);
    const int kv_compressed = kv == 2 || (kv == 0 && cap > glm53f_dense_attn_limit(c));
    const int64_t kvb_est = (int64_t)mla * cap *
                            (kv_compressed ? c->kv_lora : (int)glm53f_kv_floats_per_pos(c)) * 4;
    const int64_t idxb_est = (int64_t)mla * (int64_t)glm53f_dsa_state_floats(c, cap) * 4;
    const int span_est = cap < GLM53F_PREFILL_SPAN ? cap : GLM53F_PREFILL_SPAN;
    const int64_t work_est = (int64_t)(glm53f_layer_scratch(c, GLM53F_CPU_CHUNK, cap) +
                                       (size_t)span_est * c->hc_mult * c->hidden +
                                       (size_t)c->vocab) * 4;
    const int64_t state_est = (int64_t)(glm53f_kda_state_floats(c) * (size_t)c->n_layers) * 4;
    int64_t reserve = (int64_t)3e9;
    if (auto_size && total_ram > 0 && (int64_t)(total_ram / 5) > reserve) reserve = (int64_t)(total_ram / 5);
    const int64_t headroom = reserve + kvb_est + idxb_est + work_est + state_est;
    /* never more than every routed expert of the model */
    const int64_t all_experts = (int64_t)c->n_layers * c->n_experts * 26000000;
    if (budget > all_experts) budget = all_experts;
    if (avail > 0 && budget > (int64_t)avail - headroom) {
        const int64_t fit = (int64_t)avail - headroom;
        const int64_t least = (int64_t)(c->topk + 1) * 26000000;
        if (fit < least) {
            fprintf(stderr, "glm53f: only %.1f GB of RAM is free; the expert cache needs at "
                            "least %.1f GB. Close something or use --gpu (which frees the "
                            "trunk's %.1f GB).\n",
                    (double)avail / 1e9, (double)least / 1e9, (double)m->trunk_bytes / 1e9);
            glm53f_model_close(m);
            return -1;
        }
        if (!glm53f_quiet && auto_size)
        printf("expert cache sized from free memory: %.1f GB free, %.1f GB kept for the system "
               "and %.1f GB for this session; %.1f GB of cache (--cache-gb sets it)\n",
               (double)avail / 1e9, (double)reserve / 1e9,
               (double)(kvb_est + idxb_est + work_est + state_est) / 1e9, (double)fit / 1e9);
        else if (!glm53f_quiet)
        printf("NOTE: --cache-gb %.1f does not fit: %.1f GB of RAM is free and this session also "
               "needs%.1f GB\n      (attention caches, indexer, work buffers) plus 3 GB of headroom; "
               "using %.1f GB of cache.\n",
               cache_gb, (double)avail / 1e9,
               (double)(kvb_est + idxb_est + work_est + state_est) / 1e9, (double)fit / 1e9);
        budget = fit;
    }
    if (glm53f_cache_init(&m->cache, &m->st, c, budget, n_io, expert_i4) != 0) {
        glm53f_model_close(m); return -1;
    }
    if (expert_i4 && glm53f_i4_dir && !glm53f_quiet) {
        int nmoe = 0;
        for (L = 0; L < c->n_layers; L++) nmoe += !glm53f_is_dense(c, L);
        printf("int4 container: %d of %d MoE layers read from %s%s\n", m->cache.i4_layers, nmoe,
               glm53f_i4_dir, m->cache.i4_layers < nmoe ? "; the others are quantised on the way in" : "");
    }
    for (L = 0; L < m->n_bound; L++)
        if (!m->lay[L].w.is_dense) {
            m->lay[L].w.moe.src = &m->cache.src;
            m->lay[L].w.moe.prefetch_n = prefetch_n;
            m->lay[L].w.moe.prefill_next = m->cache.nslot >= 2 * c->n_experts;
        }
    wire_routers(m);

    m->state = (float *)calloc(glm53f_kda_state_floats(c) * (size_t)m->n_bound, sizeof(float));
    m->kv = (float **)calloc((size_t)m->n_bound, sizeof(float *));
    m->idx = (float **)calloc((size_t)m->n_bound, sizeof(float *));
    /* The compressed cache keeps the kv_lora latent per position instead of the expanded
     * keys and values (64x less memory), folding kv_b into the query and the output. Past
     * the dense-attention range the expanded form costs more RAM than a machine has, so
     * that is where it becomes the default. */
    const int compressed = kv == 2 || (kv == 0 && cap > glm53f_dense_attn_limit(c));
    if (compressed)
        if (!glm53f_quiet)
        printf("kv cache: kv_lora latent per position (%.1f KB per position, %.0fx less than "
               "expanded)\n", (double)c->kv_lora * 4 * n_mla_layers(c) / 1024.0,
               (double)glm53f_kv_floats_per_pos(c) / (double)c->kv_lora);
    if (compressed) {
        m->ckv = (float **)calloc((size_t)m->n_bound, sizeof(float *));
        if (!m->ckv) { glm53f_model_close(m); return -1; }
    }
    if (!m->state || !m->kv || !m->idx) { glm53f_model_close(m); return -1; }
    size_t kvb = 0, idxb = 0;
    for (L = 0; L < m->n_bound; L++) {
        if (!glm53f_is_mla(c, L)) continue;
        const size_t n = compressed ? (size_t)cap * c->kv_lora
                                    : (size_t)cap * glm53f_kv_floats_per_pos(c);
        const size_t ni = glm53f_dsa_state_floats(c, cap);
        if (compressed) m->ckv[L] = (float *)malloc(n * sizeof(float));
        else            m->kv[L] = (float *)malloc(n * sizeof(float));
        m->idx[L] = (float *)calloc(ni, sizeof(float));
        if ((compressed ? (void *)m->ckv[L] : (void *)m->kv[L]) == NULL || !m->idx[L]) {
            fprintf(stderr, "glm53f: cannot allocate the KV cache (%.2f GB per MLA layer)\n",
                    (double)n * 4 / 1e9);
            glm53f_model_close(m); return -1;
        }
        kvb += n * sizeof(float);
        idxb += ni * sizeof(float);
    }
    if (!glm53f_quiet)
    printf("expert cache: %d slots x %.2f MB = %.2f GB | KV cache %.2f GB + DSA index %.2f GB "
           "for %d positions | KDA state %.2f MB\n\n",
           m->cache.nslot, (double)m->cache.slot_bytes / 1e6,
           (double)m->cache.nslot * m->cache.slot_bytes / 1e9, (double)kvb / 1e9,
           (double)idxb / 1e9, cap, (double)glm53f_model_state_bytes(m) / 1e6);
    return 0;
}

void glm53f_model_close(Glm53fModel *m)
{
#ifdef GLM53F_CUDA
    if (m->gpu) glm53f_gpu_free(m->gpu);
#endif
    glm53f_mtp_close(m);
    free(m->mtp_h);
    if (m->lay) for (int L = 0; L < m->n_bound; L++) glm53f_bind_free(&m->lay[L]);
    free(m->lay);
    if (m->kv) for (int L = 0; L < m->n_bound; L++) free(m->kv[L]);
    free(m->kv);
    if (m->idx) for (int L = 0; L < m->n_bound; L++) free(m->idx[L]);
    free(m->idx);
    if (m->ckv) for (int L = 0; L < m->n_bound; L++) free(m->ckv[L]);
    free(m->ckv);
    free(m->state);
    glm53f_bind_model_free(&m->mb);
    free(m->routers);
    glm53f_cache_free(&m->cache);
    glm53f_st_close(&m->st);
    memset(m, 0, sizeof *m);
}

void glm53f_model_reset(Glm53fModel *m)
{
    memset(m->state, 0, glm53f_model_state_bytes(m));
#ifdef GLM53F_CUDA
    if (m->gpu) glm53f_gpu_reset(m->gpu);
#endif
    m->cached = 0;
}

#ifdef GLM53F_CUDA
/* With the trunk on the devices the CPU only routes and streams experts: keep each MoE
 * layer's router (gate, correction bias) and the embedding table, free everything else.
 * Weight pointers of the freed layers are cleared; the CPU forward is not used again. */
static void release_host_trunk(Glm53fModel *m)
{
    const Glm53fCfg *c = &m->cfg;
    const size_t per = (size_t)c->n_experts * ((size_t)c->hidden + 1);   /* floats per router */
    int nmoe = 0;
    for (int L = 0; L < m->n_bound; L++) nmoe += !m->lay[L].w.is_dense;
    const size_t eb = (size_t)m->mb.embed.rows * (size_t)m->mb.embed.cols *
                      (m->mb.embed.dt == GLM53F_WBF16 ? 2 : 4);
    float *routers = nmoe ? (float *)malloc((size_t)nmoe * per * sizeof(float)) : NULL;
    void *embed = malloc(eb);
    if ((nmoe && !routers) || !embed) { free(routers); free(embed); return; }
    size_t freed = 0;
    for (int L = 0, k = 0; L < m->n_bound; L++) {
        Glm53fLayerW *w = &m->lay[L].w;
        if (!w->is_dense) {
            float *r = routers + (size_t)k++ * per;
            memcpy(r, w->moe.gate, (size_t)c->n_experts * c->hidden * sizeof(float));
            memcpy(r + (size_t)c->n_experts * c->hidden, w->moe.bias, (size_t)c->n_experts * sizeof(float));
            w->moe.gate = r;
            w->moe.bias = r + (size_t)c->n_experts * c->hidden;
        }
    }
    wire_routers(m);
    for (int L = 0; L < m->n_bound; L++) {
        Glm53fLayerW *w = &m->lay[L].w;
        /* keep is_mla/is_dense and the MoE streaming fields; drop every trunk weight pointer */
        Glm53fMoeW moe = w->moe;
        const int is_mla = w->is_mla, is_dense = w->is_dense;
        memset(w, 0, sizeof *w);
        w->is_mla = is_mla; w->is_dense = is_dense;
        if (!is_dense) {
            memset(&moe.sh_gate, 0, sizeof moe.sh_gate);
            memset(&moe.sh_up, 0, sizeof moe.sh_up);
            memset(&moe.sh_down, 0, sizeof moe.sh_down);
            w->moe = moe;
        }
        freed += m->lay[L].nbytes;
        free(m->lay[L].blob);
        m->lay[L].blob = NULL;
        m->lay[L].nbytes = 0;
    }
    memcpy(embed, m->mb.embed.w, eb);
    freed += m->mb.nbytes - eb;
    free(m->mb.blob);
    m->mb.blob = embed;
    m->mb.nbytes = eb;
    m->mb.embed.w = embed;
    memset(&m->mb.lm_head, 0, sizeof m->mb.lm_head);
    m->mb.norm = NULL;
    m->routers = routers;
    if (!glm53f_quiet)
    printf("host: freed %.2f GB of trunk weights now on the GPU (kept %.2f GB: embeddings + routers)\n\n",
           (double)freed / 1e9, (double)(eb + (size_t)nmoe * per * sizeof(float)) / 1e9);
}
#endif

int glm53f_model_use_gpu(Glm53fModel *m, const int *devices, int ndev)
{
#ifdef GLM53F_CUDA
    if (glm53f_gpu_device_count() <= 0) {
        fprintf(stderr, "glm53f: no CUDA device found; running on the CPU\n");
        return -1;
    }
    if (m->ckv) {
        fprintf(stderr, "glm53f: --gpu expands keys and values on the device; GLM53F_KV=compressed "
                        "is a CPU path. Running on the CPU.\n");
        return -1;
    }
    /* The device attention kernels are dense; the DSA indexer runs on the CPU path only. */
    if (m->cap > glm53f_dense_attn_limit(&m->cfg)) {
        fprintf(stderr, "glm53f: --gpu handles up to %d positions (dense attention equals the "
                        "model's sparse attention there); this session holds %d, so the trunk "
                        "stays on the CPU, where the DSA indexer runs.\n",
                glm53f_dense_attn_limit(&m->cfg), m->cap);
        return -1;
    }
    m->gpu = glm53f_gpu_create(m, devices, ndev);
    if (!m->gpu) return -1;
    release_host_trunk(m);
    return 0;
#else
    (void)m; (void)devices; (void)ndev;
    fprintf(stderr, "glm53f: this build has no CUDA support (configure with -DGLM53F_CUDA=ON)\n");
    return -1;
#endif
}

static int argmax_(const float *v, int n)
{
    int b = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

int glm53f_model_forward(Glm53fModel *m, const int *ids, int T, float *logits, int *argmax_all)
{
    const Glm53fCfg *c = &m->cfg;
    if (T < 1) return -1;
    if (m->cached + T > m->cap) {
        fprintf(stderr, "glm53f: %d cached + %d new positions exceed the session capacity %d\n",
                m->cached, T, m->cap);
        return -1;
    }
    for (int t = 0; t < T; t++)
        if (ids[t] < 0 || ids[t] >= c->vocab) {
            fprintf(stderr, "glm53f: token id %d outside the vocabulary\n", ids[t]);
            return -1;
        }
#ifdef GLM53F_CUDA
    if (m->gpu) return glm53f_gpu_forward(m->gpu, m, ids, T, logits, argmax_all);
#endif
    /* A prompt is fed in spans of GLM53F_PREFILL_SPAN positions, each span layer by layer
     * in chunks (forward_cpu). The state a chunk leaves behind (KDA recurrence, KV cache,
     * indexer) is exactly what the next chunk needs, so neither size moves the output.
     * GLM53F_CHUNK and GLM53F_SPAN override them (both tested). */
    static int span = 0;
    if (!span) {
        const char *e = getenv("GLM53F_SPAN");
        span = e ? atoi(e) : GLM53F_PREFILL_SPAN;
        if (span < 1) span = GLM53F_PREFILL_SPAN;
    }
    for (int t0 = 0; t0 < T; t0 += span) {
        const int n = T - t0 < span ? T - t0 : span;
        const int last = t0 + n == T;
        if (forward_cpu(m, ids + t0, n, last ? logits : NULL,
                        argmax_all ? argmax_all + t0 : NULL) != 0)
            return -1;
    }
    return 0;
}

/* One span on the CPU: every chunk of at most GLM53F_CPU_CHUNK positions goes through a
 * layer before the next layer starts. The chunk sizes the working buffers; the span only
 * sizes the residual streams. Ordering the work by layer is what lets a prefill read each
 * layer's routed experts once: chunk by chunk through all 45 layers, a 372-token prompt
 * read 437 GB of experts, more than the 305 GB the model has. */
static int forward_cpu(Glm53fModel *m, const int *ids, int T, float *logits, int *argmax_all)
{
    const Glm53fCfg *c = &m->cfg;
    const int E = c->hidden, M = c->hc_mult;
    static int chunk = 0;
    if (!chunk) {
        const char *e = getenv("GLM53F_CHUNK");
        chunk = e ? atoi(e) : GLM53F_CPU_CHUNK;
        if (chunk < 1) chunk = GLM53F_CPU_CHUNK;
    }
    const int tc = T < chunk ? T : chunk;

    const size_t hsz = (size_t)T * M * E;
    float *h  = (float *)malloc(hsz * sizeof(float));
    float *sc = (float *)malloc(glm53f_layer_scratch(c, tc, m->cap) * sizeof(float));
    float *lg = (float *)malloc((size_t)c->vocab * sizeof(float));
    float *xm = (float *)malloc((size_t)2 * E * sizeof(float));
    if (!h || !sc || !lg || !xm) {
        fprintf(stderr, "glm53f: cannot allocate forward buffers for %d tokens\n", T);
        free(h); free(sc); free(lg); free(xm);
        return -1;
    }

    for (int t = 0; t < T; t++) {
        float *ht = h + (size_t)t * M * E;
        glm53f_embed_row(ht, &m->mb.embed, ids[t]);
        for (int i = 1; i < M; i++) memcpy(ht + (size_t)i * E, ht, (size_t)E * sizeof(float));
    }

    const size_t kper = glm53f_kda_state_floats(c);
    int rc = 0;
    m->layers_completed = 0;
    for (int L = 0; L < m->n_bound; L++) {
        const long drops = glm53f_expert_drops;
        for (int t0 = 0; t0 < T && glm53f_expert_drops == drops; t0 += chunk) {
            const int n = T - t0 < chunk ? T - t0 : chunk;
            glm53f_decoder_layer(h + (size_t)t0 * M * E, &m->lay[L].w, c, n,
                                 m->state + kper * (size_t)L, sc, m->kv[L], m->cached + t0,
                                 m->cap, m->idx ? m->idx[L] : NULL, m->ckv ? m->ckv[L] : NULL);
        }
        if (glm53f_expert_drops != drops) {
            fprintf(stderr, "glm53f: routed expert load failed at layer %d; refusing partial output\n", L);
            rc = -1;
            break;
        }
        m->layers_completed = L + 1;
    }

    if (rc == 0) {
        /* The MTP layer consumes the pre-norm hidden state of every position. */
        const int first = (argmax_all || m->mtp_h) ? 0 : T - 1;
        for (int t = first; t < T; t++) {
            const float *ht = h + (size_t)t * M * E;
            for (int d = 0; d < E; d++) {                 /* Glm5NextTextHyperHead: mean */
                double s = 0.0;
                for (int i = 0; i < M; i++) s += (double)ht[(size_t)i * E + d];
                xm[d] = (float)(s / M);
            }
            if (m->mtp_h) memcpy(m->mtp_h + (size_t)t * E, xm, (size_t)E * sizeof(float));
            if (!argmax_all && t < T - 1) continue;
            glm53f_rmsnorm(xm + E, xm, m->mb.norm, E, c->rms_eps);
            glm53f_mm(lg, xm + E, &m->mb.lm_head);
            if (argmax_all) argmax_all[t] = argmax_(lg, c->vocab);
        }
        if (logits) memcpy(logits, lg, (size_t)c->vocab * sizeof(float));
        m->cached += T;
    }
    free(h); free(sc); free(lg); free(xm);
    return rc;
}
