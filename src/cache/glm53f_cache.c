/* glm53f_cache.c - see glm53f_cache.h. */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"   /* first: sets _DARWIN_C_SOURCE; supplies posix_memalign on Windows */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "glm53f_cache.h"

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void fill_q(const Glm53fCache *c, int slot, Glm53fExpertQ *q)
{
    const Glm53fExpertRef *r = &c->ref[slot];
    unsigned char *base = c->arena + (size_t)slot * c->slot_bytes;
    if (c->i4) glm53f_expert_i4_view(r, base, q);
    else       glm53f_expert_view(r, c->cfg, base, c->pad[slot], glm53f_expert_weight_area(r), q);
}

/* Take a staging buffer for one FP8 read (int4 mode only); the caller must not hold mu. */
static unsigned char *stage_take(Glm53fCache *c, int *idx)
{
    pthread_mutex_lock(&c->mu);
    for (;;) {
        for (int i = 0; i < c->nstage; i++)
            if (!c->stage_busy[i]) {
                c->stage_busy[i] = 1;
                *idx = i;
                pthread_mutex_unlock(&c->mu);
                return c->stage[i];
            }
        pthread_cond_wait(&c->stage_free, &c->mu);
    }
}

static void stage_give(Glm53fCache *c, int idx)
{
    pthread_mutex_lock(&c->mu);
    c->stage_busy[idx] = 0;
    pthread_cond_broadcast(&c->stage_free);
    pthread_mutex_unlock(&c->mu);
}

/* Least recently used slot that is neither in flight nor pinned. Caller holds mu. */
static int pick_victim(Glm53fCache *c)
{
    int best = -1;
    uint64_t oldest = (uint64_t)-1;
    for (int i = 0; i < c->nslot; i++) {
        if (c->key_of[i] == GLM53F_SLOT_INFLIGHT || c->pins[i] > 0) continue;
        if (c->key_of[i] == GLM53F_SLOT_EMPTY) return i;
        if (c->used_at[i] < oldest) { oldest = c->used_at[i]; best = i; }
    }
    return best;
}

/* Reserve a slot for key and mark the read in flight. Caller holds mu. Returns the slot,
 * or -1 if every slot is pinned or in flight. */
static int reserve(Glm53fCache *c, int32_t key)
{
    const int slot = pick_victim(c);
    if (slot < 0) return -1;
    if (c->key_of[slot] >= 0) { c->slot_of[c->key_of[slot]] = -1; c->evictions++; }
    c->key_of[slot] = GLM53F_SLOT_INFLIGHT;
    c->prefetched[slot] = 0;
    c->inflight_of[key] = slot;
    return slot;
}

/* Publish (ok) or release (failed) a reserved slot. Caller holds mu. */
static void land(Glm53fCache *c, int32_t key, int slot, const Glm53fExpertRef *r, int64_t pad,
                 int ok, int background)
{
    c->inflight_of[key] = -1;
    if (ok) {
        c->ref[slot] = *r;
        c->pad[slot] = pad;
        c->key_of[slot] = key;
        c->slot_of[key] = slot;
        c->used_at[slot] = ++c->clock;
        c->prefetched[slot] = (unsigned char)background;
    } else {
        c->key_of[slot] = GLM53F_SLOT_EMPTY;
    }
    pthread_cond_broadcast(&c->landed);
}

/* Read one expert into a reserved slot. Called WITHOUT mu. */
static int load_into(Glm53fCache *c, int layer, int expert, int slot, Glm53fExpertRef *r, int64_t *pad)
{
    *pad = 0;
    if (glm53f_expert_ref(c->st, c->cfg, layer, expert, r) != 0) return 0;
    const int64_t need = c->i4 ? glm53f_expert_i4_slot_bytes(r) : glm53f_expert_slot_bytes(r);
    if (need > c->slot_bytes) {
        fprintf(stderr, "glm53f_cache: L%d expert %d needs %lld bytes, slot holds %lld\n",
                layer, expert, (long long)need, (long long)c->slot_bytes);
        return 0;
    }
    if (!c->i4)
        return glm53f_expert_load(c->st, r, c->arena + (size_t)slot * c->slot_bytes,
                                  c->slot_bytes, pad) == 0;
    int idx = 0;
    unsigned char *raw = stage_take(c, &idx);
    const int ok = glm53f_expert_load(c->st, r, raw, c->raw_bytes, pad) == 0;
    if (ok) glm53f_expert_i4_pack(r, c->cfg, raw, *pad, c->arena + (size_t)slot * c->slot_bytes);
    stage_give(c, idx);
    *pad = 0;
    return ok;
}

/* ------------------------------------------------------------------ main thread ---- */

static int cache_get(Glm53fExpertSrc *self, int layer, int expert, Glm53fExpertQ *out)
{
    Glm53fCache *c = (Glm53fCache *)self;
    if (layer < 0 || layer >= c->n_layers || expert < 0 || expert >= c->n_experts) {
        fprintf(stderr, "glm53f_cache: out of range L%d expert %d\n", layer, expert);
        return -1;
    }
    const int32_t key = layer * c->n_experts + expert;
    const double t0 = now_s();
    pthread_mutex_lock(&c->mu);
    for (;;) {
        int slot = c->slot_of[key];
        if (slot >= 0) {
            c->hits++;
            if (c->prefetched[slot]) { c->bg_used++; c->prefetched[slot] = 0; }
            c->used_at[slot] = ++c->clock;
            c->pins[slot]++;
            fill_q(c, slot, out);
            pthread_mutex_unlock(&c->mu);
            c->load_seconds += now_s() - t0;
            return 0;
        }
        if (c->inflight_of[key] >= 0) {          /* someone is reading it: wait, do not re-read */
            pthread_cond_wait(&c->landed, &c->mu);
            continue;
        }
        c->misses++;
        slot = reserve(c, key);
        if (slot < 0) {
            /* Everything is pinned or in flight; wait for a read to land and retry. */
            pthread_cond_wait(&c->landed, &c->mu);
            c->misses--;
            continue;
        }
        pthread_mutex_unlock(&c->mu);
        Glm53fExpertRef r;
        int64_t pad;
        const int ok = load_into(c, layer, expert, slot, &r, &pad);
        pthread_mutex_lock(&c->mu);
        land(c, key, slot, &r, pad, ok, 0);
        if (!ok) {
            pthread_mutex_unlock(&c->mu);
            fprintf(stderr, "glm53f_cache: short load of L%d expert %d\n", layer, expert);
            c->load_seconds += now_s() - t0;
            return -1;
        }
        c->bytes_read += (uint64_t)(r.w_bytes + r.s_bytes);
        /* loop: the slot is now resident and the hit path pins it */
        c->hits--;
    }
}

static void cache_release(Glm53fExpertSrc *self, int layer, int expert)
{
    Glm53fCache *c = (Glm53fCache *)self;
    const int32_t key = layer * c->n_experts + expert;
    pthread_mutex_lock(&c->mu);
    const int slot = c->slot_of[key];
    if (slot >= 0 && c->pins[slot] > 0) c->pins[slot]--;
    pthread_mutex_unlock(&c->mu);
}

/* Bring a batch resident with the reads issued concurrently from this thread. Experts
 * already resident or in flight are skipped (get() will serve or wait for them). */
static int cache_getmany(Glm53fExpertSrc *self, int layer, const int *ids, int n)
{
    Glm53fCache *c = (Glm53fCache *)self;
    typedef struct { int slot, expert, ok; int32_t key; Glm53fExpertRef r; int64_t pad; } Work;
    Work w[GLM53F_MAX_TOPK];
    int nw = 0;

    const double t0 = now_s();
    pthread_mutex_lock(&c->mu);
    for (int i = 0; i < n && nw < GLM53F_MAX_TOPK; i++) {
        const int e = ids[i];
        if (e < 0 || e >= c->n_experts) continue;
        const int32_t key = layer * c->n_experts + e;
        if (c->slot_of[key] >= 0 || c->inflight_of[key] >= 0) continue;
        const int slot = reserve(c, key);
        if (slot < 0) break;
        w[nw].slot = slot; w[nw].expert = e; w[nw].key = key; w[nw].ok = 0; w[nw].pad = 0;
        nw++;
    }
    pthread_mutex_unlock(&c->mu);
    if (nw == 0) return 0;

    int i;
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic, 1)
#endif
    for (i = 0; i < nw; i++)
        w[i].ok = load_into(c, layer, w[i].expert, w[i].slot, &w[i].r, &w[i].pad);

    int ok = 0;
    pthread_mutex_lock(&c->mu);
    for (int k = 0; k < nw; k++) {
        land(c, w[k].key, w[k].slot, &w[k].r, w[k].pad, w[k].ok, 0);
        if (!w[k].ok) {
            fprintf(stderr, "glm53f_cache: short prefetch of L%d expert %d; slot left empty\n",
                    layer, w[k].expert);
            continue;
        }
        c->bytes_read += (uint64_t)(w[k].r.w_bytes + w[k].r.s_bytes);
        c->prefetch_reads++;
        ok++;
    }
    pthread_mutex_unlock(&c->mu);
    c->load_seconds += now_s() - t0;
    return ok;
}

/* Replace the pending work for `layer` with these experts, in the order given. A MoE layer
 * hints its own (actual) experts first and then predictions for the layers after it, so a
 * hint for a layer not above the previous hint marks the layer now computing (the floor).
 * Pending entries not yet started are dropped when they belong to a layer below the floor
 * (already run) or to `layer` itself but are not in `ids`: a better prediction, or the
 * actual routing, cancels the earlier guesses for that layer before they cost a read.
 * The queue stays ordered by layer (stable), so nearer layers are read first. Reads
 * already in flight are never cancelled. */
static void cache_hint(Glm53fExpertSrc *self, int layer, const int *ids, int n)
{
    Glm53fCache *c = (Glm53fCache *)self;
    if (c->n_io <= 0 || layer < 0 || layer >= c->n_layers) return;
    pthread_mutex_lock(&c->mu);
    if (layer <= c->last_hint) c->hint_floor = layer;
    c->last_hint = layer;
    int kept = 0;
    for (int i = 0; i < c->qlen; i++) {
        const int32_t key = c->queue[i];
        const int kl = key / c->n_experts;
        int drop = kl < c->hint_floor;
        if (kl == layer) {
            drop = 1;
            for (int j = 0; j < n; j++) if (ids[j] == key % c->n_experts) { drop = 0; break; }
        }
        if (drop) { c->queued[key] = 0; continue; }
        c->queue[kept++] = key;
    }
    c->qlen = kept;
    /* insert after every pending entry of a layer <= `layer` */
    int at = c->qlen;
    while (at > 0 && c->queue[at - 1] / c->n_experts > layer) at--;
    for (int i = 0; i < n && c->qlen < c->qcap; i++) {
        const int e = ids[i];
        if (e < 0 || e >= c->n_experts) continue;
        const int32_t key = layer * c->n_experts + e;
        if (c->queued[key] || c->slot_of[key] >= 0 || c->inflight_of[key] >= 0) continue;
        c->queued[key] = 1;
        memmove(c->queue + at + 1, c->queue + at, (size_t)(c->qlen - at) * sizeof *c->queue);
        c->queue[at++] = key;
        c->qlen++;
    }
    pthread_cond_broadcast(&c->work);
    pthread_mutex_unlock(&c->mu);
}

/* ------------------------------------------------------------- I/O threads ---- */

static void *io_main(void *arg)
{
    Glm53fCache *c = (Glm53fCache *)arg;
    pthread_mutex_lock(&c->mu);
    for (;;) {
        while (!c->stop && c->qlen == 0) pthread_cond_wait(&c->work, &c->mu);
        if (c->stop) break;
        const int32_t key = c->queue[0];
        memmove(c->queue, c->queue + 1, (size_t)(c->qlen - 1) * sizeof *c->queue);
        c->qlen--;
        c->queued[key] = 0;
        if (c->slot_of[key] >= 0 || c->inflight_of[key] >= 0) continue;
        const int slot = reserve(c, key);
        if (slot < 0) continue;                  /* cache saturated: skip this hint */
        pthread_mutex_unlock(&c->mu);

        Glm53fExpertRef r;
        int64_t pad;
        const int ok = load_into(c, key / c->n_experts, key % c->n_experts, slot, &r, &pad);

        pthread_mutex_lock(&c->mu);
        land(c, key, slot, &r, pad, ok, 1);
        if (ok) { c->bg_reads++; c->bg_bytes += (uint64_t)(r.w_bytes + r.s_bytes); }
    }
    pthread_mutex_unlock(&c->mu);
    return NULL;
}

/* ------------------------------------------------------------------ lifecycle ---- */

int glm53f_cache_init(Glm53fCache *c, const Glm53fSt *st, const Glm53fCfg *cfg,
                      int64_t budget_bytes, int n_io, int i4)
{
    memset(c, 0, sizeof *c);
    c->src.get = cache_get;
    c->src.release = cache_release;
    c->src.getmany = getenv("GLM53F_NOPREFETCH") ? NULL : cache_getmany;
    c->src.hint = cache_hint;
    c->src.ctx = c;
    c->st = st;
    c->cfg = cfg;
    c->n_layers = cfg->n_layers + 1;   /* the MTP layer streams experts the same way */
    c->n_experts = cfg->n_experts;
    pthread_mutex_init(&c->mu, NULL);
    pthread_cond_init(&c->landed, NULL);
    pthread_cond_init(&c->work, NULL);
    pthread_cond_init(&c->stage_free, NULL);
    c->i4 = i4;

    int probe_layer = -1;
    for (int L = 0; L < cfg->n_layers; L++) if (!glm53f_is_dense(cfg, L)) { probe_layer = L; break; }
    if (probe_layer < 0) return 0;              /* no MoE layers: nothing to cache */
    Glm53fExpertRef probe;
    if (glm53f_expert_ref(st, cfg, probe_layer, 0, &probe) != 0) {
        fprintf(stderr, "glm53f_cache: cannot resolve expert 0 of layer %d\n", probe_layer);
        return -1;
    }
    c->raw_bytes = glm53f_expert_slot_bytes(&probe) + GLM53F_ST_ALIGN;
    c->slot_bytes = i4 ? glm53f_expert_i4_slot_bytes(&probe) : c->raw_bytes;
    c->nslot = (int)(budget_bytes / c->slot_bytes);
    if (c->nslot < cfg->topk + 1) {
        fprintf(stderr, "glm53f_cache: budget %.2f GB gives %d slots of %.2f MB; top-%d needs at "
                        "least %d\n", (double)budget_bytes / 1e9, c->nslot,
                (double)c->slot_bytes / 1e6, cfg->topk, cfg->topk + 1);
        return -1;
    }
    const size_t want = (size_t)c->nslot * (size_t)c->slot_bytes;
    if (posix_memalign((void **)&c->arena, GLM53F_ST_ALIGN, want) != 0) {
        fprintf(stderr, "glm53f_cache: cannot allocate %.2f GB arena\n", (double)want / 1e9);
        return -1;
    }
    if (c->i4) {
        c->nstage = n_io > 0 ? n_io + 1 : 4;
        c->stage = (unsigned char **)calloc((size_t)c->nstage, sizeof *c->stage);
        c->stage_busy = (unsigned char *)calloc((size_t)c->nstage, 1);
        if (!c->stage || !c->stage_busy) return -1;
        for (int i = 0; i < c->nstage; i++)
            if (posix_memalign((void **)&c->stage[i], GLM53F_ST_ALIGN, (size_t)c->raw_bytes) != 0) {
                fprintf(stderr, "glm53f_cache: cannot allocate the int4 staging buffers\n");
                return -1;
            }
    }
    const size_t nkey = (size_t)c->n_layers * c->n_experts;
    c->slot_of     = (int32_t *)malloc(nkey * sizeof(int32_t));
    c->inflight_of = (int32_t *)malloc(nkey * sizeof(int32_t));
    c->queued      = (unsigned char *)calloc(nkey, 1);
    c->key_of      = (int32_t *)malloc((size_t)c->nslot * sizeof(int32_t));
    c->pins        = (int32_t *)calloc((size_t)c->nslot, sizeof(int32_t));
    c->prefetched  = (unsigned char *)calloc((size_t)c->nslot, 1);
    c->used_at     = (uint64_t *)calloc((size_t)c->nslot, sizeof(uint64_t));
    c->ref         = (Glm53fExpertRef *)calloc((size_t)c->nslot, sizeof(Glm53fExpertRef));
    c->pad         = (int64_t *)calloc((size_t)c->nslot, sizeof(int64_t));
    c->qcap        = 64 * 4 * GLM53F_MAX_TOPK;   /* a 64-token prefill chunk hints up to 64 x topk per layer */
    c->queue       = (int32_t *)malloc((size_t)c->qcap * sizeof(int32_t));
    if (!c->slot_of || !c->inflight_of || !c->queued || !c->key_of || !c->pins ||
        !c->prefetched || !c->used_at || !c->ref || !c->pad || !c->queue) {
        glm53f_cache_free(c); return -1;
    }
    for (size_t i = 0; i < nkey; i++) { c->slot_of[i] = -1; c->inflight_of[i] = -1; }
    for (int i = 0; i < c->nslot; i++) c->key_of[i] = GLM53F_SLOT_EMPTY;

    if (n_io > 0) {
        c->io = (pthread_t *)calloc((size_t)n_io, sizeof(pthread_t));
        if (!c->io) { glm53f_cache_free(c); return -1; }
        for (int i = 0; i < n_io; i++) {
            if (pthread_create(&c->io[i], NULL, io_main, c) != 0) {
                fprintf(stderr, "glm53f_cache: cannot start prefetch thread %d\n", i);
                break;
            }
            c->n_io++;
        }
    }
    return 0;
}

void glm53f_cache_free(Glm53fCache *c)
{
    if (c->n_io > 0) {
        pthread_mutex_lock(&c->mu);
        c->stop = 1;
        pthread_cond_broadcast(&c->work);
        pthread_mutex_unlock(&c->mu);
        for (int i = 0; i < c->n_io; i++) pthread_join(c->io[i], NULL);
    }
    free(c->io);
    if (c->slot_of || c->arena) {
        pthread_cond_destroy(&c->stage_free);
        pthread_cond_destroy(&c->work);
        pthread_cond_destroy(&c->landed);
        pthread_mutex_destroy(&c->mu);
    }
    if (c->stage) for (int i = 0; i < c->nstage; i++) glm53f_aligned_free(c->stage[i]);
    free(c->stage); free(c->stage_busy);
    glm53f_aligned_free(c->arena);
    free(c->slot_of); free(c->inflight_of); free(c->queued); free(c->key_of); free(c->pins);
    free(c->prefetched); free(c->used_at); free(c->ref); free(c->pad); free(c->queue);
    memset(c, 0, sizeof *c);
}

void glm53f_cache_reset_stats(Glm53fCache *c)
{
    pthread_mutex_lock(&c->mu);
    c->hits = c->misses = c->evictions = c->bytes_read = c->prefetch_reads = 0;
    c->bg_reads = c->bg_bytes = c->bg_used = 0;
    c->load_seconds = 0.0;
    pthread_mutex_unlock(&c->mu);
}

void glm53f_cache_report(const Glm53fCache *c, const char *label)
{
    const uint64_t n = c->hits + c->misses;
    int resident = 0;
    for (int i = 0; i < c->nslot; i++) if (c->key_of[i] >= 0) resident++;
    if (!glm53f_quiet)
    printf("expert cache [%s]\n", label ? label : "");
    if (!glm53f_quiet)
    printf("  slots          : %d x %.2f MB = %.2f GB (%d resident), %d prefetch threads\n",
           c->nslot, (double)c->slot_bytes / 1e6, (double)c->nslot * c->slot_bytes / 1e9,
           resident, c->n_io);
    const uint64_t served = c->hits > c->prefetch_reads ? c->hits - c->prefetch_reads : 0;
    if (!glm53f_quiet)
    printf("  requests       : %llu, resident when requested: %llu (%.1f%%)\n",
           (unsigned long long)n, (unsigned long long)served, n ? 100.0 * served / n : 0.0);
    if (!glm53f_quiet)
    printf("  foreground     : %.2f GB read, %.2f s reading or waiting, %llu evictions\n",
           (double)c->bytes_read / 1e9, c->load_seconds, (unsigned long long)c->evictions);
    if (c->n_io)
        if (!glm53f_quiet)
        printf("  prefetch       : %llu experts read ahead (%.2f GB), %llu of them used\n",
               (unsigned long long)c->bg_reads, (double)c->bg_bytes / 1e9,
               (unsigned long long)c->bg_used);
}
