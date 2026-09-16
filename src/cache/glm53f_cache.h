/* glm53f_cache.h - the streaming routed-expert cache, with predictive prefetch.
 *
 * THE PROBLEM, IN NUMBERS
 *   GLM-5.3-Flash has 42 MoE layers of 288 experts, 25.2 MB each in FP8: ~305 GB. A decode
 *   step uses top-8 in each MoE layer, 336 experts. The cache keeps recently used experts
 *   resident, still FP8 (glm53f_mm multiplies the codes directly), and reads the rest from
 *   disk.
 *
 * REPLACEMENT POLICY
 *   LRU over a fixed arena of page-aligned slots. A slot is never handed out while it is
 *   PINNED (an expert the MoE is multiplying right now) or IN FLIGHT (a read into it has
 *   not landed yet).
 *
 * PREDICTIVE PREFETCH
 *   Storage throughput is a hard limit (docs/PERFORMANCE.md), so reads can only get cheaper
 *   in wall time by overlapping with compute. The MoE of layer L predicts layer L+1's
 *   routing (the next router applied to its own input, 75-90% accurate) and hands the
 *   prediction to hint(). Background I/O threads read those experts while layers L and
 *   L+1 compute. When layer L+1 then asks for an expert that is still being read, get()
 *   waits for that read instead of starting a second one.
 *
 *   Prefetch decides only WHAT is resident WHEN. Routing and arithmetic are unchanged, so
 *   output is identical with prefetch on or off.
 *
 * THREAD SAFETY
 *   All bookkeeping is under one mutex; reads happen outside it into slots reserved under
 *   it. The main thread may call get/getmany/release/hint; I/O threads only load.
 */
#ifndef GLM53F_CACHE_H
#define GLM53F_CACHE_H

#include "glm53f_portable_io.h"   /* supplies pthreads on MSVC */
#if !defined(_MSC_VER)
#include <pthread.h>
#endif

#include "glm53f.h"
#include "glm53f_load.h"
#include "glm53f_st.h"

#define GLM53F_SLOT_EMPTY     (-1)
#define GLM53F_SLOT_INFLIGHT  (-2)

typedef struct {
    Glm53fExpertSrc  src;             /* MUST be first: pass &cache->src to Glm53fMoeW */

    const Glm53fSt  *st;
    const Glm53fCfg *cfg;
    int          n_layers, n_experts;

    unsigned char *arena;         /* nslot * slot_bytes, page aligned */
    int64_t      slot_bytes;
    int          nslot;

    /* --experts int4: slots hold GLM53F_WI4 copies, quantised on the way in from the
     * FP8 read, which is staged through one of `nstage` buffers. */
    int          i4;
    int64_t      raw_bytes;       /* a staging buffer: the FP8 slot size */
    unsigned char **stage;
    unsigned char *stage_busy;
    int          nstage;
    pthread_cond_t stage_free;

    int32_t     *slot_of;         /* [n_layers*n_experts] -> resident slot, or -1  */
    int32_t     *inflight_of;     /* [n_layers*n_experts] -> slot being read, or -1 */
    int32_t     *key_of;          /* [nslot] -> key, EMPTY or INFLIGHT             */
    int32_t     *pins;            /* [nslot] experts in use; never evicted while > 0 */
    unsigned char *prefetched;    /* [nslot] loaded by prefetch and not yet used    */
    uint64_t    *used_at;         /* [nslot] LRU stamp                             */
    Glm53fExpertRef *ref;         /* [nslot] geometry of the resident expert       */
    int64_t     *pad;             /* [nslot] weight payload offset in the slot      */
    uint64_t     clock;

    pthread_mutex_t mu;
    pthread_cond_t  landed;       /* an in-flight read finished                    */
    pthread_cond_t  work;         /* prefetch queue gained entries, or stop        */

    /* prefetch queue: keys, oldest first */
    int          n_io;            /* background I/O threads, 0 = prefetch disabled */
    pthread_t   *io;
    int          stop;
    int32_t     *queue;
    int          qcap, qlen;
    int          hint_floor, last_hint;   /* layer being computed; last layer hinted */
    unsigned char *queued;        /* [n_layers*n_experts] */

    /* statistics (reset per step by glm53f_cache_reset_stats) */
    uint64_t     hits, misses, evictions, bytes_read, prefetch_reads;
    double       load_seconds;    /* main-thread time spent reading or waiting on reads */
    uint64_t     bg_reads, bg_bytes, bg_used;
} Glm53fCache;

/* budget_bytes is the arena size, rounded down to whole experts. n_io background threads
 * serve hint() (0 disables prefetch). Fails if fewer than topk+1 slots fit. */
int  glm53f_cache_init(Glm53fCache *c, const Glm53fSt *st, const Glm53fCfg *cfg,
                       int64_t budget_bytes, int n_io, int i4);
void glm53f_cache_free(Glm53fCache *c);
void glm53f_cache_reset_stats(Glm53fCache *c);
void glm53f_cache_report(const Glm53fCache *c, const char *label);

#endif /* GLM53F_CACHE_H */
