/* glm53f_model.h - a loaded GLM-5.3-Flash: resident trunk, streamed experts, carried state.
 *
 * glm53f_model_forward feeds T new tokens after the `cached` positions already consumed and
 * leaves the model ready for the next call: KDA recurrent/conv state and the MLA KV cache
 * are carried. A prefill is one call with the prompt; each decode step is one call with
 * the previous token. Logits come from the mHC head: mean over the residual streams,
 * final RMSNorm, lm_head.
 */
#ifndef GLM53F_MODEL_H
#define GLM53F_MODEL_H

#include "glm53f.h"
#include "glm53f_bind.h"
#include "glm53f_cache.h"
#include "glm53f_st.h"

typedef struct Glm53fModel {
    Glm53fCfg        cfg;
    Glm53fSt         st;
    Glm53fModelBind  mb;
    Glm53fLayerBind *lay;
    int              n_bound;        /* layers bound (== cfg.n_layers unless limited) */
    Glm53fCache      cache;

    int              cap;            /* positions the KV cache holds */
    int              cached;         /* positions consumed so far    */
    float           *state;          /* [n_bound][kda_state_floats]  */
    float          **kv;             /* [n_bound] -> [cap][H*(qk_nope+v_head)], NULL on KDA layers */
    float          **idx;            /* [n_bound] -> DSA indexer state, NULL on KDA layers   */
    float          **ckv;            /* [n_bound] -> compressed KV ([cap][kv_lora]), or NULL */

    int              layers_completed;
    double           load_seconds;
    size_t           trunk_bytes;
    struct Glm53fGpu *gpu;           /* non-NULL: forward runs the trunk on GPUs */
    float           *routers;        /* GPU mode: the MoE routers kept after the host trunk is freed */
    struct Glm53fMtp *mtp;           /* non-NULL: the MTP layer is bound (glm53f_mtp.h)   */
    float           *mtp_h;          /* [cap][hidden]: hidden states of the last forward  */
} Glm53fModel;

/* Open a checkpoint directory. cfg_path NULL means <dir>/config.json. max_layers <= 0 binds
 * every layer. cap is the most positions (prompt + generated) this session will hold.
 * prefetch_n > 0 enables predictive expert prefetch (prefill and decode): each MoE layer hints its
 * own experts and the prefetch_n most likely experts of the next layer to background
 * readers (GLM53F_IO_THREADS of them, default 2). Output does not depend on it. */
/* expert_i4 != 0: the expert cache holds routed experts re-quantised to int4 (1.8x more of
 * them in the same RAM, see glm53f.h GLM53F_WI4). The output is then an approximation of
 * the checkpoint, not the checkpoint; it is off by default. */
/* gpu_planned != 0: the caller intends to call glm53f_model_use_gpu, which frees the host
 * copy of the trunk; the expert cache may then use that RAM too.
 * kv: 0 chooses (compressed past the dense-attention range, expanded below it), 1 forces
 * the expanded keys and values, 2 forces the kv_lora latent (64x less memory). */
int  glm53f_model_open(Glm53fModel *m, const char *dir, const char *cfg_path, double cache_gb,
                       int max_layers, int cap, int prefetch_n, int expert_i4, int gpu_planned,
                       int kv);
void glm53f_model_close(Glm53fModel *m);

/* Forget every carried position. */
void glm53f_model_reset(Glm53fModel *m);

/* Feed T tokens. logits (vocab floats, may be NULL) receives the LAST position's logits;
 * argmax_all (T ints, may be NULL) receives the argmax at every position. Returns 0, or -1
 * if the forward could not be completed: outputs are then untouched, and the carried state
 * may be partially advanced, so the session must be reset before any further use. */
int  glm53f_model_forward(Glm53fModel *m, const int *ids, int T, float *logits, int *argmax_all);

/* Move the trunk to CUDA devices (ndev <= 0: all). Returns 0, or -1 if this build has no CUDA,
 * there is no device, or the trunk does not fit; the model then keeps running on the CPU.
 * On success the host copy of the trunk is freed (~14 GB for the released checkpoint): only
 * the embedding table and the MoE routers stay in RAM, so a larger --cache-gb fits. */
int  glm53f_model_use_gpu(Glm53fModel *m, const int *devices, int ndev);

/* Snapshot/restore of the carried KDA state (the KV cache needs no snapshot: rewinding
 * `cached` makes later writes overwrite the rejected rows). Used by speculative decode. */
size_t glm53f_model_state_bytes(const Glm53fModel *m);

#endif /* GLM53F_MODEL_H */
