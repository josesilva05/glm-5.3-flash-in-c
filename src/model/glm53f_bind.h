/* glm53f_bind.h - bind checkpoint tensors into the engine's weight structs.
 *
 * Every tensor is resolved, and its dtype and exact shape checked against what the config
 * implies, BEFORE any byte is read. A missing tensor or an unexpected shape fails the
 * bind: substituting NULL (or zeros) produces a model that runs and prints plausible,
 * wrong tokens.
 *
 * TWO STORAGE CLASSES
 *   Matrices keep the checkpoint's own bytes (BF16, or FP8 codes plus the F32
 *   weight_scale_inv grid) and are multiplied in place by glm53f_mm.
 *   Vectors read elementwise (norms, conv kernels, A_log, dt_bias, mHC base/scale) and
 *   the router matrix are widened to fp32.
 *
 * NAMES (prefix model.language_model.layers.L.)
 *   input_layernorm.weight, post_attention_layernorm.weight
 *   hc_{attn,ffn}_{fn,base,scale}
 *   KDA: self_attn.{q,k,v}_proj, {q,k,v}_conv1d, f_a_proj, f_b_proj, A_log, dt_bias,
 *        b_proj, g_a_proj, g_b_proj, o_norm, o_proj
 *   MLA: self_attn.q_a_proj, q_a_layernorm, q_b_proj, kv_a_proj_with_mqa, kv_a_layernorm,
 *        kv_b_proj, o_proj            (self_attn.indexer.* is present and unused, see glm53f.h)
 *   dense: mlp.{gate,up,down}_proj
 *   MoE:   mlp.gate.weight, mlp.gate.e_score_correction_bias,
 *          mlp.shared_experts.{gate,up,down}_proj     (routed experts stream via the cache)
 */
#ifndef GLM53F_BIND_H
#define GLM53F_BIND_H

#include "glm53f.h"
#include "glm53f_st.h"

typedef struct {
    void        *blob;          /* one allocation holding this layer's weights */
    size_t       nbytes;
    Glm53fLayerW w;
} Glm53fLayerBind;

/* Bytes the layer will occupy, or -1 if any tensor is missing or mis-shaped. */
int64_t glm53f_bind_layer_bytes(const Glm53fSt *s, const Glm53fCfg *c, int layer);
int     glm53f_bind_layer(const Glm53fSt *s, const Glm53fCfg *c, int layer, Glm53fLayerBind *b);
void    glm53f_bind_free(Glm53fLayerBind *b);

typedef struct {
    void        *blob;
    size_t       nbytes;
    Glm53fMat    embed, lm_head;
    const float *norm;
} Glm53fModelBind;

typedef struct {
    void        *blob;
    size_t       nbytes;
    Glm53fMtpW   w;
} Glm53fMtpBind;

/* Bind the MTP layer (checkpoint layer c->n_layers). Returns -1 if it is absent. */
int  glm53f_bind_mtp(const Glm53fSt *s, const Glm53fCfg *c, Glm53fMtpBind *b);
void glm53f_bind_mtp_free(Glm53fMtpBind *b);

int  glm53f_bind_model(const Glm53fSt *s, const Glm53fCfg *c, Glm53fModelBind *m);
void glm53f_bind_model_free(Glm53fModelBind *m);

#endif /* GLM53F_BIND_H */
