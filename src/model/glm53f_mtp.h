/* glm53f_mtp.h - the checkpoint's multi-token-prediction layer (layer n_layers, "nextn").
 *
 * The layer takes the hidden state of position i (the mean of the residual streams, before
 * the final norm) and the embedding of the token at i+1, and predicts the token at i+2:
 *
 *   x = eh_proj( [ RMSNorm(embed(tok_{i+1}), enorm) , RMSNorm(h_i, hnorm) ] )
 *   x = x + MLA(RMSNorm(x, input_layernorm))          (its own KV cache)
 *   x = x + MoE(RMSNorm(x, post_attention_layernorm))  (its own 288 experts, streamed)
 *   draft = argmax( lm_head(RMSNorm(x, shared_head.norm)) )
 *
 * transformers 5.17 does not implement this layer (it skips `layers.45.`), so there is no
 * reference to compare against. It is only ever used as a DRAFT: a speculative decoder
 * accepts a draft token solely when the full model produces the same token, so a wrong
 * draft costs speed, never output. GLM53F_MTP_STATS=1 reports how often it is right.
 */
#ifndef GLM53F_MTP_H
#define GLM53F_MTP_H

#include "glm53f_model.h"

typedef struct Glm53fMtp Glm53fMtp;

/* Bind the layer and allocate its state for `cap` positions. -1 if the checkpoint has no
 * MTP layer, the lm_head is not on the host, or memory runs out. */
int  glm53f_mtp_open(Glm53fModel *m, int cap);
void glm53f_mtp_close(Glm53fModel *m);

/* Feed T consecutive positions: h holds T hidden states (position p, p+1, ...) and
 * next_ids[t] the token that follows position p+t. Advances the layer's KV cache by T and
 * writes the draft token for the position after the last one into *draft (may be NULL).
 * Returns -1 if an expert failed to load. */
int  glm53f_mtp_feed(Glm53fModel *m, const float *h, const int *next_ids, int T, int *draft);

/* Forget every fed position (a new session). */
void glm53f_mtp_reset(Glm53fModel *m);

/* Count one draft against the token the full model produced. */
void glm53f_mtp_score(Glm53fModel *m, int draft, int actual);
void glm53f_mtp_report(const Glm53fModel *m);

#endif /* GLM53F_MTP_H */
