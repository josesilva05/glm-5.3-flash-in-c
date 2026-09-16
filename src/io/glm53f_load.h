/* glm53f_load.h - resolve and stream GLM-5.3-Flash routed experts.
 *
 * WHAT ONE EXPERT IS ON DISK
 *   layers.L.mlp.experts.E.{gate,up,down}_proj.weight            F8_E4M3 [2048,4096]/[4096,2048]
 *   layers.L.mlp.experts.E.{gate,up,down}_proj.weight_scale_inv  F32 [16,32]/[32,16]
 *
 *   Measured on the released shards: for 12,383 of 12,384 experts the three weights form
 *   ONE contiguous run (25,171,968 bytes) and the three scales form another, in the same
 *   shard. Fetching an expert is therefore two reads: one large O_DIRECT read for the
 *   weights and one tiny buffered read for the scales. The loader verifies contiguity per
 *   expert rather than assuming it; an expert that is not laid out that way (one straddles
 *   two shards) is read tensor by tensor into the same slot layout.
 *
 * SLOT LAYOUT (what the cache stores, never dequantised)
 *   [weights area: pad + weight bytes, O_DIRECT aligned] [scales area]
 *   m[i].w_off is relative to the start of the weight payload (slot + pad);
 *   m[i].s_off is relative to the start of the scales area (slot + weight_area).
 */
#ifndef GLM53F_LOAD_H
#define GLM53F_LOAD_H

#include "glm53f.h"
#include "glm53f_st.h"

typedef struct {
    int64_t w_off, w_bytes;      /* within the weight payload */
    int64_t s_off, s_bytes;      /* within the scales area    */
    int     rows, cols, srows, scols;
    int     dt;                  /* GLM53F_WF8 / GLM53F_WBF16 / GLM53F_WF32 */
} Glm53fQMat;

typedef struct {
    int     layer, expert;
    int     contiguous;          /* weights run and scales run each in one piece */
    int     shard;               /* valid when contiguous */
    int64_t w_abs, w_bytes;      /* absolute offset / size of the weights run */
    int64_t s_abs, s_bytes;      /* absolute offset / size of the scales run */
    const Glm53fTensor *tw[3], *ts[3];
    Glm53fQMat m[3];             /* gate, up, down */
} Glm53fExpertRef;

/* Resolve one expert's tensors and validate geometry against the config. */
int glm53f_expert_ref(const Glm53fSt *s, const Glm53fCfg *c, int layer, int expert,
                      Glm53fExpertRef *r);

/* Bytes a cache slot needs for this expert (weight area rounded for O_DIRECT + scales). */
int64_t glm53f_expert_weight_area(const Glm53fExpertRef *r);
int64_t glm53f_expert_slot_bytes(const Glm53fExpertRef *r);

/* Read an expert into slot (page aligned, slot_bytes long). *pad receives where the weight
 * payload starts. Returns 0 on success, -1 on any short read. */
int glm53f_expert_load(const Glm53fSt *s, const Glm53fExpertRef *r, unsigned char *slot,
                       int64_t slot_bytes, int64_t *pad);

/* --- int4 expert cache (--experts int4) ---
 * The same expert re-quantised to GLM53F_WI4, which is 14.2 MB instead of 25.2 MB, so the
 * same RAM holds 1.8x more experts. The slot holds, per matrix, the packed nibbles
 * (rows*cols/2, 64-byte aligned) followed by the group steps (rows*cols/GLM53F_I4_GROUP
 * floats). This is an approximation of the checkpoint, not the checkpoint. */
int64_t glm53f_expert_i4_slot_bytes(const Glm53fExpertRef *r);
/* Quantise an expert already read into `src` (the FP8 slot layout) into `dst`. */
void    glm53f_expert_i4_pack(const Glm53fExpertRef *r, const Glm53fCfg *c,
                              const unsigned char *src, int64_t pad, unsigned char *dst);
void    glm53f_expert_i4_view(const Glm53fExpertRef *r, const unsigned char *slot, Glm53fExpertQ *q);

/* Point q at an expert already loaded into slot. */
void glm53f_expert_view(const Glm53fExpertRef *r, const Glm53fCfg *c, const unsigned char *slot,
                        int64_t pad, int64_t weight_area, Glm53fExpertQ *q);

#endif /* GLM53F_LOAD_H */
