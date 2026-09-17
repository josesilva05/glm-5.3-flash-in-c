/* glm53f_i4file.h - an int4 expert container on disk (--write-int4, --int4-dir).
 *
 * WHY
 *   Decode time follows the bytes read per token (docs/PERFORMANCE.md). --experts int4
 *   already fits 1.8x more experts in RAM, but every miss still reads the 25.2 MB FP8
 *   expert and quantises it. A container holds the quantised experts themselves, 14.2 MB
 *   each, so a miss reads 44% fewer bytes and does no arithmetic on the way in.
 *
 * WHAT IS IN IT
 *   One file per MoE layer, <dir>/layer-LL.i4:
 *     [4096-byte header][expert 0][expert 1]...[expert n-1]
 *   Every expert takes the same slot_bytes (a multiple of 4096), holding exactly the bytes
 *   glm53f_expert_i4_pack writes into a cache slot, so a read lands straight in the slot
 *   with O_DIRECT. The output of a run from the container is therefore bit-identical to
 *   --experts int4 quantising the same experts on the way in.
 *
 *   The header carries a fingerprint of the checkpoint the layer was written from (a
 *   container from another checkpoint is refused), a CRC32 per expert for an integrity
 *   check, and a completion mark written last: a conversion that stopped midway leaves a
 *   .part file that is never used, and writing that layer again starts over.
 *
 *   The container is an APPROXIMATION of the checkpoint, like --experts int4; the checkpoint
 *   itself is never modified.
 */
#ifndef GLM53F_I4FILE_H
#define GLM53F_I4FILE_H

#include "glm53f.h"
#include "glm53f_st.h"

#define GLM53F_I4F_HEADER 4096

/* Write one MoE layer's experts to <dir>/layer-LL.i4. Skips a layer already complete and
 * valid. Returns 0 on success, 1 when skipped, -1 on failure (the .part file is removed). */
int glm53f_i4file_write_layer(const Glm53fSt *st, const Glm53fCfg *c, int layer, const char *dir);

/* Open <dir>/layer-LL.i4 for reading experts. Returns a descriptor, -1 when there is no
 * file (the layer is quantised on the way in instead), or -2 when a file is there but does
 * not belong to this checkpoint or is incomplete (a message says why). */
int glm53f_i4file_open_layer(const Glm53fSt *st, const Glm53fCfg *c, int layer, const char *dir);

/* Verify every expert of a layer against its CRC. Returns the number of bad experts, or -1
 * when the file cannot be read. */
int glm53f_i4file_verify_layer(const Glm53fSt *st, const Glm53fCfg *c, int layer, const char *dir);

/* Offset of an expert inside its layer file, and the bytes it takes. */
static inline int64_t glm53f_i4file_offset(int expert, int64_t slot_bytes)
{
    return GLM53F_I4F_HEADER + (int64_t)expert * slot_bytes;
}

#endif /* GLM53F_I4FILE_H */
