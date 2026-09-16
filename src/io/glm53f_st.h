/* glm53f_st.h - safetensors reader for the GLM-5.3-Flash checkpoint.
 *
 * FORMAT
 *   [8 bytes little-endian N] [N bytes of JSON header] [tensor data]
 *   Each header entry: {"dtype": ..., "shape": [...], "data_offsets": [start, end]}
 *   data_offsets are relative to the END of the header, so the absolute file offset is
 *   8 + N + start. The released checkpoint is 62 shards and 76,108 tensors in F8_E4M3,
 *   BF16 and F32.
 *
 * WHY A HASH INDEX IS NOT OPTIONAL
 *   Every routed-expert fetch resolves six tensors (three FP8 matrices and their scale
 *   grids), and a decode step fetches up to 336 experts, so lookups must be O(1).
 *
 * WHY pread AND NOT mmap
 *   Pages read into a buffer the engine owns never become file-backed mappings, so peak
 *   RSS tracks what is actually resident rather than the ~306 GB checkpoint. (On Windows a
 *   copy-on-write mapping of every shard would also be charged against the commit limit.)
 */
#ifndef GLM53F_ST_H
#define GLM53F_ST_H

#include <stddef.h>
#include <stdint.h>

typedef enum { GLM53F_DT_UNKNOWN = 0, GLM53F_DT_U8, GLM53F_DT_BF16, GLM53F_DT_F16, GLM53F_DT_F32,
               GLM53F_DT_F8_E4M3 } Glm53fDtype;

typedef struct {
    char     *name;
    int       shard;          /* index into Glm53fSt.fd[]                      */
    Glm53fDtype   dtype;
    int       ndim;
    int64_t   shape[8];
    int64_t   off;            /* ABSOLUTE byte offset within its shard file */
    int64_t   nbytes;
} Glm53fTensor;

/* O_DIRECT alignment. Offset, length and buffer must all be multiples of this. */
#define GLM53F_ST_ALIGN 4096

typedef struct {
    int       *fd;            /* one open descriptor per shard             */
    int       *dfd;           /* the same shards opened O_DIRECT, or -1    */
    char     **path;
    int        nshard;

    Glm53fTensor  *t;             /* every tensor, in discovery order          */
    int        nt;

    int32_t   *bucket;        /* open-addressed hash, -1 empty             */
    int        nbucket;

    char      *strpool;       /* all names, one allocation                 */
    size_t     strcap, strlen_;
} Glm53fSt;

/* Open every *.safetensors in dir and index every tensor. Returns 0 on success. */
int  glm53f_st_open(Glm53fSt *s, const char *dir);
void glm53f_st_close(Glm53fSt *s);

/* O(1) lookup. Returns NULL when absent, which callers must treat as fatal: a
 * silently missing weight reads as zeros and the model still runs, plausibly wrong. */
const Glm53fTensor *glm53f_st_find(const Glm53fSt *s, const char *name);

/* Raw bytes, exactly as stored. buf must hold t->nbytes. Returns bytes read. */
int64_t glm53f_st_read(const Glm53fSt *s, const Glm53fTensor *t, void *buf);

/* Read [off, off+nbytes) from a shard with O_DIRECT, bypassing the page cache.
 *
 * WHY THIS IS NOT JUST glm53f_st_read WITH A FLAG
 *   O_DIRECT demands that the file offset, the length and the buffer address all be
 *   multiples of GLM53F_ST_ALIGN. An expert run starts wherever the checkpoint put it, so
 *   the read is WIDENED outward to the enclosing aligned window and the caller is told
 *   where the payload actually begins inside buf. buf must therefore hold
 *   nbytes + 2*GLM53F_ST_ALIGN and be page aligned (posix_memalign).
 *
 *   Worth it because a streamed expert is read once and evicted: the page cache can only
 *   copy it twice and push out something useful. Under a 32 GB cgroup cap the buffered
 *   path measured 1,247 MB/s on a disk that does 6,400.
 *
 * Returns payload bytes available, and sets *payload_off. Falls back to a buffered read
 * at offset 0 (setting *payload_off = 0) when O_DIRECT is not available. */
int64_t glm53f_st_read_aligned(const Glm53fSt *s, int shard, int64_t off, int64_t nbytes,
                           void *buf, int64_t bufcap, int64_t *payload_off);

/* Read and widen to float32. Handles F32 (memcpy), BF16 (shift left 16), F16, and
 * U8 (raw byte value, for callers that want the quantised codes as numbers).
 * out must hold t->nbytes/elem_size floats. */
int64_t glm53f_st_read_f32(const Glm53fSt *s, const Glm53fTensor *t, float *out);

/* Elements in a tensor, and bytes per element for its dtype. */
int64_t glm53f_st_numel(const Glm53fTensor *t);
int     glm53f_st_elemsize(Glm53fDtype d);

/* bf16 -> f32 is a pure bit shift: bf16 IS the top 16 bits of an f32. No table, no
 * rounding, and unlike f16 there is no exponent rebias. */
static inline float glm53f_bf16_to_f32(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

#endif /* GLM53F_ST_H */
