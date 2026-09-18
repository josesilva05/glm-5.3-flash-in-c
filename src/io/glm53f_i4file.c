/* glm53f_i4file.c - see glm53f_i4file.h. */
#define _GNU_SOURCE            /* O_DIRECT */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "glm53f_portable_io.h"   /* first: pread, O_DIRECT, posix_memalign on Windows */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "glm53f_i4file.h"
#include "glm53f_load.h"

static const char MAGIC[8] = { 'G', 'L', 'M', '5', '3', 'F', 'I', '4' };
enum { VERSION = 1, BATCH = 8 };

static uint32_t crc_table[256];

static void crc_init(void)
{
    if (crc_table[1]) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t v = i;
        for (int k = 0; k < 8; k++) v = (v & 1) ? 0xEDB88320u ^ (v >> 1) : v >> 1;
        crc_table[i] = v;
    }
}

static uint32_t crc32_of(const unsigned char *p, size_t n)
{
    uint32_t v = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) v = crc_table[(v ^ p[i]) & 0xFF] ^ (v >> 8);
    return v ^ 0xFFFFFFFFu;
}

static uint64_t fnv(uint64_t h, int64_t v)
{
    for (int i = 0; i < 8; i++) { h ^= (uint64_t)((v >> (8 * i)) & 0xFF); h *= 1099511628211ull; }
    return h;
}

/* Identifies the checkpoint a layer file was written from: the geometry and where the
 * first and last expert of the layer sit in the shards. */
static int fingerprint(const Glm53fSt *st, const Glm53fCfg *c, int layer, int64_t *slot_bytes,
                       uint64_t *fp)
{
    uint64_t h = 1469598103934665603ull;
    h = fnv(h, layer); h = fnv(h, c->n_experts); h = fnv(h, c->hidden); h = fnv(h, c->moe_inter);
    h = fnv(h, c->fp8_block_r); h = fnv(h, c->fp8_block_c); h = fnv(h, GLM53F_I4_GROUP);
    const int probe[2] = { 0, c->n_experts - 1 };
    *slot_bytes = 0;
    for (int i = 0; i < 2; i++) {
        Glm53fExpertRef r;
        if (glm53f_expert_ref(st, c, layer, probe[i], &r) != 0) return -1;
        h = fnv(h, r.shard); h = fnv(h, r.w_abs); h = fnv(h, r.w_bytes); h = fnv(h, r.s_abs);
        for (int m = 0; m < 3; m++) { h = fnv(h, r.m[m].rows); h = fnv(h, r.m[m].cols); }
        const int64_t sb = glm53f_expert_i4_slot_bytes(&r);
        if (*slot_bytes && sb != *slot_bytes) return -1;     /* experts of a layer share a shape */
        *slot_bytes = sb;
    }
    h = fnv(h, *slot_bytes);
    *fp = h;
    return 0;
}

static void layer_path(char *buf, size_t cap, const char *dir, int layer, const char *suffix)
{
    snprintf(buf, cap, "%s/layer-%02d.i4%s", dir, layer, suffix);
}

static void put32(unsigned char *p, uint32_t v) { memcpy(p, &v, 4); }
static void put64(unsigned char *p, uint64_t v) { memcpy(p, &v, 8); }
static uint32_t get32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t get64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }

/* Read and check a header; returns the descriptor (O_DIRECT) or -1 absent / -2 invalid. */
static int open_checked(const Glm53fSt *st, const Glm53fCfg *c, int layer, const char *dir,
                        int64_t *slot_out, unsigned char *hdr_out, int quiet)
{
    char path[1024];
    layer_path(path, sizeof path, dir, layer, "");
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    unsigned char *hdr = NULL;
    if (posix_memalign((void **)&hdr, GLM53F_ST_ALIGN, GLM53F_I4F_HEADER) != 0) { close(fd); return -2; }
    int64_t slot = 0;
    uint64_t fp = 0;
    const char *why = NULL;
    if (pread(fd, hdr, GLM53F_I4F_HEADER, 0) != GLM53F_I4F_HEADER)        why = "short header";
    else if (memcmp(hdr, MAGIC, 8) != 0)                                     why = "not an int4 container";
    else if (get32(hdr + 8) != VERSION)                                      why = "another container version";
    else if (get32(hdr + 40) != 1)                                           why = "incomplete";
    else if ((int)get32(hdr + 12) != layer || (int)get32(hdr + 16) != c->n_experts ||
             get32(hdr + 20) != GLM53F_I4_GROUP)                             why = "another layer or geometry";
    else if (fingerprint(st, c, layer, &slot, &fp) != 0)                     why = "the checkpoint layer cannot be resolved";
    else if ((int64_t)get64(hdr + 24) != slot || get64(hdr + 32) != fp)      why = "written from another checkpoint";
    if (why) {
        if (!quiet) fprintf(stderr, "glm53f: %s is not usable (%s); layer %d is quantised on the way in\n",
                            path, why, layer);
        glm53f_aligned_free(hdr);
        close(fd);
        return -2;
    }
    if (hdr_out) memcpy(hdr_out, hdr, GLM53F_I4F_HEADER);
    glm53f_aligned_free(hdr);
    *slot_out = slot;
    return fd;
}

int glm53f_i4file_open_layer(const Glm53fSt *st, const Glm53fCfg *c, int layer, const char *dir)
{
    int64_t slot = 0;
    return open_checked(st, c, layer, dir, &slot, NULL, 0);
}

int glm53f_i4file_write_layer(const Glm53fSt *st, const Glm53fCfg *c, int layer, const char *dir)
{
    if (c->n_experts > (GLM53F_I4F_HEADER - 48) / 4) return -1;
    crc_init();
    {
        int64_t s = 0;
        const int fd = open_checked(st, c, layer, dir, &s, NULL, 1);
        if (fd >= 0) { close(fd); return 1; }
    }
    int64_t slot = 0;
    uint64_t fp = 0;
    if (fingerprint(st, c, layer, &slot, &fp) != 0) {
        fprintf(stderr, "glm53f: layer %d has no routed experts of one shape\n", layer);
        return -1;
    }
    char part[1024], path[1024];
    layer_path(part, sizeof part, dir, layer, ".part");
    layer_path(path, sizeof path, dir, layer, "");

    Glm53fExpertRef probe;
    if (glm53f_expert_ref(st, c, layer, 0, &probe) != 0) return -1;
    const int64_t raw_bytes = glm53f_expert_slot_bytes(&probe) + GLM53F_ST_ALIGN;
    unsigned char *raw[BATCH] = { 0 }, *dst[BATCH] = { 0 }, *hdr = NULL;
    uint32_t *crc = (uint32_t *)calloc((size_t)c->n_experts, sizeof *crc);
    int ok = crc != NULL && posix_memalign((void **)&hdr, GLM53F_ST_ALIGN, GLM53F_I4F_HEADER) == 0;
    for (int i = 0; i < BATCH && ok; i++)
        ok = posix_memalign((void **)&raw[i], GLM53F_ST_ALIGN, (size_t)raw_bytes) == 0 &&
             posix_memalign((void **)&dst[i], GLM53F_ST_ALIGN, (size_t)slot) == 0;
    FILE *f = ok ? fopen(part, "wb") : NULL;
    if (!f) {
        if (ok) fprintf(stderr, "glm53f: cannot create %s\n", part);
        ok = 0;
    }
    if (ok) {
        memset(hdr, 0, GLM53F_I4F_HEADER);
        ok = fwrite(hdr, 1, GLM53F_I4F_HEADER, f) == GLM53F_I4F_HEADER;   /* placeholder */
    }
    for (int e0 = 0; ok && e0 < c->n_experts; e0 += BATCH) {
        const int n = c->n_experts - e0 < BATCH ? c->n_experts - e0 : BATCH;
        int good[BATCH] = { 0 };
        int b;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (b = 0; b < n; b++) {
            Glm53fExpertRef r;
            int64_t pad = 0;
            memset(dst[b], 0, (size_t)slot);
            if (glm53f_expert_ref(st, c, layer, e0 + b, &r) == 0 &&
                glm53f_expert_i4_slot_bytes(&r) == slot &&
                glm53f_expert_load(st, &r, raw[b], raw_bytes, &pad) == 0) {
                glm53f_expert_i4_pack(&r, c, raw[b], pad, dst[b]);
                good[b] = 1;
            }
        }
        for (b = 0; b < n && ok; b++) {
            if (!good[b]) {
                fprintf(stderr, "glm53f: layer %d expert %d could not be read\n", layer, e0 + b);
                ok = 0;
                break;
            }
            crc[e0 + b] = crc32_of(dst[b], (size_t)slot);
            ok = fwrite(dst[b], 1, (size_t)slot, f) == (size_t)slot;
            if (!ok) fprintf(stderr, "glm53f: writing %s failed (disk full?)\n", part);
        }
    }
    if (f && fclose(f) != 0) ok = 0;
    if (ok) {   /* the header last, with the completion mark */
        memcpy(hdr, MAGIC, 8);
        put32(hdr + 8, VERSION);
        put32(hdr + 12, (uint32_t)layer);
        put32(hdr + 16, (uint32_t)c->n_experts);
        put32(hdr + 20, GLM53F_I4_GROUP);
        put64(hdr + 24, (uint64_t)slot);
        put64(hdr + 32, fp);
        put32(hdr + 40, 1);
        for (int e = 0; e < c->n_experts; e++) put32(hdr + 48 + 4 * e, crc[e]);
        FILE *h = fopen(part, "r+b");
        ok = h && fwrite(hdr, 1, GLM53F_I4F_HEADER, h) == GLM53F_I4F_HEADER;
        if (h && fclose(h) != 0) ok = 0;
    }
    if (ok) {
        remove(path);
        if (rename(part, path) != 0) { fprintf(stderr, "glm53f: cannot rename %s\n", part); ok = 0; }
    }
    if (!ok) remove(part);
    for (int i = 0; i < BATCH; i++) { glm53f_aligned_free(raw[i]); glm53f_aligned_free(dst[i]); }
    glm53f_aligned_free(hdr);
    free(crc);
    return ok ? 0 : -1;
}

int glm53f_i4file_verify_layer(const Glm53fSt *st, const Glm53fCfg *c, int layer, const char *dir)
{
    crc_init();
    int64_t slot = 0;
    unsigned char hdr[GLM53F_I4F_HEADER];
    const int fd = open_checked(st, c, layer, dir, &slot, hdr, 0);
    if (fd < 0) return -1;
    unsigned char *buf = NULL;
    if (posix_memalign((void **)&buf, GLM53F_ST_ALIGN, (size_t)slot) != 0) { close(fd); return -1; }
    int bad = 0;
    for (int e = 0; e < c->n_experts; e++) {
        if (pread(fd, buf, (size_t)slot, glm53f_i4file_offset(e, slot)) != (ssize_t)slot ||
            crc32_of(buf, (size_t)slot) != get32(hdr + 48 + 4 * e))
            bad++;
    }
    glm53f_aligned_free(buf);
    close(fd);
    return bad;
}
