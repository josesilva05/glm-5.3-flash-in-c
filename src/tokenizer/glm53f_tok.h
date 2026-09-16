/*
 * glm53f_tok.h, construct the GLM-5.3-Flash tokenizer directly from the checkpoint files.
 *
 * PURPOSE
 *   The vendored BPE implementation (third_party/tok.h) is driven by a HuggingFace
 *   tokenizer.json with a merges list. This loader instead reads the two files the
 *   checkpoint ships alongside it:
 *
 *     tiktoken.model          154,820 lines of "base64(token_bytes) rank"
 *     tokenizer_config.json   the added/special tokens, under added_tokens_decoder
 *
 *   and populates the tok.h structures from them, which keeps the engine text-in/text-out
 *   with no conversion step. Encoding and decoding remain entirely in tok.h.
 *
 * THREE INVARIANTS, EACH SILENT WHEN VIOLATED
 *   1. The vocabulary is keyed by the GPT-2 BYTE-LEVEL string (each byte mapped to a
 *      printable codepoint), not by raw token bytes. tiktoken.model supplies RAW bytes, so
 *      tk_build_bytemap() must run before any key is built; otherwise nothing hits and
 *      every piece degrades to single bytes.
 *   2. rankbpe must be 1. There is no merges list; tiktoken merges the adjacent pair whose
 *      CONCATENATION has the lowest id.
 *   3. The pre-tokenizer is the cl100k family (o200k = 0), which is what the
 *      checkpoint's tokenizer.json specifies.
 *
 * VERIFICATION
 *   tools/tok_parity_glm.py compares this loader token-for-token against the official
 *   tokenizer (Portuguese accents, code, CJK, emoji and ZWJ sequences, whitespace runs).
 */
#ifndef GLM53F_TOK_H
#define GLM53F_TOK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"

/* tok.h is a header-only library, vendored with the changes listed in NOTICE. Every function in it
 * has file scope, so a translation unit that uses only part of the API draws
 * -Wunused-function for the remainder.
 *
 * The suppression is scoped to this include rather than added to CFLAGS. A global
 * suppression would also silence unused-function warnings in first-party code, where
 * they indicate genuine dead code and should still fail the build. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "tok.h"
#pragma GCC diagnostic pop

/* Upper bound on token ids. GLM-5.3-Flash ranks occupy [0, 154820) and its added tokens
 * sit just above, inside text_config.vocab_size = 154880; ids are range-checked against
 * this bound as they are read rather than assumed. */
#define GLM53F_TOK_VOCAB 154880

/* ---------------------------------------------------------------- base64 ---- */
/* Standard alphabet, '=' padded. Returns decoded length, or -1 on a malformed group.
 * out must hold at least 3*((n+3)/4) bytes. */
static inline int glm53f_b64(const char *in, int n, unsigned char *out)
{
    static signed char t[256];
    static int init = 0;
    if (!init) {
        for (int i = 0; i < 256; i++) t[i] = -1;
        const char *A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) t[(unsigned char)A[i]] = (signed char)i;
        init = 1;
    }
    int o = 0, acc = 0, bits = 0;
    for (int i = 0; i < n; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '=') break;
        signed char v = t[c];
        if (v < 0) return -1;
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) { bits -= 8; out[o++] = (unsigned char)((acc >> bits) & 0xFF); }
    }
    return o;
}

/* Raw token bytes -> the byte-level string tok.h hashes on. Worst case two output
 * bytes per input byte (the substituted codepoints are 256..511, two UTF-8 bytes),
 * so `out` must hold 2*n+1. Returns the string length; NUL-terminates. */
static inline int glm53f_bytelevel(const Tok *T, const unsigned char *b, int n, char *out)
{
    int o = 0;
    for (int i = 0; i < n; i++) {
        memcpy(out + o, T->byte2str[b[i]], (size_t)T->byte2cp_len[b[i]]);
        o += T->byte2cp_len[b[i]];
    }
    out[o] = 0;
    return o;
}

/* ------------------------------------------------------------------ load ---- */
/* files_dir is the directory holding tiktoken.model and tokenizer_config.json
 * (the checkpoint directory). Exits on any malformed input: a tokenizer that loads
 * half-correctly is worse than one that refuses, because the failure surfaces as
 * subtly wrong text hundreds of tokens later. */
static inline void glm53f_tok_load(Tok *T, const char *files_dir)
{
    memset(T, 0, sizeof *T);

    /* Order matters: byte2str must exist before any vocab key is built (invariant 1). */
    tk_build_bytemap(T);
    T->o200k   = 0;   /* cl100k pre-tokenizer family (invariant 3) */
    T->rankbpe = 1;   /* invariant 2 */

    T->n_ids     = GLM53F_TOK_VOCAB;
    T->id2str    = (char **)calloc((size_t)T->n_ids, sizeof(char *));
    T->id_added  = (int *)calloc((size_t)T->n_ids, sizeof(int));
    T->id_special= (int *)calloc((size_t)T->n_ids, sizeof(int));
    if (!T->id2str || !T->id_added || !T->id_special) {
        fprintf(stderr, "glm53f_tok: OOM sizing %d ids\n", T->n_ids); exit(1);
    }

    /* ---- ranks ---- */
    char path[4096];
    snprintf(path, sizeof path, "%s/tiktoken.model", files_dir);
    long nbuf = 0;
    char *buf = tk_read_file(path, &nbuf);      /* tok.h helper; exits if absent */

    /* Power-of-two capacity, ~2x load factor, as tok_load does. */
    int vc = 1; while (vc < 154880 * 2) vc <<= 1;
    hm_init(&T->vocab, vc);

    int nrank = 0, maxrank = -1;
    long i = 0;
    while (i < nbuf) {
        long ls = i;
        while (i < nbuf && buf[i] != '\n') i++;
        long le = i;
        if (i < nbuf) i++;                       /* step over '\n' */
        if (le > ls && buf[le - 1] == '\r') le--;
        if (le <= ls) continue;                  /* blank line */

        /* split on the single space: "<base64> <rank>" */
        long sp = ls;
        while (sp < le && buf[sp] != ' ') sp++;
        if (sp >= le) {
            fprintf(stderr, "glm53f_tok: tiktoken.model line %d has no rank field\n", nrank);
            exit(1);
        }
        int blen = (int)(sp - ls);
        int rank = atoi(buf + sp + 1);
        if (rank < 0 || rank >= T->n_ids) {
            fprintf(stderr, "glm53f_tok: rank %d out of range at line %d\n", rank, nrank);
            exit(1);
        }

        unsigned char raw[2048];
        if (blen > (int)(sizeof(raw) * 4 / 3 + 4)) {
            fprintf(stderr, "glm53f_tok: implausibly long token at rank %d\n", rank); exit(1);
        }
        int rn = glm53f_b64(buf + ls, blen, raw);
        if (rn < 0) { fprintf(stderr, "glm53f_tok: bad base64 at rank %d\n", rank); exit(1); }

        char *key = (char *)malloc((size_t)2 * rn + 1);
        if (!key) { fprintf(stderr, "glm53f_tok: OOM on token %d\n", rank); exit(1); }
        int kl = glm53f_bytelevel(T, raw, rn, key);

        hm_put(&T->vocab, key, kl, rank);
        T->id2str[rank] = key;                   /* decode reverses this via cp2byte */
        if (rank > maxrank) maxrank = rank;
        nrank++;
    }
    free(buf);
    if (nrank == 0) { fprintf(stderr, "glm53f_tok: tiktoken.model is empty\n"); exit(1); }

    /* ---- specials ---- */
    snprintf(path, sizeof path, "%s/tokenizer_config.json", files_dir);
    long ncfg = 0;
    char *cfg = tk_read_file(path, &ncfg);
    char *arena = NULL;
    jval *root = json_parse(cfg, &arena);
    jval *adt  = json_get(root, "added_tokens_decoder");
    if (!adt) {
        fprintf(stderr, "glm53f_tok: tokenizer_config.json has no added_tokens_decoder\n");
        exit(1);
    }

    T->nsp = adt->len;
    T->sp  = (Special *)calloc((size_t)(T->nsp ? T->nsp : 1), sizeof(Special));
    if (!T->sp) { fprintf(stderr, "glm53f_tok: OOM on %d specials\n", T->nsp); exit(1); }

    for (int k = 0; k < adt->len; k++) {
        /* keys are the ids, as strings: {"154820": {"content": "<|endoftext|>", ...}} */
        int id = atoi(adt->keys[k]);
        jval *e  = adt->kids[k];
        jval *jc = json_get(e, "content");
        if (!jc || jc->t != J_STR || !jc->str) {
            fprintf(stderr, "glm53f_tok: added token %s has no content\n", adt->keys[k]);
            exit(1);
        }
        if (id < 0 || id >= T->n_ids) {
            fprintf(stderr, "glm53f_tok: added token id %d out of range\n", id); exit(1);
        }
        T->sp[k].str = jc->str;                  /* json_parse strings are independent */
        T->sp[k].len = (int)strlen(jc->str);
        T->sp[k].id  = id;
        T->id2str[id]   = jc->str;               /* added tokens decode literally */
        T->id_added[id] = 1;
        jval *sf = json_get(e, "special");
        if (sf && sf->t == J_BOOL && sf->boolean) T->id_special[id] = 1;
    }
    /* longest match first, so "<|end_of_msg|>" wins over any prefix of it */
    qsort(T->sp, (size_t)T->nsp, sizeof(Special), cmp_sp_len);

    fprintf(stderr, "[TOK] %d ranks (max id %d) + %d added tokens\n", nrank, maxrank, T->nsp);
}

#endif /* GLM53F_TOK_H */
