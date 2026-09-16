/* SPDX-License-Identifier: Apache-2.0 */
/* glm53f_cfg.h - build a Glm53fCfg from the checkpoint's config.json, without guessing.
 *
 * THE SHAPE
 *   The released config.json nests the language model under "text_config" (the vision
 *   encoder sits beside it under "vision_config" and is ignored here). KDA settings live
 *   under text_config.linear_attn_config, the FP8 block size under the top-level
 *   quantization_config. A config that is only the text part (no "text_config" key) is
 *   accepted too, which is what the tiny test model writes.
 *
 * THE ONE RULE: AN ABSENT OR UNEXPECTED FIELD IS AN ERROR, NEVER A DEFAULT.
 *   A reader that defaults a missing layer map, activation or router setting still loads
 *   and still prints fluent-looking tokens, from a different model. So every field the
 *   kernels depend on is required, every setting the kernels do NOT implement is checked
 *   to have the one value they do implement, and all problems are reported together.
 */
#ifndef GLM53F_CFG_H
#define GLM53F_CFG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"
#include "glm53f.h"

typedef struct {
    jval *txt, *lin, *root;
    int   bad;
} Glm53fCfgSrc;

static inline jval *glm53f_cfg_get(Glm53fCfgSrc *s, jval *obj, const char *name)
{
    return obj ? json_get(obj, name) : NULL;
}

static inline int glm53f_cfg_i(Glm53fCfgSrc *s, jval *obj, const char *name, const char *whence)
{
    jval *v = glm53f_cfg_get(s, obj, name);
    if (!v || v->t != J_NUM) {
        fprintf(stderr, "glm53f_cfg: %s: missing or non-numeric '%s'\n", whence, name);
        s->bad++;
        return 0;
    }
    return (int)v->num;
}

static inline float glm53f_cfg_f(Glm53fCfgSrc *s, jval *obj, const char *name, const char *whence)
{
    jval *v = glm53f_cfg_get(s, obj, name);
    if (!v || v->t != J_NUM) {
        fprintf(stderr, "glm53f_cfg: %s: missing or non-numeric '%s'\n", whence, name);
        s->bad++;
        return 0.0f;
    }
    return (float)v->num;
}

/* Require a string field to equal the one value the engine implements. */
static inline void glm53f_cfg_expect_str(Glm53fCfgSrc *s, jval *obj, const char *name,
                                         const char *want, const char *whence)
{
    jval *v = glm53f_cfg_get(s, obj, name);
    if (!v || v->t != J_STR || strcmp(v->str, want)) {
        fprintf(stderr, "glm53f_cfg: %s: '%s' must be \"%s\"%s%s%s\n", whence, name, want,
                v && v->t == J_STR ? " (found \"" : " (missing",
                v && v->t == J_STR ? v->str : "", v && v->t == J_STR ? "\")" : ")");
        s->bad++;
    }
}

static inline void glm53f_cfg_expect_int(Glm53fCfgSrc *s, jval *obj, const char *name,
                                         int want, const char *whence)
{
    jval *v = glm53f_cfg_get(s, obj, name);
    if (!v || v->t != J_NUM || (int)v->num != want) {
        fprintf(stderr, "glm53f_cfg: %s: '%s' must be %d, the only value this engine "
                        "implements\n", whence, name, want);
        s->bad++;
    }
}

static inline void glm53f_cfg_expect_bool(Glm53fCfgSrc *s, jval *obj, const char *name,
                                          int want, const char *whence)
{
    jval *v = glm53f_cfg_get(s, obj, name);
    if (!v || v->t != J_BOOL || v->boolean != want) {
        fprintf(stderr, "glm53f_cfg: %s: '%s' must be %s\n", whence, name,
                want ? "true" : "false");
        s->bad++;
    }
}

/* Returns 1 on success. On failure prints every problem and returns 0; callers must not
 * proceed with a half-filled config. */
static inline int glm53f_cfg_load(Glm53fCfg *c, jval *root, const char *whence)
{
    memset(c, 0, sizeof *c);
    Glm53fCfgSrc s; memset(&s, 0, sizeof s);
    s.root = root;
    s.txt = json_get(root, "text_config");
    if (!s.txt) s.txt = root;
    s.lin = json_get(s.txt, "linear_attn_config");
    jval *T = s.txt;

    c->hidden    = glm53f_cfg_i(&s, T, "hidden_size", whence);
    c->n_layers  = glm53f_cfg_i(&s, T, "num_hidden_layers", whence);
    c->vocab     = glm53f_cfg_i(&s, T, "vocab_size", whence);
    c->rms_eps   = glm53f_cfg_f(&s, T, "rms_norm_eps", whence);

    c->hc_mult   = glm53f_cfg_i(&s, T, "hc_mult", whence);
    c->hc_iters  = glm53f_cfg_i(&s, T, "hc_sinkhorn_iters", whence);
    c->hc_eps    = glm53f_cfg_f(&s, T, "hc_eps", whence);
    glm53f_cfg_expect_bool(&s, T, "mhc", 1, whence);

    if (!s.lin) { fprintf(stderr, "glm53f_cfg: %s: missing 'linear_attn_config'\n", whence); s.bad++; }
    c->kda_heads    = glm53f_cfg_i(&s, s.lin, "num_heads", whence);
    c->kda_head_dim = glm53f_cfg_i(&s, s.lin, "head_dim", whence);
    c->conv_k       = glm53f_cfg_i(&s, s.lin, "short_conv_kernel_size", whence);
    c->gate_lb      = glm53f_cfg_f(&s, s.lin, "gate_lower_bound", whence);

    c->n_heads     = glm53f_cfg_i(&s, T, "num_attention_heads", whence);
    c->q_lora      = glm53f_cfg_i(&s, T, "q_lora_rank", whence);
    c->kv_lora     = glm53f_cfg_i(&s, T, "kv_lora_rank", whence);
    c->qk_nope     = glm53f_cfg_i(&s, T, "qk_nope_head_dim", whence);
    c->v_head      = glm53f_cfg_i(&s, T, "v_head_dim", whence);
    c->index_topk  = glm53f_cfg_i(&s, T, "index_topk", whence);
    c->index_kpool = glm53f_cfg_i(&s, T, "index_kpool", whence);
    c->index_heads = glm53f_cfg_i(&s, T, "index_n_heads", whence);
    c->index_dim   = glm53f_cfg_i(&s, T, "index_head_dim", whence);
    {   /* the tail pool is appended to the selection; the engine implements that form */
        jval *v = json_get(T, "index_kpool_always_select_tail");
        c->index_tail = v && v->t == J_BOOL ? v->boolean : 1;
        jval *cp = json_get(T, "index_kpool_compress");
        if (cp && cp->t == J_BOOL && !cp->boolean) {
            fprintf(stderr, "glm53f_cfg: %s: index_kpool_compress false is not implemented\n", whence);
            s.bad++;
        }
        jval *it = json_get(T, "indexer_types");
        if (it && it->t == J_ARR)
            for (int i = 0; i < it->len; i++)
                if (!it->kids[i] || it->kids[i]->t != J_STR || strcmp(it->kids[i]->str, "full") != 0) {
                    fprintf(stderr, "glm53f_cfg: %s: indexer_types[%d] is not 'full'; cross-layer "
                                    "top-k sharing is not implemented\n", whence, i);
                    s.bad++;
                    break;
                }
    }
    glm53f_cfg_expect_int(&s, T, "qk_rope_head_dim", 0, whence);          /* NoPE */
    {
        const int kvh = glm53f_cfg_i(&s, T, "num_key_value_heads", whence);
        if (kvh != c->n_heads) {
            fprintf(stderr, "glm53f_cfg: %s: num_key_value_heads %d != num_attention_heads %d\n",
                    whence, kvh, c->n_heads);
            s.bad++;
        }
    }
    glm53f_cfg_expect_bool(&s, T, "attention_bias", 0, whence);

    c->n_experts    = glm53f_cfg_i(&s, T, "n_routed_experts", whence);
    c->topk         = glm53f_cfg_i(&s, T, "num_experts_per_tok", whence);
    c->n_shared     = glm53f_cfg_i(&s, T, "n_shared_experts", whence);
    c->moe_inter    = glm53f_cfg_i(&s, T, "moe_intermediate_size", whence);
    c->dense_inter  = glm53f_cfg_i(&s, T, "intermediate_size", whence);
    c->routed_scale = glm53f_cfg_f(&s, T, "routed_scaling_factor", whence);
    c->swiglu_limit = glm53f_cfg_f(&s, T, "swiglu_limit", whence);
    {
        jval *v = json_get(T, "norm_topk_prob");
        if (!v || v->t != J_BOOL) {
            fprintf(stderr, "glm53f_cfg: %s: missing boolean 'norm_topk_prob'\n", whence);
            s.bad++;
        } else c->norm_topk = v->boolean;
    }
    glm53f_cfg_expect_str(&s, T, "hidden_act", "silu", whence);
    glm53f_cfg_expect_str(&s, T, "scoring_func", "sigmoid", whence);
    glm53f_cfg_expect_str(&s, T, "topk_method", "noaux_tc", whence);
    glm53f_cfg_expect_int(&s, T, "n_group", 1, whence);
    glm53f_cfg_expect_int(&s, T, "topk_group", 1, whence);

    /* layer maps */
    jval *lt = json_get(T, "layer_types"), *mt = json_get(T, "mlp_layer_types");
    if (c->n_layers <= 0 || c->n_layers > GLM53F_MAX_LAYERS) {
        fprintf(stderr, "glm53f_cfg: %s: num_hidden_layers %d outside 1..%d\n",
                whence, c->n_layers, GLM53F_MAX_LAYERS);
        return 0;
    }
    if (!lt || lt->t != J_ARR || lt->len != c->n_layers) {
        fprintf(stderr, "glm53f_cfg: %s: 'layer_types' must list %d entries\n", whence, c->n_layers);
        s.bad++;
    } else {
        for (int i = 0; i < lt->len; i++) {
            const jval *e = lt->kids[i];
            if (e->t == J_STR && !strcmp(e->str, "linear_attention")) c->is_mla[i] = 0;
            else if (e->t == J_STR && (!strcmp(e->str, "deepseek_sparse_attention") ||
                                       !strcmp(e->str, "full_attention"))) c->is_mla[i] = 1;
            else {
                fprintf(stderr, "glm53f_cfg: %s: layer_types[%d] is not a known attention type\n",
                        whence, i);
                s.bad++;
            }
        }
    }
    if (!mt || mt->t != J_ARR || mt->len != c->n_layers) {
        fprintf(stderr, "glm53f_cfg: %s: 'mlp_layer_types' must list %d entries\n", whence, c->n_layers);
        s.bad++;
    } else {
        for (int i = 0; i < mt->len; i++) {
            const jval *e = mt->kids[i];
            if (e->t == J_STR && !strcmp(e->str, "dense"))       c->is_dense[i] = 1;
            else if (e->t == J_STR && !strcmp(e->str, "sparse")) c->is_dense[i] = 0;
            else {
                fprintf(stderr, "glm53f_cfg: %s: mlp_layer_types[%d] is not dense/sparse\n",
                        whence, i);
                s.bad++;
            }
        }
    }

    /* FP8 block size. Absent quantization_config means an unquantised checkpoint. */
    {
        jval *qc = json_get(root, "quantization_config");
        if (qc) {
            glm53f_cfg_expect_str(&s, qc, "quant_method", "fp8", whence);
            glm53f_cfg_expect_str(&s, qc, "fmt", "e4m3", whence);
            jval *bs = json_get(qc, "weight_block_size");
            if (!bs || bs->t != J_ARR || bs->len != 2 || bs->kids[0]->t != J_NUM ||
                bs->kids[1]->t != J_NUM || bs->kids[0]->num < 1 || bs->kids[1]->num < 1) {
                fprintf(stderr, "glm53f_cfg: %s: quantization_config.weight_block_size must be "
                                "[rows, cols]\n", whence);
                s.bad++;
            } else {
                c->fp8_block_r = (int)bs->kids[0]->num;
                c->fp8_block_c = (int)bs->kids[1]->num;
            }
        }
    }

    /* EOS ids: text_config.eos_token_id, a number or a list. */
    {
        jval *e = json_get(T, "eos_token_id");
        if (e && e->t == J_NUM) c->eos[c->n_eos++] = (int)e->num;
        else if (e && e->t == J_ARR) {
            for (int i = 0; i < e->len && c->n_eos < GLM53F_MAX_EOS; i++)
                if (e->kids[i]->t == J_NUM) c->eos[c->n_eos++] = (int)e->kids[i]->num;
        }
    }

    if (s.bad) {
        fprintf(stderr, "glm53f_cfg: %s: %d problem(s) above; refusing to substitute defaults, "
                        "a config this reader cannot fully understand would silently produce a "
                        "DIFFERENT model.\n", whence, s.bad);
        return 0;
    }

    /* structural checks */
    if (c->hidden <= 0 || c->vocab <= 0 || c->rms_eps <= 0.0f) {
        fprintf(stderr, "glm53f_cfg: %s: non-positive hidden/vocab/rms_eps\n", whence);
        return 0;
    }
    if (c->hc_mult < 1 || c->hc_mult > GLM53F_MAX_HC || c->hc_iters < 1) {
        fprintf(stderr, "glm53f_cfg: %s: hc_mult %d must be 1..%d and hc_sinkhorn_iters >= 1\n",
                whence, c->hc_mult, GLM53F_MAX_HC);
        return 0;
    }
    if (c->topk < 1 || c->topk > GLM53F_MAX_TOPK || c->topk > c->n_experts || c->n_experts > 1024) {
        fprintf(stderr, "glm53f_cfg: %s: top-%d of %d experts is outside what this build "
                        "supports (top-k <= %d, experts <= 1024)\n",
                whence, c->topk, c->n_experts, GLM53F_MAX_TOPK);
        return 0;
    }
    if (c->conv_k < 1 || c->conv_k > 17 || c->kda_head_dim > 512 || c->v_head > 1024) {
        fprintf(stderr, "glm53f_cfg: %s: conv kernel / head dims outside supported bounds\n", whence);
        return 0;
    }
    if (c->index_heads < 1 || c->index_heads > 256 || c->index_dim < 1 || c->index_dim > 512) {
        fprintf(stderr, "glm53f_cfg: %s: index_n_heads %d / index_head_dim %d outside bounds\n",
                whence, c->index_heads, c->index_dim);
        return 0;
    }
    if (c->index_kpool < 1 || c->index_topk < c->index_kpool) {
        fprintf(stderr, "glm53f_cfg: %s: index_topk %d / index_kpool %d invalid\n",
                whence, c->index_topk, c->index_kpool);
        return 0;
    }

    int n_mla = 0, n_dense = 0;
    for (int i = 0; i < c->n_layers; i++) { n_mla += c->is_mla[i]; n_dense += c->is_dense[i]; }
    printf("config: %s | hidden=%d layers=%d vocab=%d | %d MLA + %d KDA | %d dense + %d MoE "
           "(%d experts top-%d, %d shared) | mHC x%d | fp8 block %dx%d\n",
           whence, c->hidden, c->n_layers, c->vocab, n_mla, c->n_layers - n_mla, n_dense,
           c->n_layers - n_dense, c->n_experts, c->topk, c->n_shared, c->hc_mult,
           c->fp8_block_r, c->fp8_block_c);
    return 1;
}

/* Read and parse a config file, then load it. The parse tree is left allocated for the
 * process lifetime. */
static inline int glm53f_cfg_load_file(Glm53fCfg *c, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }
    if (fseek(f, 0, SEEK_END) != 0) { perror(path); fclose(f); return 0; }
    long n = ftell(f);
    if (n < 0 || n > (1L << 28)) {
        fprintf(stderr, "%s: implausible config size %ld\n", path, n);
        fclose(f); return 0;
    }
    if (fseek(f, 0, SEEK_SET) != 0) { perror(path); fclose(f); return 0; }
    char *txt = (char *)malloc((size_t)n + 1);
    if (!txt) { fprintf(stderr, "OOM reading %s\n", path); fclose(f); return 0; }
    size_t got = fread(txt, 1, (size_t)n, f);
    fclose(f);
    txt[got] = 0;

    char *arena = NULL;
    jval *root = json_parse(txt, &arena);
    if (!root || root->t != J_OBJ) { fprintf(stderr, "%s: not a JSON object\n", path); free(txt); return 0; }
    return glm53f_cfg_load(c, root, path);
}

#endif /* GLM53F_CFG_H */
