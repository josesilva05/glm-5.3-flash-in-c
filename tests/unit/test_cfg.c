/* test_cfg.c - the config reader accepts the released layout and REFUSES anything it would
 * otherwise have to guess.
 *
 *   test_cfg <config.json>
 *
 * The file is loaded once as-is (must pass), then re-parsed and mutated in memory one
 * field at a time; every mutation must be rejected. Each case is a way a permissive reader
 * would silently build a different model: a missing layer map, an activation or router
 * the kernels do not implement, RoPE where the engine assumes NoPE, a top-k beyond the
 * fixed routing arrays.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glm53f_cfg.h"

static char *slurp(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (!b) { fclose(f); return NULL; }
    b[fread(b, 1, (size_t)n, f)] = 0;
    fclose(f);
    return b;
}

static jval *text_cfg(jval *root)
{
    jval *t = json_get(root, "text_config");
    return t ? t : root;
}

static void drop(jval *o, const char *key)
{
    for (int i = 0; o && i < o->len; i++) if (!strcmp(o->keys[i], key)) o->keys[i] = (char *)"__dropped__";
}

static void set_num(jval *o, const char *key, double v)
{
    jval *x = json_get(o, key);
    if (x) { x->t = J_NUM; x->num = v; }
}

static void set_str(jval *o, const char *key, const char *v)
{
    jval *x = json_get(o, key);
    if (x) { x->t = J_STR; x->str = (char *)v; }
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: test_cfg <config.json>\n"); return 2; }
    char *txt = slurp(argv[1]);
    if (!txt) return 1;
    int fails = 0;

    Glm53fCfg c;
    jval *root = json_parse(txt, NULL);
    if (glm53f_cfg_load(&c, root, "unmodified")) {
        int n_mla = 0;
        for (int i = 0; i < c.n_layers; i++) n_mla += c.is_mla[i];
        printf("  PASS  unmodified config loads (%d layers, %d MLA)\n", c.n_layers, n_mla);
    } else {
        printf("  FAIL  unmodified config was rejected\n");
        fails++;
    }

    static const char *cases[] = {
        "no layer_types", "no mlp_layer_types", "hidden_act gelu", "qk_rope_head_dim 64",
        "scoring_func softmax", "n_group 4", "top-k 128", "no hc_mult", "short layer_types",
        "unknown layer type", "no linear_attn_config", "fp8 without block size",
    };
    const int ncase = (int)(sizeof cases / sizeof cases[0]);
    for (int k = 0; k < ncase; k++) {
        jval *r = json_parse(txt, NULL);
        jval *t = text_cfg(r);
        switch (k) {
        case 0:  drop(t, "layer_types"); break;
        case 1:  drop(t, "mlp_layer_types"); break;
        case 2:  set_str(t, "hidden_act", "gelu"); break;
        case 3:  set_num(t, "qk_rope_head_dim", 64); break;
        case 4:  set_str(t, "scoring_func", "softmax"); break;
        case 5:  set_num(t, "n_group", 4); break;
        case 6:  set_num(t, "num_experts_per_tok", 128); break;
        case 7:  drop(t, "hc_mult"); break;
        case 8:  { jval *lt = json_get(t, "layer_types"); if (lt) lt->len--; } break;
        case 9:  { jval *lt = json_get(t, "layer_types"); if (lt && lt->len) lt->kids[0]->str = (char *)"sliding_window"; } break;
        case 10: drop(t, "linear_attn_config"); break;
        case 11: { jval *q = json_get(r, "quantization_config"); drop(q, "weight_block_size"); } break;
        }
        printf("  ---- case: %s\n", cases[k]);
        if (glm53f_cfg_load(&c, r, cases[k])) {
            printf("  FAIL  accepted a config with %s\n", cases[k]);
            fails++;
        } else {
            printf("  PASS  rejected: %s\n", cases[k]);
        }
    }
    printf(fails ? "\nCONFIG READER: %d FAILED\n" : "\nCONFIG READER: PASS\n", fails);
    free(txt);
    return fails ? 1 : 0;
}
