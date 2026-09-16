/* glm53f_run.c - run GLM-5.3-Flash from the released checkpoint: text in, text out.
 *
 *   glm53f <model_dir> --prompt "Qual a capital do Brasil?" --gen 256
 *
 * The prompt is wrapped in the model's chat template (single user turn, generation
 * prompt, reasoning effort) unless --raw is given; --ids feeds token ids verbatim and is
 * the reproducible channel the tests use. Decoding is greedy and stops at an EOS token
 * from the config unless --no-stop is given.
 */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

#include "glm53f.h"
#include "glm53f_model.h"
#include "glm53f_mtp.h"
#include "glm53f_chat.h"
#include "glm53f_tok.h"

#ifndef GLM53F_VERSION
#define GLM53F_VERSION "2.0.0-glm"
#endif

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double peak_rss_bytes(void)
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return 0.0;
    return (double)pmc.PeakWorkingSetSize;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
#if defined(__APPLE__)
    return (double)ru.ru_maxrss;
#else
    return (double)ru.ru_maxrss * 1024.0;
#endif
#endif
}

static void json_string(FILE *f, const char *s)
{
    if (!s) { fputs("null", f); return; }
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"':  fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\n': fputs("\\n", f); break;
        case '\r': fputs("\\r", f); break;
        case '\t': fputs("\\t", f); break;
        default:
            if (*p < 0x20) fprintf(f, "\\u%04x", (unsigned)*p);
            else fputc(*p, f);
        }
    }
    fputc('"', f);
}

/* Length of the longest prefix of s[0..n) that does not end inside a UTF-8 sequence. */
static int utf8_complete(const char *s, int n)
{
    int i = n;
    int back = 0;
    while (i > 0 && back < 4 && ((unsigned char)s[i - 1] & 0xC0) == 0x80) { i--; back++; }
    if (i == 0) return n;
    const unsigned char lead = (unsigned char)s[i - 1];
    int need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
    return (back + 1 >= need) ? n : i - 1;
}

static void usage(FILE *f)
{
    fprintf(f,
"glm53f " GLM53F_VERSION ", GLM-5.3-Flash inference in portable C\n"
"\n"
"usage: glm53f <model_dir> [options]\n"
"\n"
"prompt (exactly one):\n"
"  --prompt TEXT         user message (wrapped in the chat template)\n"
"  --prompt-file PATH    the same, read from a file; use this for non-ASCII text on\n"
"                        Windows, where argv is re-encoded by the console code page\n"
"  --ids 1,2,3           raw token ids, no template, no tokenizer\n"
"  --raw                 tokenize the prompt as-is, without the chat template\n"
"  --reasoning LEVEL     max (default) | high | low, the template's Reasoning Effort\n"
"\n"
"generation:\n"
"  --gen N               tokens to generate (default 256)\n"
"  --no-stop             do not stop at EOS tokens\n"
"  --cache-gb X|auto     routed-expert cache budget in GB. auto (default) takes the\n"
"                        free RAM, keeping a fifth of the installed RAM for the system\n"
"  --prefetch N          read each MoE layer's experts and the N most likely experts\n"
"                        of the next layer (per token) in the background\n"
"                        (default 6, 0 = off; output is identical either way)\n"
"  --kv auto|expanded|compressed\n"
"                        how the MLA cache stores a position: expanded keys and values\n"
"                        (1.44 MB) or the kv_lora latent (22 KB, expanded per query).\n"
"                        auto (default) picks compressed past 2051 positions\n"
"  --experts fp8|int4    fp8 (default) streams the checkpoint own expert weights; int4\n"
"                        re-quantises them in the cache, fitting 1.8x more experts in the\n"
"                        same RAM at the cost of an APPROXIMATE output\n"
"  --gpu                 run the trunk on the CUDA devices (build with -DGLM53F_CUDA=ON);\n"
"                        routed experts stay on the CPU. The host copy of the trunk\n"
"                        is freed (~14 GB), and auto counts it as free\n"
"\n"
"diagnostics:\n"
"  --config PATH         model config (default <model_dir>/config.json)\n"
"  --tok DIR             tokenizer directory (default <model_dir>)\n"
"  --layers N            bind only the first N layers (output is NOT the model)\n"
"  --dump-logits PATH    float32 logits of the prompt's last position\n"
"  --out FILE            JSON results (default glm53f_run.json)\n"
"  --quiet               no per-step table\n"
"  --chat                interactive session: the model stays loaded and each turn\n"
"                        continues the conversation instead of re-reading it\n"
"  --ctx N               positions an interactive session may hold (default 2048,\n"
"                        or 4096 without --gpu)\n"
"  --version, --help\n"
"\n"
"Beyond index_topk + index_kpool - 1 positions (2051) the DSA indexer selects which\n"
"positions each query attends to, as the model does; --gpu is limited to that dense range.\n");
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
#ifdef _WIN32
    /* Generated text is UTF-8; without this the console renders it in the OEM code page. */
    SetConsoleOutputCP(CP_UTF8);
#endif
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(stdout); return 0; }
        if (!strcmp(argv[i], "--version")) { printf("glm53f %s\n", GLM53F_VERSION); return 0; }
    }
    if (argc < 2 || argv[1][0] == '-') { usage(stderr); return 2; }
    const char *dir = argv[1];

    const char *ids_s = NULL, *prompt_text = NULL, *prompt_file = NULL, *tok_dir = NULL;
    const char *cfg_path = NULL, *logits_path = NULL, *outp = "glm53f_run.json";
    const char *reasoning = "max";
    int gen = 256, max_layers = -1, raw = 0, no_stop = 0, quiet = 0, prefetch_n = 6, use_gpu = 0;
    int expert_i4 = 0, kv_mode = 0, chat = 0, ctx = 0;
    double cache_gb = 0.0;                       /* 0: sized from the free RAM */
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_s = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt_text = argv[++i];
        else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) prompt_file = argv[++i];
        else if (!strcmp(argv[i], "--raw")) raw = 1;
        else if (!strcmp(argv[i], "--reasoning") && i + 1 < argc) reasoning = argv[++i];
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-stop")) no_stop = 1;
        else if (!strcmp(argv[i], "--cache-gb") && i + 1 < argc) {
            const char *v = argv[++i];
            cache_gb = !strcmp(v, "auto") ? 0.0 : atof(v);
            if (cache_gb < 0.0 || (cache_gb == 0.0 && strcmp(v, "auto"))) {
                fprintf(stderr, "--cache-gb takes a size in GB or auto\n");
                return 2;
            }
        }
        else if (!strcmp(argv[i], "--prefetch") && i + 1 < argc) prefetch_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu")) use_gpu = 1;
        else if (!strcmp(argv[i], "--kv") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "expanded")) kv_mode = 1;
            else if (!strcmp(v, "compressed")) kv_mode = 2;
            else if (!strcmp(v, "auto")) kv_mode = 0;
            else { fprintf(stderr, "--kv must be auto, expanded or compressed\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "int4")) expert_i4 = 1;
            else if (!strcmp(v, "fp8")) expert_i4 = 0;
            else { fprintf(stderr, "--experts must be fp8 or int4\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
        else if (!strcmp(argv[i], "--tok") && i + 1 < argc) tok_dir = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) max_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) logits_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--chat")) chat = 1;
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else { fprintf(stderr, "unknown or incomplete option %s\n\n", argv[i]); usage(stderr); return 2; }
    }
    if (!chat && (ids_s != NULL) + (prompt_text != NULL) + (prompt_file != NULL) != 1) {
        fprintf(stderr, "exactly one of --ids, --prompt or --prompt-file is required, "
                        "or --chat for an interactive session\n");
        return 2;
    }
    if (chat && (ids_s || prompt_text || prompt_file)) {
        fprintf(stderr, "--chat starts an empty session; it takes no prompt\n");
        return 2;
    }
    if (gen < 0 || gen > GLM53F_MAX_GEN) {
        fprintf(stderr, "--gen %d is outside 0..%d\n", gen, GLM53F_MAX_GEN);
        return 2;
    }
    const char *effort = !strcmp(reasoning, "low") ? "Low" : !strcmp(reasoning, "high") ? "High"
                       : !strcmp(reasoning, "max") ? "Max" : NULL;
    if (!effort) { fprintf(stderr, "--reasoning must be max, high or low\n"); return 2; }

    /* ---- interactive session ---- */
    if (chat) {
        const int cap = ctx > 0 ? ctx : (use_gpu ? 2048 : 4096);
        Glm53fModel cm;
        Tok ctok;
        /* The session opens with the session, not with a load log. */
        glm53f_quiet = 1;
        const double t_load = now_s();
        printf("  loading the model…");
        fflush(stdout);
        glm53f_tok_load(&ctok, tok_dir ? tok_dir : dir);
        if (glm53f_model_open(&cm, dir, cfg_path, cache_gb, max_layers, cap, prefetch_n,
                              expert_i4, use_gpu, kv_mode) != 0)
            return 1;
        int on_gpu = 0;
        if (use_gpu) {
            on_gpu = glm53f_model_use_gpu(&cm, NULL, 0) == 0;
            (void)0;
            if (!on_gpu) printf("NOTE: --gpu requested but the trunk stays on the CPU.\n\n");
        }
        Glm53fChat s;
        memset(&s, 0, sizeof s);
        s.m = &cm; s.tok = &ctok; s.dir = dir;
        s.reasoning = !strcmp(reasoning, "low") ? "Low" : !strcmp(reasoning, "high") ? "High" : "Max";
        s.gen = gen; s.prefetch = prefetch_n; s.on_gpu = on_gpu; s.cache_gb = (double)cm.cache.nslot * cm.cache.slot_bytes / 1e9;
        s.cache_auto = cache_gb <= 0.0;
        s.colour = 1;
        s.load_s = now_s() - t_load;
        printf("\r%*s\r", 24, "");
        for (int L = 0; L < cm.cfg.n_layers; L++)
            if (glm53f_is_mla(&cm.cfg, L)) s.n_mla++; else s.n_kda++;
        s.ids_cap = GLM53F_MAX_PROMPT;
        s.ids = (int *)malloc((size_t)s.ids_cap * sizeof(int));
        s.out = (int *)malloc((size_t)GLM53F_MAX_GEN * sizeof(int));
        s.text_cap = GLM53F_MAX_GEN * 64 + 1;
        s.text = (char *)malloc((size_t)s.text_cap);
        s.logits = (float *)malloc((size_t)cm.cfg.vocab * sizeof(float));
        int rc = 1;
        if (s.ids && s.out && s.text && s.logits) rc = glm53f_chat_run(&s);
        free(s.ids); free(s.out); free(s.text); free(s.logits);
        glm53f_model_close(&cm);
        return rc;
    }

    /* ---- prompt ---- */
    int *prompt = (int *)malloc((size_t)GLM53F_MAX_PROMPT * sizeof(int));
    if (!prompt) return 1;
    int np = 0;
    Tok tok; int have_tok = 0;
    if (prompt_text || prompt_file) {
        glm53f_tok_load(&tok, tok_dir ? tok_dir : dir);
        have_tok = 1;
        char *user = NULL; long ulen = 0;
        if (prompt_file) user = tk_read_file(prompt_file, &ulen);
        else { ulen = (long)strlen(prompt_text); user = (char *)malloc((size_t)ulen + 1); memcpy(user, prompt_text, (size_t)ulen + 1); }
        if (!user) return 1;
        char *text;
        if (raw) {
            text = user;
        } else {
            /* chat_template.jinja for one user turn with add_generation_prompt:
             * [gMASK]<sop><|system|>Reasoning Effort: X<|user|>{msg}<|assistant|><think> */
            const size_t cap = (size_t)ulen + 256;
            text = (char *)malloc(cap);
            if (!text) return 1;
            snprintf(text, cap, "[gMASK]<sop><|system|>Reasoning Effort: %s<|user|>%s<|assistant|><think>",
                     effort, user);
            free(user);
        }
        np = tok_encode(&tok, text, (int)strlen(text), prompt, GLM53F_MAX_PROMPT);
        free(text);
    } else {
        for (const char *p = ids_s; *p && np < GLM53F_MAX_PROMPT; ) {
            prompt[np++] = (int)strtol(p, (char **)&p, 10);
            while (*p == ',' || *p == ' ') p++;
        }
    }
    if (np == 0) { fprintf(stderr, "empty prompt\n"); return 2; }
    printf("glm53f " GLM53F_VERSION " | model %s | prompt %d tokens | generating up to %d\n",
           dir, np, gen);
    /* The ids make a run reproducible with --ids; a long prompt only shows its ends. */
    printf("prompt ids:");
    if (np <= 40) {
        for (int i = 0; i < np; i++) printf("%s%d", i ? "," : " ", prompt[i]);
    } else {
        for (int i = 0; i < 12; i++) printf("%s%d", i ? "," : " ", prompt[i]);
        printf(" ... %d more ...", np - 16);
        for (int i = np - 4; i < np; i++) printf(",%d", prompt[i]);
    }
    printf("\n");

    /* ---- model ---- */
    Glm53fModel m;
    const int cap = np + gen;
    if (prefetch_n < 0 || prefetch_n > 2 * GLM53F_MAX_TOPK) {
        fprintf(stderr, "--prefetch must be 0..%d\n", 2 * GLM53F_MAX_TOPK);
        return 2;
    }
    if (glm53f_model_open(&m, dir, cfg_path, cache_gb, max_layers, cap, prefetch_n, expert_i4,
                          use_gpu, kv_mode) != 0)
        return 1;
    if (expert_i4)
        printf("NOTE: --experts int4 re-quantises routed experts in the cache: 1.8x more of them\n"
               "      fit in RAM, and the output is an APPROXIMATION of the checkpoint.\n\n");
    if (use_gpu && glm53f_model_use_gpu(&m, NULL, 0) != 0)
        printf("NOTE: --gpu requested but the trunk stays on the CPU (see the message above).\n\n");
    if (m.n_bound < m.cfg.n_layers)
        printf("NOTE: only %d of %d layers bound; the output is NOT the model's.\n\n",
               m.n_bound, m.cfg.n_layers);

    float *lg = (float *)malloc((size_t)m.cfg.vocab * sizeof(float));
    int *out = (int *)malloc((size_t)(gen + 1) * sizeof(int));
    char *textbuf = (char *)malloc((size_t)(gen + 1) * 64 + 1);
    if (!lg || !out || !textbuf) return 1;

    if (!quiet) {
        printf("%-6s %-8s %-9s %-9s %-10s %-8s\n", "STEP", "TOKEN", "SECONDS", "DISK S", "READ GB", "TOK/S");
        printf("-------------------------------------------------------\n");
    }
    int nout = 0, stopped = 0, printed = 0;
    /* GLM53F_MTP_STATS=1: run the checkpoint's MTP layer alongside decode and report how
     * often its draft equals the token the full model produced (speculative-decode study;
     * it never changes the output). Needs the lm_head on the host, so not with --gpu. */
    int mtp_on = getenv("GLM53F_MTP_STATS") && glm53f_mtp_open(&m, cap) == 0;
    int draft = -1, have_draft = 0;
    double t_mtp = 0.0;

    double t_prefill = 0.0, t_decode = 0.0, io_decode = 0.0;
    uint64_t bytes_total = 0, bg_total = 0, bg_decode = 0, fg_decode = 0;
    int next = -1;
    for (int step = 0; step < (gen > 0 ? gen : 1); step++) {
        glm53f_cache_reset_stats(&m.cache);
        const double ts = now_s();
        const int rc = step == 0 ? glm53f_model_forward(&m, prompt, np, lg, NULL)
                                 : glm53f_model_forward(&m, &next, 1, lg, NULL);
        if (rc != 0) { fprintf(stderr, "forward failed at step %d; aborting\n", step); return 1; }
        const double dt = now_s() - ts;
        if (step == 0) {
            t_prefill = dt;
            if (logits_path) {
                FILE *lf = fopen(logits_path, "wb");
                if (lf) { fwrite(lg, sizeof(float), (size_t)m.cfg.vocab, lf); fclose(lf); }
                else fprintf(stderr, "cannot write %s\n", logits_path);
            }
        } else {
            t_decode += dt;
            io_decode += m.cache.load_seconds;
        }
        bytes_total += m.cache.bytes_read;
        bg_total += m.cache.bg_bytes;
        if (step > 0) { fg_decode += m.cache.bytes_read; bg_decode += m.cache.bg_bytes; }
        if (gen == 0) break;

        int b = 0;
        for (int i = 1; i < m.cfg.vocab; i++) if (lg[i] > lg[b]) b = i;
        next = b;
        out[nout++] = next;
        if (mtp_on) {
            const double tm = now_s();
            if (have_draft) glm53f_mtp_score(&m, draft, next);
            /* Feed the positions whose following token is now known, then draft one more. */
            const int E = m.cfg.hidden;
            if (step == 0) {
                if (np > 1 && glm53f_mtp_feed(&m, m.mtp_h, prompt + 1, np - 1, NULL) != 0) mtp_on = 0;
                if (mtp_on && glm53f_mtp_feed(&m, m.mtp_h + (size_t)(np - 1) * E, &next, 1, &draft) != 0)
                    mtp_on = 0;
            } else if (glm53f_mtp_feed(&m, m.mtp_h, &next, 1, &draft) != 0) {
                mtp_on = 0;
            }
            have_draft = mtp_on;
            t_mtp += now_s() - tm;
        }
        if (!quiet)
            printf("%-6d %-8d %-9.2f %-9.2f %-10.2f %-8.3f\n", step, next, dt, m.cache.load_seconds,
                   (double)m.cache.bytes_read / 1e9, 1.0 / dt);
        int is_eos = 0;
        if (!no_stop)
            for (int e = 0; e < m.cfg.n_eos; e++) if (next == m.cfg.eos[e]) is_eos = 1;
        /* An EOS token stays in generated_ids but is not part of the text. */
        if (quiet && have_tok && !is_eos) {
            const int n = tok_decode(&tok, out, nout, textbuf, nout * 64);
            const int ok = utf8_complete(textbuf, n);
            if (ok > printed) { fwrite(textbuf + printed, 1, (size_t)(ok - printed), stdout); printed = ok; }
        }
        if (is_eos) { stopped = 1; break; }
    }
    if (quiet && have_tok) printf("\n");

    char *gtext = NULL;
    const int ntext = stopped ? nout - 1 : nout;
    if (have_tok && ntext > 0) {
        const int n = tok_decode(&tok, out, ntext, textbuf, ntext * 64);
        textbuf[n] = 0;
        gtext = textbuf;
        if (!quiet) printf("\n--- generated text ---\n%s\n----------------------\n", gtext);
    }
    const int ndec = nout > 1 ? nout - 1 : 0;
    printf("\nprefill: %d tokens in %.1f s | decode: %d tokens in %.1f s (%.2f s/token) | %s\n",
           np, t_prefill, ndec, t_decode, ndec ? t_decode / ndec : 0.0,
           stopped ? "stopped at EOS" : "reached --gen");
    if (ndec)
        printf("decode split: %.2f s/token waiting on expert reads, %.2f s/token computing\n",
               io_decode / ndec, (t_decode - io_decode) / ndec);
    if (mtp_on || t_mtp > 0.0) {
        glm53f_mtp_report(&m);
        if (ndec) printf("mtp cost: %.2f s total, %.3f s/token\n", t_mtp, t_mtp / ndec);
    }
    printf("expert bytes read: %.2f GB (%.2f GB on demand + %.2f GB read ahead) | peak RSS %.2f GB | "
           "expert drops %ld\n", (double)(bytes_total + bg_total) / 1e9, (double)bytes_total / 1e9,
           (double)bg_total / 1e9, peak_rss_bytes() / 1e9, glm53f_expert_drops);
    if (ndec)
        printf("decode reads: %.2f GB/token (%.2f on demand + %.2f read ahead)\n",
               (double)(fg_decode + bg_decode) / 1e9 / ndec, (double)fg_decode / 1e9 / ndec,
               (double)bg_decode / 1e9 / ndec);
    glm53f_cache_report(&m.cache, "last step");
    glm53f_predict_report();

    FILE *f = fopen(outp, "w");
    if (f) {
        fprintf(f, "{\"prompt_ids\":[");
        for (int i = 0; i < np; i++) fprintf(f, "%s%d", i ? "," : "", prompt[i]);
        fprintf(f, "],\"generated_ids\":[");
        for (int i = 0; i < nout; i++) fprintf(f, "%s%d", i ? "," : "", out[i]);
        fprintf(f, "],\"layers\":%d,\"stopped_at_eos\":%s,\"prefill_seconds\":%.3f,"
                   "\"decode_seconds\":%.3f,\"expert_bytes_read\":%llu,\"peak_rss_bytes\":%.0f,"
                   "\"expert_drops\":%ld,\"generated_text\":",
                m.n_bound, stopped ? "true" : "false", t_prefill, t_decode,
                (unsigned long long)bytes_total, peak_rss_bytes(), glm53f_expert_drops);
        json_string(f, gtext);
        fputs("}\n", f);
        fclose(f);
    }

    glm53f_model_close(&m);
    free(lg); free(out); free(textbuf); free(prompt);
    if (glm53f_expert_drops) {
        fprintf(stderr, "RUN INVALID: %ld routed expert load(s) failed\n", glm53f_expert_drops);
        return 4;
    }
    return 0;
}
