/* glm53f_chat.c - the interactive session: one loaded model, many turns.
 *
 * A one-shot run pays for loading the trunk and warming the expert cache every time, and
 * re-reads the whole conversation on every question. Here the model stays open: each turn
 * feeds only the new tokens, so the context the earlier turns left in the KV cache, the KDA
 * state and the DSA indexer is reused as the model itself would reuse it.
 *
 * The screen is plain text with ANSI colour: a header with the model and the settings in
 * force, the reasoning dimmed, the answer bright, and a line of per-turn numbers. Colour is
 * dropped when the output is not a terminal, so piping the session stays readable.
 */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "glm53f_chat.h"
#include "tok.h"

#define C_RESET "\033[0m"
#define C_DIM   "\033[2m"
#define C_BOLD  "\033[1m"
#define C_CYAN  "\033[36m"
#define C_GREEN "\033[32m"
#define C_GREY  "\033[90m"

static const char *cc(const Glm53fChat *s, const char *code) { return s->colour ? code : ""; }

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Length of the longest prefix of s[0..n) that does not end inside a UTF-8 sequence. */
static int utf8_complete(const char *s, int n)
{
    int i = n, back = 0;
    while (i > 0 && back < 4 && ((unsigned char)s[i - 1] & 0xC0) == 0x80) { i--; back++; }
    if (i == 0) return n;
    const unsigned char lead = (unsigned char)s[i - 1];
    const int need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
    return (back + 1 >= need) ? n : i - 1;
}

/* One line from the user as UTF-8. On Windows the console is read as UTF-16 and converted,
 * because the byte-oriented calls would hand back the OEM code page and mangle accents. */
static int read_line(char *buf, int cap)
{
#ifdef _WIN32
    const HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(in, &mode)) {
        static wchar_t w[4096];
        DWORD n = 0;
        if (!ReadConsoleW(in, w, (DWORD)(sizeof w / sizeof *w) - 1, &n, NULL)) return -1;
        w[n] = 0;
        const int got = WideCharToMultiByte(CP_UTF8, 0, w, (int)n, buf, cap - 1, NULL, NULL);
        buf[got > 0 ? got : 0] = 0;
        return got > 0 ? got : 0;
    }
#endif
    if (!fgets(buf, cap, stdin)) return -1;
    return (int)strlen(buf);
}

static void trim(char *s)
{
    int n = (int)strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ' || s[n - 1] == '\t'))
        s[--n] = 0;
}

static void header(const Glm53fChat *s)
{
    const Glm53fCfg *c = &s->m->cfg;
    printf("%s┌─ glm53f ───────────────────────────────────────────────────────────────┐%s\n",
           cc(s, C_CYAN), cc(s, C_RESET));
    printf("%s│%s %sGLM-5.3-Flash%s  %d layers · %d MLA + %d KDA · %d experts top-%d\n",
           cc(s, C_CYAN), cc(s, C_RESET), cc(s, C_BOLD), cc(s, C_RESET),
           c->n_layers, s->n_mla, s->n_kda, c->n_experts, c->topk);
    printf("%s│%s %s%s%s\n", cc(s, C_CYAN), cc(s, C_RESET), cc(s, C_GREY), s->dir, cc(s, C_RESET));
    printf("%s│%s context %d · cache %.0f GB (%d experts) · prefetch %d · %s · kv %s · experts %s\n",
           cc(s, C_CYAN), cc(s, C_RESET), s->m->cap, s->cache_gb, s->m->cache.nslot, s->prefetch,
           s->on_gpu ? "gpu" : "cpu", s->m->ckv ? "compressed" : "expanded",
           s->m->cache.i4 ? "int4" : "fp8");
    printf("%s│%s %sready in %.1f s · experts stream from disk as they are routed%s\n",
           cc(s, C_CYAN), cc(s, C_RESET), cc(s, C_GREY), s->load_s, cc(s, C_RESET));
    printf("%s└────────────────────────────────────────────────────────────────────────┘%s\n",
           cc(s, C_CYAN), cc(s, C_RESET));
    printf("%s/help for commands, /quit to leave%s\n\n", cc(s, C_GREY), cc(s, C_RESET));
}

static void help(const Glm53fChat *s)
{
    printf("%s  /reset          forget the conversation and start a new one\n"
           "  /params         the settings in force\n"
           "  /gen N          tokens to generate per answer (now %d)\n"
           "  /reasoning L    max | high | low (now %s)\n"
           "  /quit           leave%s\n\n",
           cc(s, C_GREY), s->gen, s->reasoning, cc(s, C_RESET));
}

static void params(const Glm53fChat *s)
{
    printf("%s  context %d of %d used · cache %.0f GB · prefetch %d · %s · kv %s · experts %s · "
           "reasoning %s · gen %d%s\n\n",
           cc(s, C_GREY), s->m->cached, s->m->cap, s->cache_gb, s->prefetch,
           s->on_gpu ? "gpu" : "cpu", s->m->ckv ? "compressed" : "expanded",
           s->m->cache.i4 ? "int4" : "fp8", s->reasoning, s->gen, cc(s, C_RESET));
}

/* Feed `text`, then generate until EOS or `gen` tokens, streaming what is decodable. */
static int turn(Glm53fChat *s, const char *text)
{
    Glm53fModel *m = s->m;
    const Glm53fCfg *c = &m->cfg;
    int *ids = s->ids;
    const int n = tok_encode(s->tok, text, (int)strlen(text), ids, s->ids_cap);
    if (n <= 0) { fprintf(stderr, "could not tokenise the message\n"); return -1; }
    if (m->cached + n + s->gen > m->cap) {
        printf("%s  the session holds %d positions and this turn needs %d; /reset to start over%s\n\n",
               cc(s, C_GREY), m->cap, m->cached + n + s->gen, cc(s, C_RESET));
        return 0;
    }

    /* Reading the new tokens takes seconds per position, so say so instead of going quiet. */
    printf("%s  reading %d token%s…%s", cc(s, C_GREY), n, n == 1 ? "" : "s", cc(s, C_RESET));
    fflush(stdout);
    const double t0 = now_s();
    if (glm53f_model_forward(m, ids, n, s->logits, NULL) != 0) return -1;
    const double t_prefill = now_s() - t0;
    printf("\r%*s\r", 32, "");                      /* erase that line before the answer */
    printf("%s", cc(s, C_DIM));

    int nout = 0, printed = 0, stopped = 0, in_think = 1;
    const double td = now_s();
    for (int step = 0; step < s->gen; step++) {
        int best = 0;
        for (int i = 1; i < c->vocab; i++) if (s->logits[i] > s->logits[best]) best = i;
        int is_eos = 0;
        for (int e = 0; e < c->n_eos; e++) if (best == c->eos[e]) is_eos = 1;
        if (is_eos) { stopped = 1; break; }
        s->out[nout++] = best;
        const int len = tok_decode(s->tok, s->out, nout, s->text, s->text_cap - 1);
        s->text[len > 0 ? len : 0] = 0;
        const int ok = utf8_complete(s->text, len);
        if (ok > printed) {
            /* the reasoning the model writes before </think> is dimmed, the answer is not */
            const char *close = strstr(s->text, "</think>");
            if (in_think && close) {
                const int cut = (int)(close - s->text);
                if (cut > printed) fwrite(s->text + printed, 1, (size_t)(cut - printed), stdout);
                printf("%s\n%s  ── resposta ──%s\n", cc(s, C_RESET), cc(s, C_GREY), cc(s, C_RESET));
                printed = cut + 8;
                in_think = 0;
            }
            if (ok > printed) {
                fwrite(s->text + printed, 1, (size_t)(ok - printed), stdout);
                printed = ok;
            }
            fflush(stdout);
        }
        if (step + 1 < s->gen && glm53f_model_forward(m, &best, 1, s->logits, NULL) != 0) return -1;
    }
    const double t_dec = now_s() - td;
    if (in_think) printf("%s", cc(s, C_RESET));
    printf("\n\n%s  %d tokens em %.1f s (%.2f s/token) · leitura %d em %.1f s · contexto %d/%d%s%s\n\n",
           cc(s, C_GREY), nout, t_dec, nout ? t_dec / nout : 0.0, n, t_prefill,
           m->cached, m->cap, stopped ? " · parou no EOS" : " · limite de --gen", cc(s, C_RESET));
    return 0;
}

int glm53f_chat_run(Glm53fChat *s)
{
#ifdef _WIN32
    {   /* colour needs the virtual terminal mode on Windows consoles */
        const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (GetConsoleMode(out, &mode))
            SetConsoleMode(out, mode | 0x0004 /* ENABLE_VIRTUAL_TERMINAL_PROCESSING */);
    }
#endif
    char *line = (char *)malloc(65536);
    char *text = (char *)malloc(65536 + 256);
    if (!line || !text) { free(line); free(text); return 1; }
    header(s);

    int first = 1;
    for (;;) {
        printf("%s%s you %s", cc(s, C_GREEN), cc(s, C_BOLD), cc(s, C_RESET));
        fflush(stdout);
        const int got = read_line(line, 65536);
        if (got < 0) break;                      /* end of input: leave quietly */
        trim(line);
        if (!line[0]) continue;
        if (line[0] == '/') {
            if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) break;
            if (!strcmp(line, "/help")) { help(s); continue; }
            if (!strcmp(line, "/params")) { params(s); continue; }
            if (!strcmp(line, "/reset")) {
                glm53f_model_reset(s->m);
                first = 1;
                printf("%s  conversation cleared%s\n\n", cc(s, C_GREY), cc(s, C_RESET));
                continue;
            }
            if (!strncmp(line, "/gen ", 5)) {
                const int v = atoi(line + 5);
                if (v > 0 && v <= GLM53F_MAX_GEN) s->gen = v;
                params(s);
                continue;
            }
            if (!strncmp(line, "/reasoning ", 11)) {
                const char *v = line + 11;
                if (!strcmp(v, "max") || !strcmp(v, "high") || !strcmp(v, "low")) s->reasoning = v[0] == 'm' ? "Max" : v[0] == 'h' ? "High" : "Low";
                params(s);
                continue;
            }
            printf("%s  unknown command; /help lists them%s\n\n", cc(s, C_GREY), cc(s, C_RESET));
            continue;
        }
        /* chat_template.jinja: the preamble only opens the conversation, every turn after
         * that appends its own user block, and the cache already holds what came before. */
        if (first)
            snprintf(text, 65536 + 256,
                     "[gMASK]<sop><|system|>Reasoning Effort: %s<|user|>%s<|assistant|><think>",
                     s->reasoning, line);
        else
            snprintf(text, 65536 + 256, "<|user|>%s<|assistant|><think>", line);
        first = 0;
        printf("\n");
        if (turn(s, text) != 0) { free(line); free(text); return 1; }
    }
    free(line);
    free(text);
    printf("%s  bye%s\n", cc(s, C_GREY), cc(s, C_RESET));
    return 0;
}
