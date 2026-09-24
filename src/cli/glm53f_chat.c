/* glm53f_chat.c - localcode: the interactive session, one loaded model, many turns.
 *
 * A one-shot run pays for loading the trunk and warming the expert cache every time, and
 * re-reads the whole conversation on every question. Here the model stays open: each turn
 * feeds only the new tokens, so the context the earlier turns left in the KV cache, the KDA
 * state and the DSA indexer is reused as the model itself would reuse it.
 *
 * This file is the session: commands, the chat template, one turn reported as events, and
 * the Markdown transcript behind /save. On a terminal the full-screen front end in
 * glm53f_tui.c shows it; otherwise (a pipe, NO_COLOR-style plain use) the line-by-line
 * front end at the bottom of this file does.
 */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "glm53f_chat.h"
#include "tok.h"

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void duration(char *buf, size_t cap, double sec)
{
    if (sec < 60.0) snprintf(buf, cap, "%.1fs", sec);
    else            snprintf(buf, cap, "%dm %02ds", (int)(sec / 60.0), (int)sec % 60);
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

static char *dup_str(const char *p)
{
    const size_t n = strlen(p) + 1;
    char *d = (char *)malloc(n);
    if (d) memcpy(d, p, n);
    return d;
}

static char *fmt_str(const char *fmt, const char *a, const char *b)
{
    const int n = snprintf(NULL, 0, fmt, a, b);
    char *d = (char *)malloc((size_t)n + 1);
    if (d) snprintf(d, (size_t)n + 1, fmt, a, b);
    return d;
}

/* ----------------------------------------------------------------- transcript ---- */

/* Append to the Markdown transcript; a failed allocation only costs the transcript. */
static void log_add(Glm53fChat *s, const char *p, size_t n)
{
    if (s->log_len + n + 1 > s->log_cap) {
        size_t cap = s->log_cap ? s->log_cap : 65536;
        while (s->log_len + n + 1 > cap) cap *= 2;
        char *g = (char *)realloc(s->log, cap);
        if (!g) return;
        s->log = g;
        s->log_cap = cap;
    }
    memcpy(s->log + s->log_len, p, n);
    s->log_len += n;
    s->log[s->log_len] = 0;
}

static void log_str(Glm53fChat *s, const char *p) { log_add(s, p, strlen(p)); }

/* One exchange: the message, the reasoning folded away, the answer. */
static void log_turn(Glm53fChat *s, const char *msg, const char *out, int len, const char *stats)
{
    log_str(s, "## Você\n\n");
    log_str(s, msg);
    log_str(s, "\n\n## " GLM53F_CHAT_MODEL_NAME "\n\n");
    const char *close = NULL;
    for (int i = 0; i + 8 <= len; i++)
        if (!memcmp(out + i, "</think>", 8)) { close = out + i; break; }
    const int think = close ? (int)(close - out) : len;
    if (think > 0) {
        log_str(s, "<details><summary>Raciocínio</summary>\n\n");
        log_add(s, out, (size_t)think);
        log_str(s, "\n\n</details>\n\n");
    }
    if (close) log_add(s, close + 8, (size_t)(len - think - 8));
    else       log_str(s, "*(a resposta não começou antes do fim da geração)*");
    log_str(s, "\n\n<sub>");
    log_str(s, stats);
    log_str(s, "</sub>\n\n");
}

static char *save(Glm53fChat *s, const char *arg)
{
    char name[512], msg[700];
    if (arg && *arg) {
        snprintf(name, sizeof name, "%s", arg);
    } else {
        const time_t now = time(NULL);
        char stamp[32];
        strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", localtime(&now));
        snprintf(name, sizeof name, "glm53f-chat-%s.md", stamp);
    }
    if (!s->log_len) return dup_str("nothing to save yet");
    FILE *f = fopen(name, "wb");
    if (!f) { snprintf(msg, sizeof msg, "cannot write %s", name); return dup_str(msg); }
    fprintf(f, "# Conversa com " GLM53F_CHAT_MODEL_NAME "\n\n<sub>localcode · reasoning %s · %s · kv %s · experts %s</sub>\n\n",
            s->reasoning, s->on_gpu ? "gpu" : "cpu", s->m->ckv ? "compressed" : "expanded",
            s->m->cache.i4 ? "int4 (approximate)" : "fp8");
    const int ok = fwrite(s->log, 1, s->log_len, f) == s->log_len;
    if (fclose(f) != 0 || !ok) { snprintf(msg, sizeof msg, "writing %s failed", name); return dup_str(msg); }
    snprintf(msg, sizeof msg, "saved %s", name);
    return dup_str(msg);
}

/* ------------------------------------------------------------------- commands ---- */

const Glm53fCommand glm53f_commands[] = {
    { "/file",      "PATH [question]", "send a file's text, then the question (a path with spaces in quotes)", 2 },
    { "/reasoning", "max|high|low",    "Reasoning Effort for the next turns", 2 },
    { "/gen",       "N",               "tokens to generate per answer", 2 },
    { "/thinking",  "",                "show or fold the model's reasoning", 0 },
    { "/stats",     "",                "measured speed, disk reads and expert cache of this session", 0 },
    { "/params",    "",                "the settings in force", 0 },
    { "/copy",      "",                "put the last answer on the clipboard", 0 },
    { "/save",      "[FILE]",          "write the conversation as Markdown (default glm53f-chat-DATE.md)", 1 },
    { "/reset",     "",                "forget the conversation and start a new one", 0 },
    { "/help",      "",                "the commands and keys", 0 },
    { "/quit",      "",                "leave", 0 },
};
const int glm53f_ncommands = (int)(sizeof glm53f_commands / sizeof *glm53f_commands);

static const char KEYS[] =
    "/ opens the menu: up, down to choose, tab to complete, enter to run, esc to close\n"
    "esc, ctrl+c             stop the answer being written; ctrl+c on an empty box leaves\n"
    "pgup, pgdn, wheel       scroll the conversation; end follows it again\n"
    "up, down                earlier messages";

static char *help_note(void)
{
    char buf[3072];
    int n = 0;
    for (int i = 0; i < glm53f_ncommands && n < (int)sizeof buf; i++) {
        const Glm53fCommand *c = &glm53f_commands[i];
        char head[64];
        snprintf(head, sizeof head, "%s%s%s", c->name, *c->args ? " " : "", c->args);
        n += snprintf(buf + n, sizeof buf - (size_t)n, "%-24s %s\n", head, c->help);
    }
    if (n < (int)sizeof buf) snprintf(buf + n, sizeof buf - (size_t)n, "%s", KEYS);
    return dup_str(buf);
}

void glm53f_chat_command_value(const Glm53fChat *s, const Glm53fCommand *c, char *buf, size_t cap)
{
    const char *n = c->name;
    buf[0] = 0;
    if (!strcmp(n, "/reasoning"))     snprintf(buf, cap, "%s", s->reasoning);
    else if (!strcmp(n, "/gen"))      snprintf(buf, cap, "%d", s->gen);
    else if (!strcmp(n, "/thinking")) snprintf(buf, cap, "%s", s->show_thinking ? "shown" : "folded");
    else if (!strcmp(n, "/stats") && s->last_spt > 0.0) snprintf(buf, cap, "%.2f s/token", s->last_spt);
    else if (!strcmp(n, "/copy"))     snprintf(buf, cap, "%s", s->last_answer ? "last answer" : "nothing yet");
    else if (!strcmp(n, "/save") && s->log_len) snprintf(buf, cap, "%.1f KB", (double)s->log_len / 1024.0);
    else if (!strcmp(n, "/reset"))    snprintf(buf, cap, "%d/%d", s->m->cached, s->m->cap);
    else if (!strcmp(n, "/file"))     snprintf(buf, cap, "%d left", s->m->cap - s->m->cached);
}

/* A bar per answer, higher is slower: how the speed moved as the cache warmed up. */
static void spark(const Glm53fChat *s, char *buf, size_t cap)
{
    static const char *bars[] = { "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█" };
    double lo = 1e30, hi = 0.0;
    for (int i = 0; i < s->n_spt; i++) {
        if (s->spt_hist[i] < lo) lo = s->spt_hist[i];
        if (s->spt_hist[i] > hi) hi = s->spt_hist[i];
    }
    size_t n = 0;
    buf[0] = 0;
    for (int i = 0; i < s->n_spt && n + 4 < cap; i++) {
        const int k = hi > lo ? (int)(7.0 * (s->spt_hist[i] - lo) / (hi - lo) + 0.5) : 3;
        n += (size_t)snprintf(buf + n, cap - n, "%s", bars[k]);
    }
}

static char *stats_note(const Glm53fChat *s)
{
    const Glm53fCache *c = &s->m->cache;
    char buf[1600], d1[32], d2[32], sp[80];
    int n = 0;
    if (!s->dec_tok) {
        n += snprintf(buf + n, sizeof buf - (size_t)n, "nothing measured yet: the first answer fills this in\n");
    } else {
        const double spt = s->dec_s / (double)s->dec_tok;
        duration(d1, sizeof d1, s->dec_s);
        duration(d2, sizeof d2, s->pre_s);
        spark(s, sp, sizeof sp);
        n += snprintf(buf + n, sizeof buf - (size_t)n,
                      "writing   %ld tokens in %s over %d answer%s · %.2f s/token (%.2f tokens/s) %s\n",
                      s->dec_tok, d1, s->turns, s->turns == 1 ? "" : "s", spt, 1.0 / spt, sp);
        n += snprintf(buf + n, sizeof buf - (size_t)n, "reading   %ld prompt tokens in %s · %.2f s/token\n",
                      s->pre_tok, d2, s->pre_tok ? s->pre_s / (double)s->pre_tok : 0.0);
        const double wait = s->dec_wait_s / (double)s->dec_tok;
        const double reqs = s->dec_reqs ? (double)s->dec_reqs : 1.0;
        const double ahead = (double)s->dec_ahead, demand = (double)s->dec_demand;
        const double ram = reqs - ahead - demand > 0.0 ? reqs - ahead - demand : 0.0;
        n += snprintf(buf + n, sizeof buf - (size_t)n,
                      "experts   %.0f per token: %.0f%% from RAM, %.0f%% read ahead from the SSD, "
                      "%.0f%% read on demand · %.2f GB read per token\n",
                      (double)s->dec_reqs / (double)s->dec_tok, 100.0 * ram / reqs, 100.0 * ahead / reqs,
                      100.0 * demand / reqs, s->dec_read_gb / (double)s->dec_tok);
        n += snprintf(buf + n, sizeof buf - (size_t)n, "waiting   %.2f s/token on the disk, %.0f%% of a token\n",
                      wait, spt > 0.0 ? 100.0 * wait / spt : 0.0);
        /* the finding the engine is built around (PERFORMANCE.md): decode time follows bytes read */
        n += snprintf(buf + n, sizeof buf - (size_t)n, "limit     %s\n",
                      wait > 0.5 * spt ? "the disk: tokens go as fast as experts arrive; more cache or int4 experts read less"
                                       : "compute: the experts mostly come from RAM");
    }
    snprintf(buf + n, sizeof buf - (size_t)n,
             "engine    cache %.1f GB, %d experts %s · prefetch %d · trunk on %s · context %d of %d",
             s->cache_gb, c->nslot, c->i4 ? "int4 (approximate)" : "fp8 (exact)", s->prefetch,
             s->on_gpu ? "the gpus" : "the cpu", s->m->cached, s->m->cap);
    return dup_str(buf);
}

/* The clipboard: the Windows one, or OSC 52, which most terminals pass to theirs. */
static char *copy_answer(const Glm53fChat *s)
{
    if (!s->last_answer || !*s->last_answer) return dup_str("no answer to copy yet");
    const char *t = s->last_answer;
    const size_t n = strlen(t);
#ifdef _WIN32
    const int wn = MultiByteToWideChar(CP_UTF8, 0, t, (int)n, NULL, 0);
    HGLOBAL h = GlobalAlloc(GMEM_MOVEABLE, ((size_t)wn + 1) * sizeof(wchar_t));
    if (!h) return dup_str("could not copy: out of memory");
    wchar_t *w = (wchar_t *)GlobalLock(h);
    MultiByteToWideChar(CP_UTF8, 0, t, (int)n, w, wn);
    w[wn] = 0;
    GlobalUnlock(h);
    if (!OpenClipboard(NULL)) { GlobalFree(h); return dup_str("could not open the clipboard"); }
    EmptyClipboard();
    if (!SetClipboardData(CF_UNICODETEXT, h)) { CloseClipboard(); GlobalFree(h); return dup_str("could not copy"); }
    CloseClipboard();
#else
    static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    fputs("\033]52;c;", stdout);
    for (size_t i = 0; i < n; i += 3) {
        const unsigned v = (unsigned)(unsigned char)t[i] << 16 |
                           (i + 1 < n ? (unsigned)(unsigned char)t[i + 1] << 8 : 0) |
                           (i + 2 < n ? (unsigned)(unsigned char)t[i + 2] : 0);
        const char q[4] = { b64[v >> 18 & 63], b64[v >> 12 & 63],
                            i + 1 < n ? b64[v >> 6 & 63] : '=', i + 2 < n ? b64[v & 63] : '=' };
        fwrite(q, 1, 4, stdout);
    }
    fputs("\a", stdout);
    fflush(stdout);
#endif
    char msg[96];
    snprintf(msg, sizeof msg, "copied the last answer (%zu byte%s)", n, n == 1 ? "" : "s");
    return dup_str(msg);
}

static char *params_note(const Glm53fChat *s)
{
    char line[400];
    snprintf(line, sizeof line, "context %d of %d used · cache %.1f GB · prefetch %d · %s · kv %s · "
             "experts %s · reasoning %s · thinking %s · gen %d",
             s->m->cached, s->m->cap, s->cache_gb, s->prefetch, s->on_gpu ? "gpu" : "cpu",
             s->m->ckv ? "compressed" : "expanded", s->m->cache.i4 ? "int4" : "fp8", s->reasoning,
             s->show_thinking ? "shown" : "folded", s->gen);
    return dup_str(line);
}

void glm53f_chat_input_free(Glm53fInput *in)
{
    free(in->prompt);
    free(in->shown);
    free(in->note);
    memset(in, 0, sizeof *in);
}

void glm53f_chat_input(Glm53fChat *s, const char *raw, Glm53fInput *in)
{
    memset(in, 0, sizeof *in);
    while (*raw == ' ' || *raw == '\t') raw++;
    size_t len = strlen(raw);
    while (len > 0 && (raw[len - 1] == '\n' || raw[len - 1] == '\r' || raw[len - 1] == ' ' || raw[len - 1] == '\t'))
        len--;
    if (!len) return;
    char *line = (char *)malloc(len + 1);
    if (!line) return;
    memcpy(line, raw, len);
    line[len] = 0;

    if (line[0] != '/') {
        in->kind = GLM53F_IN_MESSAGE;
        in->prompt = line;
        in->shown = dup_str(line);
        return;
    }
    in->kind = GLM53F_IN_NOTE;
    if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) {
        in->kind = GLM53F_IN_QUIT;
    } else if (!strcmp(line, "/help")) {
        in->note = help_note();
    } else if (!strcmp(line, "/stats")) {
        in->note = stats_note(s);
    } else if (!strcmp(line, "/copy")) {
        in->note = copy_answer(s);
    } else if (!strcmp(line, "/params")) {
        in->note = params_note(s);
    } else if (!strcmp(line, "/thinking")) {
        s->show_thinking = !s->show_thinking;
        in->note = dup_str(s->show_thinking ? "reasoning shown as it is written" : "reasoning folded into one line");
    } else if (!strcmp(line, "/reset") || !strcmp(line, "/clear") || !strcmp(line, "/new")) {
        glm53f_model_reset(s->m);
        s->first = 1;
        s->log_len = 0;                          /* a new conversation, a new transcript */
        in->kind = GLM53F_IN_RESET;
        in->note = dup_str("conversation cleared");
    } else if (!strcmp(line, "/save") || !strncmp(line, "/save ", 6)) {
        in->note = save(s, line[5] ? line + 6 : NULL);
    } else if (!strncmp(line, "/gen ", 5)) {
        const int v = atoi(line + 5);
        if (v > 0 && v <= GLM53F_MAX_GEN) s->gen = v;
        in->note = params_note(s);
    } else if (!strncmp(line, "/reasoning ", 11)) {
        const char *v = line + 11;
        if (!strcmp(v, "max") || !strcmp(v, "high") || !strcmp(v, "low")) {
            s->reasoning = v[0] == 'm' ? "Max" : v[0] == 'h' ? "High" : "Low";
            in->note = params_note(s);
        } else {
            in->note = dup_str("/reasoning takes max, high or low");
        }
    } else if (!strncmp(line, "/file ", 6)) {
        /* /file PATH [question]; a path with spaces goes in quotes */
        char path[1024];
        const char *p = line + 6, *q;
        while (*p == ' ') p++;
        if (*p == '"') { p++; q = strchr(p, '"'); } else { q = strchr(p, ' '); }
        size_t pl = q ? (size_t)(q - p) : strlen(p);
        if (pl >= sizeof path) pl = sizeof path - 1;
        memcpy(path, p, pl);
        path[pl] = 0;
        const char *question = q ? q + (*q == '"' ? 1 : 0) : "";
        while (*question == ' ') question++;
        FILE *f = fopen(path, "rb");
        if (!f) {
            in->note = fmt_str("cannot open %s%s", path, "");
        } else {
            fclose(f);
            long flen = 0;
            char *content = tk_read_file(path, &flen);
            const size_t bl = (size_t)flen + strlen(question) + 4;
            in->prompt = (char *)malloc(bl);
            if (in->prompt) {
                snprintf(in->prompt, bl, "%s%s%s", content, *question ? "\n\n" : "", question);
                in->kind = GLM53F_IN_MESSAGE;
                in->shown = fmt_str(*question ? "📄 %s\n\n%s" : "📄 %s%s", path, question);
                char note[1200];
                snprintf(note, sizeof note, "%s · %ld bytes", path, flen);
                in->note = dup_str(note);
            }
            free(content);
        }
    } else {
        in->note = dup_str("unknown command; /help lists them");
    }
    free(line);
}

void glm53f_chat_gauge(const Glm53fChat *s, char *buf, size_t cap)
{
    snprintf(buf, cap, "context %d/%d (%d%%)", s->m->cached, s->m->cap,
             s->m->cap ? (int)(100.0 * s->m->cached / s->m->cap + 0.5) : 0);
}

/* ----------------------------------------------------------------------- turn ---- */

/* The expert cache's counters at one moment (they only grow in a session). */
typedef struct { uint64_t hits, misses, ahead, bytes; double wait; } CacheSnap;

static void cache_snap(Glm53fCache *c, CacheSnap *o)
{
    pthread_mutex_lock(&c->mu);
    o->hits = c->hits;
    o->misses = c->misses;
    o->ahead = c->prefetch_reads + c->bg_used;   /* hits that the disk delivered just before */
    o->bytes = c->bytes_read + c->bg_bytes;
    o->wait = c->load_seconds;
    pthread_mutex_unlock(&c->mu);
}

static volatile sig_atomic_t g_stop = 0;

void glm53f_chat_stop(void) { g_stop = 1; }

int glm53f_chat_turn(Glm53fChat *s, const Glm53fInput *in, Glm53fTurnSink *k)
{
    Glm53fModel *m = s->m;
    const Glm53fCfg *c = &m->cfg;
    char line[320];

    /* chat_template.jinja: the preamble only opens the conversation, every turn after that
     * appends its own user block, and the cache already holds what came before. */
    const size_t tl = strlen(in->prompt) + 256;
    char *text = (char *)malloc(tl);
    if (!text) { k->done(k, "out of memory", 2); return 0; }
    if (s->first)
        snprintf(text, tl, "[gMASK]<sop><|system|>Reasoning Effort: %s<|user|>%s<|assistant|><think>",
                 s->reasoning, in->prompt);
    else
        snprintf(text, tl, "<|user|>%s<|assistant|><think>", in->prompt);
    const int n = tok_encode(s->tok, text, (int)strlen(text), s->ids, s->ids_cap);
    free(text);
    if (n <= 0) { k->done(k, "could not tokenise the message (is it longer than the session?)", 2); return 0; }
    if (m->cached + n + s->gen > m->cap) {
        snprintf(line, sizeof line, "the session holds %d positions and this turn needs %d; "
                 "/reset to start over, or a smaller /gen", m->cap, m->cached + n + s->gen);
        k->done(k, line, 2);
        return 0;
    }
    s->first = 0;
    g_stop = 0;

    snprintf(line, sizeof line, "Reading %d token%s", n, n == 1 ? "" : "s");
    const double t0 = now_s();
    k->status(k, line, t0);
    if (glm53f_model_forward(m, s->ids, n, s->logits, NULL) != 0) {
        k->status(k, NULL, 0.0);
        k->done(k, "the engine failed while reading the message", 2);
        return -1;
    }
    const double t_prefill = now_s() - t0;
    k->status(k, NULL, 0.0);

    int nout = 0, printed = 0, stopped = 0, in_think = 1, interrupted = 0;
    CacheSnap c0, c1;
    cache_snap(&m->cache, &c0);
    const double td = now_s();
    for (int step = 0; step < s->gen; step++) {
        if (g_stop) { interrupted = 1; break; }
        int best = 0;
        for (int i = 1; i < c->vocab; i++) if (s->logits[i] > s->logits[best]) best = i;
        int is_eos = 0;
        for (int e = 0; e < c->n_eos; e++) if (best == c->eos[e]) is_eos = 1;
        if (is_eos) { stopped = 1; break; }
        s->out[nout++] = best;
        const int len = tok_decode(s->tok, s->out, nout, s->text, s->text_cap - 1);
        s->text[len > 0 ? len : 0] = 0;
        const int ok = utf8_complete(s->text, len);
        if (in_think) {
            const char *close = strstr(s->text, "</think>");
            const int upto = close ? (int)(close - s->text) : ok;
            if (s->show_thinking) {
                if (upto > printed) k->reasoning(k, s->text + printed, upto - printed);
            } else {
                snprintf(line, sizeof line, "Thinking · %d token%s", nout, nout == 1 ? "" : "s");
                k->status(k, line, td);
            }
            if (upto > printed) printed = upto;
            if (close) {
                k->status(k, NULL, 0.0);
                k->thought(k, now_s() - td, nout);
                printed = (int)(close - s->text) + 8;
                in_think = 0;
            }
        }
        if (!in_think && ok > printed) {
            k->answer(k, s->text + printed, ok - printed);
            printed = ok;
        }
        if (step + 1 < s->gen && glm53f_model_forward(m, &best, 1, s->logits, NULL) != 0) {
            k->status(k, NULL, 0.0);
            k->done(k, "the engine failed while writing the answer", 2);
            return -1;
        }
    }
    k->status(k, NULL, 0.0);
    const double t_dec = now_s() - td;
    cache_snap(&m->cache, &c1);

    s->turns++;
    s->pre_tok += n;
    s->pre_s += t_prefill;
    if (nout) {
        s->dec_tok += nout;
        s->dec_s += t_dec;
        s->dec_wait_s += c1.wait - c0.wait;
        s->dec_read_gb += (double)(c1.bytes - c0.bytes) / 1e9;
        const uint64_t hits = c1.hits - c0.hits, ahead = c1.ahead - c0.ahead;
        s->dec_ahead += ahead < hits ? ahead : hits;
        s->dec_demand += c1.misses - c0.misses;
        s->dec_reqs += hits + (c1.misses - c0.misses);
        s->last_spt = t_dec / nout;
        if (s->n_spt == 16) { memmove(s->spt_hist, s->spt_hist + 1, 15 * sizeof *s->spt_hist); s->n_spt--; }
        s->spt_hist[s->n_spt++] = s->last_spt;
    }

    char stats[256], d[32];
    duration(d, sizeof d, t_dec);
    snprintf(stats, sizeof stats, "%d tokens · %.2f s/token · %s · read %d in %.1f s%s",
             nout, nout ? t_dec / nout : 0.0, d, n, t_prefill,
             interrupted ? " · interrupted" : stopped ? "" : " · reached /gen");
    int len = nout ? tok_decode(s->tok, s->out, nout, s->text, s->text_cap - 1) : 0;
    if (len < 0) len = 0;
    s->text[len] = 0;
    log_turn(s, in->shown ? in->shown : in->prompt, s->text, len, stats);
    {
        const char *close = strstr(s->text, "</think>");
        free(s->last_answer);
        s->last_answer = NULL;
        if (close) {
            close += 8;
            while (*close == '\n' || *close == ' ') close++;
            if (*close) s->last_answer = dup_str(close);
        }
    }
    k->done(k, stats, interrupted);
    return 0;
}

/* ------------------------------------------------------------ plain front end ---- */
/* Line by line, no colour: for pipes, captured sessions and terminals that cannot host
 * the full-screen front end. */

#ifdef _WIN32
static BOOL WINAPI on_ctrl(DWORD type)
{
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) { g_stop = 1; return TRUE; }
    return FALSE;
}
#endif

static void plain_status(Glm53fTurnSink *k, const char *text, double t0)
{
    (void)k; (void)t0;
    if (text && !strncmp(text, "Reading", 7)) printf("  %s…\n", text);
}
static void plain_reasoning(Glm53fTurnSink *k, const char *p, int n) { (void)k; fwrite(p, 1, (size_t)n, stdout); fflush(stdout); }
static void plain_thought(Glm53fTurnSink *k, double sec, int tokens)
{
    char d[32];
    int *shown = (int *)k->ctx;
    duration(d, sizeof d, sec);
    printf("%s  + Thought: %s · %d tokens\n\n", *shown ? "\n\n" : "", d, tokens);
}
static void plain_answer(Glm53fTurnSink *k, const char *p, int n) { (void)k; fwrite(p, 1, (size_t)n, stdout); fflush(stdout); }
static void plain_done(Glm53fTurnSink *k, const char *stats, int state)
{
    (void)k;
    if (state == 2) printf("  %s\n", stats);
    else            printf("\n\n  ■ localcode · " GLM53F_CHAT_MODEL_NAME " · %s\n", stats);
}

static int read_line(char *buf, int cap)
{
#ifdef _WIN32
    const HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(in, &mode)) {
        static wchar_t w[16384];
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

static int plain_run(Glm53fChat *s)
{
#ifdef _WIN32
    SetConsoleCtrlHandler(on_ctrl, TRUE);
#endif
    const int cap = 65536;
    char *line = (char *)malloc((size_t)cap);
    if (!line) return 1;
    printf("localcode · " GLM53F_CHAT_MODEL_NAME " · ready in %.1f s · /help for commands\n", s->load_s);
    int rc = 0, shown = 0;
    Glm53fTurnSink k = { plain_status, plain_reasoning, plain_thought, plain_answer, plain_done, &shown };
    for (;;) {
        printf("\n> ");
        fflush(stdout);
        if (read_line(line, cap) < 0) break;
        Glm53fInput in;
        glm53f_chat_input(s, line, &in);
        if (in.kind == GLM53F_IN_QUIT) { glm53f_chat_input_free(&in); break; }
        if (in.note) printf("  %s\n", in.note);
        if (in.kind == GLM53F_IN_MESSAGE) {
            shown = s->show_thinking;
            printf("\n");
            if (glm53f_chat_turn(s, &in, &k) != 0) rc = 1;
        }
        glm53f_chat_input_free(&in);
        if (rc) break;
    }
    free(line);
    return rc;
}

int glm53f_chat_run(Glm53fChat *s)
{
    s->first = 1;
    /* LOCALCODE_TUI=1 forces the full-screen front end (captured sessions); LOCALCODE_PLAIN=1
     * or NO_COLOR keeps the plain one. */
    const char *force = getenv("LOCALCODE_TUI");
    const int forced = force && strcmp(force, "0") != 0;
    int rc = -1;
    if (!getenv("LOCALCODE_PLAIN") && !getenv("NO_COLOR")) rc = glm53f_tui_run(s, forced);
    if (rc < 0) rc = plain_run(s);
    free(s->log);
    s->log = NULL;
    free(s->last_answer);
    s->last_answer = NULL;
    return rc;
}
