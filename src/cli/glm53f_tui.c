/* glm53f_tui.c - localcode's full-screen front end.
 *
 * The layout of the terminal coding tools it follows, in a Matrix palette: the conversation
 * fills the screen and scrolls above an input box that never leaves the bottom, with a
 * footer under it. The whole screen is redrawn from a model of the conversation (blocks of
 * text) whenever something changes, inside the terminal's alternate screen.
 *
 * The model runs on a worker thread, so the screen stays alive while it reads or writes:
 * the conversation scrolls (wheel, PgUp/PgDn), the next message can be typed and is queued,
 * and Esc or Ctrl+C stops an answer. Input is read raw, as the VT sequences the terminal
 * sends (keys, SGR mouse reports, bracketed paste), on Windows and POSIX alike.
 */
#define _POSIX_C_SOURCE 200809L

#include "glm53f_portable_io.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <poll.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif

#include "glm53f_chat.h"

/* Matrix palette, 24-bit colour */
#define C_RESET  "\033[0m"
#define C_BOLD   "\033[1m"
#define C_NOBOLD "\033[22m"
#define C_BRIGHT "\033[38;2;0;255;65m"      /* accents, "code", bold text  */
#define C_GREEN  "\033[38;2;0;190;60m"      /* secondary accents           */
#define C_DIM    "\033[38;2;50;135;70m"     /* "local", hints, numbers     */
#define C_FAINT  "\033[38;2;40;100;55m"     /* reasoning text              */
#define C_TEXT   "\033[38;2;205;255;210m"   /* answer and input text       */
#define C_CODE   "\033[38;2;140;255;170m"   /* inline code and code blocks */
#define C_PANEL  "\033[48;2;14;28;18m"      /* input box background        */

#define EDITOR_ROWS 8                        /* input rows shown before the box scrolls */

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

static int is_cont(char c) { return ((unsigned char)c & 0xC0) == 0x80; }

/* Columns of n bytes of UTF-8, one per code point (no wide glyphs are drawn). */
static int cols_of(const char *p, int n)
{
    int c = 0;
    for (int i = 0; i < n; i++) c += !is_cont(p[i]);
    return c;
}

/* Columns a string occupies on screen: code points, not counting ANSI sequences. */
static int vis_cols(const char *p)
{
    int c = 0;
    while (*p) {
        if (*p == 0x1b) {
            p++;
            if (*p == '[') {
                p++;
                while (*p && !((unsigned char)*p >= 0x40 && (unsigned char)*p <= 0x7e)) p++;
                if (*p) p++;
            }
            continue;
        }
        c += !is_cont(*p);
        p++;
    }
    return c;
}

/* ------------------------------------------------------------------ buffers ---- */

typedef struct { char *p; size_t n, cap; } Buf;

static void buf_add(Buf *b, const char *s, size_t n)
{
    if (b->n + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (b->n + n + 1 > cap) cap *= 2;
        char *g = (char *)realloc(b->p, cap);
        if (!g) return;
        b->p = g;
        b->cap = cap;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

static void buf_str(Buf *b, const char *s) { buf_add(b, s, strlen(s)); }

static void buf_fmt(Buf *b, const char *fmt, ...)
{
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n > 0) buf_add(b, tmp, (size_t)(n < (int)sizeof tmp ? n : (int)sizeof tmp - 1));
}

static void spaces(Buf *b, int n)
{
    for (int i = 0; i < n; i++) buf_add(b, " ", 1);
}

/* ----------------------------------------------------------------- terminal ---- */

static struct {
    int w, h, forced;
#ifdef _WIN32
    HANDLE in, out;
    DWORD  in_mode, out_mode;
    int    in_console, out_console;
    wchar_t high;                            /* first half of a surrogate pair */
#else
    struct termios saved;
    int    raw;
#endif
} T;

static void term_size(void)
{
    int w = 0, h = 0;
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (T.out_console && GetConsoleScreenBufferInfo(T.out, &info)) {
        w = info.srWindow.Right - info.srWindow.Left + 1;
        h = info.srWindow.Bottom - info.srWindow.Top + 1;
    }
#else
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) { w = ws.ws_col; h = ws.ws_row; }
#endif
    if (w <= 0) { const char *e = getenv("COLUMNS"); w = e ? atoi(e) : 100; }
    if (h <= 0) { const char *e = getenv("LINES"); h = e ? atoi(e) : 40; }
    T.w = w < 40 ? 40 : w;
    T.h = h < 12 ? 12 : h;
}

static void out_str(const char *s, size_t n)
{
    fwrite(s, 1, n, stdout);
    fflush(stdout);
}

static int term_open(int forced)
{
    T.forced = forced;
#ifdef _WIN32
    T.in = GetStdHandle(STD_INPUT_HANDLE);
    T.out = GetStdHandle(STD_OUTPUT_HANDLE);
    T.in_console = GetConsoleMode(T.in, &T.in_mode) != 0;
    T.out_console = GetConsoleMode(T.out, &T.out_mode) != 0;
    if (!forced && (!T.in_console || !T.out_console)) return -1;
    if (T.out_console &&
        !SetConsoleMode(T.out, T.out_mode | 0x0004 /* ENABLE_VIRTUAL_TERMINAL_PROCESSING */))
        return -1;
    /* keys arrive as VT sequences, ctrl+c as a byte, no echo, no line editing, no
     * quick-edit selection stealing the mouse */
    if (T.in_console &&
        !SetConsoleMode(T.in, 0x0200 /* ENABLE_VIRTUAL_TERMINAL_INPUT */ | 0x0008 /* WINDOW_INPUT */ |
                              0x0080 /* ENABLE_EXTENDED_FLAGS */)) {
        SetConsoleMode(T.out, T.out_mode);
        return -1;
    }
#else
    const int in_tty = isatty(STDIN_FILENO), out_tty = isatty(STDOUT_FILENO);
    if (!forced && (!in_tty || !out_tty)) return -1;
    if (in_tty && tcgetattr(STDIN_FILENO, &T.saved) == 0) {
        struct termios raw = T.saved;
        raw.c_iflag &= ~(tcflag_t)(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
        raw.c_lflag &= ~(tcflag_t)(ECHO | ICANON | IEXTEN | ISIG);
        raw.c_cc[VMIN] = 1;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) == 0) T.raw = 1;
    }
#endif
    static const char enter[] = "\033[?1049h\033[?1000h\033[?1006h\033[?2004h\033[H\033[2J";
    out_str(enter, sizeof enter - 1);
    term_size();
    return 0;
}

static void term_close(void)
{
    static const char leave[] = "\033[?2004l\033[?1006l\033[?1000l\033[0m\033[?25h\033[?1049l";
    out_str(leave, sizeof leave - 1);
#ifdef _WIN32
    if (T.in_console) SetConsoleMode(T.in, T.in_mode);
    if (T.out_console) SetConsoleMode(T.out, T.out_mode);
#else
    if (T.raw) tcsetattr(STDIN_FILENO, TCSAFLUSH, &T.saved);
#endif
}

/* Up to cap bytes of input as UTF-8: >0 bytes, 0 nothing within timeout_ms, -1 end of input. */
static int in_read(char *buf, int cap, int timeout_ms)
{
#ifdef _WIN32
    if (T.in_console) {
        if (WaitForSingleObject(T.in, (DWORD)timeout_ms) != WAIT_OBJECT_0) return 0;
        INPUT_RECORD rec[64];
        DWORD n = 0, avail = 0;
        if (!GetNumberOfConsoleInputEvents(T.in, &avail) || !avail) return 0;
        if (!ReadConsoleInputW(T.in, rec, avail < 64 ? avail : 64, &n)) return -1;
        int got = 0;
        for (DWORD i = 0; i < n; i++) {
            if (rec[i].EventType != KEY_EVENT || !rec[i].Event.KeyEvent.bKeyDown) continue;
            const wchar_t ch = rec[i].Event.KeyEvent.uChar.UnicodeChar;
            if (!ch) continue;
            for (WORD r = 0; r < rec[i].Event.KeyEvent.wRepeatCount && got + 8 < cap; r++) {
                if (ch >= 0xD800 && ch <= 0xDBFF) { T.high = ch; continue; }
                wchar_t pair[2];
                int np = 0;
                if (ch >= 0xDC00 && ch <= 0xDFFF && T.high) pair[np++] = T.high;
                pair[np++] = ch;
                T.high = 0;
                got += WideCharToMultiByte(CP_UTF8, 0, pair, np, buf + got, cap - got, NULL, NULL);
            }
        }
        return got;
    }
    {   /* a pipe or a file (a captured session) */
        DWORD avail = 0, rd = 0;
        if (GetFileType(T.in) == FILE_TYPE_PIPE) {
            if (!PeekNamedPipe(T.in, NULL, 0, NULL, &avail, NULL)) return -1;
            if (!avail) { Sleep((DWORD)timeout_ms); return 0; }
            if (avail > (DWORD)cap) avail = (DWORD)cap;
        } else {
            avail = (DWORD)cap;
        }
        if (!ReadFile(T.in, buf, avail, &rd, NULL) || rd == 0) return -1;
        return (int)rd;
    }
#else
    struct pollfd pfd = { STDIN_FILENO, POLLIN, 0 };
    const int r = poll(&pfd, 1, timeout_ms);
    if (r <= 0) return 0;
    const ssize_t n = read(STDIN_FILENO, buf, (size_t)cap);
    return n > 0 ? (int)n : -1;
#endif
}

/* ------------------------------------------------------------------- blocks ---- */
/* The conversation as the screen shows it. The worker thread appends, the UI thread draws;
 * both under g_mu. */

enum { B_SPLASH, B_USER, B_STATUS, B_REASON, B_THOUGHT, B_ANSWER, B_STATS, B_NOTE, B_ERROR };

typedef struct {
    int    kind;
    Buf    text;
    double t0;
} Block;

static pthread_mutex_t g_mu;
static Block *g_blocks;
static int    g_nblocks, g_capblocks;
static int    g_dirty = 1;

static Block *block_new(int kind)
{
    if (g_nblocks == g_capblocks) {
        const int cap = g_capblocks ? g_capblocks * 2 : 32;
        Block *g = (Block *)realloc(g_blocks, (size_t)cap * sizeof *g);
        if (!g) return NULL;
        g_blocks = g;
        g_capblocks = cap;
    }
    Block *b = &g_blocks[g_nblocks++];
    memset(b, 0, sizeof *b);
    b->kind = kind;
    g_dirty = 1;
    return b;
}

static void block_add(int kind, const char *text)
{
    Block *b = block_new(kind);
    if (b && text) buf_str(&b->text, text);
}

static void blocks_clear(void)
{
    for (int i = 0; i < g_nblocks; i++) free(g_blocks[i].text.p);
    g_nblocks = 0;
    g_dirty = 1;
}

static Block *last_block(int kind)
{
    return g_nblocks && g_blocks[g_nblocks - 1].kind == kind ? &g_blocks[g_nblocks - 1] : NULL;
}

/* ----------------------------------------------------------------- wrapping ---- */
/* Lines of the screen: ANSI text and the columns it occupies. Text is wrapped at word
 * boundaries; every line starts with the prefix, and the style in force is reapplied. */

typedef struct { char *s; } Line;
typedef struct { Line *v; int n, cap; } Lines;

static void lines_push(Lines *L, Buf *b)
{
    if (L->n == L->cap) {
        const int cap = L->cap ? L->cap * 2 : 256;
        Line *g = (Line *)realloc(L->v, (size_t)cap * sizeof *g);
        if (!g) return;
        L->v = g;
        L->cap = cap;
    }
    L->v[L->n++].s = b->p ? b->p : (char *)calloc(1, 1);
    memset(b, 0, sizeof *b);
}

static void lines_free(Lines *L)
{
    for (int i = 0; i < L->n; i++) free(L->v[i].s);
    free(L->v);
    memset(L, 0, sizeof *L);
}

typedef struct {
    Lines      *out;
    Buf         cur;
    const char *prefix;          /* ANSI + text at the start of each line      */
    int         prefix_cols;
    const char *hang;            /* prefix of continuation lines (NULL: prefix) */
    int         hang_cols;
    int         width;           /* last usable column                          */
    int         pad_to;          /* pad with spaces up to this column (bubbles) */
    int         col, content;
    char        style[96];
} Wrap;

static void wr_begin(Wrap *w, int first)
{
    const char *p = (!first && w->hang) ? w->hang : w->prefix;
    buf_str(&w->cur, p);
    w->col = (!first && w->hang) ? w->hang_cols : w->prefix_cols;
    buf_str(&w->cur, w->style);
    w->content = 0;
}

static void wr_flush(Wrap *w)
{
    if (w->pad_to > w->col) spaces(&w->cur, w->pad_to - w->col);
    buf_str(&w->cur, C_RESET);
    lines_push(w->out, &w->cur);
    wr_begin(w, 0);
}

static void wr_style(Wrap *w, const char *ansi)
{
    snprintf(w->style, sizeof w->style, "%s", ansi);
    buf_str(&w->cur, ansi);
}

static void wr_text(Wrap *w, const char *p, int n)
{
    int i = 0;
    while (i < n) {
        if (p[i] == '\n') { wr_flush(w); i++; continue; }
        if (p[i] == ' ' || p[i] == '\t') {
            if (w->content && w->col < w->width) { buf_add(&w->cur, " ", 1); w->col++; }
            i++;
            continue;
        }
        int j = i;
        while (j < n && p[j] != ' ' && p[j] != '\t' && p[j] != '\n') j++;
        int wc = cols_of(p + i, j - i);
        const int room = w->width - (w->hang ? w->hang_cols : w->prefix_cols);
        if (w->col + wc > w->width && w->content) wr_flush(w);
        if (wc > room) {                         /* longer than a line: cut by code points */
            int k = i;
            while (k < j) {
                int step = 1;
                while (k + step < j && is_cont(p[k + step])) step++;
                if (w->col + 1 > w->width) wr_flush(w);
                buf_add(&w->cur, p + k, (size_t)step);
                w->col++;
                w->content = 1;
                k += step;
            }
        } else {
            buf_add(&w->cur, p + i, (size_t)(j - i));
            w->col += wc;
            w->content = 1;
        }
        i = j;
    }
}

static void wr_end(Wrap *w)
{
    if (w->content) {
        if (w->pad_to > w->col) spaces(&w->cur, w->pad_to - w->col);
        buf_str(&w->cur, C_RESET);
        lines_push(w->out, &w->cur);
    } else {
        free(w->cur.p);
        memset(&w->cur, 0, sizeof w->cur);
    }
}

static void push_text(Lines *L, const char *ansi)
{
    Buf b = { 0 };
    buf_str(&b, ansi);
    buf_str(&b, C_RESET);
    lines_push(L, &b);
}

/* Inline Markdown: **bold** and `code` inside one line of text. */
static void wr_inline(Wrap *w, const char *p, int n, const char *base)
{
    int bold = 0, code = 0, i = 0, start = 0;
    while (i < n) {
        if (!code && p[i] == '*' && i + 1 < n && p[i + 1] == '*') {
            wr_text(w, p + start, i - start);
            bold = !bold;
            char st[96];
            snprintf(st, sizeof st, "%s%s", bold ? C_BOLD : C_NOBOLD, bold ? C_BRIGHT : base);
            wr_style(w, st);
            i += 2;
            start = i;
            continue;
        }
        if (p[i] == '`') {
            wr_text(w, p + start, i - start);
            code = !code;
            wr_style(w, code ? C_CODE : (bold ? C_BRIGHT : base));
            i++;
            start = i;
            continue;
        }
        i++;
    }
    wr_text(w, p + start, n - start);
    char st[96];
    snprintf(st, sizeof st, "%s%s", C_NOBOLD, base);
    wr_style(w, st);
}

static void render_markdown(Lines *L, const char *text, size_t len, int W)
{
    int fence = 0;
    size_t i = 0;
    while (i <= len) {
        size_t j = i;
        while (j < len && text[j] != '\n') j++;
        const char *line = text + i;
        int n = (int)(j - i);
        if (n && line[n - 1] == '\r') n--;
        int ls = 0;
        while (ls < n && line[ls] == ' ') ls++;
        const char *body = line + ls;
        int bn = n - ls;

        Wrap w = { 0 };
        w.out = L;
        w.width = W - 2;
        char prefix[64], hang[64];
        if (bn >= 3 && !strncmp(body, "```", 3)) {
            fence = !fence;
        } else if (fence) {
            Buf b = { 0 };                           /* code keeps its spaces and is not wrapped */
            buf_str(&b, "    " C_CODE);
            buf_add(&b, line, (size_t)n);
            buf_str(&b, C_RESET);
            lines_push(L, &b);
        } else if (bn == 0) {
            if (i < len) push_text(L, "");
        } else if (body[0] == '#') {
            int h = 0;
            while (h < bn && body[h] == '#') h++;
            int k = h;
            while (k < bn && body[k] == ' ') k++;
            w.prefix = "  "; w.prefix_cols = 2;
            snprintf(w.style, sizeof w.style, "%s%s", C_BOLD, C_BRIGHT);
            wr_begin(&w, 1);
            wr_text(&w, body + k, bn - k);
            wr_end(&w);
        } else if ((body[0] == '-' || body[0] == '*' || body[0] == '+') && bn > 1 && body[1] == ' ') {
            snprintf(prefix, sizeof prefix, "%*s%s•%s ", 2 + ls, "", C_BRIGHT, C_TEXT);
            snprintf(hang, sizeof hang, "%*s", 4 + ls, "");
            w.prefix = prefix; w.prefix_cols = 4 + ls;
            w.hang = hang; w.hang_cols = 4 + ls;
            snprintf(w.style, sizeof w.style, "%s", C_TEXT);
            wr_begin(&w, 1);
            wr_inline(&w, body + 2, bn - 2, C_TEXT);
            wr_end(&w);
        } else if (bn > 2 && body[0] == '>' ) {
            snprintf(prefix, sizeof prefix, "  %s│ ", C_DIM);
            w.prefix = prefix; w.prefix_cols = 4;
            snprintf(w.style, sizeof w.style, "%s", C_DIM);
            wr_begin(&w, 1);
            wr_inline(&w, body + 1, bn - 1, C_DIM);
            wr_end(&w);
        } else {
            int num = 0;
            while (num < bn && body[num] >= '0' && body[num] <= '9') num++;
            const int numbered = num > 0 && num + 1 < bn && body[num] == '.' && body[num + 1] == ' ';
            if (numbered) {
                snprintf(prefix, sizeof prefix, "%*s%s%.*s %s", 2 + ls, "", C_BRIGHT, num + 1, body, C_TEXT);
                snprintf(hang, sizeof hang, "%*s", 2 + ls + num + 2, "");
                w.prefix = prefix; w.prefix_cols = 2 + ls + num + 2;
                w.hang = hang; w.hang_cols = 2 + ls + num + 2;
                snprintf(w.style, sizeof w.style, "%s", C_TEXT);
                wr_begin(&w, 1);
                wr_inline(&w, body + num + 2, bn - num - 2, C_TEXT);
            } else {
                snprintf(prefix, sizeof prefix, "%*s", 2 + ls, "");
                w.prefix = prefix; w.prefix_cols = 2 + ls;
                snprintf(w.style, sizeof w.style, "%s", C_TEXT);
                wr_begin(&w, 1);
                wr_inline(&w, body, bn, C_TEXT);
            }
            wr_end(&w);
        }
        i = j + 1;
    }
}

/* ------------------------------------------------------------------- render ---- */

typedef struct { int w; const char *px[6]; } Glyph;
static const Glyph G_L = { 2, { "#.", "#.", "#.", "#.", "#.", "##" } };
static const Glyph G_O = { 4, { "....", "....", "####", "#..#", "#..#", "####" } };
static const Glyph G_C = { 4, { "....", "....", "####", "#...", "#...", "####" } };
static const Glyph G_A = { 4, { "....", "....", "####", "...#", "#..#", "####" } };
static const Glyph G_D = { 4, { "...#", "...#", "####", "#..#", "#..#", "####" } };
static const Glyph G_E = { 4, { "....", "....", "####", "####", "#...", "####" } };

static void centred(Lines *L, int W, const char *colour, const char *text)
{
    Buf b = { 0 };
    const int pad = (W - cols_of(text, (int)strlen(text))) / 2;
    spaces(&b, pad > 0 ? pad : 0);
    buf_str(&b, colour);
    buf_str(&b, text);
    buf_str(&b, C_RESET);
    lines_push(L, &b);
}

static void render_splash(Lines *L, const Glm53fChat *s, int W)
{
    const Glyph *word[9] = { &G_L, &G_O, &G_C, &G_A, &G_L, &G_C, &G_O, &G_D, &G_E };
    int total = -1;
    for (int i = 0; i < 9; i++) total += word[i]->w + 1;
    push_text(L, "");
    for (int row = 0; row < 3; row++) {
        Buf b = { 0 };
        spaces(&b, (W - total) / 2 > 0 ? (W - total) / 2 : 0);
        for (int i = 0; i < 9; i++) {
            buf_str(&b, i < 5 ? C_DIM : C_BRIGHT);
            for (int x = 0; x < word[i]->w; x++) {
                const int top = word[i]->px[2 * row][x] == '#', bot = word[i]->px[2 * row + 1][x] == '#';
                buf_str(&b, top && bot ? "█" : top ? "▀" : bot ? "▄" : " ");
            }
            if (i < 8) buf_add(&b, " ", 1);
        }
        buf_str(&b, C_RESET);
        lines_push(L, &b);
    }
    push_text(L, "");
    const Glm53fCfg *c = &s->m->cfg;
    char line[300];
    snprintf(line, sizeof line, GLM53F_CHAT_MODEL_NAME " · %d layers · %d MLA + %d KDA · %d experts top-%d · ready in %.1f s",
             c->n_layers, s->n_mla, s->n_kda, c->n_experts, c->topk, s->load_s);
    centred(L, W, C_GREEN, line);
    snprintf(line, sizeof line, "cache %.1f GB%s (%d experts) · prefetch %d · %s · kv %s · experts %s",
             s->cache_gb, s->cache_auto ? " auto" : "", s->m->cache.nslot, s->prefetch,
             s->on_gpu ? "gpu" : "cpu", s->m->ckv ? "compressed" : "expanded", s->m->cache.i4 ? "int4" : "fp8");
    centred(L, W, C_DIM, line);
}

static void render_bubble(Lines *L, const char *text, size_t len, int W)
{
    char prefix[96];
    snprintf(prefix, sizeof prefix, "  %s%s▌%s ", C_PANEL, C_BRIGHT, C_TEXT);
    Wrap w = { 0 };
    w.out = L;
    w.prefix = prefix; w.prefix_cols = 4;
    w.width = W - 4;
    w.pad_to = W - 2;
    snprintf(w.style, sizeof w.style, "%s", C_TEXT);
    wr_begin(&w, 1);
    wr_flush(&w);                                /* the top padding row */
    wr_text(&w, text, (int)len);
    if (w.content) wr_flush(&w);
    w.content = 1;                               /* the bottom padding row */
    wr_end(&w);
}

static void render_plain(Lines *L, const char *text, size_t len, int W, const char *style, const char *prefix)
{
    Wrap w = { 0 };
    w.out = L;
    w.prefix = prefix; w.prefix_cols = vis_cols(prefix);
    if (w.prefix_cols > 4) { w.hang = "    "; w.hang_cols = 4; }   /* long prefixes: wrap under the text */
    w.width = W - 2;
    snprintf(w.style, sizeof w.style, "%s", style);
    wr_begin(&w, 1);
    wr_text(&w, text, (int)len);
    wr_end(&w);
}

static void render_blocks(Lines *L, const Glm53fChat *s, int W)
{
    static const char *spin[] = { "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏" };
    for (int i = 0; i < g_nblocks; i++) {
        const Block *b = &g_blocks[i];
        if (i > 0) push_text(L, "");
        const char *t = b->text.p ? b->text.p : "";
        switch (b->kind) {
        case B_SPLASH: render_splash(L, s, W); break;
        case B_USER:   render_bubble(L, t, b->text.n, W); break;
        case B_STATUS: {
            char d[32], line[256];
            duration(d, sizeof d, now_s() - b->t0);
            snprintf(line, sizeof line, "  %s%s %s%s · %s", C_BRIGHT, spin[(int)(now_s() * 10.0) % 10], C_DIM, t, d);
            push_text(L, line);
            break;
        }
        case B_REASON:  render_plain(L, t, b->text.n, W, C_FAINT, "  "); break;
        case B_THOUGHT: {
            char line[256];
            snprintf(line, sizeof line, "  %s+ %s%s", C_BRIGHT, C_GREEN, t);
            push_text(L, line);
            break;
        }
        case B_ANSWER: render_markdown(L, t, b->text.n, W); break;
        case B_STATS: {
            char prefix[160];
            snprintf(prefix, sizeof prefix, "  %s■ %s%slocalcode%s · %s" GLM53F_CHAT_MODEL_NAME "%s · ",
                     C_BRIGHT, C_BOLD, C_GREEN, C_NOBOLD, C_TEXT, C_DIM);
            render_plain(L, t, b->text.n, W, C_DIM, prefix);
            break;
        }
        case B_NOTE:  render_plain(L, t, b->text.n, W, C_DIM, "  "); break;
        case B_ERROR: {
            char prefix[64];
            snprintf(prefix, sizeof prefix, "  %s! ", C_BRIGHT);
            render_plain(L, t, b->text.n, W, C_TEXT, prefix);
            break;
        }
        default: break;
        }
    }
}

/* ------------------------------------------------------------------- editor ---- */

static struct { Buf b; int cur; } E;                 /* the text being typed, cursor in bytes */
static char *g_hist[64];
static int   g_nhist, g_hpos;

static void ed_set(const char *p)
{
    E.b.n = 0;
    if (E.b.p) E.b.p[0] = 0;
    buf_str(&E.b, p);
    E.cur = (int)E.b.n;
}

static void ed_insert(const char *p, int n)
{
    buf_add(&E.b, "", 0);
    Buf nb = { 0 };
    buf_add(&nb, E.b.p ? E.b.p : "", (size_t)E.cur);
    buf_add(&nb, p, (size_t)n);
    buf_add(&nb, E.b.p ? E.b.p + E.cur : "", E.b.n - (size_t)E.cur);
    free(E.b.p);
    E.b = nb;
    E.cur += n;
}

static void ed_erase(int from, int to)
{
    if (from < 0 || to > (int)E.b.n || from >= to) return;
    memmove(E.b.p + from, E.b.p + to, E.b.n - (size_t)to + 1);
    E.b.n -= (size_t)(to - from);
    E.cur = from;
}

static int ed_prev(int i) { if (i <= 0) return 0; i--; while (i > 0 && is_cont(E.b.p[i])) i--; return i; }
static int ed_next(int i) { if (i >= (int)E.b.n) return (int)E.b.n; i++; while (i < (int)E.b.n && is_cont(E.b.p[i])) i++; return i; }

/* ------------------------------------------------------------------ session ---- */

static Glm53fChat *g_s;
static volatile int g_busy, g_worker_done, g_fatal;
static pthread_t g_worker;
static Glm53fInput g_job;
static char *g_queue[16];
static int   g_nqueue;
static int   g_offset;                               /* lines scrolled up from the bottom */
static int   g_paste;

static void sink_status(Glm53fTurnSink *k, const char *text, double t0)
{
    (void)k;
    pthread_mutex_lock(&g_mu);
    Block *b = last_block(B_STATUS);
    if (!text) {
        if (b) { free(b->text.p); g_nblocks--; g_dirty = 1; }
    } else {
        if (!b) b = block_new(B_STATUS);
        if (b) { b->text.n = 0; buf_str(&b->text, text); b->t0 = t0; g_dirty = 1; }
    }
    pthread_mutex_unlock(&g_mu);
}

static void sink_reasoning(Glm53fTurnSink *k, const char *p, int n)
{
    (void)k;
    pthread_mutex_lock(&g_mu);
    Block *b = last_block(B_REASON);
    if (!b) b = block_new(B_REASON);
    if (b) buf_add(&b->text, p, (size_t)n);
    g_dirty = 1;
    pthread_mutex_unlock(&g_mu);
}

static void sink_thought(Glm53fTurnSink *k, double sec, int tokens)
{
    (void)k;
    char d[32], line[96];
    duration(d, sizeof d, sec);
    snprintf(line, sizeof line, "Thought: %s · %d tokens", d, tokens);
    pthread_mutex_lock(&g_mu);
    block_add(B_THOUGHT, line);
    pthread_mutex_unlock(&g_mu);
}

static void sink_answer(Glm53fTurnSink *k, const char *p, int n)
{
    (void)k;
    pthread_mutex_lock(&g_mu);
    Block *b = last_block(B_ANSWER);
    if (!b) b = block_new(B_ANSWER);
    if (b) {
        if (!b->text.n) while (n > 0 && (*p == '\n' || *p == ' ')) { p++; n--; }
        buf_add(&b->text, p, (size_t)n);
    }
    g_dirty = 1;
    pthread_mutex_unlock(&g_mu);
}

static void sink_done(Glm53fTurnSink *k, const char *stats, int state)
{
    (void)k;
    pthread_mutex_lock(&g_mu);
    block_add(state == 2 ? B_ERROR : B_STATS, stats);
    pthread_mutex_unlock(&g_mu);
}

static void *worker_main(void *arg)
{
    (void)arg;
    Glm53fTurnSink k = { sink_status, sink_reasoning, sink_thought, sink_answer, sink_done, NULL };
    if (glm53f_chat_turn(g_s, &g_job, &k) != 0) g_fatal = 1;
    pthread_mutex_lock(&g_mu);
    g_worker_done = 1;
    g_dirty = 1;
    pthread_mutex_unlock(&g_mu);
    return NULL;
}

/* Carry out one submitted line on the UI thread; a message starts the worker. */
static int handle_line(const char *line)
{
    Glm53fInput in;
    glm53f_chat_input(g_s, line, &in);
    int quit = 0;
    pthread_mutex_lock(&g_mu);
    switch (in.kind) {
    case GLM53F_IN_QUIT:  quit = 1; break;
    case GLM53F_IN_NOTE:  block_add(B_NOTE, in.note); break;
    case GLM53F_IN_RESET:
        blocks_clear();
        block_add(B_SPLASH, NULL);
        block_add(B_NOTE, in.note);
        g_offset = 0;
        break;
    case GLM53F_IN_MESSAGE:
        if (g_fatal) { block_add(B_ERROR, "the engine failed earlier in this session; restart localcode"); break; }
        block_add(B_USER, in.shown);
        if (in.note) block_add(B_NOTE, in.note);
        g_job = in;
        memset(&in, 0, sizeof in);
        g_busy = 1;
        g_worker_done = 0;
        g_offset = 0;
        if (pthread_create(&g_worker, NULL, worker_main, NULL) != 0) {
            g_busy = 0;
            block_add(B_ERROR, "could not start the model thread");
            glm53f_chat_input_free(&g_job);
        }
        break;
    default: break;
    }
    g_dirty = 1;
    pthread_mutex_unlock(&g_mu);
    glm53f_chat_input_free(&in);
    return quit;
}

static void submit(void)
{
    if (!E.b.n) return;
    char *line = (char *)malloc(E.b.n + 1);
    if (!line) return;
    memcpy(line, E.b.p, E.b.n + 1);
    if (g_nhist == 64) { free(g_hist[0]); memmove(g_hist, g_hist + 1, 63 * sizeof *g_hist); g_nhist--; }
    g_hist[g_nhist] = (char *)malloc(E.b.n + 1);
    if (g_hist[g_nhist]) memcpy(g_hist[g_nhist++], line, E.b.n + 1);
    g_hpos = g_nhist;
    ed_set("");
    if (g_nqueue < 16) g_queue[g_nqueue++] = line;
    else free(line);
    g_dirty = 1;
}

/* ------------------------------------------------------------------- frame ---- */

static void draw(Glm53fChat *s)
{
    term_size();
    const int W = T.w, H = T.h;
    Lines L = { 0 };
    pthread_mutex_lock(&g_mu);
    render_blocks(&L, s, W);
    g_dirty = 0;
    pthread_mutex_unlock(&g_mu);

    /* the input box: its text wrapped at EW columns, the cursor's row and column */
    const int EW = W - 6;
    int *rows_start = (int *)malloc((E.b.n + 2) * sizeof(int));
    int nrows = 0, cur_row = 0, cur_col = 0;
    if (!rows_start) { lines_free(&L); return; }
    {
        int col = 0;
        rows_start[nrows++] = 0;
        for (int i = 0; i <= (int)E.b.n; i++) {
            if (i == (int)E.b.n) { if (i == E.cur) { cur_row = nrows - 1; cur_col = col; } break; }
            if (E.b.p[i] == '\n') {
                if (i == E.cur) { cur_row = nrows - 1; cur_col = col; }
                rows_start[nrows++] = i + 1;
                col = 0;
                continue;
            }
            if (is_cont(E.b.p[i])) continue;
            if (col == EW) { rows_start[nrows++] = i; col = 0; }
            if (i == E.cur) { cur_row = nrows - 1; cur_col = col; }
            col++;
        }
    }
    const int shown_rows = nrows < EDITOR_ROWS ? nrows : EDITOR_ROWS;
    int first_row = cur_row - shown_rows + 1;
    if (first_row < 0) first_row = 0;
    if (first_row + shown_rows > nrows) first_row = nrows - shown_rows;

    const int box = shown_rows + 3, footer = 1;
    int view = H - box - footer - 1;
    if (view < 1) view = 1;
    int maxoff = L.n - view;
    if (maxoff < 0) maxoff = 0;
    if (g_offset > maxoff) g_offset = maxoff;
    if (g_offset < 0) g_offset = 0;

    Buf f = { 0 };
    buf_str(&f, "\033[?2026h\033[?25l");
    for (int r = 0; r < view; r++) {
        const int idx = L.n - view - g_offset + r;
        buf_fmt(&f, "\033[%d;1H\033[2K", r + 1);
        if (idx >= 0 && idx < L.n) buf_str(&f, L.v[idx].s);
    }
    buf_fmt(&f, "\033[%d;1H\033[2K", view + 1);

    int row = view + 2;
    const char *bar = C_PANEL C_BRIGHT "▌" C_TEXT " ";
    buf_fmt(&f, "\033[%d;1H\033[2K  %s", row++, bar);
    spaces(&f, W - 6);
    buf_str(&f, C_RESET);
    const int editor_top = row;
    for (int r = first_row; r < first_row + shown_rows; r++) {
        buf_fmt(&f, "\033[%d;1H\033[2K  %s", row++, bar);
        int a = rows_start[r], z = r + 1 < nrows ? rows_start[r + 1] : (int)E.b.n;
        if (z > a && E.b.p[z - 1] == '\n') z--;
        int used = 0;
        if (E.b.n == 0 && r == 0) {
            const char *ph = g_busy ? "type the next message; it goes when this answer ends"
                                    : "Ask anything...  /file PATH for a document, /help for commands";
            buf_str(&f, C_DIM);
            buf_str(&f, ph);
            used = cols_of(ph, (int)strlen(ph));
            buf_str(&f, C_TEXT);
        } else if (z > a) {
            buf_add(&f, E.b.p + a, (size_t)(z - a));
            used = cols_of(E.b.p + a, z - a);
        }
        spaces(&f, W - 6 - used);
        buf_str(&f, C_RESET);
    }
    {   /* settings row, with the state on the right */
        char left[200], plain[160], right[120];
        snprintf(plain, sizeof plain, GLM53F_CHAT_MODEL_NAME " · %s · reasoning %s",
                 s->on_gpu ? "gpu" : "cpu", s->reasoning);
        snprintf(left, sizeof left, C_BRIGHT GLM53F_CHAT_MODEL_NAME C_DIM " · %s · reasoning %s",
                 s->on_gpu ? "gpu" : "cpu", s->reasoning);
        const int lc = cols_of(plain, (int)strlen(plain));
        if (g_busy) snprintf(right, sizeof right, "esc interrupt%s", g_nqueue ? " · queued" : "");
        else        snprintf(right, sizeof right, "%s", g_nqueue ? "queued" : "");
        const int rc = cols_of(right, (int)strlen(right));
        buf_fmt(&f, "\033[%d;1H\033[2K  %s%s", row++, bar, left);
        spaces(&f, W - 6 - lc - rc - 1 > 0 ? W - 6 - lc - rc - 1 : 1);
        buf_str(&f, C_DIM);
        buf_str(&f, right);
        buf_str(&f, " " C_RESET);
    }
    buf_fmt(&f, "\033[%d;1H\033[2K  %s", row++, bar);
    spaces(&f, W - 6);
    buf_str(&f, C_RESET);
    {   /* footer: where, how much context, how to scroll back */
        char cwd[512] = "", right[160], gauge[64];
#ifdef _WIN32
        if (!_getcwd(cwd, (int)sizeof cwd)) cwd[0] = 0;
#else
        if (!getcwd(cwd, sizeof cwd)) cwd[0] = 0;
#endif
        glm53f_chat_gauge(s, gauge, sizeof gauge);
        if (g_offset) snprintf(right, sizeof right, "↑ %d lines · end to follow · %s · /help", g_offset, gauge);
        else          snprintf(right, sizeof right, "%s · /help", gauge);
        const int rc = cols_of(right, (int)strlen(right));
        int room = W - 4 - rc - 2;
        int cl = (int)strlen(cwd);
        const char *cp = cwd;
        if (room < 1) room = 1;
        if (cols_of(cwd, cl) > room) { cp = cwd + cl - room + 1; while (*cp && is_cont(*cp)) cp++; }
        buf_fmt(&f, "\033[%d;1H\033[2K  %s%s", row, C_DIM, cp);
        spaces(&f, W - 4 - cols_of(cp, (int)strlen(cp)) - rc > 0 ? W - 4 - cols_of(cp, (int)strlen(cp)) - rc : 1);
        buf_str(&f, right);
        buf_str(&f, C_RESET);
    }
    buf_fmt(&f, "\033[%d;%dH\033[?25h\033[?2026l", editor_top + (cur_row - first_row), 5 + cur_col);
    out_str(f.p, f.n);
    free(f.p);
    free(rows_start);
    lines_free(&L);
}

/* -------------------------------------------------------------------- input ---- */

/* Handle a chunk of input bytes; returns 1 to leave. */
static int on_input(const char *p, int n, int view_rows)
{
    int i = 0;
    while (i < n) {
        const unsigned char c = (unsigned char)p[i];
        if (g_paste) {
            if (c == 0x1b && i + 5 < n && !memcmp(p + i, "\033[201~", 6)) { g_paste = 0; i += 6; continue; }
            if (c == '\r') { ed_insert("\n", 1); if (i + 1 < n && p[i + 1] == '\n') i++; }
            else if (c >= 0x20 || c == '\n' || c == '\t') ed_insert(p + i, 1);
            i++;
            continue;
        }
        if (c == 0x1b) {
            if (i + 1 < n && (p[i + 1] == '[' || p[i + 1] == 'O')) {
                const int ss3 = p[i + 1] == 'O';
                int j = i + 2;
                while (j < n && !((unsigned char)p[j] >= 0x40 && (unsigned char)p[j] <= 0x7e)) j++;
                if (j >= n) return 0;                    /* an incomplete sequence: drop it */
                const char fin = p[j];
                char par[32];
                int pl = j - (i + 2);
                if (pl > 31) pl = 31;
                memcpy(par, p + i + 2, (size_t)pl);
                par[pl] = 0;
                i = j + 1;
                if (!ss3 && par[0] == '<' && (fin == 'M' || fin == 'm')) {     /* SGR mouse */
                    const int b = atoi(par + 1);
                    if (b == 64) g_offset += 3;
                    if (b == 65) g_offset -= 3;
                    g_dirty = 1;
                    continue;
                }
                if (!strcmp(par, "200") && fin == '~') { g_paste = 1; continue; }
                switch (fin) {
                case 'A':
                    if (g_hpos > 0) { g_hpos--; ed_set(g_hist[g_hpos]); }
                    break;
                case 'B':
                    if (g_hpos < g_nhist - 1) { g_hpos++; ed_set(g_hist[g_hpos]); }
                    else { g_hpos = g_nhist; ed_set(""); }
                    break;
                case 'C': E.cur = ed_next(E.cur); break;
                case 'D': E.cur = ed_prev(E.cur); break;
                case 'H': E.cur = 0; break;
                case 'F': E.cur = (int)E.b.n; g_offset = 0; break;
                case '~': {
                    const int k = atoi(par);
                    if (k == 3) ed_erase(E.cur, ed_next(E.cur));
                    else if (k == 5) g_offset += view_rows - 2;
                    else if (k == 6) g_offset -= view_rows - 2;
                    else if (k == 1 || k == 7) E.cur = 0;
                    else if (k == 4 || k == 8) { E.cur = (int)E.b.n; g_offset = 0; }
                    break;
                }
                default: break;
                }
                g_dirty = 1;
                continue;
            }
            if (g_busy) glm53f_chat_stop();              /* esc: stop the answer */
            i++;
            g_dirty = 1;
            continue;
        }
        if (c == 0x03) {                                 /* ctrl+c */
            if (g_busy) glm53f_chat_stop();
            else if (E.b.n) ed_set("");
            else return 1;
            i++;
            g_dirty = 1;
            continue;
        }
        if (c == 0x04 && !E.b.n && !g_busy) return 1;   /* ctrl+d on an empty box */
        if (c == '\r' || c == '\n') {
            int next = i + 1;
            if (c == '\r' && next < n && p[next] == '\n') next++;
            /* an unbracketed paste: more text right behind this line break */
            if (c == '\r' && next < n && (unsigned char)p[next] >= 0x20) ed_insert("\n", 1);
            else submit();
            i = next;
            g_dirty = 1;
            continue;
        }
        if (c == 0x7f || c == 0x08) { ed_erase(ed_prev(E.cur), E.cur); i++; g_dirty = 1; continue; }
        if (c == 0x01) { E.cur = 0; i++; g_dirty = 1; continue; }
        if (c == 0x05) { E.cur = (int)E.b.n; i++; g_dirty = 1; continue; }
        if (c == 0x15) { ed_set(""); i++; g_dirty = 1; continue; }
        if (c == 0x17) {                                 /* ctrl+w: the word before the cursor */
            int a = E.cur;
            while (a > 0 && E.b.p[a - 1] == ' ') a--;
            while (a > 0 && E.b.p[a - 1] != ' ') a--;
            ed_erase(a, E.cur);
            i++;
            g_dirty = 1;
            continue;
        }
        if (c == '\t') { ed_insert("    ", 4); i++; g_dirty = 1; continue; }
        if (c < 0x20) { i++; continue; }
        int j = i;
        while (j < n && (unsigned char)p[j] >= 0x20 && p[j] != 0x7f) j++;
        ed_insert(p + i, j - i);
        i = j;
        g_dirty = 1;
    }
    return 0;
}

/* --------------------------------------------------------------------- main ---- */

int glm53f_tui_run(Glm53fChat *s, int forced)
{
    if (term_open(forced) != 0) return -1;
    g_s = s;
    pthread_mutex_init(&g_mu, NULL);
    block_add(B_SPLASH, NULL);

    char in[4096];
    int quit = 0, eof = 0;
    double last_draw = 0.0;
    while (!quit) {
        const int got = eof ? 0 : in_read(in, sizeof in, g_busy ? 100 : 400);
        if (got < 0) eof = 1;
        if (got > 0) {
            const int view = T.h - 6;
            if (on_input(in, got, view > 3 ? view : 3)) quit = 1;
        }
        if (g_busy) {
            pthread_mutex_lock(&g_mu);
            const int done = g_worker_done;
            pthread_mutex_unlock(&g_mu);
            if (done) {
                pthread_join(g_worker, NULL);
                glm53f_chat_input_free(&g_job);
                g_busy = 0;
                g_dirty = 1;
            }
        }
        if (!g_busy && g_nqueue && !quit) {
            char *line = g_queue[0];
            memmove(g_queue, g_queue + 1, (size_t)(--g_nqueue) * sizeof *g_queue);
            if (handle_line(line)) quit = 1;
            free(line);
        }
        if (eof && !g_busy && !g_nqueue) quit = 1;
        const int old_w = T.w, old_h = T.h;
        term_size();
        if (T.w != old_w || T.h != old_h) { out_str("\033[2J", 4); g_dirty = 1; }
        const double t = now_s();
        if (g_dirty || (g_busy && t - last_draw > 0.12)) {
            draw(s);
            last_draw = t;
        }
    }
    if (g_busy) {                                        /* leaving mid-answer: stop it first */
        glm53f_chat_stop();
        pthread_join(g_worker, NULL);
        glm53f_chat_input_free(&g_job);
        g_busy = 0;
    }
    term_close();
    pthread_mutex_lock(&g_mu);
    blocks_clear();
    pthread_mutex_unlock(&g_mu);
    free(g_blocks);
    g_blocks = NULL;
    g_capblocks = 0;
    for (int i = 0; i < g_nhist; i++) free(g_hist[i]);
    for (int i = 0; i < g_nqueue; i++) free(g_queue[i]);
    free(E.b.p);
    printf("localcode · bye\n");
    return g_fatal ? 1 : 0;
}
