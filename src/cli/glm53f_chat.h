/* glm53f_chat.h - localcode, the interactive session (glm53f_chat.c, glm53f_tui.c).
 *
 * The caller opens the model, loads the tokenizer and hands both over with the settings it
 * chose; the session owns the screen until the user leaves. Nothing here changes what the
 * model computes: a turn feeds the new tokens to the same forward pass a one-shot run uses.
 *
 * glm53f_chat.c holds the session itself (commands, the chat template, a turn, the
 * transcript) and a plain line-by-line front end for pipes; glm53f_tui.c is the full-screen
 * front end used on a terminal. Both drive a turn through the same events.
 */
#ifndef GLM53F_CHAT_H
#define GLM53F_CHAT_H

#include <stddef.h>

#include "glm53f_model.h"
#include "glm53f_tok.h"

#define GLM53F_CHAT_MODEL_NAME "GLM-5.3-Flash"

typedef struct {
    Glm53fModel *m;
    Tok         *tok;
    const char  *dir;
    const char  *reasoning;      /* "Max" | "High" | "Low", as the template spells it */
    int          gen;            /* tokens per answer            */
    int          prefetch;
    int          on_gpu;
    int          show_thinking;  /* 1: show the reasoning text, 0: fold it into one line */
    double       cache_gb;       /* the cache actually allocated */
    int          cache_auto;     /* 1 when it was sized from the free RAM */
    double       load_s;         /* seconds spent opening the model */
    int          n_mla, n_kda;

    int         *ids, ids_cap;   /* scratch owned by the caller */
    int         *out;
    char        *text;
    int          text_cap;
    float       *logits;

    /* owned by the session */
    int          first;          /* the next message opens the conversation */
    char        *log;            /* the conversation as Markdown, for /save */
    size_t       log_len, log_cap;
} Glm53fChat;

int glm53f_chat_run(Glm53fChat *s);

/* ---- shared by the front ends ---- */

/* Events of one turn, in order: status (reading, thinking; NULL text turns it off), the
 * reasoning text if shown, the thought summary, answer text, and done. */
typedef struct Glm53fTurnSink {
    void (*status)(struct Glm53fTurnSink *k, const char *text, double t0);
    void (*reasoning)(struct Glm53fTurnSink *k, const char *bytes, int n);
    void (*thought)(struct Glm53fTurnSink *k, double seconds, int tokens);
    void (*answer)(struct Glm53fTurnSink *k, const char *bytes, int n);
    void (*done)(struct Glm53fTurnSink *k, const char *stats, int interrupted);
    void *ctx;
} Glm53fTurnSink;

/* What a line typed by the user asks for. */
typedef struct {
    int   kind;                  /* GLM53F_IN_*                                   */
    char *prompt;                /* MESSAGE: the text for the model, templated    */
    char *shown;                 /* MESSAGE: what the transcript shows            */
    char *note;                  /* NOTE, or a remark before a MESSAGE (/file)    */
} Glm53fInput;

enum { GLM53F_IN_NONE, GLM53F_IN_MESSAGE, GLM53F_IN_NOTE, GLM53F_IN_QUIT, GLM53F_IN_RESET };

/* Interpret one line: a command is carried out (settings, /save, /reset) and answered with
 * a note; a message is wrapped in the chat template. Free with glm53f_chat_input_free. */
void glm53f_chat_input(Glm53fChat *s, const char *line, Glm53fInput *in);
void glm53f_chat_input_free(Glm53fInput *in);

/* Run one turn. g_stop (glm53f_chat_stop) ends it early; returns -1 if the engine failed. */
int  glm53f_chat_turn(Glm53fChat *s, const Glm53fInput *in, Glm53fTurnSink *k);
void glm53f_chat_stop(void);

/* "context 88/2048 (4%)" and the settings line, for status rows. */
void glm53f_chat_gauge(const Glm53fChat *s, char *buf, size_t cap);

/* Full-screen front end; returns 0, or -1 when the terminal cannot host it. */
int  glm53f_tui_run(Glm53fChat *s, int forced);

#endif /* GLM53F_CHAT_H */
