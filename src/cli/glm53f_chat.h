/* glm53f_chat.h - the interactive session (see glm53f_chat.c).
 *
 * The caller opens the model, loads the tokenizer and hands both over with the settings it
 * chose; the session owns the screen until the user leaves. Nothing here changes what the
 * model computes: a turn feeds the new tokens to the same forward pass a one-shot run uses.
 */
#ifndef GLM53F_CHAT_H
#define GLM53F_CHAT_H

#include "glm53f_model.h"
#include "glm53f_tok.h"

typedef struct {
    Glm53fModel *m;
    Tok         *tok;
    const char  *dir;
    const char  *reasoning;      /* "Max" | "High" | "Low", as the template spells it */
    int          gen;            /* tokens per answer            */
    int          prefetch;
    int          on_gpu;
    int          colour;         /* 0 when the output is not a terminal */
    double       cache_gb;
    double       load_s;         /* seconds spent opening the model */
    int          n_mla, n_kda;

    int         *ids, ids_cap;   /* scratch owned by the caller */
    int         *out;
    char        *text;
    int          text_cap;
    float       *logits;
} Glm53fChat;

int glm53f_chat_run(Glm53fChat *s);

#endif /* GLM53F_CHAT_H */
