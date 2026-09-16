#!/usr/bin/env python3
"""Token-for-token parity of the C tokenizer (test_tok encodefile) against the official
GLM-5.3-Flash tokenizer (transformers AutoTokenizer) on a varied corpus.

  .venv/Scripts/python tools/tok_parity_glm.py <model_dir> <test_tok executable>
"""
import os
import subprocess
import sys
import tempfile

from transformers import AutoTokenizer

CASES = [
    "Qual a capital do Brasil?",
    "Olá! A ação, a emoção e o coração: pão, maçã, avô, você, órgão, também.",
    "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Explique a fotossíntese.<|assistant|><think>",
    "def soma(a, b):\n    return a + b  # comentário\n\n\tprint(soma(1, 2))\n",
    "    leading spaces, trailing spaces    \n\n\n",
    "It's a test: don't, won't, I'm, they'll, we've, she'd.",
    "数字 12345678 和 3.14159，中文标点。日本語のテキスト。한국어 문장.",
    "Emoji 👍🏽 família 👨‍👩‍👧‍👦 e bandeira 🇧🇷!",
    "URL https://example.com/path?x=1&y=2 e e-mail nome.sobrenome@dominio.com.br",
    "Números: 1000000, 3,14; 2026-09-15; R$ 1.234,56; 100%",
    " espaço não separável em-space​zero-width",
    "ÀÉÎÕÜ àéîõü ÇÑ çñ ß ø å æ œ",
]


def main():
    model_dir, exe = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(model_dir)
    fails = 0
    for i, text in enumerate(CASES):
        want = tok.encode(text, add_special_tokens=False)
        with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".txt") as f:
            f.write(text.encode("utf-8"))
            path = f.name
        try:
            out = subprocess.run([exe, model_dir, "encodefile", path], capture_output=True,
                                 text=True, check=True).stdout.strip()
        finally:
            os.unlink(path)
        got = [int(x) for x in out.split(",") if x]
        ok = got == want
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  case {i:2d}: {len(want)} ids")
        if not ok:
            k = next((j for j in range(min(len(got), len(want))) if got[j] != want[j]), min(len(got), len(want)))
            print(f"        first difference at id {k}: C {got[k:k + 6]} vs reference {want[k:k + 6]}")
    print("TOKENIZER PARITY:", "PASS" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
