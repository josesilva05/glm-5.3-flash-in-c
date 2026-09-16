#!/usr/bin/env python3
"""
convert_tokenizer_json_to_tiktoken.py

Converts HuggingFace tokenizer.json (ByteLevel BPE) to:
1. tiktoken.model (base64(raw_bytes) rank)
2. updates tokenizer_config.json with added_tokens_decoder
"""
import base64
import json
import os
import sys

def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

def token_to_raw_bytes(tok_str: str) -> bytes:
    return bytes([byte_decoder[c] for c in tok_str])

def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\JoseS\model\GLM-5.3-Flash"
    tok_json_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.isfile(tok_json_path):
        print(f"Error: {tok_json_path} not found")
        return 1

    print(f"Reading {tok_json_path}...")
    with open(tok_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["model"]["vocab"]
    print(f"Found {len(vocab)} vocabulary tokens")

    # Sort vocab by rank
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])

    tiktoken_path = os.path.join(model_dir, "tiktoken.model")
    print(f"Writing {tiktoken_path}...")
    with open(tiktoken_path, "w", encoding="utf-8") as f:
        for tok_str, rank in sorted_vocab:
            raw_b = token_to_raw_bytes(tok_str)
            b64 = base64.b64encode(raw_b).decode("ascii")
            f.write(f"{b64} {rank}\n")

    print(f"Successfully generated {tiktoken_path} with {len(sorted_vocab)} tokens")

    # Generate or update added_tokens_decoder in tokenizer_config.json
    tok_cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    cfg = {}
    if os.path.isfile(tok_cfg_path):
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            try:
                cfg = json.load(f)
            except Exception:
                cfg = {}

    added_tokens = data.get("added_tokens", [])
    decoder = {}
    for a in added_tokens:
        decoder[str(a["id"])] = {
            "content": a["content"],
            "special": a.get("special", True)
        }

    cfg["added_tokens_decoder"] = decoder
    with open(tok_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"Updated {tok_cfg_path} with {len(decoder)} added tokens in added_tokens_decoder")
    return 0

if __name__ == "__main__":
    sys.exit(main())
