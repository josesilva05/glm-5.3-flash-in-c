# Validation

The engine is graded against the official implementation, `modeling_glm5_next.py` in
transformers 5.17.0. Nothing here compares the engine with itself.

## 1. Weightless tests (CI, seconds)

```bash
ctest --test-dir build -C Release --output-on-failure
```

| test | what it proves |
|---|---|
| `tiny_oracle` (`test_glm_tiny`) | a 5-layer random model with the released architecture, stored in the released names and dtypes, matches transformers: last-position logits within atol 1e-5 / rtol 1e-4, 6/6 greedy tokens with incremental decode (KDA state + KV cache carried), 18/18 teacher-forced positions, a 40-position session whose length forces the DSA indexer to select matches the reference at every position, prefetch leaves logits bit-identical (GATE 5, 20 runs), and in a CUDA build the same reference checks pass with the trunk on the GPU split across devices (GATE G1-G3) |
| `fp8_kernel` (`test_fp8`) | the FP8 matvec equals a scalar reference with the same arithmetic bit for bit, on expert shapes, ragged shapes and all 256 codes |
| `tiny_oracle_chunked` | the same gates with the prompt fed in chunks of 7 positions: a chunked prefill leaves the state a single pass would |
| `tiny_oracle_spans` | the same gates in chunks of 7 inside spans of 17, taken layer by layer: neither the order of the work nor the span boundaries move the results |
| `config_tiny`, `config_released` (`test_cfg`) | the reader accepts both configs and rejects 12 mutated ones (missing layer maps, other activation, RoPE, router grouping, top-k, mHC, attention type, FP8 block size) |
| `safetensors` (`test_st`) | header parsing of adversarial shards: inf/NaN bit patterns, f16 subnormals, FP8, rank 0-4, zero-element tensors, escaped names, `__metadata__`, two shards |
| `tokenizer` | roundtrip over README.md; runs when `GLM53F_TOK_FILES` points at the model directory |

The tiny oracle was checked against deliberate bugs: removing the SwiGLU clamp,
transposing the mHC mixer, or skipping the Sinkhorn iterations each fail it by thousands
of times the tolerance. Its `swiglu_limit` is 0.5 so the clamp actually binds.

Regenerate its fixture with `python tools/make_glm_tiny.py tests/fixtures/glm_tiny`.

## 2. Real checkpoint

### 2.1 Reference forward

`tools/glm_reference.py` builds one `Glm5NextTextDecoderLayer` at a time on the meta device,
assigns that layer's weights from the shards (FP8 dequantised exactly as transformers
does), runs it and frees it. Only the routed experts are replaced by a loader with the
same arithmetic as `Glm5NextTextExperts.forward`. It reads shards with positioned reads,
not mmap, which on Windows would charge every shard against the commit limit.

```bash
python tools/glm_reference.py <model_dir> --prompt "Qual a capital do Brasil?" --out ref
build/Release/glm53f <model_dir> --prompt-file prompt.txt --gen 1 --dump-logits c_logits.bin
python -c "import numpy as np; r=np.fromfile('ref/logits.bin','f4'); c=np.fromfile('c_logits.bin','f4'); print(r.argmax(), c.argmax(), abs(r-c).max())"
```

### 2.2 Results (2026-09-15, Ryzen 7 5700X3D, MSVC build)

| check | result |
|---|---|
| first 4 layers, 18-token chat prompt | same argmax and top-5; max logit difference 8.0e-6 |
| all 45 layers, same prompt | same next token (785, "The") and identical top-10; max difference 1.5e-5 (7e-7 relative) |
| decode: prompt + 5 generated tokens fed to the reference | reference predicts the engine's 6th token (41949) |
| full generation, 234 tokens | coherent reasoning and the correct answer ("A capital do Brasil é Brasília..."), stopped at EOS |
| `--kv compressed`, all 45 layers, same prompt | same next token and top-10; 1.5e-5 from the reference, 7.6e-06 from the expanded cache; the tiny oracle passes every gate in this mode, including the 40-position sparse one |
| `--gpu`, all 45 layers, same prompt | same next token and top-10; max difference 2.1e-5 from the reference, 1.2e-5 from the CPU engine |
| `--gpu`, 40 greedy tokens (12 runs: 24/38/44 GB caches, keep-warm on and off) | generated ids identical to the CPU run |
| `--experts int4`, 40 greedy tokens | 38 of 40 tokens equal to the exact run, same next-token argmax; logits RMS 0.77 from the exact path. This mode is an approximation by construction and is off by default |
| `GLM53F_MTP_STATS=1`, 12- and 80-token runs | generated ids unchanged; the MTP layer only drafts (transformers 5.17 has no reference for it, and a draft never reaches the output) |

### 2.3 Tokenizer parity

```bash
python tools/tok_parity_glm.py <model_dir> <path to test_tok>
```

12/12 cases token-identical to the official tokenizer: Portuguese accents, the chat
template string, code with tabs, whitespace runs, contractions, CJK, emoji with skin tone
and ZWJ sequences, URLs, numbers, non-breaking and zero-width spaces, Latin diacritics.
The engine's own tokenization of the chat prompt produced the same 18 ids as
`apply_chat_template`.

## 3. Python environment

The tools need `torch`, `transformers`, `safetensors` and `numpy` (see `pyproject.toml`).
The repository's `.venv` (not committed) was created with:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.14.0 transformers==5.17.0 safetensors numpy
```
