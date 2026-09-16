# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Interactive session, `--chat` (with `--ctx N`): the model stays loaded and each turn feeds
  only its new tokens, reusing the KV cache, the KDA state and the indexer; reasoning dimmed,
  answer after a separator, per-turn numbers; `/reset`, `/params`, `/gen`, `/reasoning`,
  `/help`, `/quit`, and `/save [FILE]`, which writes the conversation as Markdown with the
  reasoning folded.
- `--cache-gb auto`, now the default: the expert cache takes the RAM free when the model
  opens, minus the session's other buffers and a fifth of the installed RAM kept for the
  system. The size used is printed.

### Changed

- Prefill goes layer by layer: every chunk of the prompt passes a layer before the next layer
  starts, so each layer's routed experts are read once per prompt instead of once per chunk.
  372-token prompt, `--gpu --cache-gb 16`: 196.1, 191.0, 192.2 s -> 118.6, 119.0, 120.1 s
  (38% faster), 431 GB -> 257 GB of experts read, logits byte-identical. Prompts
  up to 4096 positions are held at once (`GLM53F_SPAN`); the next layer's experts are
  predicted during a prefill only when the cache holds two layers of them.

- CUDA backend (`-DGLM53F_CUDA=ON`, `--gpu`): the trunk (KDA, MLA, mHC, dense and shared
  MLPs, lm_head) runs on NVIDIA GPUs, routed experts stay on the CPU. Decode 2.22 ->
  1.80 s/token at equal peak RSS (40 GB) on 2x RTX 3060, generated tokens identical,
  45-layer logits within 2.1e-5 of the transformers reference. It includes:
  - cuBLAS products on fp32, with persistent fp32 copies in spare device memory;
  - GPU kernels for RMSNorm, KDA recurrence and MLA attention with short per-thread loops
    (the KDA and MLA ones keep the CPU's order of sums);
  - placement proportional to each device's free memory;
  - keep-warm, which holds the cards' performance state while the CPU computes experts
    (`GLM53F_GPU_WARM=0` disables it);
  - the host copy of the trunk is freed after upload (13.9 GB), leaving room for cache;
  - `GLM53F_GPU_EXACT`, `GLM53F_GPU_WIDEN`, `GLM53F_GPU_PROFILE`, `GLM53F_GPU_LAYERS_PER_DEV`;
  - `test_glm_tiny` GATE G1-G3: the reference checks with the trunk on the GPU.
- FP8 matvec decodes E4M3 codes arithmetically instead of with a table gather: 2.2x faster
  kernel, bit-identical (new `fp8_kernel` test; real 45-layer logits byte-identical).
  Decode 3-5% faster in interleaved A/B; the disk now limits (docs/PERFORMANCE.md).
- The expert cache is capped by the free physical memory at open time (3 GB of headroom,
  the trunk's RAM counted as free when `--gpu` is requested): asking for more than fits
  used to page the machine to a standstill instead of failing.
- Prefills are fed in chunks of 256 positions, so the working buffers no longer grow with
  the prompt (a 7,598-position session now peaks at the same buffers as a short one), and
  the memory guard accounts for the attention caches, the indexer state and those buffers
  instead of only the expert cache.
- `--kv compressed`: the MLA cache can hold the `kv_lora` latent per position (22 KB)
  instead of the expanded keys and values (1.44 MB), by folding `kv_b` into the query and
  the output. 64x less memory, the same value to 7.6e-06, chosen automatically past 2051
  positions, where a 32k session now needs 0.72 GB of cache instead of 47 GB.
- DeepSeek Sparse Attention (the checkpoint's DSA indexer) is implemented, so sessions are
  no longer limited to 2051 positions: the indexer picks the 512 best pools of 4 positions
  plus the tail for every query, exactly as the reference does (new GATE 4 on a 40-position
  tiny session). `--gpu` still runs dense attention and refuses longer sessions.
- `--experts int4`: the expert cache can hold routed experts re-quantised to int4
  (group of 64), 1.8x more experts in the same RAM and 13% faster decode, at the cost of an
  approximate output (38 of 40 greedy tokens unchanged). Off by default; the checkpoint on
  disk is never converted. New `GLM53F_WI4` matrix format, AVX2 kernel and quantiser, both
  covered by `fp8_kernel`.
- The checkpoint's MTP layer (layer 45) is implemented and measured under
  `GLM53F_MTP_STATS=1`: 88.6% of its drafts equal the token the full model produces.
  Speculative decode was not built: with reads setting the pace, verifying two tokens per
  pass does not lower the bytes read per token (docs/PERFORMANCE.md).
- `GLM53F_ROUTE_TRACE`: per-layer routing and read/compute timing trace.
- `GLM53F_PREDICT_STATS` also reports how well the routers of the layers 2, 3 and 4 ahead
  predict, applied to the current layer's input.
- The prefetch queue is ordered by layer instead of by insertion, and cancellation of
  pending reads is anchored on the layer being computed. Prefetching more than one layer
  ahead was measured and rejected (docs/PERFORMANCE.md).
- The CLI reports expert bytes read ahead as well as on demand, and reads per decode token.
- Predictive expert prefetch (`--prefetch N`, default 6): each MoE layer predicts the next
  layer's experts with that layer's router and background readers fetch them while
  compute continues; wrong predictions are cancelled before they are read. Decode
  2.96 -> 2.29 s/token, prefill of an 18-token prompt 45.8 -> 30.8 s (3 runs each), output
  bit-identical.
- Thread-safe expert cache: pins for experts in use, in-flight tracking so a request waits
  for a read already underway, background reader pool (`GLM53F_IO_THREADS`).
- `tools/bench_expert_io.c`: expert read throughput versus concurrency.
- `GLM53F_PREDICT_STATS=1`: report how predictable each layer's routing is.
- `DISK S` column and a decode disk/compute split in the CLI report.
- `docs/`: USAGE, ARCHITECTURE, VALIDATION, PERFORMANCE.

### Fixed

- Windows: read-only files are opened with `FILE_FLAG_OVERLAPPED`; a synchronous handle
  serialises concurrent reads on the file object.
- Windows console showed UTF-8 output in the OEM code page ("BrasÃ­lia"); the CLI now
  sets the console output code page to UTF-8.

## [2.0.0] - 2026-09-15

First version that implements GLM-5.3-Flash itself and runs the released checkpoint.
Validated on the released weights against the official transformers implementation
(`modeling_glm5_next.py`): identical next token and top-10 on a real chat prompt, logits
within 1.5e-5 (7e-7 relative) after all 45 layers, and the same token after five decode
steps.

### Added

- GLM-5.3-Flash architecture, each kernel ported from `modeling_glm5_next.py`:
  Manifold-Constrained Hyper-Connections (4 residual streams, Sinkhorn mixer), KDA with
  the low-rank output gate, NoPE MLA without output gate, clamped SwiGLU, routed MoE
  (288 experts, top-8, sigmoid router with correction bias) plus shared expert.
- FP8 E4M3 block-scaled matrices (`weight_scale_inv`, 128x128) multiplied directly from
  the codes; matrices are self-describing (`Glm53fMat`), since one layer mixes FP8, BF16
  and fp32.
- Expert streaming for the GLM shard layout: weights and scales are two contiguous runs
  per expert, so a cache miss costs two reads.
- `glm53f_model.[ch]`: loading, incremental forward and logits, shared by the CLI and tests.
- Chat template (`[gMASK]<sop><|system|>Reasoning Effort: ...<|user|>...<|assistant|><think>`),
  `--reasoning`, EOS stop, streamed text with `--quiet`.
- `tools/glm_reference.py`, `tools/make_glm_tiny.py` + `tests/unit/test_glm_tiny.c`
  (weightless oracle), `tests/unit/test_cfg.c` (12 refusal cases),
  `tools/tok_parity_glm.py` (C tokenizer vs official, 12/12).

### Changed

- The config reader requires every field the kernels use and refuses settings they do not
  implement (activation, router, RoPE, grouping, attention types).
- Attention is dense. It equals DSA exactly up to index_topk + index_kpool - 1 = 2051
  positions; longer sessions are refused rather than computed differently.
- Tokenizer loader fixed to the cl100k pre-tokenizer family the checkpoint specifies.

### Removed

- Code, tests, fixtures, tools, scripts and documentation of the engine this project was
  derived from, which targeted a different model: trunk streaming, memory presets,
  ultra-low-memory mode, draft/speculative decode, saved state, MXFP4 kernels, and the
  corresponding pre-tokenizer path in `third_party/tok.h`.
- The NEON paths of the removed kernels; ARM builds use the portable scalar kernels.

### Fixed

- safetensors reader: header-length overflow (`8 + hlen` wrap), escape at buffer end in
  `skip_value`, unchecked `realloc` in shard listing, unvalidated data offsets, shapes and
  digit strings.
- The binder no longer replaces missing tensors with NULL (which crashed or silently
  changed the model); a missing or mis-shaped tensor fails the load.
