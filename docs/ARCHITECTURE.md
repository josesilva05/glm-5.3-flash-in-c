# Architecture

This document describes the model the engine implements and how the engine is built.
The reference for every numeric detail is `transformers/models/glm5_next/modeling_glm5_next.py`
(transformers 5.17.0); each kernel in `src/core/glm53f_ops.c` names the module it mirrors.

## 1. The model: GLM-5.3-Flash (text path)

| property | value |
|---|---|
| decoder layers | 45 (plus one multi-token-prediction layer, index 45, not used) |
| hidden size | 4096 |
| vocabulary | 154,880 |
| attention | 34 KDA layers + 11 MLA layers (`layer_types`; MLA at 3, 7, 11, ..., 43) |
| feed-forward | 3 dense SwiGLU layers (0-2), then 42 MoE layers |
| MoE | 288 routed experts, top-8, 1 shared expert, expert width 2048 |
| residual | mHC, 4 streams |
| parameters | ~314 B total, ~18 B active per token |
| checkpoint | 62 safetensors shards, ~306 GB: FP8 E4M3 + BF16 + F32 |

### 1.1 Residual stream: Manifold-Constrained Hyper-Connections (mHC)

The residual stream is not one vector but `hc_mult = 4` copies of the hidden state,
`h = [T][4][4096]`. The embedding is copied into all four. Each sub-block (attention and
feed-forward) has its own mHC site with learned `fn [24][16384]`, `base [24]`, `scale [3]`:

```
flat  = RMSNorm_unweighted(h.flatten())            over 4 x 4096
mix   = fn @ flat                                  24 values
pre   = sigmoid(mix[0:4]  * scale[0] + base[0:4]) + eps
post  = 2 * sigmoid(mix[4:8] * scale[1] + base[4:8])
comb  = softmax_row(mix[8:24].view(4,4) * scale[2] + base[8:24]) + eps
        column-normalise, then 19 x (row-normalise, column-normalise)   (Sinkhorn)
x     = sum_i pre[i] * h[i]                        collapse to one vector
y     = sub_block(RMSNorm(x))
h[j]' = post[j] * y + sum_i comb[i][j] * h[i]      expand back into 4 streams
```

After the last layer the four streams are averaged, then the final RMSNorm and `lm_head`.

### 1.2 KDA layers (linear attention, 34 layers)

Gated delta-rule attention with a fixed-size recurrent state, `Glm5NextTextLinearAttention`:

1. `q, k, v = Linear(x)` (each `[64 x 128][4096]`), short causal depthwise conv (k=4) with SiLU.
2. Forget gate: `g = -5 * sigmoid(exp(A_log[h]) * (f_b(f_a(x)) + dt_bias))`, decay `alpha = exp(g)`
   per head and channel.
3. `beta = sigmoid(b_proj(x))`, one per head.
4. `q, k` L2-normalised per head (`x / sqrt(sum x^2 + 1e-6)`), `q *= 128^-0.5`.
5. Recurrence per head: `S *= alpha; u = S^T k; S += k (beta (v - u))^T; o = S^T q`.
6. Output: `o_proj( RMSNorm_head(o) * w * sigmoid(g_b(g_a(x))) )` (low-rank output gate).

State per layer: 64 x 128 x 128 recurrent floats + 3 x 8192 x 3 conv floats (4.5 MB);
202 MB for all 34 layers, independent of context length.

### 1.3 MLA layers (11 layers)

`Glm5NextTextAttention`: DeepSeek-style latent attention with **no positional encoding**
(`qk_rope_head_dim = 0`) and no output gate.

```
q   = q_b(RMSNorm(q_a(x)))          [64][256]
kv  = kv_b(RMSNorm(kv_a(x)))        [64][256 + 256]   -> k, v
out = o_proj( softmax(q.k / sqrt(256)) causal . v )
```

The checkpoint also carries a DeepSeek Sparse Attention (DSA) indexer. It selects
`index_topk / index_kpool = 512` pools of 4 tokens plus the incomplete tail, so for a
sequence of at most `index_topk + index_kpool - 1 = 2051` positions it selects every
visible token and the sparse attention equals dense causal attention exactly. Up to there
the engine computes dense attention; past it the indexer selects (2.6).

### 1.4 Feed-forward

- SwiGLU with clamping: `down( silu(min(gate, 10)) * clamp(up, -10, 10) )`.
- Router: `scores = sigmoid(W x)`; select top-8 of `scores + e_score_correction_bias`;
  weights are the unbiased `scores` of the selected experts, renormalised to sum 1 and
  multiplied by 2.5.
- MoE output: `sum_j w_j * expert_j(x) + shared_expert(x)`.

### 1.5 Storage formats

| tensors | dtype |
|---|---|
| KDA projections, `kv_b_proj`, mHC `fn`, router, norms, embeddings, `lm_head` | BF16 |
| MLA `q_a`, `q_b`, `kv_a`, `o_proj`; dense MLP; shared and routed experts | FP8 E4M3 + F32 `weight_scale_inv` |
| `A_log`, `dt_bias`, mHC `base`/`scale`, router correction bias | F32 |

FP8 weights are block-quantised: element `(i, j)` is `E4M3(code) * scale_inv[i/128][j/128]`.

## 2. The engine

```
include/glm53f/glm53f.h       types (config, self-describing matrices, weight structs), kernel API
include/glm53f/glm53f_cfg.h   strict config.json reader
src/core/glm53f_ops.c         kernels: matmul (fp32/BF16/FP8), norms, SwiGLU, conv, mHC, KDA, MLA, router, MoE, decoder layer
src/io/glm53f_st.c/.h         safetensors index (hash table over 76k tensors), positioned and O_DIRECT reads
src/io/glm53f_load.c/.h       routed-expert geometry and two-read expert loading
src/io/glm53f_i4file.c/.h     the int4 expert container: writing, checking, opening per layer
src/io/glm53f_portable_io.h   POSIX shims for Windows/macOS (pread, O_DIRECT, aligned allocation)
src/cache/glm53f_cache.c/.h   LRU cache of routed experts, parallel batch prefetch
src/model/glm53f_bind.c/.h    checkpoint tensors -> weight structs, with dtype and shape checks
src/model/glm53f_model.c/.h   open, incremental forward, logits
src/tokenizer/glm53f_tok.h    tokenizer from tiktoken.model + tokenizer_config.json
src/gpu/glm53f_gpu.c/.h       CUDA backend, host side: placement, fp32 copies, keep-warm, forward (optional)
src/gpu/glm53f_gpu_kernels.*  CUDA kernels mirroring glm53f_ops.c
src/cli/glm53f_run.c          command line: chat template, greedy decode, EOS, reports
src/cli/glm53f_chat.c/.h      interactive session (localcode, --chat): commands, a turn as events, /save
src/cli/glm53f_tui.c          localcode's full-screen terminal interface (worker thread, raw VT input)
third_party/                  json.h, tok.h (BPE), Unicode tables
```

### 2.1 Where the bytes live

| data | location | size |
|---|---|---|
| trunk: all 45 layers except routed experts | RAM, loaded at start | 12.63 GB |
| embeddings + lm_head (BF16) | RAM | 2.54 GB |
| routed experts (12,096 x 25.19 MB, FP8) | NVMe, streamed | ~305 GB |
| expert cache | RAM, `--cache-gb` | e.g. 24 GB = 952 experts |
| with `--gpu`: trunk + lm_head | GPU memory; the host keeps embeddings + routers (1.47 GB) | 12.7 GB (+7.3 GB fp32 copies) |
| KDA recurrent state | RAM | 202 MB |
| MLA KV cache (expanded k and v, fp32) | RAM | 1.44 MB per position |

### 2.2 Matrices

Every matrix is a `Glm53fMat` carrying its own dtype (`GLM53F_WF32`, `GLM53F_WBF16`,
`GLM53F_WF8`), shape and, for FP8, a pointer to its scale grid. One layer mixes all three
formats, so the dtype cannot live on the layer struct. `glm53f_mm` dispatches:

- **BF16**: widened inside the dot product (`bf16 -> f32` is a 16-bit shift), never stored as fp32.
- **FP8**: codes decoded inside the dot product; each (row, column-block) sum is scaled
  once. Never dequantised to a matrix. With AVX2 eight codes are decoded at a time without
  a table: |v| = (m + 8[e != 0]) << max(e, 1) times 2^-10, sign from bit 7, NaN for
  0x7f/0xff; the integer is exact in float and 2^-10 a power of two, so the values equal
  the 256-entry table bit for bit, at half the cost of a gather.
- **int4** (`--experts int4`, optional): the expert cache re-quantises FP8 experts on the
  way in, one step per group of 64 columns, `step = max|w| / 7` and `level = round(w/step)`
  clamped to [-8, 7], two levels per byte. 14.16 MB instead of 25.19 MB per expert, so the
  same RAM holds 1.8x more of them. Since a group lies inside one FP8 scale block, the
  block scale cancels out of the level and only the stored step carries it. The result is
  an approximation of the checkpoint (PERFORMANCE.md), so it is off by default.
- Sums use double accumulators in a fixed reduction order; OpenMP splits only over output
  rows, so results do not depend on the thread count.

### 2.3 Forward pass

`glm53f_model_forward(m, ids, T, logits, argmax_all)` appends T positions to the session:

1. embedding rows copied into 4 streams;
2. per layer `glm53f_decoder_layer`: mHC site -> RMSNorm -> KDA (state carried in place)
   or MLA (writes positions `[cached, cached+T)` into the KV cache, attends causally) ->
   mHC expand; mHC site -> RMSNorm -> dense MLP or MoE -> mHC expand;
3. mean of streams -> final RMSNorm -> `lm_head`.

A prompt is fed in chunks of 256 positions (`GLM53F_CPU_CHUNK`): what a chunk leaves behind
(KDA recurrence, KV cache, indexer state) is exactly what the next one needs, so the working
buffers stay the size of one chunk instead of growing with the prompt.

The chunks go **layer by layer**: every chunk passes layer L before any chunk enters layer
L+1. Nothing a chunk computes at layer L depends on layer L+1, so the order changes no sum,
but it decides how often routed experts are read. Chunk by chunk through all 45 layers, each
chunk routed to nearly all 288 experts of every layer and evicted the previous chunk's, and a
372-token prompt read 431 GB, more than the 305 GB of experts the model has. Layer by layer,
a layer's experts are read once for the whole prompt (PERFORMANCE.md, Prefill). The residual
streams of up to `GLM53F_PREFILL_SPAN` = 4096 positions (65 KB each) wait between layers;
longer prompts go span by span. With `--gpu` the streams wait in host memory and each chunk
visits the layer's device.

For the same reason a prefill hints the next layer's predicted experts (2.5) only when the
cache holds two layers' worth of experts: with room for one, those reads would evict experts
the current layer's later chunks still need.

`GLM53F_CHUNK` and `GLM53F_SPAN` override both sizes; `tiny_oracle_chunked` (chunks of 7)
and `tiny_oracle_spans` (chunks of 7 in spans of 17, layers split across two GPUs in a CUDA
build) run the whole oracle to show the results do not depend on them. Each decode step is
one call with one token.

### 2.4 MoE and expert streaming

`glm53f_moe` works expert-major over chunks of up to 64 tokens: route every token, collect
the distinct experts, fetch them in batches of 16 through the cache (`getmany` issues the
reads concurrently), apply each expert to every (token, slot) that selected it, then sum
each token's contributions in top-k order plus the shared expert.

On disk, each expert's three weight matrices form one contiguous run (25,171,968 bytes)
and its three scale grids another, so a miss is one O_DIRECT read plus one small read.
The cache is a fixed arena of page-aligned slots with LRU eviction; a slot being filled
by a batch read is marked in-flight so no other read can claim it.

### 2.5 Predictive prefetch

Storage throughput is fixed (~2.6 GB/s on the reference SSD, independent of concurrency),
so expert reads can only get cheaper in wall time by overlapping with compute. In decode
the MoE of layer L, right after computing its real routing:

1. hints its own experts to the cache's background readers (`src->hint`); pending
   predictions for layer L that turned out wrong are cancelled before they are read;
2. applies layer L+1's router to its own input and hints the top N experts for L+1
   (the next router predicts 75% of the actual top-8, 90% of it within the top-16);
3. computes the shared expert while the readers work;
4. takes each routed expert with `get()`, which waits for a read already in flight
   instead of starting a second one, and releases it after use.

The cache keeps all bookkeeping under one mutex; reads happen outside it into slots
reserved as in-flight, and experts being multiplied are pinned so no background read can
evict them. Prefetch only changes when experts become resident: the experts multiplied,
and the order of every sum, are the same, so output is bit-identical with prefetch on or
off (gated by `test_glm_tiny` GATE 5, 20 runs with a 9-slot cache).

Prefill works the same way over a chunk of tokens: the layer hints the union of experts
its tokens routed to, predicts the next layer's top N for every token, and hints that
union ordered by how many tokens voted for each expert. The main thread then multiplies
each expert as it lands, so expert compute overlaps the reads still underway. (The
shared expert stays at the end in prefill; computing it first only pays off in decode.)
`GLM53F_NO_PREFILL_PREFETCH` and `GLM53F_PREFILL_PRED` exist for A/B measurement.

Wasted reads are expensive because the disk is the bottleneck: predicting 16 experts per
layer made decode slower than no prefetch at all. N = 6 with 2 readers measured best for
decode and is also the best measured per-token prediction count for prefill.

### 2.6 DSA: which positions a query attends to

`Glm5NextTextIndexer`, implemented in `src/core/glm53f_dsa.c`. Per MLA layer and position it
keeps a light key and a gate vector:

```
k_j    = LayerNorm(wk . x_j, k_norm, eps 1e-6)                 [index_dim = 128]
g_j    = index_kpool_compress_gate . x_j                       [index_dim]
pool p = positions [4p, 4p+4), complete pools only
P_p[c] = sum_j softmax_j(g_[4p+j][c] + ape[j][c]) * k_[4p+j][c]   (softmax per channel)
q_t    = wq_b . q_resid_t                                      [32][128]
w_t    = (weights_proj . x_t) / sqrt(32)                       [32]
s_t[p] = sum_h w_t[h] * relu(q_t[h] . P_p / sqrt(128))
```

The `index_topk / index_kpool` = 512 best pools become 2048 positions, and the incomplete
tail pool (up to 3 positions) is always appended, so a query attends to at most 2051 of
them. While the complete pools fit in the budget every visible position is selected: that
is the dense regime the engine was limited to before, and the selection is skipped there,
which keeps those sessions bit-identical to the previous engine.

State per MLA layer is `2*cap*128 + cap/4*128` floats (1.15 KB per position per layer,
12.7 KB per position over the 11 MLA layers), against 1.44 MB per position for the KV
cache, which remains what bounds a long session.

Validation: `test_glm_tiny` GATE 4 runs a 40-position session on the tiny model, whose
`index_topk` of 16 makes the indexer select from position 19 on, and matches the reference
at all 40 positions. `--gpu` keeps its dense kernels, so it refuses sessions longer than
the dense-equivalent range and leaves the trunk on the CPU.

### 2.7 KV cache: expanded or compressed (`--kv`)

MLA compresses a position into a `kv_lora` latent of 512 floats and expands it into 64
heads of keys and values through `kv_b`. The cache can hold either form:

| form | per position | how attention runs |
|---|---|---|
| expanded (`--kv expanded`) | 1.44 MB over the 11 MLA layers | scores against stored keys, sums stored values |
| compressed (`--kv compressed`) | 22 KB, 64x less | folds `kv_b` into the query and the output |

The compressed form uses the identity that makes MLA cheap:

```
q . (W_k c)         = (W_k^T q) . c            score against the latent itself
sum_j p_j (W_v c_j) = W_v (sum_j p_j c_j)      expand once per query, not per key
```

so `kv_b` is applied twice per query and head (once to the query, once to the weighted sum
of latents) instead of once per stored position. Per attended position the attention does
2 x kv_lora multiply-adds instead of qk_nope + v_head, which is twice the work, and the
cache stops being what limits a long session: 32,768 positions cost 0.72 GB instead of
47 GB. Exact arithmetic gives the same value; in floating point the order of the sums
differs, and on the released checkpoint the two forms land 7.6e-06 apart, against a
distance of 1.5e-05 from the reference itself.

`--kv auto`, the default, picks compressed only past the dense-attention range (2051
positions), where the expanded cache would not fit anyway. `--gpu` expands on the device
and takes the expanded form.

### 2.8 CUDA backend (`--gpu`)

Built with `-DGLM53F_CUDA=ON`. The trunk moves to the GPUs; routed experts stay where the
bytes are, on the CPU with the cache and prefetch of 2.4-2.5.

**Placement.** Layers go to devices in order, each device taking a share of the trunk
proportional to its free memory (keeping 768 MB spare plus its scratch buffers); the
last device also holds the final norm and lm_head. The residual streams cross between
devices once per forward through pinned host memory. Once uploaded, the host copy of the
trunk is freed except the embedding table (looked up on the host) and the MoE routers
(the CPU routes and predicts experts): ~13.9 GB returned to the expert cache.

**Per layer.** mHC collapse, RMSNorm, KDA or MLA, mHC expand, mHC collapse, RMSNorm run on
the device. A dense layer runs its MLP there too. A MoE layer copies the normalised input
to the host (4096 floats per token), launches the shared expert on the device, runs
`glm53f_moe_routed` (routing, prefetch hints, routed experts) on the CPU meanwhile, copies
the routed sum back and adds it.

**Matrix products.** cuBLAS SGEMM on fp32 values. Stored BF16 and FP8 matrices are widened
to fp32 on the device (BF16 exactly; FP8 as transformers dequantises: code value times
block scale). After upload, the spare memory of every device holds persistent fp32
copies (lm_head first, then layers in order; 7.3 GB on two 12 GB cards), which removes
the per-token widening for those matrices; the rest are widened per product into a
256 MB buffer, a block of rows at a time for matrices larger than it.

**Kernels.** GPUs run long sequential loops inside one thread slowly, so the kernels are
written with short loops per thread: RMSNorm sums squares per block of 256 and scales per
element; the KDA delta rule is four launches per token (decay, read, write, output), one
thread per matrix element or column; MLA attention is four launches (scores per
position, softmax normaliser per head, probabilities, output per value dimension). The
KDA and MLA kernels keep the CPU's order of every sum, so they are bit-identical to the
per-head kernels they replaced; RMSNorm and cuBLAS change the last bits of sums.
`GLM53F_GPU_EXACT=1` selects kernels that repeat the CPU arithmetic bit for bit.

**Keep-warm.** A decode step alternates ~0.25 s of device work with ~1.6 s of routed
experts on the CPU. At that duty cycle the driver lowers the cards' performance state
(P3: memory clocked at 5 GHz instead of 7.3), and the same trunk work then takes ~0.6 s.
While the CPU computes experts, a host thread keeps each device busy with a throwaway
4096 x 4096 SGEMV at a time, waiting on a blocking CUDA event (no CPU spin); model kernels
queue behind at most one of them (~0.2 ms). The cards stay in P2 at ~85-90 W each.
`GLM53F_GPU_WARM=0` disables it.

**Routed experts on a device (optional).** With `GLM53F_GPU_EXPERTS=N` and int4 experts, a
decode MoE layer hands up to N of its routed experts to one device (`Glm53fExpertOffload`,
glm53f.h) and multiplies the others on the CPU at the same time. The weights stay in the
host cache and cross PCIe per use, which only beats the CPU from page-locked memory, so
the first `GLM53F_GPU_EXPERTS_PIN_GB` (default 24) of the cache arena are registered with
`cudaHostRegister` and only experts whose slot lies there are sent. The device kernel
repeats `matmul_i4` operation for operation (float accumulators fed by fmaf, the same
double reduction tree, groups summed in order) and contributions are still summed in top-k
order, so the output does not depend on where an expert ran. If the device fails, the CPU
computes those experts and keeps them from then on.

Validation: `test_glm_tiny` GATE G1-G3 repeat the reference checks with the trunk on the
GPU and layers split across devices; on the real checkpoint the 45-layer logits stay within
2.1e-5 of the reference and 40 greedy tokens match the CPU run (VALIDATION.md 2.2).

### 2.9 MTP layer (multi-token prediction)

`GLM53F_MTP_STATS=1` binds checkpoint layer `n_layers` ("nextn"), which the released
transformers does not implement (it skips `layers.45.`). From the hidden state of position
i (the mean of the residual streams, before the final norm) and the embedding of the token
at i+1 it predicts the token at i+2:

```
x     = eh_proj([ RMSNorm(embed(tok_i+1), enorm), RMSNorm(h_i, hnorm) ])
x     = x + MLA(RMSNorm(x, input_layernorm))            own KV cache, dense causal
x     = x + MoE(RMSNorm(x, post_attention_layernorm))   own 288 experts, same stream cache
draft = argmax(lm_head(RMSNorm(x, shared_head.norm)))
```

Unlike the trunk, the layer has no mHC: it is a classic pre-norm residual block. It is only
a draft: a speculative decoder accepts it solely when the full model produces the same
token, so a wrong draft costs speed, never output - which is what makes a layer with no
reference implementation safe to run. Today the engine only measures it (88.6% of drafts
match; see PERFORMANCE.md for why speculation is not worth it on this machine). The layer
needs the lm_head on the host, so it is unavailable together with `--gpu`.

### 2.10 Failure policy

A wrong model that prints fluent text is the failure this engine is built to avoid:

- the config reader refuses missing fields and settings the kernels do not implement;
- the binder refuses missing or mis-shaped tensors;
- an expert that fails to load makes the forward fail (`glm53f_expert_drops`);
- sessions beyond the dense-attention limit are refused;
- allocation failure inside a kernel aborts rather than returning stale buffers.
