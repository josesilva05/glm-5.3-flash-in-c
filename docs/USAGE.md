# Usage

## Requirements

- x86-64 CPU with AVX2 + FMA (ARM builds use portable scalar kernels).
- RAM: ~16 GB for the trunk plus whatever you give the expert cache. 64 GB with
  `--cache-gb 24` is comfortable; 30 GB of cache on 64 GB leaves the system at ~96% RAM.
- The GLM-5.3-Flash checkpoint on a fast local NVMe (routed experts are read during decode).
- CMake 3.16+ and MSVC, GCC or Clang; OpenMP.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

On Linux/macOS/MSYS2 `make -j` works too. The binary is `build/Release/glm53f(.exe)`
(MSVC) or `build/glm53f` / `bin/glm53f`.

### With the CUDA backend (`--gpu`)

Needs the CUDA Toolkit (12 or newer; measured with 13.3) and an NVIDIA GPU of compute
capability 8.6 (RTX 30 series; set `CMAKE_CUDA_ARCHITECTURES` for others).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGLM53F_CUDA=ON
cmake --build build --config Release -j
```

With MSVC, delete `build/Release/*.exe` before rebuilding after changing library sources:
MSBuild can rebuild `glm53f.lib` without relinking the executables.

## Run

```bash
build/Release/glm53f <model_dir> --prompt-file prompt.txt --gen 400 --gpu --quiet
```

The prompt is wrapped in the model's chat template for one user turn:

```
[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{prompt}<|assistant|><think>
```

so the model first writes its reasoning (in `<think>`, usually in English) and then the
answer. Generation stops at an EOS token from the config (`<|endoftext|>`, `<|user|>`,
`<|observation|>`).

### Options

| option | meaning |
|---|---|
| `--prompt TEXT` | user message, wrapped in the chat template |
| `--prompt-file PATH` | the same, read from a UTF-8 file (use this for non-ASCII text on Windows) |
| `--ids 1,2,3` | raw token ids: no template, no tokenizer |
| `--raw` | tokenize the prompt without the chat template |
| `--reasoning max\|high\|low` | the template's Reasoning Effort line (default max) |
| `--gen N` | tokens to generate (default 256) |
| `--no-stop` | do not stop at EOS |
| `--cache-gb X\|auto` | routed-expert cache in GB. `auto` (default) takes the RAM free when the model opens, minus what the session needs (attention caches, indexer, work buffers) and a fifth of the installed RAM kept for the system; the size used is printed. With `--gpu` the trunk's RAM counts as free |
| `--kv auto\|expanded\|compressed` | how the MLA cache stores a position: expanded keys and values (1.44 MB per position) or the `kv_lora` latent (22 KB, expanded once per query, ARCHITECTURE.md 2.7). `auto` (default) uses compressed past 2051 positions, where the expanded cache no longer fits |
| `--experts fp8\|int4` | `fp8` (default) multiplies the checkpoint's own expert weights. `int4` re-quantises them inside the cache (1.8x more experts in the same RAM, ~13% faster decode) and makes the output an approximation of the model, not the model |
| `--int4-dir DIR` | read the routed experts from an int4 container written by `--write-int4` (below): 14.2 MB per read instead of 25.2 MB, bit-identical to `--experts int4` and like it an approximation of the checkpoint. Implies `--experts int4`; layers missing from the container are quantised on the way in |
| `--gpu` | run the trunk (attention, mHC, dense and shared-expert MLPs, lm_head) on all CUDA devices; routed experts stay on the CPU. Needs a `-DGLM53F_CUDA=ON` build. Falls back to the CPU with a message if there is no device or the trunk does not fit. The host copy of the trunk is then freed, so raise `--cache-gb` by ~13 GB (see below) |
| `--prefetch N` | predictive expert prefetch in prefill and decode: background readers fetch each MoE layer's experts and the N most likely experts of the next layer (per token) while compute continues (default 6, `0` = off; output is identical either way). `GLM53F_IO_THREADS` sets the reader count (default 2). With `--gpu`, where compute is faster, `--prefetch 4` measured marginally better (PERFORMANCE.md) |
| `--config PATH` | config file (default `<model_dir>/config.json`) |
| `--tok DIR` | tokenizer directory (default `<model_dir>`) |
| `--layers N` | bind only the first N layers (diagnostics; output is not the model's) |
| `--dump-logits PATH` | float32 logits of the prompt's last position |
| `--out FILE` | JSON report (default `glm53f_run.json`) |
| `--quiet` | stream the text instead of the per-step table |
| `--chat` | interactive session, as `localcode` (below); takes no prompt |
| `--ctx N` | positions an interactive session may hold (default 2048, 4096 without `--gpu`) |

Limits: prompt + generated tokens <= 32,768 (`--gpu` handles up to 2051, the range where
dense attention equals the model's sparse attention); `--gen` <= 8192. Past 2051 positions
the KV cache switches to the compressed form (22 KB per position instead of 1.44 MB), so
an 8k session holds ~180 MB of it and a 32k session ~0.7 GB.

### The int4 expert container

```bash
build/Release/glm53f <model_dir> --write-int4 <container_dir>            # all 42 MoE layers, ~171 GB
build/Release/glm53f <model_dir> --write-int4 <container_dir> --i4-layers 3-27
build/Release/glm53f <model_dir> --verify-int4 <container_dir>
build/Release/localcode <model_dir> --int4-dir <container_dir>
```

The container holds the routed experts already quantised to int4, one file per MoE layer
(4.08 GB each). Reading a layer's experts from it costs 44% fewer bytes than the FP8
checkpoint and no arithmetic, which is what sets decode speed (PERFORMANCE.md). The
checkpoint is never changed, and a run without `--int4-dir` is the exact model again.

Writing is resumable: a layer already complete is skipped, and a stopped conversion leaves
a `.part` file that is never used. Each file carries a fingerprint of the checkpoint it came
from (a container from another checkpoint is refused) and a CRC per expert, which
`--verify-int4` checks. A partial container works: its layers are read from it, the others
quantised on the way in. Writing reads the FP8 experts at full speed; a drive without a
heatsink or with a small SLC cache can slow down after tens of GB.

### localcode: the interactive session

```bash
build/Release/localcode <model_dir>
```

`localcode` is the same program opening straight into a conversation, on the GPUs when the
build has CUDA (`--cpu` keeps the trunk on the CPU). The model directory can also come from
`LOCALCODE_MODEL`. `glm53f <model_dir> --chat --gpu` is the same session.

The model stays loaded between messages, and each turn feeds only its new tokens: what the
earlier turns left in the KV cache, the KDA state and the indexer is reused, so a follow-up
question does not re-read the conversation.

On a terminal it takes the whole window, in the layout of terminal coding tools and a green
palette: the conversation scrolls above an input box that stays at the bottom, with a footer
showing the working directory and the context used. The reasoning is folded into one live
line (`Thinking · 42 tokens · 38s`, then `+ Thought: 81s`), the answer's Markdown is
rendered as it streams, and each answer ends with its tokens, s/token and time. The model
runs on its own thread, so the screen stays responsive while it works. Answers may run to
2048 tokens (`--gen`).

| key | effect |
|---|---|
| enter | send the message; while an answer is being written it waits in a queue |
| esc, ctrl+c | stop the answer being written (the conversation keeps what was written) |
| ctrl+c on an empty box, ctrl+d | leave |
| pgup, pgdn, mouse wheel | scroll the conversation; end follows it again |
| up, down | earlier messages |
| left, right, home, end, ctrl+a, ctrl+e, ctrl+u, ctrl+w | edit the message |
| paste | text with line breaks stays one message |

| command | effect |
|---|---|
| `/file PATH [question]` | send a file's text followed by the question; a path with spaces goes in quotes |
| `/reset` (`/new`, `/clear`) | forget the conversation and start a new one |
| `/save [FILE]` | write the conversation as Markdown: each message, the reasoning in a folded `<details>` block, the answer and its numbers (default `glm53f-chat-YYYYMMDD-HHMMSS.md`) |
| `/thinking` | show the reasoning as it is written, or fold it again |
| `/reasoning max\|high\|low` | Reasoning Effort for the following turns |
| `/gen N` | tokens to generate per answer |
| `/params` | the settings in force and the context used |
| `/help`, `/quit` | |

When input or output is not a terminal (a pipe, a captured session) the session runs line
by line without colour. `LOCALCODE_PLAIN=1` (or `NO_COLOR`) asks for that mode on a terminal
too; `LOCALCODE_TUI=1` forces the full screen, sized by `COLUMNS` and `LINES`, which is how
the screen is tested from a script.

### Output

Without `--quiet`, one row per generated token:

```
STEP   TOKEN    SECONDS   DISK S    READ GB    TOK/S
0      785      53.17     30.42     71.71      0.019     <- prefill of the whole prompt
1      1196     4.30      2.87      7.30       0.233
```

`DISK S` is time spent waiting for expert reads, `READ GB` the expert bytes read. The run
ends with the decoded text, prefill/decode timings, the disk/compute split, the expert
bytes read (on demand and read ahead) with the average per decode token, peak RSS and a
cache report. The JSON report holds prompt ids, generated ids and text, and the timings.

## Windows notes

- Write prompt files as UTF-8 **without BOM** and without a trailing newline, or those bytes
  become part of the prompt. In Windows PowerShell 5.1:
  `[System.IO.File]::WriteAllText("$PWD\pergunta.txt", "Qual a capital do Brasil?")`
- The CLI switches the console to UTF-8 itself.

## Diagnostics environment variables

| variable | meaning |
|---|---|
| `GLM53F_PREDICT_STATS=1` | report how predictable each layer's routing is |
| `GLM53F_MTP_STATS=1` | bind the checkpoint's MTP layer, draft a token per decode step and report how often the draft equals the token the model produced (CPU path only; does not change the output) |
| `GLM53F_CHUNK=N` | positions per prefill chunk (default 256); the output does not depend on it |
| `GLM53F_SPAN=N` | positions a prefill takes layer by layer at once (default 4096, ARCHITECTURE.md 2.3); the output does not depend on it |
| `GLM53F_IO_PRIORITY=low` | (Windows) mark the checkpoint's reads low priority, so the system and other programs, including the page file on the same drive, go first and the machine stays responsive while the model reads. Off by default: Windows also slows low-priority reads on an idle drive, and decode measured 13% slower (PERFORMANCE.md) |
| `GLM53F_ROUTE_TRACE=path` | append one line per decode MoE layer: layer, the 8 routed experts, microseconds waiting for reads, microseconds computing |

## GPU environment variables

| variable | meaning |
|---|---|
| `GLM53F_GPU_WARM=0` | turn off keep-warm (see ARCHITECTURE.md 2.6); saves ~120 W while decoding and costs speed |
| `GLM53F_GPU_WIDEN=0` | do not keep fp32 copies of weights in spare device memory |
| `GLM53F_GPU_EXACT=1` | use the fused-multiply-add kernels that repeat the CPU arithmetic bit for bit (slow; for debugging differences) |
| `GLM53F_GPU_PROFILE=1` | at exit, print decode time split into GPU trunk, CPU routed experts and head |
| `GLM53F_GPU_PROFILE=2` | also synchronise after every section of a decode step and print per-section times (perturbs timing) |
| `GLM53F_GPU_LAYERS_PER_DEV=N` | cap layers per device (forces a multi-device placement; tests use it) |

## Tuning memory

- By default (`--cache-gb auto`) the cache takes what is free when the model opens, keeping
  a fifth of the installed RAM (12.8 GB on 64 GB) for the system and whatever you open
  later. Pass a size to choose yourself. More cache means fewer disk reads per token.
- With `--gpu` only ~1.5 GB of the trunk stays in RAM (embeddings and MoE routers), so the
  cache can take the other ~14 GB: on 64 GB, `--gpu --cache-gb 38` has the same peak RSS
  (~40 GB) as `--cache-gb 24` on the CPU.
- Do not push the system to its commit limit: paging costs far more than a smaller cache.
  The engine checks the free physical memory when it opens the model and shrinks the cache
  to fit (leaving 3 GB of headroom), reporting what it used; with `--gpu` it counts the
  trunk's RAM as free, since the upload releases it. The check sees the memory free at that
  moment, so leave room for whatever you open later: on 64 GB with a browser and an editor
  running, `--cache-gb 30` to `36` is comfortable, and 44 is not.
