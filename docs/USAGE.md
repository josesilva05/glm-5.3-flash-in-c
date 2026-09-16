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
build/Release/glm53f <model_dir> --prompt-file prompt.txt --gen 400 --cache-gb 24 --quiet
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
| `--cache-gb X` | routed-expert cache in GB (default 16) |
| `--experts fp8\|int4` | `fp8` (default) multiplies the checkpoint's own expert weights. `int4` re-quantises them inside the cache (1.8x more experts in the same RAM, ~13% faster decode) and makes the output an approximation of the model, not the model |
| `--gpu` | run the trunk (attention, mHC, dense and shared-expert MLPs, lm_head) on all CUDA devices; routed experts stay on the CPU. Needs a `-DGLM53F_CUDA=ON` build. Falls back to the CPU with a message if there is no device or the trunk does not fit. The host copy of the trunk is then freed, so raise `--cache-gb` by ~13 GB (see below) |
| `--prefetch N` | predictive expert prefetch in prefill and decode: background readers fetch each MoE layer's experts and the N most likely experts of the next layer (per token) while compute continues (default 6, `0` = off; output is identical either way). `GLM53F_IO_THREADS` sets the reader count (default 2). With `--gpu`, where compute is faster, `--prefetch 4` measured marginally better (PERFORMANCE.md) |
| `--config PATH` | config file (default `<model_dir>/config.json`) |
| `--tok DIR` | tokenizer directory (default `<model_dir>`) |
| `--layers N` | bind only the first N layers (diagnostics; output is not the model's) |
| `--dump-logits PATH` | float32 logits of the prompt's last position |
| `--out FILE` | JSON report (default `glm53f_run.json`) |
| `--quiet` | stream the text instead of the per-step table |

Limits: prompt + generated tokens <= 2051 (see ARCHITECTURE.md, MLA); `--gen` <= 8192.

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

- Give the expert cache the RAM that is left after ~16 GB for the trunk and a few GB for
  the operating system. More cache means fewer disk reads per token.
- With `--gpu` only ~1.5 GB of the trunk stays in RAM (embeddings and MoE routers), so the
  cache can take the other ~14 GB: on 64 GB, `--gpu --cache-gb 38` has the same peak RSS
  (~40 GB) as `--cache-gb 24` on the CPU.
- Do not push the system to its commit limit: paging costs far more than a smaller cache.
  The engine checks the free physical memory when it opens the model and shrinks the cache
  to fit (leaving 3 GB of headroom), reporting what it used; with `--gpu` it counts the
  trunk's RAM as free, since the upload releases it. The check sees the memory free at that
  moment, so leave room for whatever you open later: on 64 GB with a browser and an editor
  running, `--cache-gb 30` to `36` is comfortable, and 44 is not.
