# Performance

All figures measured on one machine: AMD Ryzen 7 5700X3D (8 cores / 16 threads, AVX2),
64 GB DDR4, NVMe SSD, 2x RTX 3060 12 GB (one on PCIe 3.0 x4, one on 4.0 x16), Windows 11,
MSVC Release build, CUDA 13.3, 18-token chat prompt.

## Measured

| run | cache | prefill | decode | notes |
|---|---|---|---|---|
| 234 generated tokens | 30 GB | 54.7 s | 2.90 s/token | peak RSS 45.7 GB; system RAM at ~96% |
| 60 generated tokens | 24 GB | 53.2 s | 3.27 s/token | peak RSS 39.5 GB |

Loading: 7 s for the 15.2 GB trunk (layers read in parallel).

## Where decode time goes (24 GB cache, 59 decode steps)

| part | s/token | share |
|---|---|---|
| waiting for routed-expert reads | 1.87 | 57% |
| computing on the CPU | 1.40 | 43% |

The cache served 53% of expert requests from RAM; the rest were read at ~2.6 GB/s,
about 4-5 GB per token. Routing is spread across all 12,096 experts, so the hit rate grows
slowly with cache size.

Estimated multiply-adds per token, by component:

| component | billions | share |
|---|---|---|
| routed experts (42 layers x 8, FP8) | 8.46 | 51% |
| KDA attention (34 layers, BF16) | 4.68 | 28% |
| MLA attention (11 layers) | 1.29 | 8% |
| shared experts (42 layers, FP8) | 1.06 | 6% |
| lm_head | 0.63 | 4% |
| dense MLP (3 layers) | 0.45 | 3% |
| mHC mixers | 0.04 | <1% |
| **total** | **16.6** | |

## What limits speed on this hardware

- **Disk.** While routed experts that are not cached must be read from the SSD, the time
  per token cannot fall below the read time, whatever computes the rest.
- **Memory bandwidth.** Even with every expert in RAM, a token touches ~23 GB of weights
  (15 GB trunk + 8.5 GB active experts). Dual-channel DDR4 moves ~50 GB/s, so a
  CPU-only engine is bounded to roughly 2 tokens/s here.
- **VRAM.** Two 12 GB GPUs hold the trunk (see "CUDA backend" below), but not the ~305 GB
  of experts.

## Storage throughput

(Measured in September 2026 the checkpoint's files read slower; see "Reading the FP8 checkpoint got slower" below.)

`build/Release/bench_expert_io <model_dir> [experts]` reads random routed experts through the
engine's O_DIRECT path at 1-32 concurrent reads.

| concurrent reads | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| GB/s (one process) | 2.45 | 2.32 | 2.53 | 2.40 | 2.41 |

Two processes at once get ~1.3 GB/s each, so ~2.6 GB/s is the drive's own sustained limit
(an OEM PCIe 4.0 x4 NVMe, `PCH-ALDLP-1TB`), not a software queue. Read-only handles on
Windows are nevertheless opened with `FILE_FLAG_OVERLAPPED`, since a synchronous handle
serialises concurrent reads on the file object.

## Expert predictability

With `GLM53F_PREDICT_STATS=1` the CLI reports how much of each layer's routing is known
before the layer runs (40 decode tokens, 1,558 layer-steps):

| predictor | share of actual experts |
|---|---|
| router of layer L applied to layer L-1's MoE input, top-8 | 75.0% |
| same, top-16 | 90.2% |
| experts layer L used for the previous token | 32.3% |
| top-8 prediction or previous token | 80.7% |

Since storage throughput is fixed, reads can only get cheaper in wall time by overlapping
with compute: prefetching a layer's predicted experts while the previous layer computes.

## Predictive prefetch (`--prefetch N`)

40 generated tokens, 24 GB cache, same prompt; generated ids identical in every run.

| configuration | decode s/token |
|---|---|
| no prefetch (3 runs) | 2.96, 2.98, 2.96 |
| N = 16, 8 readers | 4.66 |
| N = 8, 8 readers | 2.74 |
| N = 8, 1 reader, cancellation of wrong predictions | 2.58 |
| N = 8, 2 readers, cancellation | 2.56 |
| N = 4, 2 readers, cancellation | 2.32 |
| **N = 6, 2 readers, cancellation (3 runs) — default** | **2.29, 2.28, 2.29** |

Decode is 23% faster with the default. With N = 6 about 90% of the experts read ahead
are used; the main thread's wait on expert reads drops from 1.88 to ~1.1 s/token.

## Prefill

The 18-token prompt reads ~72 GB of experts because different tokens route to different
experts. Without prefetch that is ~28 s of waiting inside a ~46 s prefill.

Prefill with prefetch (`--gen 1`, 24 GB cache; logits bit-identical in every run):

| configuration | prefill | main-thread wait on reads |
|---|---|---|
| no prefetch (3 runs) | 45.65, 45.95, 45.85 s | 28.2 s |
| this layer's experts only, no prediction | 36.24 s | 18.0 s |
| + next-layer prediction, 2 per token | 30.82 s | 12.4 s |
| + next-layer prediction, 4 per token | 30.36 s | 11.4 s |
| **+ next-layer prediction, 6 per token (3 runs) — default** | **30.35, 30.51, 31.44 s** | ~11.6 s |

Prefill is 33% faster with the default; ~97% of the experts read ahead are used.

### Layer by layer (a 372-token document)

A prompt is fed in chunks of 256 positions. Taken chunk by chunk through all 45 layers,
every chunk routes to nearly all 288 experts of every layer, and by the time the next chunk
reaches a layer the cache has moved on to the layers after it: the 372-token prompt read
431 GB of experts, more than the 305 GB the model has. Taken layer by layer (ARCHITECTURE.md
2.3), a layer's experts are read once for the whole prompt.

`--gpu --cache-gb 16`, `--gen 1`, interleaved runs, the same prompt; logits of the last
position byte-identical between the two:

| order | prefill (3 runs) | experts read | of which on demand |
|---|---|---|---|
| chunk by chunk (before) | 196.1, 191.0, 192.2 s | 431 GB | 39 GB |
| **layer by layer** | **118.6, 119.0, 120.1 s** | **257 GB** | 4 GB |

38% faster. The gain grows with the prompt: chunk by chunk the reads grow with the number of
chunks, layer by layer they are bounded by the experts the prompt actually uses.

With the first 6 layers only (`--layers 6`, 3 MoE layers) the same prompt read 46 GB before
and 21.7 GB after, and showed the one thing layer-major ordering changes for prefetch: with an
8 GB cache (one layer of experts) hinting the next layer's predicted experts evicted experts
the current layer still needed, and the reads stayed at 43 GB. A prefill therefore predicts
the next layer only when the cache holds two layers' experts (576 slots, 14.5 GB).

| `--layers 6 --gpu` | 8 GB cache | 16 GB cache |
|---|---|---|
| chunk by chunk | 20.5 s, 46.2 GB | 18.1 s, 40.4 GB |
| layer by layer | 14.2 s, 21.7 GB | 11.7 s, 21.7 GB |

## The int4 container and the drive's temperature

`--int4-dir` reads experts already quantised to int4 (USAGE.md). Checked first on 6 layers
(`--gpu`, the 372-token document): the logits are byte-identical to `--experts int4`
quantising on the way in, and the experts read fall from 21.70 to 12.20 GB (-44%), the
prefill from 14.1 to 9.6 s.

In `localcode` with `--gpu`, a container holding 25 of the 42 MoE layers (3-27; the rest
quantised on the way in, because the drive had no room for more), cache sized
automatically (37.8 GB, 2,671 experts), reasoning Max:

| mode | s/token | drive temperature during a long answer |
|---|---|---|
| FP8, exact (the checkpoint) | 1.77 with the drive cool; ~5 once it reaches 78 °C | climbed to 78 °C and held there |
| `--experts int4`, no container (earlier, 44 GB cache) | 1.49 | - |
| **int4 container, 25 of 42 layers** | **1.12, 1.17** (two answers, 148 and 242 tokens) | **stayed at or below 70 °C** |

Two effects add up. Fewer bytes per token is the direct one. The other is thermal: this
drive (an OEM PCIe 4.0 NVMe without a heatsink) reads the exact model continuously, climbs
to 78 °C within minutes and then holds that temperature by reading slower, which took the
exact path from 1.8 to ~5 s/token in a long answer. Reading 44% less, it stayed below 70 °C.
With the container the drive is no longer the busiest part: the task manager showed the
disk at 68-91% and the CPU at 78-89% during decode, so int4 expert compute is the next
limit.

The first ~13 layers of the conversion took 15-20 s each; after about 40 GB written in a
row the drive slowed to 50-136 s per layer, its write cache exhausted, which only affects
writing the container.

## Low I/O priority (`GLM53F_IO_PRIORITY=low`, not adopted as default)

Marking the checkpoint's reads low priority lets the system and other programs use the
drive first, which keeps the machine responsive while the exact path reads continuously.
Interleaved runs, FP8 exact, `--gpu --cache-gb 36`, 40 tokens, identical tokens throughout:

| priority | decode s/token | waiting on reads | read per token |
|---|---|---|---|
| low | 2.83, 2.66 | 2.16, 2.02 s | 4.31, 4.32 GB |
| normal (default) | **2.44, 2.35** | 1.76, 1.69 s | 4.34, 4.35 GB |

Windows slows low-priority reads even when nothing else uses the drive, so the option stays
opt-in. (These runs are slower than earlier ones because the drive was 98% full and warm;
only the interleaved comparison is meaningful.)

## CUDA backend (`--gpu`)

The trunk runs on the two GPUs, routed experts on the CPU (ARCHITECTURE.md 2.6). 40
generated tokens, same prompt, prefetch 6; the generated ids are identical to the CPU run
in every row.

| configuration | peak RSS | prefill | decode s/token |
|---|---|---|---|
| CPU only, 24 GB cache (3 runs) | 39.6 GB | 31.8, 31.1, 31.4 s | 2.23, 2.22, 2.22 |
| `--gpu`, keep-warm off, 24 GB cache (3 runs) | 39.7 GB | 29.7, 29.6, 30.5 s | 2.25, 2.26, 2.26 |
| `--gpu`, 24 GB cache (3 runs) | 39.7 GB | 29.6, 29.4, 29.5 s | 2.08, 2.08, 2.08 |
| **`--gpu --cache-gb 38` (3 runs): same RSS as the CPU row** | 40.0 GB | 30.4, 30.1, 29.7 s | **1.80, 1.81, 1.81** |
| `--gpu --cache-gb 44` | 46.0 GB | 30.6 s | 1.70 |

At equal memory use decode is 19% faster than the CPU path (2.22 -> 1.80 s/token). Of a
1.80 s step the GPU trunk takes ~0.22 s; the rest is the CPU side (routing, prefetch,
waiting for reads, routed experts): the disk and the routed experts remain the limit.

### How the trunk got from ~0.9 to 0.22 s per decode step

Decode with a 24 GB cache, `GLM53F_GPU_PROFILE=1`:

| change | trunk s/step | decode s/token |
|---|---|---|
| first version: one kernel per row, arithmetic copied from the CPU | ~0.9 | 2.59 |
| cuBLAS SGEMM on fp32 widened per product | 0.73 | 2.46 |
| RMSNorm per block, KDA recurrence per element (were loops of 16k-64k steps in one GPU thread) | 0.61 | 2.38 |
| MLA attention per score/element; layers balanced across devices; fp32 copies in spare memory (7.3 GB) | 0.61 | 2.25 |
| keep-warm | **0.22** | **2.08** |

Two findings drove this:

- **Saturated vs real.** With routed experts skipped (a diagnostic run with wrong output),
  the same trunk took 0.25 s per step; inserting a 38 ms sleep per MoE layer, as the CPU
  experts do, made it 0.82 s, every section uniformly 3-5x slower. `nvidia-smi` showed both
  cards in P3 (memory at 5001 MHz) during real decode. Keeping them busy while the CPU works
  holds P2 (7301 MHz) at ~85-90 W per card and restores the saturated speed. Lighter
  keep-warm variants (a small product every millisecond, or batches on a separate CUDA
  stream) either did not hold P2 or competed with the model's kernels.
- **Widening costs as much as multiplying.** On an RTX 3060 an 8192 x 4096 BF16 product
  takes 0.64 ms to widen to fp32 plus 0.40 ms for SGEMV; fused BF16 kernels (float or
  double accumulators, 2-D blocks) measured 1.1-2.0 ms. Stored fp32 copies skip the first
  part, which is why spare VRAM is filled with them.

Keep-warm slows the CPU side by ~0.2 s per step (routed-expert time 1.63 -> 1.85 s) but
saves 0.39 s of trunk time. Launching 4 or 16 products per wake-up did not change that
(2.09, 2.10 s/token).

Per decode step with `GLM53F_GPU_PROFILE=2` (synchronised after each section, 38 GB cache):

| section | ms |
|---|---|
| KDA (34 layers), of which input products 90, output product 29 | 135 |
| MLA (11 layers) | 44 |
| mHC collapse + norm, both sites (45 layers) | 40 |
| shared experts (42 layers) | 36 |
| final norm + lm_head | 8 |
| dense MLP (3 layers) | 5 |
| host/device copies, stream moves, mHC expand | 8 |
| routed experts on the CPU (including waiting for reads) | 1805 |

## Routed experts: compute, GPU offload and the disk

### Where a decode token goes (GPU trunk, 38 GB cache)

`GLM53F_ROUTE_TRACE` recorded 233 decode tokens (9,786 MoE layer-steps): routed-expert
compute on the CPU took 0.70 s/token (2.08 ms per expert) and waiting for expert reads
0.72 s/token. About 130-164 of the 336 experts a token needs are not in the 38 GB cache
when the token starts, 3.3-4.1 GB that the SSD delivers at ~2.6 GB/s in ~1.3-1.6 s.

### Experts on the GPUs (not implemented: slower on this machine)

(With int4 experts and page-locked memory the balance changes; see "Routed experts on a device" below.)

| measurement | GPU 0 (PCIe 3.0 x4) | GPU 1 (PCIe 4.0 x16) |
|---|---|---|
| copy one expert (25.2 MB) to the device | 8.9 ms | 1.0 ms |
| 8 experts of one layer, FP8 (dequantise + SGEMV, or fused block kernel) | 6.2-6.4 ms | 6.6 ms |

On the GPU an expert takes ~0.8 ms against 2.08 ms on the CPU, so at best ~0.43 s/token
could be saved. But each device serves 168 experts per token; a VRAM expert cache simulated
on the trace has a 0% hit rate up to 160 slots (3.9 GB), 27% / 47% (device 0 / 1) with 5.9
GB and 40% / 63% with the whole 11.8 GB. At 5.9 GB device 0 would copy ~122 experts per
token over its x4 link, ~1.1 s, more than the compute saved. A device-1-only variant was
estimated at ~0.1 s/token and not built.

### FP8 kernel without gather

The FP8 matvec decoded codes through a 256-entry table with `_mm256_i32gather_ps`. It now
decodes them arithmetically (ARCHITECTURE.md 2.2), bit-identical (`test_fp8`; 45-layer
logits byte-identical to the previous build on the CPU and with `--gpu`):

| 2048 x 4096 FP8 matvec | 1 thread | 16 threads |
|---|---|---|
| table gather | 3.72 ms | 0.74 ms |
| arithmetic decode | 1.90 ms | 0.34 ms |

End to end the gain is small because reads, not compute, set the pace: compute dropped
from 1.19 to 0.84 s/token on the CPU path and the time waiting for reads rose by about the
same amount. Interleaved A/B, `--gpu --cache-gb 38`, 40 tokens, identical ids:

| run pair | gather | arithmetic |
|---|---|---|
| 1 | 1.94 | 1.88 |
| 2 | 1.92 | 1.85 |
| 3 | 1.93 | 1.84 |

(These pairs ran later than the table above, with the machine measuring ~5% slower
overall; only the pairwise comparison is meaningful.)

### Cache policy

Misses per token on the same trace (after 20 warm-up tokens):

| slots | LRU (the engine) | LFU, aging 0.98 | LFU, aging 0.9 | Belady (clairvoyant bound) |
|---|---|---|---|---|
| 952 (24 GB) | 161.8 | 211.3 | 153.2 | 95.0 |
| 1508 (38 GB) | 129.6 | 159.5 | 128.2 | 71.1 |

Frequency-based eviction does not beat LRU meaningfully; even perfect knowledge of the
future leaves 71 misses (1.8 GB) per token.

### Reads are the clock (measured, 119 decode tokens, `--gpu --cache-gb 38`)

A decode token reads 3.90 GB of experts (0.11 GB on demand, 3.79 GB read ahead): 1.50 s of
SSD inside a 1.74 s step, so the drive is busy ~86% of the time. Decode time tracks bytes
read, which is why a faster expert kernel alone does not show up end to end.

### Predicting more than one layer ahead (measured, not adopted)

The router of a layer applied to the current layer's MoE input, 40 decode tokens
(`GLM53F_PREDICT_STATS`):

| routers applied this many layers early | top-8 | top-16 |
|---|---|---|
| 1 (what prefetch uses) | 75.0% | 90.2% |
| 2 | 66.8% | 82.7% |
| 3 | 61.9% | 77.5% |
| 4 | 57.3% | 72.5% |

Accuracy decays slowly, but with the disk already ~86% busy, earlier reads can only fill
the idle ~14%, while every wrong guess costs a whole 25 MB read. Measured over 60 decode
tokens, `--gpu --cache-gb 38`, generated ids identical everywhere:

| configuration | decode s/token | reads per token |
|---|---|---|
| prefetch 6 (default) | 1.91 | 4.09 GB |
| + 4 experts 2 layers ahead | 1.84 | 4.17 GB |
| + 8 experts 2 layers ahead | 1.89 | 4.39 GB |
| + 4 and 4, two and three layers ahead | 1.86 | 4.23 GB |
| + 6, 4, 2 two, three and four layers ahead | 1.85 | 4.30 GB |
| prefetch 4, no deeper prediction | 1.80 | 3.82 GB |
| prefetch 8 | 1.87 | 4.24 GB |

Deeper prediction reads more and gains nothing, so it was removed. What the sweep does show
is that the best prediction count moved with the faster expert kernel; two interleaved
rounds of 60 tokens each:

| `--prefetch` | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| `--gpu`, 38 GB cache | 1.88, 1.83 | 1.83, 1.83 | 1.79, 1.82 | 1.79, 1.82 | 1.84, 1.82 |
| reads per token | 3.69 GB | 3.74 GB | 3.81 GB | 3.95 GB | 4.09 GB |

On the CPU-only path (24 GB cache, 40 tokens) the order reverses, because compute there
leaves more room to overlap: 6 gives 2.31 and 2.32 s/token, 4 gives 2.33 and 2.35. Both
differences are about 1%, inside the run-to-run spread, so the default stays 6; with
`--gpu` you can try `--prefetch 4`.

Two changes from this investigation were kept, since they are right regardless of depth:
the read queue is ordered by layer (nearer layers first) instead of by insertion, and
cancellation of pending reads is anchored on the layer being computed rather than on the
layer just hinted.

### Speculative decode with the MTP layer (measured, not adopted)

The checkpoint carries a multi-token-prediction layer (layer 45: MLA + MoE with its own 288
experts, plus `eh_proj`, `enorm`, `hnorm` and the shared head norm). `GLM53F_MTP_STATS=1`
binds it and drafts a token per step without changing the output (ARCHITECTURE.md 2.7).

| measurement | value |
|---|---|
| draft equals the token the full model produced | 70 of 79 (88.6%), 80-token run |
| cost of running the layer per token (compute and its own expert reads) | 0.135 s |

The acceptance rate is high, but speculation does not pay here. Replaying the routing trace
with two tokens verified per pass gives 130.5 expert reads per token against 129.7 one token
at a time (1508 slots; 160.9 against 162.0 at 952). Consecutive tokens do share experts -
32% of a layer's experts are the ones the previous token used - but those repeats are
already cache hits today, so merging two tokens nearly doubles the reads of a pass while
producing at most two tokens. Add the MTP layer's own ~8 expert reads per token and the
bytes per token come out equal or slightly worse. Since decode time tracks bytes read, the
expected gain is around zero; the machinery (state snapshot and rollback for the KDA
recurrence, the KV caches and the GPU copies) was therefore not built.

It would pay on a machine where reads are not the wall: with the experts in RAM, or on
storage several times faster, compute dominates and 1.9 tokens per forward becomes a real
speed-up. The measurement code stays for that case.

### An int4 expert cache (`--experts int4`, off by default)

Reads set the pace, so the way to go faster is to need fewer of them. The cache can hold
routed experts re-quantised to int4 with a step per group of 64 columns (ARCHITECTURE.md
2.2): 14.16 MB instead of 25.19 MB, so the same RAM holds 1.8x more experts and fewer
tokens miss. The reads themselves stay FP8 - the checkpoint is not converted - and each
miss is quantised on the reader thread on its way into the cache.

40 tokens, `--gpu --cache-gb 44`, interleaved runs:

| experts in the cache | slots | reads per token | decode s/token |
|---|---|---|---|
| FP8, the checkpoint's own values (default) | 1,747 | 3.73 GB | 1.77, 1.70 |
| int4, re-quantised | 3,107 | 2.51 GB | 1.54, 1.49 |

That is 13% faster for an output that is no longer the checkpoint's: over 40 greedy tokens
38 matched the exact run (the first difference at token 27), the next-token argmax and the
top-1 stayed the same, but the logits moved (RMS 0.77, correlation 0.92 against the exact
path). The quantiser is vectorised because it sits in the read path: the first, scalar
version made decode 3.27 s/token, slower than not quantising at all.

### Cache size, measured (40 tokens, `--gpu`, one idle machine)

| experts | cache | slots | reads per token | decode s/token | peak RSS |
|---|---|---|---|---|---|
| FP8 | 38 GB | 1,508 | 3.66 GB | 1.70 | 40.0 GB |
| FP8 | 46 GB | 1,826 | 3.64 GB | 1.74 | 48.0 GB |
| FP8 | 52 GB | 2,064 | 3.45 GB | 1.65 | 54.0 GB |
| int4 | 52 GB | 3,673 | 2.19 GB | 1.38 | 54.1 GB |

More cache buys less and less: 14 GB more RAM moved the exact path by 3%, because a token
draws 336 experts out of 12,096 (305 GB) and a cache of this size covers about a sixth of
them. On 64 GB, 38 GB of cache is the comfortable setting and 52 GB leaves too little for
the rest of the system; the int4 mode is the one that turns extra RAM into speed.

### What did not work on the exact path

Three attempts to read fewer FP8 bytes without touching a single weight, all measured and
rejected:

| attempt | result |
|---|---|
| better replacement policies (2Q, SLRU, LRU-2, LFU with decay) | at best 2% fewer reads than LRU |
| cycle-aware eviction (evict the layer just computed, keep the layers coming) | 12-15% *more* reads: the cache already holds about five tokens of experts, so the wrap-around never bites |
| evict read-ahead slots that have not been used yet | 8.23 GB per token and 3.73 s/token: those slots are precisely the next layer's experts |

The trace explains why LRU is hard to beat here. Over 233 decode tokens the 336 experts a
token uses are spread over 8,073 distinct experts; the 1,747 most used cover 64.4% of all
uses, which is exactly the hit rate LRU achieves with 1,747 slots. Belady's 80.8% comes
from knowing the future, not from a hot set an online policy could learn.

Remaining levers on the exact path: more memory for the cache, faster storage, and a
lossless container (E4M3 codes carry less than 8 bits of entropy, but a converted copy
needs disk space this machine does not have).

## September 2026: the full int4 container, the SSD, and the GPU as a second worker

Same machine, `--gpu`, 38 GB cache unless stated, 18-token chat prompt, 64 generated
tokens; s/token is the median over steps 3-63.

### The full int4 container

The container was completed (layers 28-44 written in 629 s after freeing disk space; all 42
layers pass `--verify-int4`). With every layer read from it:

| run | s/token | reads per token | wait on reads | compute |
|---|---|---|---|---|
| container with 25 of 42 layers (the rest quantised on the way in) | 1.01 | 2.00 GB | 0.51 s | ~0.56 s |
| **container with all 42 layers** | **0.69-0.72** | **1.58 GB** | **0.14-0.16 s** | 0.56-0.58 s |
| `localcode`, same container, two answers (66 and 144 tokens) | 0.66-0.69 | - | - | - |

The generated text is identical with the partial and the full container (it is the same
int4 arithmetic; only the bytes come from a different file). For comparison, a `localcode`
session on the exact FP8 path the same day took 3.55 s/token: the automatic cache came out
at 33.7 GB because other programs held RAM, and every read came from the slow FP8 files
described next.

Right after a large write the drive stalls reads for minutes: the first run after writing
the 69 GB of layers 28-44 had steps of up to 29 s, and a run just after a 4 GB test write
had seven steps of 13-39 s. Runs a few minutes later were clean.

### Reading the FP8 checkpoint got slower

`bench_expert_io` gave 1.66-1.89 GB/s at every queue depth on 2026-09-23/24, against
2.32-2.53 GB/s in the table above, and FP8 decode went from 1.80 to 2.48 s/token. Ruled out:
free space (98-117 GB free after cleaning up), TRIM (`Optimize-Volume -ReTrim`), the PCIe
link (Gen4 x4, its maximum), PCIe link power management (off, no change), temperature
(59 °C idle, 67-73 °C under load), wear (0%), and reader threads (2, 4 and 8 all slower
or equal).

Unbuffered random 25 MB reads, per file set:

| files | written | GB/s |
|---|---|---|
| a file written seconds before | same day | 3.7-5.9 (partly the drive's write cache) |
| the int4 container | 2026-09-17 | 3.2-3.3 |
| the FP8 checkpoint | 2026-09-14/15 (download) | 1.5-1.7 |

Per FP8 shard (8 random reads each) the speed ranges from 0.51 to 4.10 GB/s, median 1.63;
47 of the 62 shards read below 2 GB/s. A sequential copy of one shard (SHA-256 identical)
read 3.8-3.9 GB/s after 15 idle minutes against 2.6-2.8 for the original, so how the
download was laid out on the drive costs reads, not the drive itself. The checkpoint was
not rewritten.

### Prefetch retuned for int4 (container with 25 of 42 layers)

| configuration | s/token |
|---|---|
| `--prefetch 6`, 2 readers (default) | **1.01** |
| `--prefetch 4` | 1.02 |
| `--prefetch 8` | 1.08 |
| `--prefetch 10` | 1.13 |
| 3 readers | 1.07 |
| 4 readers | 1.09 |

The defaults chosen for FP8 are still the best; all six runs produced the same text.

### Routed experts on a device (`GLM53F_GPU_EXPERTS`, off by default)

With the full container, the route trace put a token at 0.51 s waiting on reads, 0.30 s
multiplying routed experts on the CPU and ~0.26 s in the trunk, so the experts were moved
to the idle GPU. A standalone benchmark on 64 real int4 experts on cuda:1 (PCIe 4.0 x16):

| step | per 14.16 MB expert |
|---|---|
| CPU product (from the trace) | ~0.89 ms |
| copy host -> device, pageable | 1.09-1.25 ms (11-13 GB/s) |
| copy host -> device, page-locked | 0.53 ms (26.6 GB/s) |
| the product on the device, weights already in VRAM | 0.17 ms |
| one layer of 8 experts, copies overlapping products, page-locked | 4.3 ms per layer (0.18 s per token) |
| the same from pageable memory | 8.7-10.6 ms per layer |

`cudaHostRegister` accepted 8, 16 and 24 GB of the arena and refused 30 and 38 GB. In the
engine (ARCHITECTURE.md 2.8), one run each, same text as without it in every run:

| arm | s/token | wait on reads | compute |
|---|---|---|---|
| off | 0.690 | 0.14 s | 0.58 s |
| N = 3 | 0.650 | 0.19 s | - |
| N = 5 | 0.680 | 0.21 s | 0.47 s |
| N = 8 | 0.700 | 0.21 s | - |
| off, keep-warm off | 0.680 | 0.15 s | - |
| N = 5, keep-warm off | 0.660 | 0.21 s | - |
| N = 8, keep-warm off | 0.660 | 0.22 s | - |

Compute fell by 0.11 s and the wait on reads rose by about as much. A token reads 1.58 GB,
~0.5 s of this SSD at ~3.2 GB/s; prefetch used to hide most of it behind compute, and with
less compute there is less to hide it behind. At ~5% at best, within run-to-run noise, it
stays off by default.

### Cache-aware routing (simulated, not built)

With reads setting the pace again, the next lever is to need fewer of them.
`GLM53F_ROUTE_TOP16` recorded 299 decode tokens and `tools/route_cache_sim.py` replayed
them through a 2,684-slot LRU cache (38 GB of int4), 40 tokens of warm-up. "swap" replaces
a missing chosen expert whose normalised weight is below the threshold by the best
resident candidate among the router's 9th-16th; "drop" skips chosen experts below it.

| policy | reads per token | vs exact | experts per token | routing weight moved | layers changed |
|---|---|---|---|---|---|
| exact | 92.0 | 100% | 336 | 0% | 0% |
| swap < 0.05 | 80.6 | 88% | 336 | 1.1% | 19% |
| swap < 0.10 | 42.9 | 47% | 336 | 9.8% | 80% |
| swap < 0.15 | 29.7 | 32% | 336 | 16.9% | 93% |
| drop < 0.05 | 78.9 | 86% | 307 | 2.7% | 27% |
| drop < 0.08 | 50.5 | 55% | 233 | 14.4% | 81% |
| swap everything that has a resident substitute | 21.8 | 24% | 336 | 33.1% | 97% |

The simulation is open loop (the trace is the exact model's routing; a policy that changes
the output also changes later routing), so it sizes the SSD traffic at stake, not the
quality. Halving the reads moves ~10% of the routing weight in 80% of the layers, which is
likely to change the text; the mild settings save 12-14%. None of it was built.

### Where this leaves the machine

On 64 GB of RAM and this SSD the int4 path is bound at roughly 0.5 s per token by reads
(~2 tokens/s), whatever computes the rest: the GPU offload above showed it. Engines that
report 15-25 tokens/s on comparable models (FreeToken, llama.cpp with smaller MoE models)
hold every expert in RAM. What would move this machine: more RAM (the board takes 128 GB),
a second or faster SSD, or an approximate routing policy validated against the exact path.
