#!/usr/bin/env python3
"""Reference forward for GLM-5.3-Flash on the REAL checkpoint, using the official
transformers modules (models/glm5_next/modeling_glm5_next.py) layer by layer.

The full model does not fit in RAM as float32, so this script never instantiates it.
It builds ONE Glm5NextTextDecoderLayer at a time on the meta device, assigns that layer's
weights (FP8 blocks dequantised exactly as transformers' Fp8Dequantize does), runs it,
and frees it. The routed experts are the only module replaced: LazyExperts loads just
the experts the router selected and applies the same arithmetic as
Glm5NextTextExperts.forward. Everything else -- mHC, KDA, MLA + DSA indexer, router,
shared expert, norms -- is the upstream code, unmodified.

Outputs (under --out):
  ref.json          prompt ids, greedy next token, top-10 logits of the last position
  logits.bin        float32 logits of the LAST position (vocab)
  hidden_L{n}.bin   optional, float32 [T][hc_mult][hidden] after layer n (--dump-layers)

Usage:
  .venv/Scripts/python tools/glm_reference.py C:/path/GLM-5.3-Flash --prompt "..." --out ref/
  .venv/Scripts/python tools/glm_reference.py DIR --ids 1,2,3 --layers 4 --dump-layers 0,3
"""
import argparse
import json
import os
import struct
import sys
import time

import torch
import torch.nn.functional as F

from transformers.models.glm5_next import modeling_glm5_next as M
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

PRE = "model.language_model."


class Shards:
    """name -> (shard, offset, dtype, shape) from the headers; tensors read with plain
    positioned reads.

    Not safe_open: on Windows it maps each shard copy-on-write, and every mapped shard
    counts its full size against the commit limit. Touching all 62 shards commits ~305 GB
    and the process dies partway through the layer stack."""

    DT = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16,
          "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8, "I64": torch.int64}

    def __init__(self, root):
        self.root = root
        self.meta = {}
        for fn in sorted(os.listdir(root)):
            if not fn.endswith(".safetensors"):
                continue
            with open(os.path.join(root, fn), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for k, v in hdr.items():
                if k != "__metadata__":
                    s, e = v["data_offsets"]
                    self.meta[k] = (fn, 8 + n + s, e - s, v["dtype"], v["shape"])
        self.fds = {}

    def _read(self, fn, off, nbytes):
        if fn not in self.fds:
            self.fds[fn] = open(os.path.join(self.root, fn), "rb", buffering=0)
        fh = self.fds[fn]
        fh.seek(off)
        buf = bytearray(nbytes)
        view = memoryview(buf)
        got = 0
        while got < nbytes:
            r = fh.readinto(view[got:])
            if not r:
                raise IOError(f"short read in {fn} at {off + got}")
            got += r
        return buf

    def has(self, name):
        return name in self.meta

    def get(self, name):
        fn, off, nb, dt, shape = self.meta[name]
        t = torch.frombuffer(self._read(fn, off, nb), dtype=self.DT[dt])
        return t.reshape(shape) if shape else t.reshape(())

    def rows(self, name, idx):
        fn, off, nb, dt, shape = self.meta[name]
        esz = torch.empty((), dtype=self.DT[dt]).element_size()
        rb = esz * shape[1]
        return torch.stack([torch.frombuffer(self._read(fn, off + i * rb, rb), dtype=self.DT[dt])
                            for i in idx])

    def weight(self, name):
        """A .weight tensor as float32, dequantising FP8 blocks with weight_scale_inv."""
        w = self.get(name)
        if w.dtype == torch.float8_e4m3fn:
            s = self.get(name + "_scale_inv").float()
            rows, cols = w.shape
            bm = -(-rows // s.shape[0])
            bn = -(-cols // s.shape[1])
            s = s.repeat_interleave(bm, 0)[:rows].repeat_interleave(bn, 1)[:, :cols]
            return w.float() * s
        return w.float()


class LazyExperts(torch.nn.Module):
    """Same math as Glm5NextTextExperts.forward, loading only the selected experts."""

    def __init__(self, config, shards, layer):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.swiglu_limit = config.swiglu_limit
        self.shards, self.layer = shards, layer

    _apply_gate = M.Glm5NextTextExperts._apply_gate

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        base = f"{PRE}layers.{self.layer}.mlp.experts."
        for expert_idx in hit:
            e = int(expert_idx[0])
            gate_up = torch.cat([self.shards.weight(f"{base}{e}.gate_proj.weight"),
                                 self.shards.weight(f"{base}{e}.up_proj.weight")], dim=0)
            down = self.shards.weight(f"{base}{e}.down_proj.weight")
            top_k_pos, token_idx = torch.where(mask[e])
            current = self._apply_gate(F.linear(hidden_states[token_idx], gate_up))
            current = F.linear(current, down) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


def layer_state_dict(shards, layer_mod, L):
    """Map every parameter/buffer of the HF layer module to its checkpoint tensor."""
    sd = {}
    lp = f"{PRE}layers.{L}."
    for name, _ in list(layer_mod.named_parameters()) + list(layer_mod.named_buffers()):
        if name.startswith("mlp.experts."):
            continue
        src = lp + name
        src = src.replace("self_attn.forget_gate.", "self_attn.")
        src = src.replace("attn_hc.fn", "hc_attn_fn").replace("attn_hc.base", "hc_attn_base")
        src = src.replace("attn_hc.scale", "hc_attn_scale").replace("ffn_hc.fn", "hc_ffn_fn")
        src = src.replace("ffn_hc.base", "hc_ffn_base").replace("ffn_hc.scale", "hc_ffn_scale")
        if name == "self_attn.conv1d.weight":
            sd[name] = torch.cat([shards.get(lp + f"self_attn.{c}_conv1d.weight").float()
                                  for c in "qkv"], dim=0)
            continue
        if name.endswith(".weight") and shards.has(src + "_scale_inv"):
            sd[name] = shards.weight(src)
        else:
            sd[name] = shards.get(src).float()
    return sd


def build_layer(config, shards, L):
    with torch.device("meta"):
        layer = M.Glm5NextTextDecoderLayer(config, L)
    if config.mlp_layer_types[L] == "sparse":
        layer.mlp.experts = LazyExperts(config, shards, L)
    sd = layer_state_dict(shards, layer, L)
    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if not m.startswith("mlp.experts.")]
    if missing or unexpected:
        raise RuntimeError(f"layer {L}: missing {missing} unexpected {unexpected}")
    return layer.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt")
    ap.add_argument("--ids")
    ap.add_argument("--raw", action="store_true", help="no chat template")
    ap.add_argument("--layers", type=int, default=0, help="run only the first N layers")
    ap.add_argument("--dump-layers", default="")
    ap.add_argument("--out", default="ref")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    cfg_all = json.load(open(os.path.join(args.model, "config.json")))
    config = Glm5NextTextConfig(**cfg_all["text_config"])
    config._attn_implementation = "eager"
    NL = args.layers or config.num_hidden_layers

    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        if args.raw:
            ids = tok.encode(args.prompt, add_special_tokens=False)
        else:
            enc = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                          tokenize=True, add_generation_prompt=True)
            ids = list(enc["input_ids"] if hasattr(enc, "keys") else enc)
    T = len(ids)
    print(f"prompt: {T} ids {ids}", flush=True)

    shards = Shards(args.model)
    dump = {int(x) for x in args.dump_layers.split(",") if x != ""}

    with torch.no_grad():
        emb = shards.rows(PRE + "embed_tokens.weight", ids).float()          # [T, D]
        h = emb.unsqueeze(0).unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()
        mask = torch.ones(1, T, dtype=torch.bool)
        topk = None
        t_all = time.time()
        for L in range(NL):
            t0 = time.time()
            layer = build_layer(config, shards, L)
            t1 = time.time()
            h, topk = layer(h, attention_mask=mask, position_ids=None, past_key_values=None,
                            position_embeddings=None, prev_topk_indices=topk)
            print(f"layer {L:2d} {config.layer_types[L][:6]} {config.mlp_layer_types[L]:6s} "
                  f"load {t1 - t0:5.1f}s run {time.time() - t1:5.1f}s  |h|max {h.abs().max():.3f}",
                  flush=True)
            if L in dump:
                h[0].contiguous().numpy().astype("float32").tofile(
                    os.path.join(args.out, f"hidden_L{L}.bin"))
            del layer
        x = M.Glm5NextTextHyperHead()(h)
        norm_w = shards.get(PRE + "norm.weight").float()
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps) * norm_w
        lm = shards.weight("lm_head.weight")
        logits = F.linear(x[0, -1], lm)
        logits.numpy().astype("float32").tofile(os.path.join(args.out, "logits.bin"))
        top = torch.topk(logits, 10)
        res = {
            "ids": ids, "layers": NL, "next_id": int(top.indices[0]),
            "top10": [[int(i), float(v)] for v, i in zip(top.values, top.indices)],
            "seconds": time.time() - t_all,
        }
        try:
            from transformers import AutoTokenizer
            tk = AutoTokenizer.from_pretrained(args.model)
            res["next_text"] = tk.decode([res["next_id"]])
            res["top10_text"] = [tk.decode([i]) for i, _ in res["top10"]]
        except Exception:
            pass
        json.dump(res, open(os.path.join(args.out, "ref.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(json.dumps(res))


if __name__ == "__main__":
    sys.exit(main())
