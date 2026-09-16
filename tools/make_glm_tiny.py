#!/usr/bin/env python3
"""Build the tiny GLM-5.3-Flash oracle fixture: a random 5-layer model with the released
architecture, saved in the released checkpoint's tensor names and dtypes, plus the
reference outputs of the official transformers implementation.

The model exercises every path the engine has: KDA and MLA (with the DSA indexer, which
the reference runs), dense and MoE feed-forward, mHC, FP8 block-quantised matrices,
BF16 matrices and F32 vectors. Weights are stored the way the real checkpoint stores
them, and the reference is computed from the DEQUANTISED stored weights, so the engine
and the reference see exactly the same effective model.

  .venv/Scripts/python tools/make_glm_tiny.py tests/fixtures/glm_tiny

Writes config.json, model.safetensors and ref.json (prompt ids, last-position logits,
greedy continuation, teacher-forced argmax at every position).
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file

from transformers.models.glm5_next import modeling_glm5_next as M
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

FP8_MAX = 448.0
BLOCK = [4, 4]
PRE = "model.language_model."

TEXT = dict(
    vocab_size=64, hidden_size=32, intermediate_size=24, moe_intermediate_size=8,
    num_hidden_layers=5, num_attention_heads=2, num_key_value_heads=2,
    n_shared_experts=1, n_routed_experts=6, num_experts_per_tok=2, routed_scaling_factor=2.5,
    kv_lora_rank=8, q_lora_rank=12, qk_rope_head_dim=0, v_head_dim=8, qk_nope_head_dim=8,
    n_group=1, topk_group=1, norm_topk_prob=True, hidden_act="silu", rms_norm_eps=1e-5,
    layer_types=["linear_attention", "linear_attention", "deepseek_sparse_attention",
                 "linear_attention", "deepseek_sparse_attention"],
    mlp_layer_types=["dense", "sparse", "sparse", "sparse", "sparse"],
    indexer_types=["full"] * 5,
    index_topk=16, index_head_dim=8, index_n_heads=2, index_kpool=4,
    swiglu_limit=0.5, hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20,
    linear_attn_config={"num_heads": 2, "head_dim": 8, "short_conv_kernel_size": 4,
                        "gate_lower_bound": -5.0},
    eos_token_id=[1, 2], pad_token_id=0,
)


def fp8_quant(w):
    """transformers' Fp8Quantize for a 2-D weight: per-block max-abs scale, clamp, cast."""
    rows, cols = w.shape
    br, bc = BLOCK
    assert rows % br == 0 and cols % bc == 0, (rows, cols)
    t = w.float().reshape(rows // br, br, cols // bc, bc)
    mx = t.abs().amax(dim=(1, 3))
    safe = torch.where(mx > 0, mx, torch.ones_like(mx))
    scale = torch.where(mx > 0, FP8_MAX / safe, torch.ones_like(mx))
    q = torch.clamp(t * scale[:, None, :, None], -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    inv = (1.0 / scale).float()
    deq = (q.float() * inv[:, None, :, None]).reshape(rows, cols)
    return q.reshape(rows, cols), inv, deq


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/glm_tiny"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(1234)

    cfg = Glm5NextTextConfig(**TEXT)
    cfg._attn_implementation = "eager"
    model = M.Glm5NextTextModel(cfg).float().eval()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    # Random weights with enough spread that every mechanism matters: the default init
    # zeroes mHC base, sets its scale to 1 and zeroes A_log, which would leave the
    # stream mixer and the decay nearly uniform and let a wrong kernel pass.
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + [("lm_head.weight", lm_head.weight)]:
            if name.endswith(("norm.weight", "layernorm.weight")) or "o_norm" in name:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            elif name.endswith("attn_hc.scale") or name.endswith("ffn_hc.scale"):
                p.copy_(0.5 + torch.rand_like(p))
            elif name.endswith(".base"):
                p.copy_(torch.randn_like(p))
            elif name.endswith("A_log"):
                p.copy_(torch.randn_like(p))
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * (0.5 / p.shape[-1] ** 0.5 if p.dim() > 1 else 0.3))
        for L, layer in enumerate(model.layers):
            if cfg.mlp_layer_types[L] == "sparse":
                layer.mlp.gate.e_score_correction_bias.copy_(0.1 * torch.randn(cfg.n_routed_experts))

    tensors = {}

    def bf16(name, t):
        """Store BF16; return the float32 the reference must use (a lossy round trip)."""
        b = t.detach().to(torch.bfloat16)
        tensors[name] = b.contiguous()
        return b.float()

    def f32(name, t):
        tensors[name] = t.detach().float().contiguous()
        return tensors[name]

    def fp8(name, t):
        q, inv, deq = fp8_quant(t.detach())
        tensors[name] = q.contiguous()
        tensors[name + "_scale_inv"] = inv.contiguous()
        return deq

    with torch.no_grad():
        sd = {}
        model.embed_tokens.weight.copy_(bf16(PRE + "embed_tokens.weight", model.embed_tokens.weight))
        model.norm.weight.copy_(bf16(PRE + "norm.weight", model.norm.weight))
        lm_head.weight.copy_(bf16("lm_head.weight", lm_head.weight))
        for L, layer in enumerate(model.layers):
            lp = f"{PRE}layers.{L}."
            layer.input_layernorm.weight.copy_(bf16(lp + "input_layernorm.weight", layer.input_layernorm.weight))
            layer.post_attention_layernorm.weight.copy_(
                bf16(lp + "post_attention_layernorm.weight", layer.post_attention_layernorm.weight))
            for site, hc in (("attn", layer.attn_hc), ("ffn", layer.ffn_hc)):
                hc.fn.copy_(bf16(lp + f"hc_{site}_fn", hc.fn))
                hc.base.copy_(f32(lp + f"hc_{site}_base", hc.base))
                hc.scale.copy_(f32(lp + f"hc_{site}_scale", hc.scale))
            a = layer.self_attn
            if cfg.layer_types[L] == "linear_attention":
                for n in ("q_proj", "k_proj", "v_proj", "b_proj", "g_a_proj", "g_b_proj", "o_proj"):
                    getattr(a, n).weight.copy_(bf16(lp + f"self_attn.{n}.weight", getattr(a, n).weight))
                for n in ("f_a_proj", "f_b_proj"):
                    getattr(a.forget_gate, n).weight.copy_(
                        bf16(lp + f"self_attn.{n}.weight", getattr(a.forget_gate, n).weight))
                a.forget_gate.A_log.copy_(f32(lp + "self_attn.A_log", a.forget_gate.A_log))
                a.forget_gate.dt_bias.copy_(f32(lp + "self_attn.dt_bias", a.forget_gate.dt_bias))
                a.o_norm.weight.copy_(bf16(lp + "self_attn.o_norm.weight", a.o_norm.weight))
                P = cfg.linear_num_heads * cfg.linear_head_dim
                w = a.conv1d.weight.detach()
                parts = [bf16(lp + f"self_attn.{c}_conv1d.weight", w[i * P:(i + 1) * P]) for i, c in enumerate("qkv")]
                a.conv1d.weight.copy_(torch.cat(parts, 0))
            else:
                for n in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
                    getattr(a, n).weight.copy_(fp8(lp + f"self_attn.{n}.weight", getattr(a, n).weight))
                a.kv_b_proj.weight.copy_(bf16(lp + "self_attn.kv_b_proj.weight", a.kv_b_proj.weight))
                for n in ("q_a_layernorm", "kv_a_layernorm"):
                    getattr(a, n).weight.copy_(bf16(lp + f"self_attn.{n}.weight", getattr(a, n).weight))
                ix = a.indexer
                for n in ("wq_b", "wk", "weights_proj"):
                    getattr(ix, n).weight.copy_(bf16(lp + f"self_attn.indexer.{n}.weight", getattr(ix, n).weight))
                ix.k_norm.weight.copy_(bf16(lp + "self_attn.indexer.k_norm.weight", ix.k_norm.weight))
                ix.k_norm.bias.copy_(bf16(lp + "self_attn.indexer.k_norm.bias", ix.k_norm.bias))
                ix.index_kpool_compress_ape.copy_(
                    bf16(lp + "self_attn.indexer.index_kpool_compress_ape", ix.index_kpool_compress_ape))
                ix.index_kpool_compress_gate.copy_(
                    bf16(lp + "self_attn.indexer.index_kpool_compress_gate", ix.index_kpool_compress_gate))
            mlp = layer.mlp
            if cfg.mlp_layer_types[L] == "dense":
                for n in ("gate_proj", "up_proj", "down_proj"):
                    getattr(mlp, n).weight.copy_(fp8(lp + f"mlp.{n}.weight", getattr(mlp, n).weight))
            else:
                mlp.gate.weight.copy_(bf16(lp + "mlp.gate.weight", mlp.gate.weight))
                f32(lp + "mlp.gate.e_score_correction_bias", mlp.gate.e_score_correction_bias)
                for n in ("gate_proj", "up_proj", "down_proj"):
                    s = mlp.shared_experts
                    getattr(s, n).weight.copy_(fp8(lp + f"mlp.shared_experts.{n}.weight", getattr(s, n).weight))
                I = cfg.moe_intermediate_size
                ex = mlp.experts
                for e in range(cfg.n_routed_experts):
                    g = fp8(lp + f"mlp.experts.{e}.gate_proj.weight", ex.gate_up_proj[e, :I])
                    u = fp8(lp + f"mlp.experts.{e}.up_proj.weight", ex.gate_up_proj[e, I:])
                    ex.gate_up_proj[e].copy_(torch.cat([g, u], 0))
                    ex.down_proj[e].copy_(fp8(lp + f"mlp.experts.{e}.down_proj.weight", ex.down_proj[e]))

    ids = [5, 17, 3, 42, 9, 9, 60, 1, 33, 12, 7, 21]
    n_gen = 6

    def logits_all(seq):
        with torch.no_grad():
            h = model(input_ids=torch.tensor([seq])).last_hidden_state
            return lm_head(h)[0]

    lg = logits_all(ids)
    seq = list(ids)
    for _ in range(n_gen):
        seq.append(int(logits_all(seq)[-1].argmax()))
    tf = logits_all(seq).argmax(-1).tolist()

    save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    full = {"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next",
            "text_config": dict(TEXT, dtype="bfloat16", mhc=True, scoring_func="sigmoid",
                                topk_method="noaux_tc", attention_bias=False),
            "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3",
                                    "quant_method": "fp8", "weight_block_size": BLOCK}}
    json.dump(full, open(os.path.join(out, "config.json"), "w"), indent=1)
    json.dump({"prompt_ids": ids, "logits_last": [float(x) for x in lg[-1]],
               "generated_ids": seq[len(ids):], "full_ids": seq, "tf_argmax": tf},
              open(os.path.join(out, "ref.json"), "w"), indent=1)
    print(f"wrote {out}: {len(tensors)} tensors, prompt {len(ids)} ids, generated {seq[len(ids):]}")


if __name__ == "__main__":
    sys.exit(main())
