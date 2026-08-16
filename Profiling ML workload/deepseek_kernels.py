#!/usr/bin/env python3
"""Kernel-level decomposition of DeepSeek-V4-Flash, on the same schema as
specdecode_kernels.py and vision_kernels.py (kernels.py). Fills in the
worked example that was previously only discussed in prose.

Source: DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token
Context Intelligence" (arXiv:2606.19348), Section 4.2.1 "Model Setups" for
DeepSeek-V4-Flash, and Section 2.3 for the CSA/HCA attention mechanics.

WHERE THIS DIFFERS FROM specdecode_kernels.py, structurally:
  1. Attention is a multi-stage pipeline (compress -> index -> select ->
     attend -> grouped-project), not one fused kernel. CSA and HCA share
     most of that pipeline but differ in whether they sparsify (CSA: top-k
     over compressed entries, via a separate low-rank FP4 indexer) or just
     densely attend to everything compressed (HCA).
  2. The MoE FFN's expert selection is genuinely stochastic for 40 of 43
     layers (learned top-6-of-256 routing) and genuinely NOT stochastic for
     the other 3 (Hash routing -- a deterministic function of token ID).
     Both produce the identical DOWNSTREAM compute (6 routed-expert FFN
     calls + 1 shared-expert FFN call); they differ only in the SELECTION
     kernel's op_type (matmul+reduce vs. lookup) and in whether that
     selection needs profiling to characterize.
  3. Three coexisting precisions by ROLE, not just by tensor: BF16/FP8 for
     the compressed KV entries (paper: RoPE dims BF16, rest FP8), FP4 for
     the indexer's QK path and routed-expert weights, BF16 elsewhere.

SIMPLIFICATIONS (flagged, not hidden): the paper's compression step
(Eqs. 9-12) is a softmax-weighted combination over 2m raw KV entries per
compressed entry; here it's modeled as 4 linear projections at roughly that
cost, without reproducing the exact softmax-combine coefficient. The
grouped output projection (paper's "Grouped Output Projection" in 2.3.1) is
modeled as two chained matmuls at the stated group/intermediate dimensions
rather than the precise per-group einsum. Exact CSA/HCA layer interleaving
pattern beyond "the first two layers are SWA, the rest interleave CSA and
HCA" is not given in the paper to per-layer precision -- alternating
starting after layer 2 is ASSUMED. None of these change the ORDER of
magnitude; they would need revisiting for a publication-grade FLOPs count.
"""

from __future__ import annotations

from kernels import (Kernel, BYTES_PER_ELEM, GENERIC_CHIPLET,
                     write_kernel_table, write_latency_table)

# ---------------------------------------------------------------------------
# Model constants -- DeepSeek-V4-Flash (Section 4.2.1)
# ---------------------------------------------------------------------------
N_LAYERS = 43
D_MODEL = 4096
N_SWA_ONLY = 2                       # first 2 layers: pure sliding-window attention
N_HYBRID = N_LAYERS - N_SWA_ONLY     # 41 layers interleave CSA/HCA
N_CSA = (N_HYBRID + 1) // 2          # ASSUMED alternating split -- 21
N_HCA = N_HYBRID - N_CSA             # 20

# CSA / HCA shared attention dims
N_H = 64                 # query heads (both CSA and HCA)
C = 512                  # head dim
D_C = 1024               # query compression (latent) dim
G = 8                    # output-projection groups
D_G = 1024               # per-group intermediate dim
N_WIN = 128               # sliding-window branch, tokens

# CSA-only
M_CSA = 4                 # compression ratio
N_I_H = 64                # indexer query heads
C_I = 128                 # indexer head dim
TOPK = 512                # sparse-attention top-k

# HCA-only
M_HCA = 128                # compression ratio (much heavier)

# MoE
N_ROUTED = 256
N_SHARED = 1
EXPERT_WIDTH = 2048
EXPERTS_PER_TOK = 6
N_HASH_ROUTED_LAYERS = 3     # first 3 MoE layers use deterministic Hash routing

VOCAB_SIZE = 128_000        # paper states vocab "remains 128K" -- not given more precisely
MTP_DEPTH = 1

PRECISION_MAIN = "bf16"      # weights / general activations
PRECISION_KV = "fp8"         # compressed KV entries, non-RoPE dims (paper: mixed BF16 RoPE + FP8 rest;
                             # modeled here as uniformly FP8, the dominant share -- simplification)
PRECISION_INDEXER = "fp4"    # indexer QK path (paper, explicit)
BYTES_PER_ELEM["fp4"] = 0.5   # 4-bit, same byte width as int4 in kernels.py's table

B_MAIN = BYTES_PER_ELEM[PRECISION_MAIN]
B_KV = BYTES_PER_ELEM[PRECISION_KV]
B_IDX = BYTES_PER_ELEM[PRECISION_INDEXER]

CONTEXT_REGIMES = {
    "short_chat_L1500":   {"L": 1500,    "tokens": 1},   # decode: 1 new token/step
    "long_agent_L128000": {"L": 128000,  "tokens": 1},
    "long_agent_L1000000":{"L": 1000000, "tokens": 1},   # the model's headline 1M-context case
}


def _mm(tokens, d_in, d_out):
    return 2.0 * tokens * d_in * d_out


KERNELS: list[Kernel] = []


def _add(**kw):
    KERNELS.append(Kernel(**kw))


# ---- Attention: pure-SWA layers (first 2) ----------------------------------
_add(id="ATTN.swa.qkv_proj", task="ATTN", layer_class="swa_only_layer", op_type="matmul",
     multiplicity_fn=lambda c: N_SWA_ONLY,
     flops_fn=lambda c: _mm(c["tokens"], D_MODEL, 3 * N_H * C),
     weight_bytes_fn=lambda c: D_MODEL * 3 * N_H * C * B_MAIN,
     io_bytes_fn=lambda c: c["tokens"] * (D_MODEL + 3 * N_H * C) * B_MAIN,
     precision=PRECISION_MAIN, note="First 2 layers: plain windowed attention, no compression.")
_add(id="ATTN.swa.attention", task="ATTN", layer_class="swa_only_layer", op_type="attention",
     multiplicity_fn=lambda c: N_SWA_ONLY,
     flops_fn=lambda c: 4.0 * (N_H * C) * c["tokens"] * min(c["L"], N_WIN),
     weight_bytes_fn=lambda c: 0.0,
     io_bytes_fn=lambda c: min(c["L"], N_WIN) * C * 2 * B_KV,
     precision=PRECISION_KV, note=f"window={N_WIN} tokens.")
_add(id="ATTN.swa.out_proj", task="ATTN", layer_class="swa_only_layer", op_type="matmul",
     multiplicity_fn=lambda c: N_SWA_ONLY,
     flops_fn=lambda c: _mm(c["tokens"], N_H * C, D_MODEL),
     weight_bytes_fn=lambda c: N_H * C * D_MODEL * B_MAIN,
     io_bytes_fn=lambda c: c["tokens"] * (N_H * C + D_MODEL) * B_MAIN,
     precision=PRECISION_MAIN, note="")

# ---- Attention: CSA layers --------------------------------------------------
for lc, count in [("csa_layer", N_CSA)]:
    _add(id="ATTN.csa.kv_compress", task="ATTN", layer_class=lc, op_type="matmul",
         multiplicity_fn=lambda c, n=count: n,
         flops_fn=lambda c: _mm(c["tokens"], D_MODEL, 4 * C),   # 4 projections: W_a^KV,W_b^KV,W_a^Z,W_b^Z
         weight_bytes_fn=lambda c: D_MODEL * 4 * C * B_MAIN,
         io_bytes_fn=lambda c: c["tokens"] * (D_MODEL + 4 * C) * B_KV,
         precision=PRECISION_MAIN,
         note="Eqs 9-10: per-raw-token KV/weight projections feeding the "
              "m=4-to-1 compression (SIMPLIFIED: linear cost, softmax-combine "
              "coefficient not modeled).")
    _add(id="ATTN.csa.indexer_proj", task="ATTN", layer_class=lc, op_type="matmul",
         multiplicity_fn=lambda c, n=count: n,
         flops_fn=lambda c: _mm(c["tokens"], D_MODEL, D_C) + _mm(c["tokens"], D_C, C_I * N_I_H),
         weight_bytes_fn=lambda c: (D_MODEL * D_C + D_C * C_I * N_I_H) * B_IDX,
         io_bytes_fn=lambda c: c["tokens"] * (D_MODEL + C_I * N_I_H) * B_IDX,
         precision=PRECISION_INDEXER,
         note="Eqs 13-14: low-rank down/up projection for indexer queries, FP4.")
    _add(id="ATTN.csa.indexer_score", task="ATTN", layer_class=lc, op_type="attention",
         multiplicity_fn=lambda c, n=count: n,
         flops_fn=lambda c: 2.0 * N_I_H * C_I * c["tokens"] * (min(c["L"], N_WIN) if False else c["L"] / M_CSA),
         weight_bytes_fn=lambda c: 0.0,
         io_bytes_fn=lambda c: (c["L"] / M_CSA) * C_I * B_IDX,
         precision=PRECISION_INDEXER,
         note="Eq 16: index score vs. every compressed block -- this is the "
              "term that scales with context length L, analogous to global "
              "attention's linear-in-L cost in the spec-decode workload.")
    _add(id="ATTN.csa.topk_select", task="ATTN", layer_class=lc, op_type="reduce_argmax",
         multiplicity_fn=lambda c, n=count: n,
         flops_fn=lambda c: c["tokens"] * (c["L"] / M_CSA),
         weight_bytes_fn=lambda c: 0.0,
         io_bytes_fn=lambda c: (c["L"] / M_CSA) * 4 * B_MAIN,
         precision="fp32", note="Eq 17: top-512 selector over compressed indexer scores.")
    _add(id="ATTN.csa.core_attention", task="ATTN", layer_class=lc, op_type="attention",
         multiplicity_fn=lambda c, n=count: n,
         flops_fn=lambda c: (_mm(c["tokens"], D_C, N_H * C) +
                             4.0 * (N_H * C) * c["tokens"] * (TOPK + N_WIN)),
         weight_bytes_fn=lambda c: D_C * N_H * C * B_MAIN,
         io_bytes_fn=lambda c: (TOPK + N_WIN) * C * 2 * B_KV,
         precision=PRECISION_KV,
         note=f"MQA over top-{TOPK} selected compressed entries + {N_WIN}-token "
              f"sliding window, shared K/V across all {N_H} query heads.")
    _add(id="ATTN.csa.grouped_out_proj", task="ATTN", layer_class=lc, op_type="matmul",
         multiplicity_fn=lambda c, n=count: n,
         flops_fn=lambda c: _mm(c["tokens"], N_H * C, D_G) + _mm(c["tokens"], D_G * G, D_MODEL),
         weight_bytes_fn=lambda c: (N_H * C * D_G + D_G * G * D_MODEL) * B_MAIN,
         io_bytes_fn=lambda c: c["tokens"] * (N_H * C + D_MODEL) * B_MAIN,
         precision=PRECISION_MAIN,
         note=f"{G} groups -> {D_G}-dim intermediate -> combine to d_model "
              f"(SIMPLIFIED two-matmul stand-in for the per-group einsum).")

# ---- Attention: HCA layers (same pipeline minus indexer/top-k) ------------
for lc, count in [("hca_layer", N_HCA)]:
    _add(id="ATTN.hca.kv_compress", task="ATTN", layer_class=lc, op_type="matmul",
         multiplicity_fn=lambda c, n=count: n,
         flops_fn=lambda c: _mm(c["tokens"], D_MODEL, 2 * C),   # HCA: single-stream compress, 2 projections
         weight_bytes_fn=lambda c: D_MODEL * 2 * C * B_MAIN,
         io_bytes_fn=lambda c: c["tokens"] * (D_MODEL + 2 * C) * B_KV,
         precision=PRECISION_MAIN, note=f"Eqs 20-21, compression ratio m'={M_HCA}.")
    _add(id="ATTN.hca.core_attention", task="ATTN", layer_class=lc, op_type="attention",
         multiplicity_fn=lambda c, n=count: n,
         flops_fn=lambda c: (_mm(c["tokens"], D_C, N_H * C) +
                             4.0 * (N_H * C) * c["tokens"] * (c["L"] / M_HCA + N_WIN)),
         weight_bytes_fn=lambda c: D_C * N_H * C * B_MAIN,
         io_bytes_fn=lambda c: (c["L"] / M_HCA + N_WIN) * C * 2 * B_KV,
         precision=PRECISION_KV,
         note="No sparse selection -- dense MQA over ALL compressed entries "
              f"(L/{M_HCA}) plus the {N_WIN}-token sliding window.")
    _add(id="ATTN.hca.grouped_out_proj", task="ATTN", layer_class=lc, op_type="matmul",
         multiplicity_fn=lambda c, n=count: n,
         flops_fn=lambda c: _mm(c["tokens"], N_H * C, D_G) + _mm(c["tokens"], D_G * G, D_MODEL),
         weight_bytes_fn=lambda c: (N_H * C * D_G + D_G * G * D_MODEL) * B_MAIN,
         io_bytes_fn=lambda c: c["tokens"] * (N_H * C + D_MODEL) * B_MAIN,
         precision=PRECISION_MAIN, note="")

# ---- MoE FFN: shared expert (always fires, all 43 layers) -----------------
_add(id="MOE.shared_expert_ffn", task="MOE", layer_class="all_moe_layers", op_type="matmul",
     multiplicity_fn=lambda c: N_LAYERS * N_SHARED,
     flops_fn=lambda c: 6.0 * c["tokens"] * D_MODEL * EXPERT_WIDTH,   # gate+up+down, SwiGLU
     weight_bytes_fn=lambda c: 3 * D_MODEL * EXPERT_WIDTH * B_MAIN,
     io_bytes_fn=lambda c: c["tokens"] * (D_MODEL + EXPERT_WIDTH) * B_MAIN,
     precision=PRECISION_MAIN,
     note="Matches the paper's own stated per-expert cost exactly: 6*h*d "
          "FLOPs/token-expert-pair (Section 3.1) -- good cross-check on the "
          "general matmul-decomposition recipe.")

# ---- MoE FFN: routed experts, LEARNED routing (40 of 43 layers) -----------
_add(id="MOE.router_learned", task="MOE", layer_class="learned_routing_layers", op_type="matmul",
     multiplicity_fn=lambda c: N_LAYERS - N_HASH_ROUTED_LAYERS,
     flops_fn=lambda c: _mm(c["tokens"], D_MODEL, N_ROUTED),
     weight_bytes_fn=lambda c: D_MODEL * N_ROUTED * B_MAIN,
     io_bytes_fn=lambda c: c["tokens"] * (D_MODEL + N_ROUTED) * B_MAIN,
     precision=PRECISION_MAIN,
     note="THE real stochastic branch: affinity score per token over 256 "
          "routed experts, then top-6 selected. Needs a routing histogram "
          "from profiling to characterize which experts, not just how many.")
_add(id="MOE.router_learned_select", task="MOE", layer_class="learned_routing_layers", op_type="reduce_argmax",
     multiplicity_fn=lambda c: N_LAYERS - N_HASH_ROUTED_LAYERS,
     flops_fn=lambda c: c["tokens"] * N_ROUTED,
     weight_bytes_fn=lambda c: 0.0,
     io_bytes_fn=lambda c: N_ROUTED * 4 * BYTES_PER_ELEM["fp8"],
     precision="fp32", note="Top-6 selection over the affinity scores.")
_add(id="MOE.routed_expert_ffn_learned", task="MOE", layer_class="learned_routing_layers", op_type="matmul",
     multiplicity_fn=lambda c: (N_LAYERS - N_HASH_ROUTED_LAYERS) * EXPERTS_PER_TOK,
     flops_fn=lambda c: 6.0 * c["tokens"] * D_MODEL * EXPERT_WIDTH,
     weight_bytes_fn=lambda c: 3 * D_MODEL * EXPERT_WIDTH * BYTES_PER_ELEM["fp4"],  # QAT: routed weights FP4
     io_bytes_fn=lambda c: c["tokens"] * (D_MODEL + EXPERT_WIDTH) * B_MAIN,
     precision="fp4",
     note="6 of 256 experts fire per token, chosen by the router above. "
          "Weights are FP4 (post-training QAT, Section 5.2.1), dequantized "
          "to FP8 for the actual matmul -- modeled here at FP4 read bytes.")

# ---- MoE FFN: routed experts, HASH routing (first 3 layers) ---------------
_add(id="MOE.router_hash", task="MOE", layer_class="hash_routing_layers", op_type="lookup",
     multiplicity_fn=lambda c: N_HASH_ROUTED_LAYERS,
     flops_fn=lambda c: 0.0,
     weight_bytes_fn=lambda c: 0.0,
     io_bytes_fn=lambda c: c["tokens"] * 4,
     precision="int32",
     note="NOT stochastic: which experts a token hits is a deterministic "
          "hash of its token ID. Zero FLOPs, no profiling needed -- contrast "
          "directly with MOE.router_learned above, same layer TYPE, "
          "different (and non-probabilistic) selection mechanism.")
_add(id="MOE.routed_expert_ffn_hash", task="MOE", layer_class="hash_routing_layers", op_type="matmul",
     multiplicity_fn=lambda c: N_HASH_ROUTED_LAYERS * EXPERTS_PER_TOK,
     flops_fn=lambda c: 6.0 * c["tokens"] * D_MODEL * EXPERT_WIDTH,
     weight_bytes_fn=lambda c: 3 * D_MODEL * EXPERT_WIDTH * BYTES_PER_ELEM["fp4"],
     io_bytes_fn=lambda c: c["tokens"] * (D_MODEL + EXPERT_WIDTH) * B_MAIN,
     precision="fp4", note="Identical downstream compute to the learned-routing case.")

# ---- mHC: the op that doesn't fit the existing vocabulary -----------------
_add(id="MHC.sinkhorn_iterate", task="MHC", layer_class="all_layers", op_type="iterative_normalize",
     multiplicity_fn=lambda c: N_LAYERS,
     flops_fn=lambda c: 20 * 4 * 4 * 4.0,     # 20 iterations x row+col normalize on a 4x4 matrix
     weight_bytes_fn=lambda c: 0.0,
     io_bytes_fn=lambda c: 4 * 4 * 4 * BYTES_PER_ELEM["fp32"],   # the 4x4 matrix, resident on-chip
     precision="fp32",
     note="NOT bandwidth- or compute-bound -- bound by 20 sequential "
          "dependent steps. See explanation.txt for why this needs a new "
          "resource axis (sequential depth x round-trip latency) instead of "
          "the compute/bandwidth roofline every other kernel here uses.")

# ---- Embedding, LM head, MTP ------------------------------------------------
_add(id="OUT.embed_lookup", task="OUT", layer_class="embedding", op_type="lookup",
     multiplicity_fn=lambda c: 1, flops_fn=lambda c: 0.0,
     weight_bytes_fn=lambda c: c["tokens"] * D_MODEL * B_MAIN,
     io_bytes_fn=lambda c: c["tokens"] * D_MODEL * B_MAIN,
     precision=PRECISION_MAIN, note="")
_add(id="OUT.lm_head", task="OUT", layer_class="output", op_type="matmul",
     multiplicity_fn=lambda c: 1,
     flops_fn=lambda c: _mm(c["tokens"], D_MODEL, VOCAB_SIZE),
     weight_bytes_fn=lambda c: D_MODEL * VOCAB_SIZE * B_MAIN,
     io_bytes_fn=lambda c: c["tokens"] * (D_MODEL + VOCAB_SIZE) * B_MAIN,
     precision=PRECISION_MAIN, note="")
_add(id="OUT.mtp_extra_head", task="OUT", layer_class="mtp", op_type="matmul",
     multiplicity_fn=lambda c: MTP_DEPTH,
     flops_fn=lambda c: _mm(c["tokens"], D_MODEL, VOCAB_SIZE),
     weight_bytes_fn=lambda c: D_MODEL * VOCAB_SIZE * B_MAIN,
     io_bytes_fn=lambda c: c["tokens"] * (D_MODEL + VOCAB_SIZE) * B_MAIN,
     precision=PRECISION_MAIN,
     note="MTP depth=1: one extra lookahead-token prediction. This is what "
          "vLLM/SGLang attach speculative decoding to (branded 'DSpark' on "
          "the model card) -- reuse specdecode_kernels.py's B/C/D loop "
          "shape on TOP of this if modeling the full inference-serving DAG "
          "rather than just one forward pass.")


def bottleneck_summary(hw=GENERIC_CHIPLET) -> list[dict]:
    rows = []
    for regime, ctx in CONTEXT_REGIMES.items():
        for task in ["ATTN", "MOE", "MHC", "OUT"]:
            tks = [k for k in KERNELS if k.task == task]
            totals = [(k, k.expected_latency_ms(ctx, hw) * k.multiplicity(ctx)) for k in tks]
            totals.sort(key=lambda t: -t[1])
            task_total = sum(t for _, t in totals)
            top_k, top_ms = totals[0]
            bound = top_k.latency_scenarios(ctx, hw)[0][2]
            rows.append({
                "context_regime": regime, "task": task,
                "task_expected_latency_ms": f"{task_total:.4g}",
                "dominant_kernel": top_k.id,
                "dominant_kernel_share": f"{top_ms/task_total:.1%}" if task_total else "n/a",
                "dominant_bound_type": bound,
            })
    return rows


def write_bottleneck_csv(path: str = "deepseek_bottleneck_summary.csv") -> None:
    import csv
    rows = bottleneck_summary()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


OP_COLORS = {
    "matmul": "#2a78d6", "attention": "#eb6834", "norm": "#9a9890",
    "elementwise": "#c3c2b7", "lookup": "#1baf7a", "reduce_argmax": "#e34948",
    "iterative_normalize": "#4a3aa7",
}


def draw_kernel_figure(basename: str = "deepseek_dag_figure") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch

    ink, muted, accent = "#0b0b0b", "#52514e", "#2a78d6"
    ctx_ref = CONTEXT_REGIMES["short_chat_L1500"]

    def chain_for(task, group_order):
        chain = []
        for lc in group_order:
            lc_kernels = [k for k in KERNELS if k.task == task and k.layer_class == lc]
            if lc_kernels:
                chain.append((lc, lc_kernels))
        return chain

    rows = [
        ("ATTN: hybrid attention -- SWA (x2) then CSA/HCA interleaved (x41)",
         chain_for("ATTN", ["swa_only_layer", "csa_layer", "hca_layer"])),
        ("MOE: router (stochastic for 40 layers, deterministic hash for 3) -> "
         "1 shared + 6-of-256 routed expert FFNs, every layer",
         chain_for("MOE", ["all_moe_layers", "learned_routing_layers", "hash_routing_layers"])),
        ("MHC: residual mixing, a NEW op type (sequential-depth bound, not "
         "compute/bandwidth bound)",
         chain_for("MHC", ["all_layers"])),
        ("OUT: embedding, LM head, +1 MTP lookahead head (the DSpark speculative-decoding hook)",
         chain_for("OUT", ["embedding", "output", "mtp"])),
    ]

    fig, axes = plt.subplots(len(rows), 1, figsize=(17, 13))
    for ax, (title, chain) in zip(axes, rows):
        ax.set_xlim(0, 17); ax.set_ylim(0, 1.6); ax.axis("off")
        ax.set_title(title, fontsize=11, color=ink, loc="left", pad=4)
        x = 0.3
        for layer_class, lc_kernels in chain:
            mult = lc_kernels[0].multiplicity(ctx_ref)
            gx0 = x
            for k in lc_kernels:
                w = 1.75
                box = FancyBboxPatch((x, 0.35), w, 0.85, boxstyle="round,pad=0.04",
                                     fc=OP_COLORS.get(k.op_type, "#ccc"), ec="white", lw=1.0)
                ax.add_patch(box)
                short = k.id.split(".", 1)[-1].replace("_", "\n")
                ax.text(x + w / 2, 0.775, short, ha="center", va="center",
                        fontsize=6.6, color="white", fontweight="bold")
                x += w + 0.12
            gw = x - gx0 - 0.12
            ax.add_patch(plt.Rectangle((gx0 - 0.08, 0.28), gw + 0.16, 1.0,
                                       fill=False, ec=muted, lw=1.1, linestyle=(0, (3, 2))))
            ax.text(gx0 + gw / 2, 1.36, f"x {mult:g}  ({layer_class})",
                    ha="center", fontsize=8, color=muted, fontweight="bold")
            x += 0.28
        ax.set_xlim(0, max(x + 0.3, 4))

    handles = [Patch(fc=c, label=op) for op, c in OP_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=9, frameon=False,
              bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Kernel-level pipeline -- DeepSeek-V4-Flash (43 layers, 284B/13B activated)",
                fontsize=14, color=ink, y=0.995)
    fig.text(0.5, 0.965,
             "Contrast with Glimmer's kernel_dag_figure.png: attention is a multi-stage "
             "pipeline (not one fused op), and the MoE router is the one REAL structural "
             "branch across both workloads -- except the first 3 layers, which aren't stochastic at all.",
             ha="center", fontsize=8.8, color=muted)
    fig.tight_layout(rect=[0, 0.03, 1, 0.955])
    fig.savefig(f"{basename}.png", dpi=200, facecolor="white")
    fig.savefig(f"{basename}.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    write_kernel_table(KERNELS, CONTEXT_REGIMES["short_chat_L1500"],
                       "deepseek_kernel_table.csv", context_label="short_chat_L1500")
    write_latency_table(KERNELS, CONTEXT_REGIMES, "deepseek_latency_scenarios.csv")
    write_bottleneck_csv("deepseek_bottleneck_summary.csv")
    draw_kernel_figure("deepseek_dag_figure")
    print(f"[kernels]  {len(KERNELS)} kernels across ATTN/MOE/MHC/OUT")
    print("[files]    deepseek_kernel_table.csv, deepseek_latency_scenarios.csv, "
          "deepseek_bottleneck_summary.csv, deepseek_dag_figure.png/svg")
